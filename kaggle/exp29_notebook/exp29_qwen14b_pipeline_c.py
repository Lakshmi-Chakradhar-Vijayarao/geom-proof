"""
GEOM-PROOF -- Exp 29: Pipeline C (matched-methodology, real-generation
SQuAD-EM/F1 scoring) at Qwen2.5-14B-Instruct-AWQ, closing the paper's
own disclosed gap: "this is the Pipeline-A validation only, not the
matched-methodology (Pipeline C) scale curve... which remains fit on
four points (0.5B-7B) and does not include a 14B observation."

Identical protocol to Exp 15 (exp15_matched_generation.py): HaluEval,
n=500, bare "Q: ...\\nA:" template, greedy decode, SQuAD-style
normalized exact-match/F1 scoring (score_generation, copied verbatim),
last-token hidden states at every layer from a clean forward pass over
prompt+generation. The model is the full-precision Qwen2.5-14B-Instruct
checkpoint, quantized on load via bitsandbytes 4-bit -- not the
pre-quantized AWQ checkpoint used in the Pipeline-A validation and the
causal-intervention replication at this scale (Exp 20/21). Three
successive attempts to load the AWQ checkpoint via its required
gptqmodel backend each failed a different way (torch/CUDA-kernel
mismatch, huggingface_hub/transformers version skew, a missing
transitive dependency after pinning), all downstream of gptqmodel's
own dependency resolution having drifted since Exp 20/21 last used it
successfully. bitsandbytes is a stable, core-transformers-integrated
alternative that avoids this dependency chain entirely; this Pipeline-C
point uses a different 4-bit scheme than the causal-intervention
experiments as a result, disclosed here and at first use in the paper.

Outputs feed directly into this paper's existing certificate-validation
pipeline (Fisher J + AUROC bound via fisher_ratio_pca / auroc_bound,
identical to every other Pipeline-C table entry) -- computed here too,
so this is a genuine fifth point on the matched-methodology scale
curve, not just a raw generation log.
"""
import gc
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

try:
    import bitsandbytes  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import norm as scipy_norm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device -- Qwen2.5-14B needs a GPU.")
try:
    _a = torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")
    print(f"GPU sanity check passed: {torch.cuda.get_device_name(0)}")
except Exception as e:
    raise RuntimeError(f"GPU sanity check failed: {e}")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
N_SAMPLES = 500
MAX_NEW_TOKENS = 40
N_COMPONENTS = 100
RANDOM_STATE = 42
LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)


def normalize_answer(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, reference):
    p, r = normalize_answer(prediction), normalize_answer(reference)
    return bool(r) and (p == r or r in p)


def token_f1(prediction, reference):
    pt, rt = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = Counter(pt) & Counter(rt)
    num_same = sum(common.values())
    if not pt or not rt or num_same == 0:
        return 0.0
    prec, rec = num_same / len(pt), num_same / len(rt)
    return 2 * prec * rec / (prec + rec)


def score_generation(generated, right_answer):
    em = exact_match(generated, right_answer)
    f1 = token_f1(generated, right_answer)
    return (1 if (em or f1 >= 0.5) else 0), max(float(em), f1)


def load_halueval(n):
    ds = None
    for attempt in [
        lambda: load_dataset("HaluEval/halueval", "qa", split="data", trust_remote_code=True),
        lambda: load_dataset("pminervini/HaluEval", "qa_samples"),
        lambda: load_dataset("pminervini/HaluEval", "qa"),
    ]:
        try:
            ds = attempt()
            break
        except Exception as e:
            print(f"  HaluEval loader fallback: {e}")
    if ds is None:
        raise RuntimeError("Could not load HaluEval from any known path.")
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        ds = ds[list(ds.keys())[0]]

    Q_KEYS = ["question", "input", "query"]
    RIGHT_KEYS = ["right_answer", "answer", "correct_answer", "gold_answer"]
    HALL_KEYS = ["hallucinated_answer", "hallucination", "wrong_answer", "model_answer", "output", "negative"]

    def pick(row, keys):
        for k in keys:
            v = row.get(k, "")
            if v and str(v).strip():
                return str(v).strip()
        return ""

    out = []
    for row in ds:
        if len(out) >= n:
            break
        q, right, hall = pick(row, Q_KEYS), pick(row, RIGHT_KEYS), pick(row, HALL_KEYS)
        if q and right and hall and right != hall:
            out.append({"question": q, "right_answer": right, "hallucinated_answer": hall})
    print(f"  Loaded {len(out)} valid pairs from {ds.num_rows} rows")
    return out[:n]


def fisher_ratio_pca(H_L, y, k=N_COMPONENTS):
    k = min(k, H_L.shape[1], H_L.shape[0] - 2)
    H_L = H_L.astype(np.float64)
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    H_proj = pca.fit_transform(H_L)
    mu_c = H_proj[y == 1].mean(0)
    mu_h = H_proj[y == 0].mean(0)
    Sigma_w = LedoitWolf().fit(np.vstack([H_proj[y == 1] - mu_c, H_proj[y == 0] - mu_h])).covariance_
    diff = mu_c - mu_h
    try:
        J = float(diff @ np.linalg.solve(Sigma_w, diff))
    except np.linalg.LinAlgError:
        J = 0.0
    return max(J, 0.0)


def best_layer_probe(H, y):
    n, n_layers, d = H.shape
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    best_auroc, best_layer = 0.0, 0
    for L in range(n_layers):
        H_L = H[:, L, :]
        oof = np.zeros(n)
        for tr, te in skf.split(H_L, y):
            sc = StandardScaler()
            pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0] - 1, H_L.shape[1]), random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))
            clf = LogisticRegression(**LR_KWARGS)
            clf.fit(X_tr, y[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]
        auroc = float(roc_auc_score(y, oof))
        if auroc > best_auroc:
            best_auroc, best_layer = auroc, L
    return best_layer, best_auroc


def main():
    t0 = time.time()
    print(f"Loading {MODEL_ID} (bitsandbytes 4-bit)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Single-GPU placement still OOM'd (14.47/14.56GiB) at the identical
    # allocation point -- this is peak *transient* memory during weight
    # materialization/quantization itself exceeding one T4's capacity, not
    # a device-splitting bug. Fix: give accelerate explicit CPU headroom to
    # spill into during loading via max_memory + cpu offload, rather than
    # trying to fit the whole materialization process in 14.6GB of VRAM.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "12GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "28GiB"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", max_memory=max_memory,
        quantization_config=bnb_config, trust_remote_code=True)
    model.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    n_layers = model.config.num_hidden_layers
    print(f"{n_layers} layers, d={model.config.hidden_size}")

    samples = load_halueval(N_SAMPLES)
    print(f"{len(samples)} HaluEval samples loaded")

    all_hs, all_labels, all_scores, generations = [], [], [], []
    for i, sample in enumerate(samples):
        if i % 25 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(samples) - i) / 60
            print(f"  [{i}/{len(samples)}] elapsed={elapsed/60:.1f}min ETA={eta:.0f}min", flush=True)

        prompt = f"Q: {sample['question']}\nA:"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                gen_out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                          num_beams=1, pad_token_id=tok.pad_token_id,
                                          return_dict_in_generate=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at sample {i} -- skipping.")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        full_ids = gen_out.sequences[0]
        gen_ids = full_ids[prompt_len:]
        generated_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        label, score = score_generation(generated_text, sample["right_answer"])

        with torch.no_grad():
            fwd_out = model(input_ids=full_ids.unsqueeze(0).to("cuda"), output_hidden_states=True)
        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in fwd_out.hidden_states])

        all_hs.append(hs)
        all_labels.append(label)
        all_scores.append(score)
        generations.append({"question": sample["question"], "generated": generated_text,
                             "right_answer": sample["right_answer"], "label": label, "score": score})

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"\nGeneration complete: N={len(y)}, n_correct={int(y.sum())}, hall_rate={1-y.mean():.3f}")

    np.savez_compressed(OUT / "29_hidden_states_qwen14b.npz", hidden_states=H, labels=y)
    with open(OUT / "29_generations_qwen14b.json", "w") as f:
        json.dump(generations, f, indent=2)

    print("\nFitting certificate (best-layer probe + Fisher J + AUROC bound)...")
    best_layer, best_auroc = best_layer_probe(H, y)
    J = fisher_ratio_pca(H[:, best_layer, :], y)
    bound = float(scipy_norm.cdf(np.sqrt(max(J, 0.0) / 2)))
    print(f"Best layer: L{best_layer}  Probe AUROC: {best_auroc:.4f}  Fisher J: {J:.4f}  AUROC bound: {bound:.4f}  "
          f"Bound error: {bound - best_auroc:+.4f}")

    result = {
        "model": MODEL_ID, "quantization": "bitsandbytes-4bit-nf4",
        "n_params": "14.8B", "n_samples": int(len(y)),
        "hall_rate": float(1 - y.mean()), "n_layers": n_layers,
        "best_layer": best_layer, "probe_auroc": best_auroc,
        "fisher_J": J, "auroc_bound": bound, "bound_error": bound - best_auroc,
    }
    with open(OUT / "29_result_qwen14b.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT}/29_result_qwen14b.json")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
