"""
GEOM-PROOF -- Exp 32: third architecture family (Pythia / GPT-NeoX), resolving
Priority-2 item P2.1 from the frontier-lab review (FRONTIER_LAB_REVIEW_RESULT.md):
"All six models used throughout the paper are only two architecture families
(GPT-2 and Qwen2.5). A finding that holds across two families sharing similar
design lineage is weaker evidence of generality than a finding that also
holds on a third, independently-designed family. Add Llama or Pythia."

This is a DIRECT extension of the matched-methodology certificate protocol in
exp15_matched_generation.py -- identical prompt format, identical scoring
(SQuAD-style EM/F1 against HaluEval right_answer), identical Fisher-J /
corrected-AUROC-bound / 5-fold CV probe machinery -- restricted to three
Pythia checkpoints spanning a comparable parameter range to the paper's
existing GPT-2/Qwen sweep (410M / 1.4B / 2.8B). Pythia (EleutherAI, GPT-NeoX
architecture: parallel attention+MLP, rotary embeddings, no gating) is
architecturally distinct from both GPT-2 (learned positional embeddings,
sequential attention->MLP) and Qwen2.5 (Llama-style RMSNorm + SwiGLU + GQA),
and its checkpoints are fully open (no gated-access token needed, unlike
Llama), which is why it was chosen over Llama for this addition.

No chat template is used (matching exp15's raw "Q: ... \nA:" prompt format,
which exp15 applied uniformly even to the Qwen-Instruct models) since none of
these Pythia checkpoints are instruction-tuned.

Runtime estimate on T4: ~10-30 min per model (all three are <=2.8B), so the
whole script is expected to finish comfortably inside a single Kaggle GPU
session.

Outputs (download to results/logs/ and results/hidden_states/, matching
exp15's naming convention with a `32_` prefix instead of `15_`):
  32_third_architecture_pythia.json      -- consolidated results, all 3 models
  32_hidden_states_<model_key>.npz       -- per-model hidden states + labels
  32_generations_<model_key>.json        -- per-model generations + scores
"""

import gc
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

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
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

RESUME_DIR_CANDIDATES = [
    Path("/kaggle/input/exp32-third-architecture-pythia-run1"),
]
JSON_PATH = OUT / "32_third_architecture_pythia.json"

N_SAMPLES = 500
N_COMPONENTS = 100
N_SPLITS = 5
RANDOM_STATE = 42
MAX_NEW_TOKENS = 40

LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)

MODELS = {
    "pythia_410m": {"model_id": "EleutherAI/pythia-410m", "n_params": "410M"},
    "pythia_1.4b": {"model_id": "EleutherAI/pythia-1.4b", "n_params": "1.4B"},
    "pythia_2.8b": {"model_id": "EleutherAI/pythia-2.8b", "n_params": "2.8B"},
}


# ---- SQuAD-style scoring, identical to exp15_matched_generation.py ----

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, reference: str) -> bool:
    norm_pred = normalize_answer(prediction)
    norm_ref = normalize_answer(reference)
    if not norm_ref:
        return False
    return norm_pred == norm_ref or norm_ref in norm_pred


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_generation(generated: str, right_answer: str) -> tuple:
    em = exact_match(generated, right_answer)
    f1 = token_f1(generated, right_answer)
    correct = em or f1 >= 0.5
    return (1 if correct else 0), max(float(em), f1)


def load_halueval(n: int) -> list:
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
        q = pick(row, Q_KEYS)
        right = pick(row, RIGHT_KEYS)
        hall = pick(row, HALL_KEYS)
        if q and right and hall and right != hall:
            out.append({"question": q, "right_answer": right, "hallucinated_answer": hall})

    print(f"  Loaded {len(out)} valid pairs from {ds.num_rows} rows")
    return out[:n]


def run_model(model_key: str, cfg: dict, samples: list) -> dict:
    hs_path = OUT / f"32_hidden_states_{model_key}.npz"
    result_path = OUT / f"32_result_{model_key}.json"

    for cand_dir in RESUME_DIR_CANDIDATES:
        cand_hs = cand_dir / hs_path.name
        cand_res = cand_dir / result_path.name
        if cand_hs.exists() and cand_res.exists():
            print(f"\n[{model_key}] found completed result in {cand_dir} -- copying forward, skipping.")
            import shutil
            shutil.copy(cand_hs, hs_path)
            shutil.copy(cand_res, result_path)
            with open(result_path) as f:
                return json.load(f)

    if result_path.exists():
        print(f"\n[{model_key}] already completed this session -- skipping.")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'=' * 70}")
    print(f"Model: {model_key} ({cfg['model_id']})")
    print(f"{'=' * 70}")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    if DEVICE != "cuda":
        model = model.to(DEVICE)

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {n_layers} layers, d={d}")

    all_hs, all_labels, all_scores, generations = [], [], [], []
    skipped = 0

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(samples) - i) / 60
            print(f"  [{i:4d}/{len(samples)}] elapsed={elapsed / 60:.1f}min ETA={eta:.0f}min")

        prompt = f"Q: {sample['question']}\nA:"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256, padding=False)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tok.pad_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at sample {i} -- clearing cache and skipping.")
            torch.cuda.empty_cache()
            gc.collect()
            skipped += 1
            continue

        full_ids = gen_out.sequences[0]
        gen_ids = full_ids[prompt_len:]
        generated_text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        label, score = score_generation(generated_text, sample["right_answer"])

        with torch.no_grad():
            full_inputs = {"input_ids": full_ids.unsqueeze(0).to(DEVICE)}
            fwd_out = model(**full_inputs, output_hidden_states=True)
        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in fwd_out.hidden_states])

        all_hs.append(hs)
        all_labels.append(label)
        all_scores.append(score)
        generations.append({
            "question": sample["question"],
            "right_answer": sample["right_answer"],
            "generated": generated_text,
            "label": label,
            "score": round(score, 4),
        })

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    total_min = (time.time() - t0) / 60
    print(f"  Done in {total_min:.1f} min. Shape={H.shape} hall_rate={1 - y.mean():.3f} skipped={skipped}")

    np.savez_compressed(hs_path, hidden_states=H, labels=y)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    n_samp, n_lay, dd = H.shape
    J_per_layer, bound_per_layer = [], []
    for L in range(n_lay):
        H_L = H[:, L, :].astype(np.float64)
        k = min(N_COMPONENTS, H_L.shape[1], H_L.shape[0] - 2)
        pca = PCA(n_components=k, random_state=RANDOM_STATE)
        H_proj = pca.fit_transform(H_L)
        mu_c = H_proj[y == 1].mean(0) if (y == 1).any() else np.zeros(k)
        mu_h = H_proj[y == 0].mean(0) if (y == 0).any() else np.zeros(k)
        H_c = H_proj[y == 1] - mu_c
        H_h = H_proj[y == 0] - mu_h
        if len(H_c) < 2 or len(H_h) < 2:
            J_per_layer.append(0.0)
            bound_per_layer.append(0.5)
            continue
        Sigma_w = LedoitWolf().fit(np.vstack([H_c, H_h])).covariance_
        diff = mu_c - mu_h
        try:
            w = np.linalg.solve(Sigma_w, diff)
            J = float(diff @ w)
        except np.linalg.LinAlgError:
            J = float(diff @ np.linalg.lstsq(Sigma_w, diff, rcond=None)[0])
        J = max(J, 0.0)
        bound = float(scipy_norm.cdf(np.sqrt(J / 2)))
        J_per_layer.append(J)
        bound_per_layer.append(bound)

    linear_aurocs = []
    if (y == 1).sum() >= N_SPLITS and (y == 0).sum() >= N_SPLITS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for L in range(n_lay):
            H_L = H[:, L, :]
            oof = np.zeros(n_samp)
            for tr, te in skf.split(H_L, y):
                sc = StandardScaler()
                pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0] - 1, H_L[tr].shape[1]),
                          random_state=RANDOM_STATE)
                X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
                X_te = pca.transform(sc.transform(H_L[te]))
                clf = LogisticRegression(**LR_KWARGS)
                clf.fit(X_tr, y[tr])
                oof[te] = clf.predict_proba(X_te)[:, 1]
            linear_aurocs.append(float(roc_auc_score(y, oof)))
    else:
        print("  WARNING: class too imbalanced for 5-fold CV -- probe AUROC skipped.")
        linear_aurocs = [float("nan")] * n_lay

    best_J_layer = int(np.argmax(J_per_layer))
    valid_probe = [a for a in linear_aurocs if not np.isnan(a)]
    best_probe_layer = int(np.argmax(linear_aurocs)) if valid_probe else None
    best_probe_auroc = float(np.max(valid_probe)) if valid_probe else None

    result = {
        "model_key": model_key,
        "model_id": cfg["model_id"],
        "n_params": cfg["n_params"],
        "architecture_family": "pythia_gpt_neox",
        "n_samples": int(n_samp),
        "n_layers": int(n_lay),
        "d": int(dd),
        "hall_rate": float(1 - y.mean()),
        "n_correct": int(y.sum()),
        "skipped": skipped,
        "J_per_layer": J_per_layer,
        "auroc_bound_per_layer": bound_per_layer,
        "best_J": float(J_per_layer[best_J_layer]),
        "best_J_layer": best_J_layer,
        "best_auroc_bound": bound_per_layer[best_J_layer],
        "best_depth_fraction_J": best_J_layer / (n_lay - 1),
        "linear_auroc_per_layer": linear_aurocs,
        "best_probe_auroc": best_probe_auroc,
        "best_probe_layer": best_probe_layer,
        "best_depth_fraction_probe": (best_probe_layer / (n_lay - 1)) if best_probe_layer is not None else None,
        "bound_error_at_best_J_layer": abs(bound_per_layer[best_J_layer] - linear_aurocs[best_J_layer]) if not np.isnan(linear_aurocs[best_J_layer]) else None,
        "total_runtime_min": round(total_min, 1),
        "methodology": "real_generation_squad_em_f1",
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(OUT / f"32_generations_{model_key}.json", "w") as f:
        json.dump(generations, f, indent=2)

    print(f"  Saved: {result_path.name}, {hs_path.name}")
    print(f"  hall_rate={result['hall_rate']:.3f}  best_J={result['best_J']:.2f} @ L{best_J_layer}  "
          f"best_bound={result['best_auroc_bound']:.4f}  best_probe_auroc={best_probe_auroc}")

    return result


def main():
    t_start = time.time()
    print("=" * 70)
    print("Exp 32 -- Third Architecture Family: Pythia (410M / 1.4B / 2.8B)")
    print("=" * 70)

    print("\nLoading HaluEval ...")
    samples = load_halueval(N_SAMPLES)
    print(f"  {len(samples)} QA pairs loaded.")

    all_results = {}
    if JSON_PATH.exists():
        with open(JSON_PATH) as f:
            all_results = json.load(f)

    for model_key, cfg in MODELS.items():
        all_results[model_key] = run_model(model_key, cfg, samples)
        with open(JSON_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Checkpoint saved: {JSON_PATH}")

    total_min = (time.time() - t_start) / 60
    print(f"\nAll models done in {total_min:.1f} min total.")
    print("=== SUMMARY: does the certificate hold on a third architecture family? ===")
    for k, r in all_results.items():
        print(f"  {k} ({r['n_params']}): best_bound={r['best_auroc_bound']:.4f}  "
              f"best_probe_auroc={r['best_probe_auroc']}  "
              f"bound_error={r['bound_error_at_best_J_layer']}")


if __name__ == "__main__":
    main()
