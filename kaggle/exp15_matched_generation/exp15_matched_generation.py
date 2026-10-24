"""
GEOM-PROOF — Kaggle Exp 15: Matched-Methodology Generation Certificate
========================================================================

WHY THIS EXPERIMENT EXISTS
---------------------------
Second-round adversarial review found that the paper's "in-distribution"
numbers (Table 1, Table 4 scale curve, the Wasserstein crossover in Sec 6)
were never actually comparable across models:

  - Qwen 2.5 3B / GPT-2 Medium / Qwen 2.5 7B (Exp 00): hidden states extracted
    via NO-GENERATION forced-choice -- the model is simply forward-passed on
    the pre-written correct answer string (label=1) or the pre-written
    hallucinated answer string (label=0). No generation, no ROUGE-L, an
    artificially easy template-injection task.
  - Qwen 2.5 0.5B / 1.5B (Exp 11): hidden states extracted from REAL
    generation on TruthfulQA, scored via ROUGE-L >= 0.4 (near-degenerate:
    n_correct = 8/400 and 4/400 respectively).
  - The paper's "Scope Condition" (Sec 3.2) claims ALL results condition on
    ROUGE-L labels -- false for the Exp 00 models, whose labels are balanced
    by construction (497/497 confident splits) and have nothing to do with
    ROUGE-L at all.

This is not a labeling technicality: forced-choice injection and real
generation are different TASKS. Comparing them as "in-distribution" points
on the same scale curve, or as the "in-distribution" half of an
in-distribution-vs-OOD collapse story, is an item-difficulty confound.

WHAT THIS EXPERIMENT DOES
---------------------------
One consistent pipeline, all 6 models, real generation, real scoring:

  1. Load HaluEval QA pairs (question + right_answer + hallucinated_answer).
  2. For each of N_SAMPLES questions, GENERATE a real completion given ONLY
     the question (never shown right_answer or hallucinated_answer) --
     prompt: "Q: {question}\nA:", greedy decoding, max_new_tokens=40.
  3. Score the generation against right_answer using SQuAD-style normalized
     exact-match / token-F1 (same methodology already validated and used to
     fix Paper 3's labeling artifact -- more robust than a single ROUGE-L
     threshold, which produces near-degenerate labels on short answers).
     label = 1 (correct) if EM or F1 >= 0.5, else 0 (hallucinated).
  4. Extract last-token hidden states of the GENERATED completion at every
     layer (not of injected reference text).
  5. Compute Fisher J(L), the corrected AUROC bound Phi(sqrt(J/2)), and
     5-fold OOF linear-probe AUROC at every layer -- identical statistical
     machinery to Exp 01/08/11/13, so the ONLY thing that changes relative
     to those experiments is that every model now sits on the same task.

Models (same 6 used across the paper): GPT-2 (117M), GPT-2 Medium (345M),
Qwen 2.5 0.5B / 1.5B / 3B / 7B (all -Instruct where applicable).

Runtime estimate on T4: ~3-5 hours total (small models ~10-20 min each,
7B ~60-90 min). Per-model checkpointing: each model's results are saved to
/kaggle/working/ immediately after that model finishes, so a session
timeout does not lose completed models -- rerun with the same
kernel_sources self-reference to resume (see RESUME logic below).

Outputs (download to results/logs/ and results/hidden_states/):
  15_matched_generation.json          -- consolidated results, all 6 models
  15_hidden_states_<model_key>.npz    -- per-model hidden states + labels
"""

import os, sys, json, gc, time, re, string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.stats import norm as scipy_norm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

# Resume support: if a previous run's output was attached as a kernel_source,
# copy already-completed model results forward so we don't redo them.
RESUME_DIR_CANDIDATES = [
    Path("/kaggle/input/exp15-matched-generation-run1"),
    Path("/kaggle/input/exp15-matched-generation-run2"),
]
JSON_PATH = OUT / "15_matched_generation.json"

N_SAMPLES    = 500      # QA pairs -> 500 real generations per model
N_COMPONENTS = 100
N_SPLITS     = 5
RANDOM_STATE = 42
MAX_NEW_TOKENS = 40

LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)

MODELS = {
    "gpt2_117m":     {"model_id": "gpt2",                          "n_params": "117M"},
    "gpt2_medium":   {"model_id": "openai-community/gpt2-medium",  "n_params": "345M"},
    "qwen_0.5b":     {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",    "n_params": "494M"},
    "qwen_1.5b":     {"model_id": "Qwen/Qwen2.5-1.5B-Instruct",    "n_params": "1.54B"},
    "qwen_3b":       {"model_id": "Qwen/Qwen2.5-3B-Instruct",      "n_params": "3.09B"},
    "qwen_7b":       {"model_id": "Qwen/Qwen2.5-7B-Instruct",      "n_params": "7.6B"},
}


# ─────────────────────────────────────────────────────────────────────────────
# SQuAD-style scoring (identical methodology to papers-2026 Paper 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


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


# ─────────────────────────────────────────────────────────────────────────────
# HaluEval loader (same as Exp 00 / Exp 13)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Per-model: generate, score, extract hidden states
# ─────────────────────────────────────────────────────────────────────────────

def run_model(model_key: str, cfg: dict, samples: list) -> dict:
    hs_path = OUT / f"15_hidden_states_{model_key}.npz"
    result_path = OUT / f"15_result_{model_key}.json"

    # Resume: check prior-run kernel_sources for already-completed results
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

    print(f"\n{'='*70}")
    print(f"Model: {model_key} ({cfg['model_id']})")
    print(f"{'='*70}")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
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
            print(f"  [{i:4d}/{len(samples)}] elapsed={elapsed/60:.1f}min ETA={eta:.0f}min")

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

        # Last-token hidden states at every layer, from a clean forward pass
        # over the full (prompt + generated) sequence -- avoids relying on
        # generate()'s incremental hidden-state cache structure.
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
    print(f"  Done in {total_min:.1f} min. Shape={H.shape} hall_rate={1-y.mean():.3f} skipped={skipped}")

    np.savez_compressed(hs_path, hidden_states=H, labels=y)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Fisher curve + corrected AUROC bound + linear probe ─────────────────
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
        bound = float(scipy_norm.cdf(np.sqrt(J / 2)))   # corrected binormal AUROC identity
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
                pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0]-1, H_L[tr].shape[1]),
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
    with open(OUT / f"15_generations_{model_key}.json", "w") as f:
        json.dump(generations, f, indent=2)

    print(f"  Saved: {result_path.name}, {hs_path.name}")
    print(f"  hall_rate={result['hall_rate']:.3f}  best_J={result['best_J']:.2f} @ L{best_J_layer}  "
          f"best_bound={result['best_auroc_bound']:.4f}  best_probe_auroc={best_probe_auroc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("="*70)
    print("Exp 15 -- Matched-Methodology Generation Certificate (all 6 models)")
    print("="*70)

    print("\nLoading HaluEval ...")
    samples = load_halueval(N_SAMPLES)
    print(f"  {len(samples)} QA pairs loaded.")

    all_results = {}
    if JSON_PATH.exists():
        with open(JSON_PATH) as f:
            all_results = json.load(f)

    for model_key, cfg in MODELS.items():
        if model_key in all_results:
            print(f"\n[{model_key}] already in consolidated JSON -- skipping.")
            continue
        try:
            r = run_model(model_key, cfg, samples)
            all_results[model_key] = r
        except Exception as e:
            print(f"\n[{model_key}] FAILED: {e}")
            all_results[model_key] = {"error": str(e)}

        with open(JSON_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Checkpoint saved -> {JSON_PATH}")

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"ALL MODELS DONE. Total runtime: {total_min:.1f} min")
    print(f"{'='*70}")
    for k, r in all_results.items():
        if "error" in r:
            print(f"  {k:15s}  ERROR: {r['error']}")
        else:
            print(f"  {k:15s}  hall_rate={r['hall_rate']:.3f}  best_J={r['best_J']:.2f}  "
                  f"bound={r['best_auroc_bound']:.4f}  probe_auroc={r['best_probe_auroc']}")

    print("\nFILES TO DOWNLOAD from /kaggle/working/:")
    for p in sorted(OUT.glob("15_*")):
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

    return all_results


if __name__ == "__main__":
    main()
