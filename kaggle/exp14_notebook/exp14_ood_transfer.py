"""
GEOM-PROOF — Kaggle Exp 14: OOD Transfer Test (Qwen 2.5 7B + GPT-2 Medium)
============================================================================

Tests whether the conformal guarantee calibrated on HaluEval transfers to
TruthfulQA — the same OOD test done for Qwen 3B in Exp 10, now extended
to Qwen 7B and GPT-2 Medium.

Design (identical to Exp 10 OOD section):
  - Train probe on HaluEval (Exp 13 / Exp 00 hidden states)
  - Calibrate conformal threshold on HaluEval calibration split
  - Apply threshold to TruthfulQA hidden states
  - Measure: AUROC collapse, conformal violation rate

For Qwen 7B: needs fresh TruthfulQA extraction (no GPU generation —
  forward pass only on question+answer strings from TruthfulQA dataset,
  ROUGE-L labeled).
For GPT-2 Medium: TruthfulQA hidden states already exist as
  qwen3b_truthfulqa_ood.npz... wait, that's Qwen 3B only.
  GPT-2 Med TruthfulQA needs extraction too.

Both extractions are fast on T4: GPT-2 Med ~3 min, Qwen 7B ~8 min.

Outputs:
  /kaggle/working/14_ood_transfer.json
  /kaggle/working/14_ood_transfer.png
  /kaggle/working/14_truthfulqa_qwen7b.npz
  /kaggle/working/14_truthfulqa_gpt2med.npz

HOW TO USE:
  1. New Kaggle notebook, GPU T4 x1
  2. Add previous Exp 13 output as data source (for HaluEval 7B states):
       chakradharvijayarao/geom-proof-exp13-qwen7b
  3. Cell 1: !pip install -q transformers datasets accelerate scikit-learn rouge-score
  4. Cell 2: paste this script
  5. Runtime: ~30-40 min total
"""

import os, json, gc, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_lib

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

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

# Input: HaluEval hidden states from prior experiments
# Exp 13 output for 7B, uploaded as dataset
HALUEVAL_7B_PATH  = Path("/kaggle/input/geom-proof-exp13-qwen7b/13_halueval_qwen7b.npz")
HALUEVAL_3B_PATH  = Path("/kaggle/input/geom-proof-extraction-all-gpu-jobs/00_halueval_qwen3b.npz")
HALUEVAL_GPT2_PATH = Path("/kaggle/input/geom-proof-extraction-all-gpu-jobs/00_halueval_gpt2med.npz")

# Fallback: if input datasets not attached, use working dir (re-extract)
if not HALUEVAL_7B_PATH.exists():
    HALUEVAL_7B_PATH = OUT / "13_halueval_qwen7b.npz"
if not HALUEVAL_GPT2_PATH.exists():
    HALUEVAL_GPT2_PATH = OUT / "00_halueval_gpt2med.npz"

N_COMPONENTS  = 100
N_SPLITS      = 5
RANDOM_STATE  = 42
ROUGE_THRESH  = 0.4
N_TRUTHFULQA  = 400

LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)

# ── Shared utilities ───────────────────────────────────────────────────────────

def load_truthfulqa(n: int) -> list[dict]:
    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for row in ds:
        if len(out) >= n:
            break
        correct = [a for a in row["correct_answers"] if a.strip()]
        if correct:
            out.append({"question": row["question"],
                        "correct_answers": correct})
    return out[:n]


def rouge_label(response: str, correct_answers: list) -> int:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return int(max(
        scorer.score(ref, response)["rougeL"].fmeasure
        for ref in correct_answers
    ) >= ROUGE_THRESH)


def extract_truthfulqa_hs(model_id: str, out_path: Path,
                           dtype=torch.float16) -> None:
    """Extract TruthfulQA hidden states via generation + ROUGE-L labeling."""
    if out_path.exists():
        print(f"  {out_path.name} already exists — skipping.")
        return

    print(f"\n  Loading {model_id} for TruthfulQA extraction ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {n_layers} layers, d={d}")

    samples = load_truthfulqa(N_TRUTHFULQA)
    scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    all_hs, all_labels, skipped = [], [], 0

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"    [{i}/{len(samples)}]")
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        response = tok.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        label = rouge_label(response, sample["correct_answers"])

        # Hidden states at last prompt token
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        if out.hidden_states is None:
            skipped += 1
            continue

        hs = np.stack([h[0, -1, :].float().cpu().numpy()
                       for h in out.hidden_states])
        all_hs.append(hs)
        all_labels.append(label)

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"  Shape: {H.shape}  hall_rate={1-y.mean():.3f}  skipped={skipped}")
    np.savez_compressed(out_path, hidden_states=H, labels=y)
    print(f"  Saved: {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Fisher + probe helpers ─────────────────────────────────────────────────────

def fisher_ratio_pca(H_L, y, k=100):
    k = min(k, H_L.shape[1], H_L.shape[0] - 2)
    H_L = H_L.astype(np.float64)
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    H_proj = pca.fit_transform(H_L)
    mu_c = H_proj[y == 1].mean(0)
    mu_h = H_proj[y == 0].mean(0)
    Sigma_w = LedoitWolf().fit(
        np.vstack([H_proj[y==1] - mu_c, H_proj[y==0] - mu_h])
    ).covariance_
    diff = mu_c - mu_h
    try:
        J = float(diff @ np.linalg.solve(Sigma_w, diff))
    except np.linalg.LinAlgError:
        J = 0.0
    return max(J, 0.0)


def best_layer_probe(H_train, y_train):
    """Find best layer by 5-fold OOF AUROC on training data."""
    n, n_layers, d = H_train.shape
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_auroc, best_layer = 0.0, 0
    for L in range(n_layers):
        H_L = H_train[:, L, :]
        oof = np.zeros(n)
        for tr, te in skf.split(H_L, y_train):
            sc  = StandardScaler()
            pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0]-1,
                                        H_L.shape[1]), random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))
            clf  = LogisticRegression(**LR_KWARGS)
            clf.fit(X_tr, y_train[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]
        auroc = float(roc_auc_score(y_train, oof))
        if auroc > best_auroc:
            best_auroc, best_layer = auroc, L
    return best_layer, best_auroc


def ood_transfer_test(H_train, y_train, H_ood, y_ood, model_name):
    """
    Train probe on H_train (HaluEval), calibrate threshold,
    apply to H_ood (TruthfulQA), report AUROC + conformal violation.
    """
    print(f"\n  OOD transfer: {model_name}")
    n_train = len(y_train)
    n_layers_train = H_train.shape[1]
    n_layers_ood   = H_ood.shape[1]

    # Use min layers (in case architectures differ)
    n_layers = min(n_layers_train, n_layers_ood)

    # Step 1: find best layer on training data
    print(f"    Finding best probe layer on HaluEval ({n_train} samples) ...")
    best_layer, best_id_auroc = best_layer_probe(H_train, y_train)
    print(f"    Best layer: L{best_layer}  in-dist AUROC: {best_id_auroc:.4f}")

    # Step 2: train final probe on full training set
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(n_train)
    n_tr  = int(0.75 * n_train)
    tr_idx  = idx[:n_tr]
    cal_idx = idx[n_tr:]

    H_L_train = H_train[:, best_layer, :]
    sc  = StandardScaler()
    pca = PCA(n_components=min(N_COMPONENTS, H_L_train[tr_idx].shape[0]-1,
                                H_L_train.shape[1]), random_state=RANDOM_STATE)
    X_tr  = pca.fit_transform(sc.fit_transform(H_L_train[tr_idx]))
    X_cal = pca.transform(sc.transform(H_L_train[cal_idx]))

    clf = LogisticRegression(**LR_KWARGS)
    clf.fit(X_tr, y_train[tr_idx])

    # Step 3: calibrate conformal threshold on HaluEval calibration split
    scores_cal = clf.predict_proba(X_cal)[:, 1]
    best_alpha, alpha_star, accept_rate_id, emp_hall_id = None, None, None, None
    for alpha in np.arange(0.01, 0.50, 0.01):
        thresh   = np.quantile(scores_cal[y_train[cal_idx] == 0], 1 - alpha)
        accepted = scores_cal > thresh
        if accepted.sum() == 0:
            continue
        hall_rate = float((y_train[cal_idx][accepted] == 0).mean())
        if hall_rate <= alpha:
            alpha_star     = float(round(alpha, 2))
            accept_rate_id = float(accepted.mean())
            emp_hall_id    = hall_rate
            thresh_star    = thresh
            break

    print(f"    Conformal α*={alpha_star}  accept_rate={accept_rate_id:.3f}"
          f"  emp_hall={emp_hall_id:.4f}")

    # Step 4: apply to OOD (TruthfulQA) — same layer, same probe
    # Handle case where OOD has fewer layers
    ood_layer = min(best_layer, n_layers_ood - 1)
    H_L_ood = H_ood[:, ood_layer, :]
    X_ood   = pca.transform(sc.transform(H_L_ood))

    scores_ood = clf.predict_proba(X_ood)[:, 1]

    # OOD AUROC
    try:
        ood_auroc = float(roc_auc_score(y_ood, scores_ood))
    except ValueError:
        ood_auroc = float("nan")

    # OOD conformal violation
    if alpha_star is not None:
        accepted_ood = scores_ood > thresh_star
        n_accepted_ood = accepted_ood.sum()
        if n_accepted_ood > 0:
            emp_hall_ood = float((y_ood[accepted_ood] == 0).mean())
            accept_rate_ood = float(accepted_ood.mean())
            violation_factor = emp_hall_ood / alpha_star if alpha_star > 0 else float("nan")
        else:
            emp_hall_ood = float("nan")
            accept_rate_ood = 0.0
            violation_factor = float("nan")
    else:
        emp_hall_ood = accept_rate_ood = violation_factor = float("nan")

    print(f"    OOD AUROC:          {ood_auroc:.4f}  (in-dist: {best_id_auroc:.4f})")
    print(f"    OOD hall rate:      {emp_hall_ood:.4f}  (α*={alpha_star})")
    print(f"    OOD accept rate:    {accept_rate_ood:.3f}")
    print(f"    Violation factor:   {violation_factor:.1f}×  "
          f"({'COLLAPSE' if violation_factor > 3 else 'holds'})")

    return {
        "model":              model_name,
        "best_layer":         best_layer,
        "in_dist_auroc":      best_id_auroc,
        "alpha_star":         alpha_star,
        "in_dist_accept_rate": accept_rate_id,
        "in_dist_emp_hall":   emp_hall_id,
        "ood_auroc":          ood_auroc,
        "ood_emp_hall":       emp_hall_ood,
        "ood_accept_rate":    accept_rate_ood,
        "violation_factor":   violation_factor,
        "ood_collapse":       violation_factor > 3 if not np.isnan(violation_factor) else None,
        "auroc_drop":         float(best_id_auroc - ood_auroc),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "="*60)
    print("Exp 14 — OOD Transfer: HaluEval → TruthfulQA")
    print("Models: Qwen 2.5 7B + GPT-2 Medium")
    print("="*60)

    results = {}

    # ── Job A: Extract TruthfulQA hidden states ───────────────────────────────
    print("\n" + "="*60)
    print("Job A: TruthfulQA extraction")
    print("="*60)

    tqa_7b_path   = OUT / "14_truthfulqa_qwen7b.npz"
    tqa_gpt2_path = OUT / "14_truthfulqa_gpt2med.npz"

    extract_truthfulqa_hs("Qwen/Qwen2.5-7B-Instruct",
                           tqa_7b_path, dtype=torch.float16)
    extract_truthfulqa_hs("openai-community/gpt2-medium",
                           tqa_gpt2_path, dtype=torch.float32)

    # ── Job B: OOD transfer tests ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("Job B: OOD transfer analysis")
    print("="*60)

    # Qwen 2.5 7B
    if HALUEVAL_7B_PATH.exists() and tqa_7b_path.exists():
        H_id  = np.load(HALUEVAL_7B_PATH)["hidden_states"].astype(np.float32)
        y_id  = np.load(HALUEVAL_7B_PATH)["labels"].astype(int)
        H_ood = np.load(tqa_7b_path)["hidden_states"].astype(np.float32)
        y_ood = np.load(tqa_7b_path)["labels"].astype(int)
        print(f"\n  7B  in-dist: {H_id.shape}  hall={1-y_id.mean():.3f}")
        print(f"  7B  OOD:     {H_ood.shape}  hall={1-y_ood.mean():.3f}")
        results["Qwen 2.5 7B"] = ood_transfer_test(
            H_id, y_id, H_ood, y_ood, "Qwen 2.5 7B"
        )
    else:
        print("  Qwen 7B: missing hidden states, skipping.")

    # GPT-2 Medium
    if HALUEVAL_GPT2_PATH.exists() and tqa_gpt2_path.exists():
        H_id  = np.load(HALUEVAL_GPT2_PATH)["hidden_states"].astype(np.float32)
        y_id  = np.load(HALUEVAL_GPT2_PATH)["labels"].astype(int)
        H_ood = np.load(tqa_gpt2_path)["hidden_states"].astype(np.float32)
        y_ood = np.load(tqa_gpt2_path)["labels"].astype(int)
        print(f"\n  GPT-2 Med in-dist: {H_id.shape}  hall={1-y_id.mean():.3f}")
        print(f"  GPT-2 Med OOD:     {H_ood.shape}  hall={1-y_ood.mean():.3f}")
        results["GPT-2 Medium"] = ood_transfer_test(
            H_id, y_id, H_ood, y_ood, "GPT-2 Medium"
        )
    else:
        print("  GPT-2 Medium: missing hidden states, skipping.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results["total_runtime_min"] = round((time.time() - t0) / 60, 1)
    out_json = OUT / "14_ood_transfer.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, r in results.items():
        if not isinstance(r, dict):
            continue
        print(f"\n  {model}:")
        print(f"    In-dist AUROC:   {r['in_dist_auroc']:.4f}")
        print(f"    OOD AUROC:       {r['ood_auroc']:.4f}  "
              f"(drop: {r['auroc_drop']:.4f})")
        print(f"    Violation:       {r['violation_factor']:.1f}×  "
              f"({'COLLAPSE' if r.get('ood_collapse') else 'holds'})")

    print(f"\nTotal runtime: {results['total_runtime_min']} min")

    print("\nFILES TO DOWNLOAD:")
    for f in [out_json,
              OUT/"14_ood_transfer.png",
              OUT/"14_truthfulqa_qwen7b.npz",
              OUT/"14_truthfulqa_gpt2med.npz"]:
        if Path(f).exists():
            print(f"  {Path(f).name:45s}  {Path(f).stat().st_size/1e6:.1f} MB")


def _plot(results: dict) -> None:
    models = [k for k, v in results.items() if isinstance(v, dict) and "ood_auroc" in v]
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: AUROC in-dist vs OOD
    ax = axes[0]
    x = np.arange(len(models))
    id_aurocs  = [results[m]["in_dist_auroc"] for m in models]
    ood_aurocs = [results[m]["ood_auroc"] for m in models]
    bars1 = ax.bar(x - 0.2, id_aurocs,  0.35, label="In-dist (HaluEval)",
                   color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + 0.2, ood_aurocs, 0.35, label="OOD (TruthfulQA)",
                   color="tomato", alpha=0.85)
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.set_title("AUROC: In-Distribution vs OOD", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    for bar, val in zip(bars1, id_aurocs):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, ood_aurocs):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Right: Violation factor
    ax2 = axes[1]
    vf = [results[m]["violation_factor"] for m in models]
    alpha_stars = [results[m]["alpha_star"] for m in models]
    colors = ["tomato" if v > 3 else "steelblue" for v in vf]
    bars3 = ax2.bar(models, vf, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="green", ls="--", lw=1.5,
                label="Guarantee threshold (1×)")
    ax2.axhline(3.0, color="orange", ls=":", lw=1.5,
                label="Collapse threshold (3×)")
    ax2.set_ylabel("Violation factor (emp_hall / α*)", fontsize=10)
    ax2.set_title("Conformal Guarantee Violation Under OOD Shift", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    for bar, val, a in zip(bars3, vf, alpha_stars):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.1,
                 f"{val:.1f}×\n(α*={a})", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Exp 14: OOD Transfer — HaluEval (in-dist) → TruthfulQA (OOD)\n"
        "Conformal guarantee calibrated on HaluEval, applied to TruthfulQA",
        fontsize=11
    )
    plt.tight_layout()
    out = OUT / "14_ood_transfer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}")


if __name__ == "__main__":
    main()
