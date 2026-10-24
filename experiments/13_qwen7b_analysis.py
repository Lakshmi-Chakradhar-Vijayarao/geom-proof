"""
Exp 13 — Qwen 2.5 7B: Full Certificate Validation

Runs the complete analysis pipeline on Qwen 2.5 7B — the largest model in the
study — using the same balanced HaluEval extraction as Exp 00 (1000 QA pairs →
2000 balanced hidden-state vectors, 50/50 classes, no ROUGE noise).

This directly addresses the main reviewer concern: "Does the Fisher certificate
hold beyond ~3B parameters?"

Pipeline (zero new datasets, same design choices throughout):
  1. Extract hidden states:  same as Exp 00 (balanced HaluEval, last-token, all layers)
  2. Fisher certificate:     same as Exp 01 (PCA-100, Ledoit-Wolf, Φ(√J/2))
  3. Linear probe:           same as Exp 01 (LR C=1.0, 5-fold StratifiedKFold)
  4. MLP probe:              same as Exp 12 (64 hidden, ReLU, Adam, early stopping)
  5. Conformal coverage:     same as Exp 10 (60/20/20 split, α* calibration)
  6. Scale update:           adds 7B actual AUROC to the scale curve table

Model: Qwen/Qwen2.5-7B-Instruct
  - 28 transformer layers + embedding = 29 layers
  - d = 3584, params ≈ 7.6B
  - Runs in bfloat16 to fit in ~14GB RAM (use 'cpu' device_map)

Outputs:
  results/hidden_states/13_halueval_qwen7b.npz
  results/logs/13_qwen7b_analysis.json
  results/plots/13_qwen7b_certificate.png

Cost: CPU-only, ~4–8 hours for extraction, ~20 min for analysis.
      On M2/M3 Mac with 16GB+: feasible. On 8GB: use n=400 (set N_SAMPLES=200).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import fisher_curve

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.special import ndtr

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR   = ROOT / "results" / "plots"
HS_DIR      = ROOT / "results" / "hidden_states"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
HS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
HS_PATH      = HS_DIR / "13_halueval_qwen7b.npz"
N_SAMPLES    = 1000   # pairs → 2000 balanced hidden-state vectors (same as Exp 00)
                      # Reduce to 200 on 8GB RAM machines (→ 400 vectors)
N_COMPONENTS = 100    # PCA — same as all other experiments
N_SPLITS     = 5      # 5-fold CV — same as all other experiments
RANDOM_STATE = 42

MLP_KWARGS = dict(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    tol=1e-4,
)
LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)


# ── Step 1: Extraction ─────────────────────────────────────────────────────────

def load_halueval(n: int) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("HaluEval/halueval", "qa", split="data",
                          trust_remote_code=True)
    except Exception:
        ds = load_dataset("pminervini/HaluEval", split="qa_samples")

    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        question      = row.get("question", row.get("input", ""))
        right         = row.get("right_answer", row.get("answer", ""))
        hallucinated  = row.get("hallucinated_answer", row.get("output", ""))
        if question and right and hallucinated and right != hallucinated:
            samples.append({"question": question,
                            "right_answer": right,
                            "hallucinated_answer": hallucinated})
    return samples[:n]


def extract_hidden_states() -> None:
    """Extract and save balanced HaluEval hidden states for Qwen 7B."""
    if HS_PATH.exists():
        print(f"  Hidden states already at {HS_PATH} — skipping extraction.")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"Extracting: {MODEL_ID}")
    print(f"  N pairs: {N_SAMPLES} → {2*N_SAMPLES} vectors (balanced 50/50)")
    print(f"  dtype: bfloat16  device: cpu")
    print(f"  ETA: ~4–8 hours on CPU")
    print(f"{'='*60}")

    samples = load_halueval(N_SAMPLES)
    print(f"  Loaded {len(samples)} HaluEval QA pairs.")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d        = model.config.hidden_size
    print(f"  Architecture: {n_layers} transformer layers, d={d}")

    all_hs     = []
    all_labels = []
    skipped    = 0

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  [{i}/{len(samples)}] extracting ...")

        for text, label in [
            (f"Q: {sample['question']}\nA: {sample['right_answer']}",       1),
            (f"Q: {sample['question']}\nA: {sample['hallucinated_answer']}", 0),
        ]:
            inputs = tok(text, return_tensors="pt", truncation=True,
                         max_length=256, padding=False)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            if out.hidden_states is None:
                skipped += 1
                continue
            hs = np.stack([
                h[0, -1, :].float().cpu().numpy()
                for h in out.hidden_states
            ])  # (n_layers+1, d)
            all_hs.append(hs)
            all_labels.append(label)

    H = np.stack(all_hs).astype(np.float32)   # (2N, n_layers+1, d)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n  Extracted: H={H.shape}  hall_rate={1-y.mean():.3f}  skipped={skipped}")
    np.savez_compressed(HS_PATH, hidden_states=H, labels=y)
    print(f"  Saved: {HS_PATH}")

    del model
    import gc; gc.collect()


# ── Step 2: Analysis helpers ───────────────────────────────────────────────────

def oof_auroc_per_layer(H, y, probe_cls, probe_kwargs, label="probe"):
    n_samples, n_layers, d = H.shape
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    aurocs = []
    for L in range(n_layers):
        H_L = H[:, L, :]
        oof  = np.zeros(n_samples)
        for tr, te in skf.split(H_L, y):
            sc  = StandardScaler()
            pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0]-1,
                                       H_L[tr].shape[1]),
                      random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))
            clf  = probe_cls(**probe_kwargs)
            clf.fit(X_tr, y[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]
        aurocs.append(float(roc_auc_score(y, oof)))
    best = float(np.max(aurocs))
    print(f"    [{label}] best AUROC: {best:.4f} at L{np.argmax(aurocs)}")
    return aurocs


def conformal_calibration(H, y, best_layer):
    """
    Split conformal: 60/20/20 train/cal/test.
    Finds smallest α* s.t. empirical hall rate ≤ α* at 95% coverage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    n = len(y)
    idx = np.random.RandomState(RANDOM_STATE).permutation(n)
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)
    tr_idx  = idx[:n_tr]
    cal_idx = idx[n_tr:n_tr+n_cal]
    te_idx  = idx[n_tr+n_cal:]

    H_L = H[:, best_layer, :]
    sc  = StandardScaler()
    pca = PCA(n_components=min(N_COMPONENTS, H_L[tr_idx].shape[0]-1,
                               H_L.shape[1]), random_state=RANDOM_STATE)
    X_tr  = pca.fit_transform(sc.fit_transform(H_L[tr_idx]))
    X_cal = pca.transform(sc.transform(H_L[cal_idx]))
    X_te  = pca.transform(sc.transform(H_L[te_idx]))

    clf = LogisticRegression(**LR_KWARGS)
    clf.fit(X_tr, y[tr_idx])

    scores_cal = clf.predict_proba(X_cal)[:, 1]
    scores_te  = clf.predict_proba(X_te)[:, 1]

    best_alpha = None
    for alpha in np.arange(0.01, 0.50, 0.01):
        thresh = np.quantile(scores_cal[y[cal_idx] == 0], 1 - alpha)
        accepted = scores_te > thresh
        if accepted.sum() == 0:
            continue
        hall_rate_accepted = (y[te_idx][accepted] == 0).mean()
        if hall_rate_accepted <= alpha:
            best_alpha = float(round(alpha, 2))
            accept_rate = float(accepted.mean())
            emp_hall = float(hall_rate_accepted)
            break

    return {
        "alpha_star":   best_alpha,
        "accept_rate":  accept_rate if best_alpha else None,
        "emp_hall_rate": emp_hall   if best_alpha else None,
    }


# ── Step 3: Main ───────────────────────────────────────────────────────────────

def run() -> dict:
    print("\n" + "="*60)
    print("Exp 13 — Qwen 2.5 7B: Full Certificate Validation")
    print("="*60)

    # ── Extraction ────────────────────────────────────────────────────────────
    extract_hidden_states()

    data = np.load(HS_PATH)
    H    = data["hidden_states"].astype(np.float32)
    y    = data["labels"].astype(int)
    n, n_layers, d = H.shape
    print(f"\nLoaded: H={H.shape}  hall_rate={1-y.mean():.3f}")

    # ── Fisher certificate ────────────────────────────────────────────────────
    print("\nComputing Fisher certificate Φ(√J/2) at every layer ...")
    curve = fisher_curve(H, y, method="pca", n_components=N_COMPONENTS,
                         verbose=True)
    fisher_bounds = curve["auroc_bound"]
    J_per_layer   = curve["J"]
    best_J_layer  = int(curve["best_layer"])
    best_J        = float(curve["best_J"])
    best_bound    = float(curve["best_auroc_bound"])

    # ── Linear probe ─────────────────────────────────────────────────────────
    print("\nTraining linear probes (5-fold) ...")
    linear_aurocs = oof_auroc_per_layer(H, y, LogisticRegression, LR_KWARGS,
                                        label="linear")

    # ── MLP probe ─────────────────────────────────────────────────────────────
    print("\nTraining MLP probes (5-fold) ...")
    mlp_aurocs = oof_auroc_per_layer(H, y, MLPClassifier, MLP_KWARGS,
                                     label="mlp")

    # ── Certificate validation ─────────────────────────────────────────────
    best_linear_auroc = float(np.max(linear_aurocs))
    best_linear_layer = int(np.argmax(linear_aurocs))
    best_mlp_auroc    = float(np.max(mlp_aurocs))
    best_mlp_layer    = int(np.argmax(mlp_aurocs))

    # Bound errors (all layers)
    errors_linear = [abs(b - a) for b, a in zip(fisher_bounds, linear_aurocs)
                     if not np.isnan(b)]
    max_error   = float(np.max(errors_linear))
    mean_error  = float(np.mean(errors_linear))

    # Violations: does any probe AUROC exceed the bound?
    linear_violations = [L for L in range(n_layers)
                         if linear_aurocs[L] > fisher_bounds[L] + 1e-6]
    mlp_violations    = [L for L in range(n_layers)
                         if mlp_aurocs[L]    > fisher_bounds[L] + 1e-6]

    # MLP gain
    mlp_gain = float(best_mlp_auroc - best_linear_auroc)

    # Mean bound gaps
    gaps_linear = [b - a for b, a in zip(fisher_bounds, linear_aurocs)
                   if not np.isnan(b)]
    gaps_mlp    = [b - a for b, a in zip(fisher_bounds, mlp_aurocs)
                   if not np.isnan(b)]

    print(f"\n{'─'*50}")
    print(f"  Fisher bound (best):   {best_bound:.4f}  at L{best_J_layer}")
    print(f"  Linear AUROC (best):   {best_linear_auroc:.4f}  at L{best_linear_layer}")
    print(f"  MLP AUROC (best):      {best_mlp_auroc:.4f}  at L{best_mlp_layer}")
    print(f"  Bound error max/mean:  {max_error:.4f} / {mean_error:.4f}")
    print(f"  Linear violations:     {len(linear_violations)} / {n_layers}")
    print(f"  MLP violations:        {len(mlp_violations)} / {n_layers}")
    print(f"  MLP gain over linear:  {mlp_gain:+.4f}")

    # ── Conformal calibration ─────────────────────────────────────────────────
    print("\nRunning conformal calibration ...")
    conformal = conformal_calibration(H, y, best_linear_layer)
    print(f"  α* = {conformal['alpha_star']}  "
          f"accept_rate = {conformal['accept_rate']:.3f}  "
          f"emp_hall = {conformal['emp_hall_rate']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    result = {
        "model":                  MODEL_ID,
        "n_samples":              n,
        "n_layers":               n_layers,
        "d":                      d,
        "n_params_approx":        "7.6B",
        # Fisher
        "best_J":                 best_J,
        "best_J_layer":           best_J_layer,
        "best_bound":             best_bound,
        "J_per_layer":            [float(x) for x in J_per_layer],
        "fisher_bound_per_layer": [float(x) for x in fisher_bounds],
        # Linear probe
        "best_linear_auroc":      best_linear_auroc,
        "best_linear_layer":      best_linear_layer,
        "linear_auroc_per_layer": linear_aurocs,
        "linear_bound_error_max":  max_error,
        "linear_bound_error_mean": mean_error,
        "linear_violations":      len(linear_violations),
        # MLP probe
        "best_mlp_auroc":         best_mlp_auroc,
        "best_mlp_layer":         best_mlp_layer,
        "mlp_auroc_per_layer":    mlp_aurocs,
        "mlp_violations":         len(mlp_violations),
        "mlp_gain":               mlp_gain,
        "mean_gap_linear":        float(np.mean(gaps_linear)),
        "mean_gap_mlp":           float(np.mean(gaps_mlp)),
        # Conformal
        "conformal":              conformal,
        # Certificate holds?
        "certificate_holds_linear": len(linear_violations) == 0,
        "certificate_holds_mlp":    len(mlp_violations) == 0,
    }

    out_path = RESULTS_DIR / "13_qwen7b_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(result)
    return result


def _plot(r: dict) -> None:
    n_layers = r["n_layers"]
    depth    = [L / (n_layers - 1) for L in range(n_layers)]
    color    = "mediumseagreen"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Fisher bound + linear + MLP
    ax = axes[0]
    ax.plot(depth, r["fisher_bound_per_layer"],
            color=color, lw=2.2, label="Fisher bound $\\Phi(\\sqrt{J}/2)$")
    ax.plot(depth, r["linear_auroc_per_layer"],
            color=color, lw=1.8, ls="--", label="Linear probe (LR)")
    ax.plot(depth, r["mlp_auroc_per_layer"],
            color=color, lw=1.8, ls=":", label="MLP probe (64h, relu)")
    ax.fill_between(depth, r["mlp_auroc_per_layer"],
                    r["fisher_bound_per_layer"],
                    color=color, alpha=0.12, label="Bound slack (MLP)")
    ax.axvline(r["best_linear_layer"] / (n_layers-1),
               color="gray",  ls=":", alpha=0.5, lw=1)
    ax.axvline(r["best_mlp_layer"] / (n_layers-1),
               color="black", ls=":", alpha=0.5, lw=1)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("Normalised Depth (layer / max)", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(
        f"Qwen 2.5 7B — Certificate vs Probes\n"
        f"Linear: {r['best_linear_auroc']:.4f}  MLP: {r['best_mlp_auroc']:.4f}  "
        f"Bound: {r['best_bound']:.4f}",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.01)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.2)

    # Right: Bound error per layer (linear)
    ax2 = axes[1]
    errors = [abs(b - a) for b, a in zip(r["fisher_bound_per_layer"],
                                          r["linear_auroc_per_layer"])]
    ax2.bar(depth, errors, width=1/(n_layers-1)*0.8,
            color=color, alpha=0.7, label="Bound error (linear)")
    ax2.axhline(r["linear_bound_error_mean"], color="red", ls="--",
                label=f"Mean error = {r['linear_bound_error_mean']:.4f}")
    ax2.axhline(r["linear_bound_error_max"], color="darkred", ls=":",
                label=f"Max error  = {r['linear_bound_error_max']:.4f}")
    ax2.set_xlabel("Normalised Depth", fontsize=11)
    ax2.set_ylabel("|Bound − AUROC|", fontsize=11)
    ax2.set_title("Bound Error per Layer (Qwen 2.5 7B)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0.0, 1.0)
    ax2.grid(True, alpha=0.2)

    plt.suptitle(
        "Exp 13: Qwen 2.5 7B — Fisher Certificate Validation\n"
        f"(n={r['n_samples']}, {n_layers} layers, d={r['d']})",
        fontsize=12
    )
    plt.tight_layout()
    out = PLOTS_DIR / "13_qwen7b_certificate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    run()
