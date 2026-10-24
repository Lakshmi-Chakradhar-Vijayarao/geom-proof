"""
GEOM-PROOF — Kaggle Exp 13: Qwen 2.5 7B Full Certificate Validation
=====================================================================

Self-contained Kaggle notebook script. Runs entirely on T4 GPU (15GB VRAM).
Estimated runtime: ~2.5–3.5 hours total on T4.

What this does (in order):
  STEP 1  Balanced HaluEval extraction for Qwen 2.5 7B
            1000 QA pairs → 2000 hidden-state vectors (50/50 balanced)
            Same protocol as Exp 00 (no generation, forward pass only)
            ~2–2.5 hours on T4

  STEP 2  Fisher certificate Φ(√(J/2)) at every layer
            PCA(100) + Ledoit-Wolf — same as all other experiments
            ~5 min

  STEP 3  Linear probe (LogisticRegression, 5-fold StratifiedKFold)
            Same hyperparams as Exp 01/12: C=1.0, max_iter=1000
            ~15 min

  STEP 4  MLP probe (64 hidden, ReLU, Adam, early stopping)
            Same hyperparams as Exp 12
            ~20 min

  STEP 5  Conformal calibration (60/20/20 split, α* search)
            Same as Exp 10
            ~2 min

  STEP 6  Scale curve update
            Adds 7B actual AUROC to the Qwen family table
            ~1 min

Outputs saved to /kaggle/working/:
  13_halueval_qwen7b.npz         ← hidden states (download → results/hidden_states/)
  13_qwen7b_analysis.json        ← full results  (download → results/logs/)
  13_qwen7b_certificate.png      ← plot          (download → results/plots/)

HOW TO USE ON KAGGLE:
  1. Create new notebook → Accelerator: GPU T4 x1
  2. Session: set to 9h (maximum)
  3. Paste this entire file into a single code cell (or upload as script)
  4. Run
  5. Download the three output files from the Output panel

INSTALL CELL (run first, separately):
  !pip install -q transformers datasets accelerate scikit-learn matplotlib scipy

NOTE: Qwen 2.5 7B in float16 needs ~14GB VRAM — fits on T4 (15GB).
      If OOM: reduce N_SAMPLES = 500 below (→ 1000 vectors instead of 2000).
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, json, gc, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.stats import norm as scipy_norm
from scipy.special import ndtr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

HS_PATH   = OUT / "13_halueval_qwen7b.npz"
JSON_PATH = OUT / "13_qwen7b_analysis.json"
PLOT_PATH = OUT / "13_qwen7b_certificate.png"

# ── Configuration (same as all other experiments) ────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES    = 1000    # QA pairs → 2000 balanced vectors. Reduce to 500 if OOM.
N_COMPONENTS = 100     # PCA components — same as Exp 01/12
N_SPLITS     = 5       # StratifiedKFold folds — same as Exp 01/12
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

print(f"\nConfiguration:")
print(f"  Model     : {MODEL_ID}")
print(f"  N pairs   : {N_SAMPLES} → {2*N_SAMPLES} vectors (balanced 50/50)")
print(f"  PCA k     : {N_COMPONENTS}")
print(f"  CV folds  : {N_SPLITS}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. HALUEVAL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_halueval(n: int) -> list[dict]:
    """Load n balanced QA pairs from HaluEval. Same loader as Exp 00."""
    ds = None
    for attempt in [
        lambda: load_dataset("HaluEval/halueval", "qa", split="data",
                             trust_remote_code=True),
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
        split_name = list(ds.keys())[0]
        print(f"  DatasetDict — using split '{split_name}'")
        ds = ds[split_name]

    first = ds[0]
    print(f"  HaluEval columns: {list(first.keys())}")

    Q_KEYS    = ["question", "input", "query"]
    RIGHT_KEYS = ["right_answer", "answer", "correct_answer", "gold_answer"]
    HALL_KEYS  = ["hallucinated_answer", "hallucination", "wrong_answer",
                  "model_answer", "output", "negative"]

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
        q    = pick(row, Q_KEYS)
        right = pick(row, RIGHT_KEYS)
        hall  = pick(row, HALL_KEYS)
        if q and right and hall and right != hall:
            out.append({"question": q,
                        "right_answer": right,
                        "hallucinated_answer": hall})

    print(f"  Loaded {len(out)} valid pairs from {ds.num_rows} rows")
    return out[:n]


# ─────────────────────────────────────────────────────────────────────────────
# 2. HIDDEN STATE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_hidden_states() -> None:
    """Extract balanced HaluEval hidden states for Qwen 2.5 7B."""
    if HS_PATH.exists():
        print(f"\nStep 1: Hidden states already at {HS_PATH} — skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Step 1: Extracting hidden states")
    print(f"  Model   : {MODEL_ID}")
    print(f"  N pairs : {N_SAMPLES} → {2*N_SAMPLES} vectors")
    print(f"  dtype   : float16  device: {DEVICE}")
    print(f"{'='*60}")

    t0 = time.time()

    print("\nLoading HaluEval ...")
    samples = load_halueval(N_SAMPLES)
    print(f"  {len(samples)} pairs loaded.")

    print(f"\nLoading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d        = model.config.hidden_size
    print(f"  Loaded: {n_layers} transformer layers, d={d}")
    if DEVICE == "cuda":
        print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    all_hs     = []
    all_labels = []
    skipped    = 0

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(samples) - i) / 60
            print(f"  [{i:4d}/{len(samples)}]  elapsed={elapsed/60:.1f}min  "
                  f"ETA={eta:.0f}min")

        for text, label in [
            (f"Q: {sample['question']}\nA: {sample['right_answer']}",       1),
            (f"Q: {sample['question']}\nA: {sample['hallucinated_answer']}", 0),
        ]:
            inputs = tok(text, return_tensors="pt", truncation=True,
                         max_length=256, padding=False)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at sample {i} — clearing cache and retrying ...")
                torch.cuda.empty_cache()
                gc.collect()
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)

            if out.hidden_states is None:
                skipped += 1
                continue

            # Last token, all layers → (n_layers+1, d)
            hs = np.stack([
                h[0, -1, :].float().cpu().numpy()
                for h in out.hidden_states
            ])
            all_hs.append(hs)
            all_labels.append(label)

    H = np.stack(all_hs).astype(np.float32)   # (2N, n_layers+1, d)
    y = np.array(all_labels, dtype=np.int32)

    total_min = (time.time() - t0) / 60
    print(f"\n  Extraction complete in {total_min:.1f} min")
    print(f"  Shape: {H.shape}  hall_rate={1-y.mean():.3f}  skipped={skipped}")

    np.savez_compressed(HS_PATH, hidden_states=H, labels=y)
    print(f"  Saved: {HS_PATH}  ({HS_PATH.stat().st_size/1e6:.0f} MB)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    if DEVICE == "cuda":
        print(f"  VRAM freed. Current: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FISHER CERTIFICATE
# ─────────────────────────────────────────────────────────────────────────────

def fisher_ratio_pca(H_L: np.ndarray, y: np.ndarray, k: int = 100) -> float:
    """Compute Fisher J at one layer via PCA(k) + Ledoit-Wolf. Same as src/fisher.py."""
    k = min(k, H_L.shape[1], H_L.shape[0] - 2)
    H_L = H_L.astype(np.float64)

    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    H_proj = pca.fit_transform(H_L)

    mu_c = H_proj[y == 1].mean(0)
    mu_h = H_proj[y == 0].mean(0)
    H_c  = H_proj[y == 1] - mu_c
    H_h  = H_proj[y == 0] - mu_h

    Sigma_w = LedoitWolf().fit(np.vstack([H_c, H_h])).covariance_
    diff    = mu_c - mu_h
    try:
        w = np.linalg.solve(Sigma_w, diff)
        J = float(diff @ w)
    except np.linalg.LinAlgError:
        J = float(diff @ np.linalg.lstsq(Sigma_w, diff, rcond=None)[0])
    return max(J, 0.0)


def compute_fisher_curve(H: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    print(f"\n{'='*60}")
    print("Step 2: Fisher certificate Φ(√(J/2)) at every layer")
    print(f"{'='*60}")
    n_layers = H.shape[1]
    J_list, bound_list = [], []
    for L in range(n_layers):
        J = fisher_ratio_pca(H[:, L, :], y, k=N_COMPONENTS)
        b = float(scipy_norm.cdf(np.sqrt(J / 2)))  # corrected binormal AUROC identity
        J_list.append(J)
        bound_list.append(b)
        print(f"  L{L:3d}: J={J:.4f}  bound={b:.4f}")
    best = int(np.argmax(J_list))
    print(f"\n  Best J layer: L{best}  J={J_list[best]:.4f}  bound={bound_list[best]:.4f}")
    return J_list, bound_list


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROBE (linear + MLP) — OOF AUROC PER LAYER
# ─────────────────────────────────────────────────────────────────────────────

def oof_auroc_per_layer(H: np.ndarray, y: np.ndarray,
                        probe_cls, probe_kwargs: dict,
                        label: str = "probe") -> list[float]:
    """5-fold OOF AUROC per layer. Identical to Exp 01 / Exp 12."""
    n_samples, n_layers, d = H.shape
    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
    aurocs = []

    for L in range(n_layers):
        H_L = H[:, L, :]
        oof = np.zeros(n_samples)

        for tr, te in skf.split(H_L, y):
            sc  = StandardScaler()
            pca = PCA(n_components=min(N_COMPONENTS,
                                       H_L[tr].shape[0] - 1,
                                       H_L[tr].shape[1]),
                      random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))
            clf  = probe_cls(**probe_kwargs)
            clf.fit(X_tr, y[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]

        auroc = float(roc_auc_score(y, oof))
        aurocs.append(auroc)
        print(f"  L{L:3d}: AUROC={auroc:.4f}")

    best = float(np.max(aurocs))
    print(f"\n  [{label}] best AUROC: {best:.4f}  at L{int(np.argmax(aurocs))}")
    return aurocs


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONFORMAL CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def conformal_calibration(H: np.ndarray, y: np.ndarray,
                           best_layer: int) -> dict:
    """
    60/20/20 split conformal. Finds smallest α* s.t. hall_rate ≤ α*
    among accepted queries at 95% nominal coverage.
    Identical design to Exp 10.
    """
    print(f"\n{'='*60}")
    print(f"Step 5: Conformal calibration (layer L{best_layer})")
    print(f"{'='*60}")

    n   = len(y)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(n)
    n_tr  = int(0.6 * n)
    n_cal = int(0.2 * n)
    tr_idx  = idx[:n_tr]
    cal_idx = idx[n_tr:n_tr + n_cal]
    te_idx  = idx[n_tr + n_cal:]

    H_L   = H[:, best_layer, :]
    sc    = StandardScaler()
    pca   = PCA(n_components=min(N_COMPONENTS,
                                  H_L[tr_idx].shape[0] - 1,
                                  H_L.shape[1]),
                random_state=RANDOM_STATE)
    X_tr  = pca.fit_transform(sc.fit_transform(H_L[tr_idx]))
    X_cal = pca.transform(sc.transform(H_L[cal_idx]))
    X_te  = pca.transform(sc.transform(H_L[te_idx]))

    clf = LogisticRegression(**LR_KWARGS)
    clf.fit(X_tr, y[tr_idx])

    scores_cal = clf.predict_proba(X_cal)[:, 1]
    scores_te  = clf.predict_proba(X_te)[:, 1]

    best_alpha  = None
    accept_rate = None
    emp_hall    = None

    for alpha in np.arange(0.01, 0.50, 0.01):
        thresh   = np.quantile(scores_cal[y[cal_idx] == 0], 1 - alpha)
        accepted = scores_te > thresh
        if accepted.sum() == 0:
            continue
        hall_rate = float((y[te_idx][accepted] == 0).mean())
        if hall_rate <= alpha:
            best_alpha  = float(round(alpha, 2))
            accept_rate = float(accepted.mean())
            emp_hall    = hall_rate
            break

    result = {
        "alpha_star":    best_alpha,
        "accept_rate":   accept_rate,
        "emp_hall_rate": emp_hall,
    }
    print(f"  α*          = {best_alpha}")
    print(f"  accept_rate = {accept_rate:.3f}" if accept_rate else "  no valid α found")
    print(f"  emp_hall    = {emp_hall:.4f}"    if emp_hall    else "")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(r: dict) -> None:
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
    ax.fill_between(depth,
                    r["mlp_auroc_per_layer"],
                    r["fisher_bound_per_layer"],
                    color=color, alpha=0.12, label="Bound slack (MLP)")
    ax.axvline(r["best_linear_layer"] / (n_layers - 1),
               color="gray",  ls=":", alpha=0.5, lw=1,
               label=f"Best linear (L{r['best_linear_layer']})")
    ax.axvline(r["best_mlp_layer"] / (n_layers - 1),
               color="black", ls=":", alpha=0.5, lw=1,
               label=f"Best MLP (L{r['best_mlp_layer']})")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("Normalised Depth (layer / max)", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    viol = r["mlp_violations"]
    ax.set_title(
        f"Qwen 2.5 7B — Certificate vs Probes\n"
        f"Linear: {r['best_linear_auroc']:.4f}  "
        f"MLP: {r['best_mlp_auroc']:.4f}  "
        f"Bound: {r['best_bound']:.4f}  "
        f"Viol: {viol}",
        fontsize=10
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.4, 1.01)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.2)

    # Right: Bound error per layer
    ax2 = axes[1]
    errors = [abs(b - a) for b, a
              in zip(r["fisher_bound_per_layer"], r["linear_auroc_per_layer"])]
    ax2.bar(depth, errors, width=1/(n_layers-1)*0.8,
            color=color, alpha=0.7)
    ax2.axhline(r["linear_bound_error_mean"], color="red", ls="--",
                lw=1.5, label=f"Mean = {r['linear_bound_error_mean']:.4f}")
    ax2.axhline(r["linear_bound_error_max"], color="darkred", ls=":",
                lw=1.5, label=f"Max  = {r['linear_bound_error_max']:.4f}")
    ax2.set_xlabel("Normalised Depth", fontsize=11)
    ax2.set_ylabel("|Bound $-$ AUROC|", fontsize=11)
    ax2.set_title("Bound Error per Layer (Qwen 2.5 7B)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0.0, 1.0)
    ax2.grid(True, alpha=0.2)

    plt.suptitle(
        f"Exp 13: Qwen 2.5 7B — Fisher Certificate Validation\n"
        f"(n={r['n_samples']}, {n_layers} layers, d={r['d']}, "
        f"params≈{r['n_params_approx']})",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {PLOT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("\n" + "="*60)
    print("Exp 13 — Qwen 2.5 7B: Full Certificate Validation")
    print("="*60)

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    extract_hidden_states()

    data  = np.load(HS_PATH)
    H     = data["hidden_states"].astype(np.float32)
    y     = data["labels"].astype(int)
    n, n_layers, d = H.shape
    print(f"\nLoaded: H={H.shape}  hall_rate={1-y.mean():.3f}")
    print(f"  Balanced classes: correct={y.sum()}  hallucinated={(y==0).sum()}")

    # ── Step 2: Fisher certificate ────────────────────────────────────────────
    J_per_layer, fisher_bounds = compute_fisher_curve(H, y)
    best_J_layer = int(np.argmax(J_per_layer))
    best_J       = float(J_per_layer[best_J_layer])
    best_bound   = float(fisher_bounds[best_J_layer])

    # ── Step 3: Linear probe ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3: Linear probes (LogisticRegression, 5-fold OOF)")
    print(f"{'='*60}")
    linear_aurocs = oof_auroc_per_layer(H, y, LogisticRegression, LR_KWARGS,
                                        label="linear")

    # ── Step 4: MLP probe ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4: MLP probes (64 hidden, ReLU, Adam, 5-fold OOF)")
    print(f"{'='*60}")
    mlp_aurocs = oof_auroc_per_layer(H, y, MLPClassifier, MLP_KWARGS,
                                     label="mlp")

    # ── Summary statistics ────────────────────────────────────────────────────
    best_linear_auroc = float(np.max(linear_aurocs))
    best_linear_layer = int(np.argmax(linear_aurocs))
    best_mlp_auroc    = float(np.max(mlp_aurocs))
    best_mlp_layer    = int(np.argmax(mlp_aurocs))

    errors_linear = [abs(b - a) for b, a in zip(fisher_bounds, linear_aurocs)
                     if not np.isnan(b)]
    max_error  = float(np.max(errors_linear))
    mean_error = float(np.mean(errors_linear))

    linear_violations = [L for L in range(n_layers)
                         if linear_aurocs[L] > fisher_bounds[L] + 1e-6]
    mlp_violations    = [L for L in range(n_layers)
                         if mlp_aurocs[L]    > fisher_bounds[L] + 1e-6]
    mlp_gain          = float(best_mlp_auroc - best_linear_auroc)

    gaps_linear = [b - a for b, a in zip(fisher_bounds, linear_aurocs)]
    gaps_mlp    = [b - a for b, a in zip(fisher_bounds, mlp_aurocs)]

    print(f"\n{'─'*50}")
    print(f"  Fisher bound (best):    {best_bound:.4f}  at L{best_J_layer}")
    print(f"  Linear AUROC (best):    {best_linear_auroc:.4f}  at L{best_linear_layer}")
    print(f"  MLP AUROC (best):       {best_mlp_auroc:.4f}  at L{best_mlp_layer}")
    print(f"  Bound error max/mean:   {max_error:.4f} / {mean_error:.4f}")
    print(f"  Linear violations:      {len(linear_violations)} / {n_layers}")
    print(f"  MLP violations:         {len(mlp_violations)} / {n_layers}")
    print(f"  MLP gain over linear:   {mlp_gain:+.4f}")

    # ── Step 5: Conformal ─────────────────────────────────────────────────────
    conformal = conformal_calibration(H, y, best_linear_layer)

    # ── Step 6: Build result dict ─────────────────────────────────────────────
    result = {
        "experiment":             "13_qwen7b_full_analysis",
        "model":                  MODEL_ID,
        "n_params_approx":        "7.6B",
        "n_samples":              n,
        "n_layers":               n_layers,
        "d":                      d,
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
        "linear_violation_layers": linear_violations,
        # MLP probe
        "best_mlp_auroc":         best_mlp_auroc,
        "best_mlp_layer":         best_mlp_layer,
        "mlp_auroc_per_layer":    mlp_aurocs,
        "mlp_violations":         len(mlp_violations),
        "mlp_violation_layers":   mlp_violations,
        "mlp_gain":               mlp_gain,
        "mean_gap_linear":        float(np.mean(gaps_linear)),
        "mean_gap_mlp":           float(np.mean(gaps_mlp)),
        # Conformal
        "conformal":              conformal,
        # Certificate verdict
        "certificate_holds_linear": len(linear_violations) == 0,
        "certificate_holds_mlp":    len(mlp_violations) == 0,
        # Runtime
        "total_runtime_min":      round((time.time() - t_start) / 60, 1),
    }

    with open(JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {JSON_PATH}")

    # ── Step 7: Plot ──────────────────────────────────────────────────────────
    print("\nGenerating plot ...")
    make_plot(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — Qwen 2.5 7B")
    print(f"{'='*60}")
    print(f"  Fisher bound (best):         {best_bound:.4f}")
    print(f"  Linear probe AUROC (best):   {best_linear_auroc:.4f}  at L{best_linear_layer}")
    print(f"  MLP probe AUROC (best):      {best_mlp_auroc:.4f}  at L{best_mlp_layer}")
    print(f"  Bound max error:             {max_error:.4f}")
    print(f"  Bound mean error:            {mean_error:.4f}")
    print(f"  Certificate holds (linear):  {result['certificate_holds_linear']}")
    print(f"  Certificate holds (MLP):     {result['certificate_holds_mlp']}")
    print(f"  MLP gain over linear:        {mlp_gain:+.4f}")
    if conformal["alpha_star"]:
        print(f"  Conformal α*:                {conformal['alpha_star']}")
        print(f"  Accept rate:                 {conformal['accept_rate']:.3f}")
    print(f"  Total runtime:               {result['total_runtime_min']} min")

    print(f"\n{'='*60}")
    print("FILES TO DOWNLOAD:")
    print(f"{'='*60}")
    for path in [HS_PATH, JSON_PATH, PLOT_PATH]:
        if path.exists():
            print(f"  {path.name:45s}  {path.stat().st_size/1e6:.1f} MB")

    print("""
NEXT STEPS (after download):
  1. Copy files to local paths:
       13_halueval_qwen7b.npz      → geom-proof/results/hidden_states/
       13_qwen7b_analysis.json     → geom-proof/results/logs/
       13_qwen7b_certificate.png   → geom-proof/results/plots/

  2. Fill in the --- rows in main.tex Tables 2, 6, 7 with actual values
     from 13_qwen7b_analysis.json:
       best_linear_auroc, best_linear_layer
       best_mlp_auroc, best_mlp_layer, mlp_gain, mlp_violations
       best_bound, linear_bound_error_max, linear_bound_error_mean

  3. Recompile on Overleaf and submit.
""")
    return result


if __name__ == "__main__":
    main()
