"""
Exp 12 — MLP Probe vs Linear Probe vs Fisher Certificate

Tests whether the Fisher AUROC certificate Φ(√J/2) bounds a nonlinear
(MLP) probe as well as it bounds the linear probe.

Design:
  - Same PCA(100) projection used by all other experiments (fair comparison)
  - Same 5-fold StratifiedKFold setup (random_state=42)
  - Linear probe: LogisticRegression(C=1.0, max_iter=1000)  [baseline]
  - MLP probe:   MLPClassifier(64 hidden, relu, adam, early_stopping)

Per-layer output for each model:
  - Fisher bound:     Φ(√J(L)/2)
  - Linear AUROC:     OOF AUROC with logistic regression
  - MLP AUROC:        OOF AUROC with shallow MLP

Key questions:
  Q1. Does MLP exceed linear probe AUROC? (at how many layers, by how much?)
  Q2. Does the Fisher bound still hold for MLP? (does MLP exceed Φ(√J/2)?)
  Q3. Is the bound gap different for linear vs MLP?

Data used (pre-extracted, zero new inference):
  - results/hidden_states/00_halueval_qwen3b.npz    (2000 × 37 × 2048)
  - results/hidden_states/00_halueval_gpt2med.npz   (2000 × 25 × 1024)

Cost: CPU-only, ~10–20 minutes total.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.special import ndtr  # Φ(x) = ndtr(x)
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import fisher_curve

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR   = ROOT / "results" / "plots"
HS_DIR      = ROOT / "results" / "hidden_states"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS = {
    "Qwen 2.5 3B": {
        "path": HS_DIR / "00_halueval_qwen3b.npz",
        "color": "steelblue",
    },
    "GPT-2 Medium": {
        "path": HS_DIR / "00_halueval_gpt2med.npz",
        "color": "darkorange",
    },
}

N_COMPONENTS = 100   # PCA projection — same as all other experiments
N_SPLITS     = 5     # 5-fold CV — same as all other experiments
RANDOM_STATE = 42

# MLP architecture: one hidden layer, relu, adam with early stopping
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

# Linear probe kwargs — identical to all other experiments
LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)


# ── Core functions ─────────────────────────────────────────────────────────────

def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    H = data["hidden_states"].astype(np.float32)
    y = data["labels"].astype(int)
    n = min(H.shape[0], y.shape[0])
    return H[:n], y[:n]


def oof_auroc_per_layer(
    H: np.ndarray,
    y: np.ndarray,
    probe_cls,
    probe_kwargs: dict,
    n_components: int = N_COMPONENTS,
    n_splits:    int = N_SPLITS,
    label:       str = "probe",
) -> list[float]:
    """
    For each layer L, fit probe on PCA(n_components) projected hidden states
    using n_splits-fold stratified CV and return OOF AUROC.
    """
    n_samples, n_layers, d = H.shape
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aurocs = []

    for L in range(n_layers):
        H_L = H[:, L, :]                          # (n, d)
        oof = np.zeros(n_samples)

        for fold_idx, (tr, te) in enumerate(skf.split(H_L, y)):
            # StandardScaler + PCA — identical preprocessing to Exp 01
            sc  = StandardScaler()
            pca = PCA(n_components=min(n_components, H_L[tr].shape[0] - 1,
                                       H_L[tr].shape[1]),
                      random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))

            clf = probe_cls(**probe_kwargs)
            clf.fit(X_tr, y[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]

        aurocs.append(float(roc_auc_score(y, oof)))

    print(f"    [{label}] best AUROC: {max(aurocs):.4f} at L{np.argmax(aurocs)}")
    return aurocs


def fisher_bounds_per_layer(H: np.ndarray, y: np.ndarray) -> list[float]:
    """Compute Φ(√J/2) at each layer using same PCA(100) method."""
    curve = fisher_curve(H, y, method="pca", n_components=N_COMPONENTS, verbose=False)
    return curve["auroc_bound"]


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> dict:
    all_results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        if not cfg["path"].exists():
            print(f"  SKIP — {cfg['path']} not found")
            continue

        H, y = load_npz(cfg["path"])
        n_samples, n_layers, d = H.shape
        print(f"  Hidden states: {H.shape}  hall_rate={1 - y.mean():.3f}")

        # ── Fisher certificate ────────────────────────────────────────────────
        print("\n  Computing Fisher certificate (Φ(√J/2)) ...")
        fisher_bounds = fisher_bounds_per_layer(H, y)

        # ── Linear probe ─────────────────────────────────────────────────────
        print("\n  Training linear probes (LogisticRegression) ...")
        linear_aurocs = oof_auroc_per_layer(
            H, y, LogisticRegression, LR_KWARGS, label="linear"
        )

        # ── MLP probe ─────────────────────────────────────────────────────────
        print("\n  Training MLP probes (MLPClassifier 64 hidden, relu, adam) ...")
        mlp_aurocs = oof_auroc_per_layer(
            H, y, MLPClassifier, MLP_KWARGS, label="mlp"
        )

        # ── Analysis ──────────────────────────────────────────────────────────
        best_linear_auroc = float(np.max(linear_aurocs))
        best_mlp_auroc    = float(np.max(mlp_aurocs))
        best_fisher_bound = float(np.max(fisher_bounds))

        # Layers where MLP > linear
        mlp_wins = [L for L in range(n_layers) if mlp_aurocs[L] > linear_aurocs[L]]
        # Layers where MLP exceeds Fisher bound (certificate violation)
        mlp_exceeds_bound = [
            L for L in range(n_layers)
            if (not np.isnan(fisher_bounds[L])) and mlp_aurocs[L] > fisher_bounds[L]
        ]

        # Mean bound gap: linear vs MLP
        gaps_linear = [b - a for b, a in zip(fisher_bounds, linear_aurocs)
                       if not np.isnan(b)]
        gaps_mlp    = [b - a for b, a in zip(fisher_bounds, mlp_aurocs)
                       if not np.isnan(b)]
        mean_gap_linear = float(np.mean(gaps_linear))
        mean_gap_mlp    = float(np.mean(gaps_mlp))

        result = {
            "model_name":           model_name,
            "n_samples":            n_samples,
            "n_layers":             n_layers,
            "d":                    d,
            # Per-layer curves
            "fisher_bound_per_layer":  [float(x) for x in fisher_bounds],
            "linear_auroc_per_layer":  linear_aurocs,
            "mlp_auroc_per_layer":     mlp_aurocs,
            # Summary stats
            "best_fisher_bound":       best_fisher_bound,
            "best_linear_auroc":       best_linear_auroc,
            "best_linear_layer":       int(np.argmax(linear_aurocs)),
            "best_mlp_auroc":          best_mlp_auroc,
            "best_mlp_layer":          int(np.argmax(mlp_aurocs)),
            # Q1: MLP vs Linear
            "n_layers_mlp_beats_linear":   len(mlp_wins),
            "mlp_win_layers":              mlp_wins,
            "mlp_gain_at_best_layer":      float(best_mlp_auroc - best_linear_auroc),
            # Q2: Certificate bound for MLP
            "n_layers_mlp_exceeds_bound":  len(mlp_exceeds_bound),
            "mlp_exceeds_bound_layers":    mlp_exceeds_bound,
            "fisher_bound_holds_for_mlp":  len(mlp_exceeds_bound) == 0,
            # Q3: Bound gap
            "mean_bound_gap_linear":       mean_gap_linear,
            "mean_bound_gap_mlp":          mean_gap_mlp,
            "bound_gap_reduction_mlp":     float(mean_gap_linear - mean_gap_mlp),
        }

        print(f"\n  {'─'*50}")
        print(f"  Fisher bound (best):   {best_fisher_bound:.4f}")
        print(f"  Linear AUROC (best):   {best_linear_auroc:.4f}  at L{result['best_linear_layer']}")
        print(f"  MLP AUROC (best):      {best_mlp_auroc:.4f}  at L{result['best_mlp_layer']}")
        print(f"  MLP gain over linear:  {result['mlp_gain_at_best_layer']:+.4f}")
        print(f"  Layers MLP > linear:   {len(mlp_wins)} / {n_layers}")
        print(f"  MLP exceeds bound:     {len(mlp_exceeds_bound)} layers  "
              f"({'VIOLATION' if mlp_exceeds_bound else 'bound holds'})")
        print(f"  Mean gap (linear):     {mean_gap_linear:.4f}")
        print(f"  Mean gap (MLP):        {mean_gap_mlp:.4f}  "
              f"(reduction: {result['bound_gap_reduction_mlp']:+.4f})")

        all_results[model_name] = result

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "12_mlp_probe.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(all_results)
    return all_results


def _plot(results: dict) -> None:
    computed = {k: v for k, v in results.items() if "fisher_bound_per_layer" in v}
    if not computed:
        return

    n_models = len(computed)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]

    model_colors = {
        "Qwen 2.5 3B":   "steelblue",
        "GPT-2 Medium":  "darkorange",
    }

    for ax, (model_name, r) in zip(axes, computed.items()):
        n_layers = r["n_layers"]
        depth    = [L / (n_layers - 1) for L in range(n_layers)]
        mc       = model_colors.get(model_name, "purple")

        ax.plot(depth, r["fisher_bound_per_layer"],
                color=mc, linewidth=2.2, label="Fisher bound $\\Phi(\\sqrt{J}/2)$")
        ax.plot(depth, r["linear_auroc_per_layer"],
                color=mc, linewidth=1.8, linestyle="--",
                label="Linear probe (LR)")
        ax.plot(depth, r["mlp_auroc_per_layer"],
                color=mc, linewidth=1.8, linestyle=":",
                label="MLP probe (64h, relu)")

        # Shade region between Fisher bound and MLP
        ax.fill_between(
            depth,
            r["mlp_auroc_per_layer"],
            r["fisher_bound_per_layer"],
            color=mc, alpha=0.10,
            label="Bound slack (MLP)"
        )

        # Mark best layers
        bl = r["best_linear_layer"]
        bm = r["best_mlp_layer"]
        ax.axvline(bl / (n_layers - 1), color="gray",  linestyle=":", alpha=0.5, linewidth=1)
        ax.axvline(bm / (n_layers - 1), color="black", linestyle=":", alpha=0.5, linewidth=1)

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Normalised Depth (layer / max)", fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(
            f"{model_name}\n"
            f"MLP gain: {r['mlp_gain_at_best_layer']:+.4f}  |  "
            f"Bound holds: {'✓' if r['fisher_bound_holds_for_mlp'] else '✗'}",
            fontsize=10
        )
        ax.legend(fontsize=8)
        ax.set_ylim(0.4, 1.01)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Fisher Bound vs Linear Probe vs MLP Probe — All Layers\n"
        "(Exp 12: Does Φ(√J/2) bound nonlinear probes?)",
        fontsize=12
    )
    plt.tight_layout()
    out = PLOTS_DIR / "12_mlp_probe_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()
