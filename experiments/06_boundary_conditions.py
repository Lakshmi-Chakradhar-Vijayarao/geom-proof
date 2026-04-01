"""
Exp 06 — Boundary Conditions: Where Does the Certificate Fail?

The central figure for the project: Fisher ratio (x-axis) vs actual probe AUROC (y-axis),
plotted for EVERY layer across ALL models. This shows the mapping from geometry to
predictability and where the certificate succeeds or fails.

Also tests:
  1. Null boundary: GPT-2 117M — bound should predict ≈0.50
  2. Weak boundary: GPT-2 Medium 345M — bound should predict ≈0.55–0.60
  3. Strong boundary: Qwen 2.5 3B — bound should predict ≈0.77
  4. OOD degradation: certificate on wrong layer (not best) — graceful?
  5. SSM transfer: Mamba 130M — does the bound hold on a non-transformer?

Cost: $0 — uses all prior outputs.
Hardware: CPU (~1 hour).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import auroc_bound

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXP01 = RESULTS_DIR / "01_fisher_analysis.json"
EXP04 = RESULTS_DIR / "04_mamba_transfer.json"

MODEL_META = {
    "GPT-2 117M":        {"params": 117e6,  "color": "gray",       "marker": "o", "architecture": "Transformer"},
    "GPT-2 Medium 345M": {"params": 345e6,  "color": "steelblue",  "marker": "s", "architecture": "Transformer"},
    "Qwen 2.5 3B":       {"params": 3e9,    "color": "darkorange", "marker": "^", "architecture": "Transformer"},
    "Mamba 130M":        {"params": 130e6,  "color": "green",      "marker": "D", "architecture": "SSM"},
}


def theoretical_curve(J_range: np.ndarray) -> np.ndarray:
    """AUROC_bound = Phi(sqrt(J)/2)."""
    return norm.cdf(np.sqrt(np.maximum(0, J_range)) / 2)


def run() -> dict:
    print("=" * 60)
    print("Exp 06 — Boundary Conditions + Central Figure")
    print("=" * 60)

    all_points = {}  # model_name -> {J_per_layer, probe_auroc_per_layer}

    # Load Exp 01
    if EXP01.exists():
        with open(EXP01) as f:
            exp01 = json.load(f)
        for model_name, result in exp01.items():
            if result.get("fisher_computed", False):
                all_points[model_name] = {
                    "J_per_layer": result["J_per_layer"],
                    "auroc_bound_per_layer": result["auroc_bound_per_layer"],
                    "probe_auroc_per_layer": result["probe_auroc_per_layer"],
                }
    else:
        print("Exp 01 not found. Run Exp 01 first.")

    # Load Exp 04
    if EXP04.exists():
        with open(EXP04) as f:
            exp04 = json.load(f)
        all_points["Mamba 130M"] = {
            "J_per_layer": exp04["J_per_layer"],
            "auroc_bound_per_layer": exp04["auroc_bound_per_layer"],
            "probe_auroc_per_layer": exp04["probe_auroc_per_layer"],
        }

    if not all_points:
        print("No data. Run Exp 01 and Exp 04 first.")
        return {}

    # ── Analysis: boundary conditions ─────────────────────────────────────────
    boundary_analysis = {}
    for model_name, data in all_points.items():
        J = np.array(data["J_per_layer"])
        bound = np.array(data["auroc_bound_per_layer"])
        actual = np.array(data["probe_auroc_per_layer"])
        errors = np.abs(bound - actual)

        best_layer = int(np.argmax(J))
        worst_layer = int(np.argmax(errors))

        boundary_analysis[model_name] = {
            "best_layer": best_layer,
            "best_J": float(J[best_layer]),
            "best_bound": float(bound[best_layer]),
            "best_actual": float(actual[best_layer]),
            "best_error": float(errors[best_layer]),
            "mean_error_all_layers": float(errors.mean()),
            "max_error_all_layers": float(errors.max()),
            "worst_layer": worst_layer,
            "worst_error": float(errors[worst_layer]),
            "layers_within_005": int((errors < 0.05).sum()),
            "layers_within_010": int((errors < 0.10).sum()),
            "total_layers": len(J),
        }
        print(f"\n{model_name}:")
        print(f"  Best layer: L{best_layer} | J={J[best_layer]:.4f} | "
              f"bound={bound[best_layer]:.4f} | actual={actual[best_layer]:.4f} | "
              f"error={errors[best_layer]:.4f}")
        print(f"  All layers: mean_error={errors.mean():.4f}, max_error={errors.max():.4f}")
        print(f"  Within ±0.05: {(errors < 0.05).sum()}/{len(errors)} layers")
        print(f"  Within ±0.10: {(errors < 0.10).sum()}/{len(errors)} layers")

    results = {
        "experiment": "06_boundary_conditions",
        "boundary_analysis": boundary_analysis,
    }
    out_path = RESULTS_DIR / "06_boundary_conditions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    _plot_central_figure(all_points)
    return results


def _plot_central_figure(all_points: dict) -> None:
    """
    Central figure: Fisher ratio (x) vs probe AUROC (y), all layers, all models.
    Theoretical curve overlaid. This is the project's signature plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_main, ax_best = axes

    # ── Main panel: all layers, all models ────────────────────────────────────
    J_all = []
    for model_name, data in all_points.items():
        J = np.array(data["J_per_layer"])
        actual = np.array(data["probe_auroc_per_layer"])
        meta = MODEL_META.get(model_name, {"color": "black", "marker": "o", "architecture": "?"})

        ax_main.scatter(J, actual,
                        color=meta["color"], marker=meta["marker"],
                        alpha=0.5, s=30, label=f"{model_name} ({meta['architecture']})")
        J_all.extend(J.tolist())

    # Theoretical curve
    J_range = np.linspace(0, max(J_all) * 1.1, 300)
    auroc_theory = theoretical_curve(J_range)
    ax_main.plot(J_range, auroc_theory, "k-", linewidth=2.5,
                 label="Theory: Φ(√J/2)", zorder=10)

    ax_main.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_main.axhline(0.70, color="green", linestyle="--", linewidth=1, alpha=0.6, label="Governance threshold")
    ax_main.set_xlabel("Fisher Ratio J(L)", fontsize=12)
    ax_main.set_ylabel("Probe AUROC (OOF)", fontsize=12)
    ax_main.set_title("Fisher Ratio → Probe AUROC\n(all layers, all models)", fontsize=13)
    ax_main.legend(fontsize=8)
    ax_main.set_ylim(0.4, 1.0)

    # ── Right panel: best-layer only, bound vs actual ────────────────────────
    for model_name, data in all_points.items():
        J = np.array(data["J_per_layer"])
        bound = np.array(data["auroc_bound_per_layer"])
        actual = np.array(data["probe_auroc_per_layer"])
        meta = MODEL_META.get(model_name, {"color": "black", "marker": "o"})
        best_L = int(np.argmax(J))

        ax_best.scatter(bound[best_L], actual[best_L],
                        color=meta["color"], marker=meta["marker"],
                        s=200, zorder=5, edgecolors="white", linewidths=2,
                        label=f"{model_name}\n  bound={bound[best_L]:.3f}, actual={actual[best_L]:.3f}")

    lo = 0.45
    hi = 0.90
    ax_best.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, alpha=0.5, label="Perfect calibration")
    ax_best.fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05],
                          alpha=0.1, color="green", label="±0.05 band")
    ax_best.set_xlabel("AUROC Bound (Fisher, best layer)", fontsize=12)
    ax_best.set_ylabel("Actual Probe AUROC (best layer)", fontsize=12)
    ax_best.set_title("Certificate Accuracy at Best Layer\n(all models)", fontsize=13)
    ax_best.legend(fontsize=8)
    ax_best.set_xlim(lo, hi)
    ax_best.set_ylim(lo, hi)

    plt.tight_layout()
    out = PLOTS_DIR / "06_central_figure.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Central figure saved: {out}")


if __name__ == "__main__":
    run()
