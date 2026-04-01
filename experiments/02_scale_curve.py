"""
Exp 02 — Scale Curve Formalization

Fits a log-linear relationship between model scale (log10 params) and best-layer AUROC.
Uses three confirmed data points from MECH-INT and HaRP.
Extrapolates to 7B with bootstrapped confidence intervals.
Produces the 7B prediction that will be pre-registered for GUARDIAN.

Cost: $0 — pure computation on prior results.
Hardware: CPU (~30 minutes).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.scale_curve import fit_scale_curve, bootstrap_prediction, SCALE_DATA

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET_MODELS = {
    "Mistral 7B": 7e9,
    "Llama 3 8B": 8e9,
}

# If Exp 01 ran, use its best-probe AUROCs (more accurate than known values).
# Otherwise, fall back to SCALE_DATA defaults.
EXP01_RESULTS = RESULTS_DIR / "01_fisher_analysis.json"


def load_exp01_aurocs() -> dict | None:
    if not EXP01_RESULTS.exists():
        return None
    with open(EXP01_RESULTS) as f:
        data = json.load(f)
    result = {}
    for model, v in data.items():
        if v.get("fisher_computed", False) and "best_probe_auroc" in v:
            result[model] = {"params": v["n_params"], "auroc": v["best_probe_auroc"]}
    return result if result else None


def run() -> dict:
    print("=" * 60)
    print("Exp 02 — Scale Curve Formalization")
    print("=" * 60)

    # Use Exp 01 outputs if available, else fall back to known values
    exp01 = load_exp01_aurocs()
    if exp01:
        print(f"  Using Exp 01 OOF probe AUROCs: {list(exp01.keys())}")
        # Merge: use Exp 01 for models we have, SCALE_DATA for any missing
        scale_data = dict(SCALE_DATA)
        for model, vals in exp01.items():
            if model in scale_data:
                scale_data[model]["auroc"] = vals["auroc"]
    else:
        print("  Exp 01 not found — using known AUROCs from MECH-INT and HaRP.")
        scale_data = SCALE_DATA

    print("\nData points:")
    for model, v in scale_data.items():
        print(f"  {model}: {v['params']/1e6:.0f}M params, AUROC={v['auroc']:.4f}")

    # Fit scale curve
    fit = fit_scale_curve(scale_data)
    print(f"\nFit: AUROC = sigmoid({fit['a']:.4f} * log10(params) + {fit['b']:.4f})")
    print(f"R² = {fit['r_squared']:.4f}")
    print("\nResiduals:")
    for model, residual in fit["residuals"].items():
        print(f"  {model}: {residual:+.4f}")

    # 7B predictions
    predictions = {}
    for target_name, n_params in TARGET_MODELS.items():
        print(f"\n{'─'*40}")
        print(f"Prediction for {target_name} ({n_params/1e9:.0f}B params):")
        pred = bootstrap_prediction(n_params, scale_data, n_bootstrap=10_000, ci=0.90)
        predictions[target_name] = {"n_params": n_params, **pred}
        print(f"  Point estimate: AUROC = {pred['point_estimate']:.4f}")
        print(f"  90% CI: [{pred['ci_lower']:.4f}, {pred['ci_upper']:.4f}]")
        print(f"  (CI is wide — only 3 data points. This is honest.)")

    results = {
        "fit": {k: v for k, v in fit.items() if k != "predict"},
        "data_points": {k: {"params": v["params"], "auroc": v["auroc"]} for k, v in scale_data.items()},
        "predictions": predictions,
    }

    out_path = RESULTS_DIR / "02_scale_curve.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    _plot(scale_data, fit, predictions)
    return results


def _plot(scale_data: dict, fit: dict, predictions: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    # Smooth curve
    log_range = np.linspace(7.5, 10.5, 300)  # 30M to 10B
    from scipy.special import expit as sigmoid
    smooth_auroc = sigmoid(fit["a"] * log_range + fit["b"])
    ax.plot(10 ** log_range / 1e9, smooth_auroc,
            color="steelblue", linewidth=2.5, label="Fitted curve", zorder=3)

    # Known data points
    colors = {"GPT-2 117M": "gray", "GPT-2 Medium 345M": "steelblue", "Qwen 2.5 3B": "darkorange"}
    for model, v in scale_data.items():
        ax.scatter(v["params"] / 1e9, v["auroc"],
                   color=colors.get(model, "black"),
                   s=120, zorder=5, label=f"{model} ({v['auroc']:.3f})", edgecolors="white", linewidths=1.5)

    # 7B predictions
    for target_name, pred in predictions.items():
        x = pred["n_params"] / 1e9
        ax.errorbar(
            x, pred["point_estimate"],
            yerr=[[pred["point_estimate"] - pred["ci_lower"]],
                  [pred["ci_upper"] - pred["point_estimate"]]],
            fmt="D", color="crimson", markersize=10, capsize=6, linewidth=2,
            label=f"{target_name} (predicted: {pred['point_estimate']:.3f}, 90% CI [{pred['ci_lower']:.3f}, {pred['ci_upper']:.3f}])",
            zorder=6,
        )

    ax.axhline(0.70, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Governance threshold (0.70)")
    ax.axhline(0.55, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Above-chance threshold (0.55)")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance (0.50)")

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (Billions of Parameters)", fontsize=12)
    ax.set_ylabel("Best-Layer Probe AUROC", fontsize=12)
    ax.set_title("Scale Curve: Hallucination Signal vs Model Size", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.45, 0.95)
    ax.set_xlim(0.05, 20)

    plt.tight_layout()
    out = PLOTS_DIR / "02_scale_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()
