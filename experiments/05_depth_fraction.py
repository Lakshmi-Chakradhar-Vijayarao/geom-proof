"""
Exp 05 — Depth Fraction Universality

Across all models (GPT-2 117M, GPT-2 Medium 345M, Qwen 2.5 3B, Mamba 130M),
plots Fisher ratio J(L) and probe AUROC as a function of normalized depth (L/L_max).

Key question: Does the ≈89% depth fraction pattern hold across architectures?
Expected finding: For models with strong signal (3B+), yes. For null-signal
models (117M), the peak may differ — which is itself a finding.

Cost: $0 — uses Exp 01 and Exp 04 outputs.
Hardware: CPU (~1 hour).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EXP01 = RESULTS_DIR / "01_fisher_analysis.json"
EXP04 = RESULTS_DIR / "04_mamba_transfer.json"

MODEL_META = {
    "GPT-2 117M":        {"params": 117e6, "architecture": "Transformer", "color": "gray"},
    "GPT-2 Medium 345M": {"params": 345e6, "architecture": "Transformer", "color": "steelblue"},
    "Qwen 2.5 3B":       {"params": 3e9,   "architecture": "Transformer", "color": "darkorange"},
    "Mamba 130M":        {"params": 130e6, "architecture": "SSM",         "color": "green"},
}


def run() -> dict:
    print("=" * 60)
    print("Exp 05 — Depth Fraction Universality")
    print("=" * 60)

    model_data = {}

    # Load Exp 01 results
    if EXP01.exists():
        with open(EXP01) as f:
            exp01 = json.load(f)
        for model_name, result in exp01.items():
            if result.get("fisher_computed", False):
                n_layers = result["n_layers"]
                model_data[model_name] = {
                    "n_layers": n_layers,
                    "J_per_layer": result["J_per_layer"],
                    "probe_auroc_per_layer": result["probe_auroc_per_layer"],
                    "best_J_layer": result["best_J_layer"],
                    "best_probe_layer": result["best_probe_layer"],
                    "depth_fraction_J": result["best_depth_fraction_fisher"],
                    "depth_fraction_probe": result["best_depth_fraction_probe"],
                }
                print(f"  Loaded {model_name}: {n_layers} layers, "
                      f"peak depth={result['best_depth_fraction_probe']:.3f}")
    else:
        print("  Exp 01 results not found. Run Exp 01 first.")

    # Load Exp 04 results (Mamba)
    if EXP04.exists():
        with open(EXP04) as f:
            exp04 = json.load(f)
        n_layers = exp04["n_layers"]
        model_data["Mamba 130M"] = {
            "n_layers": n_layers,
            "J_per_layer": exp04["J_per_layer"],
            "probe_auroc_per_layer": exp04["probe_auroc_per_layer"],
            "best_J_layer": exp04["fisher_best_layer"],
            "best_probe_layer": exp04["probe_best_layer"],
            "depth_fraction_J": exp04["fisher_depth_fraction"],
            "depth_fraction_probe": exp04["probe_depth_fraction"],
        }
        print(f"  Loaded Mamba 130M: {n_layers} layers, "
              f"peak depth={exp04['probe_depth_fraction']:.3f}")
    else:
        print("  Exp 04 results not found. Run Exp 04 first.")

    if not model_data:
        print("No data available. Run Exp 01 and Exp 04 first.")
        return {}

    # Summary table
    print("\nDepth Fraction Summary:")
    print(f"{'Model':<25} {'Arch':<15} {'Probe peak':<12} {'Fisher peak':<12} {'≈89%?'}")
    print("─" * 70)
    depth_summary = {}
    for model_name, data in model_data.items():
        arch = MODEL_META.get(model_name, {}).get("architecture", "?")
        df_probe = data["depth_fraction_probe"]
        df_fisher = data["depth_fraction_J"]
        close_to_89 = "YES" if abs(df_probe - 0.89) < 0.05 else "no"
        print(f"  {model_name:<23} {arch:<15} {df_probe:.3f}        {df_fisher:.3f}        {close_to_89}")
        depth_summary[model_name] = {
            "architecture": arch,
            "depth_fraction_probe": df_probe,
            "depth_fraction_fisher": df_fisher,
            "close_to_89_pct": close_to_89 == "YES",
        }

    results = {
        "experiment": "05_depth_fraction",
        "depth_summary": depth_summary,
    }
    out_path = RESULTS_DIR / "05_depth_fraction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    _plot(model_data)
    return results


def _plot(model_data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, data in model_data.items():
        meta = MODEL_META.get(model_name, {})
        color = meta.get("color", "black")
        n_layers = data["n_layers"]
        norm_depth = [L / (n_layers - 1) for L in range(n_layers)]

        # Normalize J to [0,1] for comparison across models
        J = np.array(data["J_per_layer"])
        J_norm = (J - J.min()) / (J.max() - J.min() + 1e-10)
        probe_aurocs = data["probe_auroc_per_layer"]

        ax1.plot(norm_depth, J_norm, label=model_name, color=color, linewidth=2)
        ax2.plot(norm_depth, probe_aurocs, label=model_name, color=color, linewidth=2)

    for ax in (ax1, ax2):
        ax.axvline(0.89, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="89% depth")
        ax.set_xlabel("Normalized Depth (L / L_max)", fontsize=12)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Normalized Fisher Ratio J(L)", fontsize=12)
    ax1.set_title("Fisher Ratio vs Depth — All Models", fontsize=13)
    ax1.set_ylim(0, 1.05)

    ax2.set_ylabel("Probe AUROC (OOF)", fontsize=12)
    ax2.set_title("Probe AUROC vs Depth — All Models", fontsize=13)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylim(0.4, 1.0)

    plt.tight_layout()
    out = PLOTS_DIR / "05_depth_fraction_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()
