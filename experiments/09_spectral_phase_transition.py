"""
Exp 09 — Spectral Phase Transition Analysis (Direction A)

The RMT hypothesis: the emergence of the hallucination signal at layer ~89%
depth is a Baik-Ben Arous-Péché (BBP) phase transition in the spectral
structure of the hidden-state covariance.

Below the BBP threshold (signal eigenvalue < σ²√(d/n)), the class-mean
difference is buried in the noise bulk of the ESD and no linear probe can
detect it. Above the threshold, a spike eigenvalue emerges from the bulk
and the probe works.

This experiment:
  1. Computes ESD at every layer for all three models
  2. Measures KL divergence from Marchenko-Pastur (departure from null/noise)
  3. Tracks spike count and spike ratio above the bulk upper edge λ_+
  4. Plots the spectral KL curve alongside the W₂ curve and probe AUROC curve
  5. Tests whether the three curves exhibit a coincident phase transition

Key observational prediction:
  - GPT-2 117M: spectral KL ≈ 0 at all layers (no signal above bulk) → null AUROC
  - GPT-2 Medium 345M: spectral KL rises weakly → weak probe AUROC
  - Qwen 2.5 3B: spectral KL rises sharply at L28-L32 → AUROC lifts from chance

If these three curves (spectral KL, W₂, probe AUROC) all show the same
phase transition at the same layer, this is the first empirical evidence
of a BBP-type transition controlling hallucination detectability.

Cost: $0 — uses existing hidden states.
Hardware: CPU (~2–3 hours).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.spectral import spectral_curve, bbp_threshold

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HS_DIR = ROOT / "results" / "hidden_states"

MODELS = {
    "Qwen 2.5 3B": {
        "hs_path": HS_DIR / "00_halueval_qwen3b.npz",
        "labels_path": None,  # embedded
        "labels_fallback": None,
        "n_params": 3e9,
        "color": "darkorange",
    },
    "GPT-2 Medium 345M": {
        "hs_path": HS_DIR / "00_halueval_gpt2med.npz",
        "labels_path": None,  # embedded
        "labels_fallback": None,
        "n_params": 345e6,
        "color": "steelblue",
    },
    "Mamba-2 130M": {
        "hs_path": ROOT / "results" / "hidden_states" / "04_mamba_hidden_states.npz",
        "labels_path": None,
        "labels_fallback": None,
        "n_params": 130e6,
        "color": "green",
    },
}


def load_hs(cfg: dict) -> np.ndarray | None:
    if not cfg["hs_path"].exists():
        return None
    data = np.load(cfg["hs_path"])
    key = "hidden_states" if "hidden_states" in data else list(data.keys())[0]
    return data[key].astype(np.float32)


def run() -> dict:
    print("=" * 70)
    print("Exp 09 — Spectral Phase Transition Analysis (RMT / BBP)")
    print("=" * 70)

    all_results = {}

    # Load Exp 08 OT results if available (for direct curve comparison)
    exp08_results = {}
    exp08_path = RESULTS_DIR / "08_ot_certificate.json"
    if exp08_path.exists():
        with open(exp08_path) as f:
            exp08_results = json.load(f)
        print("  Loaded Exp 08 OT certificate curves for comparison.")

    for model_name, cfg in MODELS.items():
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")
        H = load_hs(cfg)
        if H is None:
            print(f"  Skipping — data not found.")
            continue

        n, n_layers, d = H.shape
        gamma = d / n
        theta_star = bbp_threshold(d, n)
        print(f"  Shape: {H.shape}, γ=d/n={gamma:.3f}, BBP threshold θ*={theta_star:.4f}")

        print("\n  Computing spectral curve (ESD + Marchenko-Pastur KL) ...")
        spec = spectral_curve(H, verbose=True)

        result = {
            "model_name": model_name,
            "n_params": cfg["n_params"],
            "n_layers": int(n_layers),
            "n_samples": int(n),
            "d": int(d),
            "gamma": float(gamma),
            "bbp_threshold": float(theta_star),
            "kl_per_layer": spec["kl_per_layer"],
            "n_spikes_per_layer": spec["n_spikes_per_layer"],
            "spike_ratio_per_layer": spec["spike_ratio_per_layer"],
            "best_spectral_layer": spec["best_layer"],
            "best_kl": spec["best_kl"],
            "best_depth_fraction": spec["best_depth_fraction"],
        }

        # If Exp 08 ran, check whether spectral KL and SW₂ peaks coincide
        if model_name in exp08_results:
            r08 = exp08_results[model_name]
            sw2_best = r08["best_layer"]["SW2"]["layer"]
            probe_best = r08["best_layer"]["Probe"]["layer"]
            spec_best = spec["best_layer"]
            result["spectral_vs_sw2_agreement"] = abs(spec_best - sw2_best) <= 2
            result["spectral_vs_probe_agreement"] = abs(spec_best - probe_best) <= 2
            print(f"\n  Phase transition coincidence:")
            print(f"    Spectral KL peak: L{spec_best} ({spec['best_depth_fraction']:.3f})")
            print(f"    SW₂ peak:         L{sw2_best}")
            print(f"    Probe AUROC peak: L{probe_best}")
            print(f"    All within ±2 layers: "
                  f"{'YES ✓' if result['spectral_vs_probe_agreement'] else 'NO'}")

        all_results[model_name] = result

    out = RESULTS_DIR / "09_spectral_phase_transition.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    _plot(all_results, exp08_results)
    return all_results


def _plot(results: dict, exp08_results: dict) -> None:
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(8*n, 6))
    if n == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, results.items()):
        cfg_color = {"Qwen 2.5 3B": "darkorange",
                     "GPT-2 Medium 345M": "steelblue",
                     "Mamba-2 130M": "green"}.get(model_name, "black")
        n_layers = r["n_layers"]
        nd = [L/(n_layers-1) for L in range(n_layers)]

        def norm01(x):
            x = np.array(x, dtype=float)
            finite = x[np.isfinite(x)]
            if len(finite) == 0: return x
            return (x - finite.min()) / (finite.max() - finite.min() + 1e-10)

        # Spectral KL curve
        kl_vals = r["kl_per_layer"]
        ax.plot(nd, norm01(kl_vals), color=cfg_color, lw=2.5,
                label="Spectral KL (ESD from MP)")

        # SW₂ and probe from Exp 08 if available and same layer count
        if model_name in exp08_results:
            r08 = exp08_results[model_name]
            if len(r08["sw2_per_layer"]) == len(nd):
                ax.plot(nd, norm01(r08["sw2_per_layer"]), color="purple", lw=2,
                        ls="--", label="SW₂ (normalized)")
                ax.plot(nd, r08["probe_auroc_per_layer"], color="black", lw=1.5,
                        ls=":", label="Probe AUROC")

        ax.axvline(0.89, color="red", ls=":", lw=1, alpha=0.5, label="89% depth")
        ax.set_xlabel("Normalized Depth")
        ax.set_ylabel("Normalized Value")
        ax.set_title(f"{model_name}\nγ=d/n={r['gamma']:.2f}, "
                     f"BBP θ*={r['bbp_threshold']:.3f}")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.1)

    plt.suptitle("Spectral Phase Transition: ESD Departure from Marchenko-Pastur",
                 fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "09_spectral_phase_transition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()
