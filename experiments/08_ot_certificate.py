"""
Exp 08 — The Wasserstein Generalization of the Fisher Certificate

This experiment is the heart of the B direction.

Core claim: The Fisher certificate J(L) is the Wasserstein-2 distance
between the class-conditional hidden-state distributions — in whitened,
Gaussian-approximated space.

This experiment:
  1. Verifies the identity: J ≈ W₂²(P_c, P_h) in whitened space  [mathematical]
  2. Computes non-Gaussian alternatives: SW₂, MMD, Bures W₂       [empirical]
  3. Compares all four as predictors of probe AUROC across layers   [comparison]
  4. Tests the comparison on Mamba-2 if Exp 04 has run             [architecture transfer]

Key question: In layers where the class-conditional distributions are
non-Gaussian (early layers, small models), does SW₂ predict probe AUROC
better than J?

Cost: $0 — uses existing hidden states.
Hardware: CPU (~3–4 hours, SW₂ is the bottleneck).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.wasserstein import (
    ot_certificate_curve, fisher_as_whitened_w2, bures_w2_equal_cov
)
from src.fisher import fisher_curve

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
    },
    "GPT-2 Medium 345M": {
        "hs_path": HS_DIR / "00_halueval_gpt2med.npz",
        "labels_path": None,  # embedded
        "labels_fallback": None,
        "n_params": 345e6,
    },
    "Mamba-2 130M": {
        "hs_path": RESULTS_DIR.parent / "hidden_states" / "04_mamba_hidden_states.npz",
        "labels_path": RESULTS_DIR.parent / "hidden_states" / "04_mamba_hidden_states.npz",
        "labels_fallback": None,
        "n_params": 130e6,
    },
}


def load_model_data(cfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if not cfg["hs_path"].exists():
        return None
    data = np.load(cfg["hs_path"])
    key = "hidden_states" if "hidden_states" in data else list(data.keys())[0]
    H = data[key].astype(np.float32)

    # Labels: embedded in npz, or from external file
    if "labels" in data:
        y = data["labels"].astype(int)
    elif cfg["labels_path"] is not None and cfg["labels_path"].exists():
        y = np.load(cfg["labels_path"]).astype(int)
    elif cfg["labels_fallback"] and Path(cfg["labels_fallback"]).exists():
        y = np.load(cfg["labels_fallback"]).astype(int)
    else:
        return None

    n = min(H.shape[0], y.shape[0])
    return H[:n], y[:n]


def probe_auroc_per_layer(H: np.ndarray, y: np.ndarray) -> list[float]:
    n_samples, n_layers, d = H.shape
    aurocs, skf = [], StratifiedKFold(5, shuffle=True, random_state=42)
    for L in range(n_layers):
        H_L = H[:, L, :]
        oof = np.zeros(n_samples)
        for tr, te in skf.split(H_L, y):
            sc = StandardScaler()
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(sc.fit_transform(H_L[tr]), y[tr])
            oof[te] = lr.predict_proba(sc.transform(H_L[te]))[:, 1]
        aurocs.append(float(roc_auc_score(y, oof)))
    return aurocs


def run() -> dict:
    print("=" * 70)
    print("Exp 08 — OT Certificate: Wasserstein Generalization of Fisher")
    print("=" * 70)

    all_results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")
        loaded = load_model_data(cfg)
        if loaded is None:
            print(f"  Skipping — data not found.")
            continue
        H, y = loaded
        print(f"  Shape: {H.shape}, hall_rate: {1-y.mean():.3f}")

        if len(np.unique(y)) < 2:
            print(f"  Skipping — single-class labels (hall_rate={1-y.mean():.3f}). "
                  "Cannot compute Fisher certificate or probe AUROC.")
            continue

        # 1. Verify the identity: J = W₂ in whitened space
        print("\n  [1] Verifying Fisher = W₂ in whitened space ...")
        identity_checks = []
        # Check at 5 representative layers
        check_layers = np.linspace(0, H.shape[1]-1, 5, dtype=int)
        for L in check_layers:
            check = fisher_as_whitened_w2(H[:, L, :], y)
            identity_checks.append({
                "layer": int(L),
                "J": round(check["J"], 5),
                "w2_whitened": round(check["w2_whitened"], 5),
                "relative_error": round(check["relative_error"], 5),
                "identity_holds": check["identity_holds"],
            })
            print(f"    L{L:3d}: J={check['J']:.5f}  W₂_white={check['w2_whitened']:.5f}  "
                  f"err={check['relative_error']:.4f}  "
                  f"{'✓' if check['identity_holds'] else '✗'}")

        # 2. Compute Fisher J(L) curve
        print("\n  [2] Fisher J(L) curve ...")
        fisher_result = fisher_curve(H, y, method="pca", n_components=100, verbose=False)

        # 3. Compute OT certificate curve (SW₂, MMD, Bures W₂)
        print("\n  [3] OT certificate curve (SW₂, MMD, Bures W₂) ...")
        ot_result = ot_certificate_curve(H, y, n_projections=100, n_components=50,
                                          verbose=True)

        # 4. OOF probe AUROC at each layer
        print("\n  [4] OOF probe per layer ...")
        probe_aurocs = probe_auroc_per_layer(H, y)

        # 5. Compare: which certificate best predicts probe AUROC?
        print("\n  [5] Correlation analysis ...")
        J_vals = np.array(fisher_result["J"])
        sw2_vals = np.array(ot_result["sw2_per_layer"])
        mmd_vals = np.array(ot_result["mmd_per_layer"])
        w2_eq_vals = np.array(ot_result["w2_eq_per_layer"])
        probe_vals = np.array(probe_aurocs)

        # Spearman correlation: certificate vs probe AUROC across layers
        r_J = float(spearmanr(J_vals, probe_vals).statistic)
        r_sw2 = float(spearmanr(sw2_vals, probe_vals).statistic)
        r_mmd = float(spearmanr(mmd_vals, probe_vals).statistic)
        r_w2eq = float(spearmanr(w2_eq_vals, probe_vals).statistic)

        print(f"    Spearman r (Fisher J):     {r_J:.4f}")
        print(f"    Spearman r (Sliced W₂):    {r_sw2:.4f}")
        print(f"    Spearman r (Bures W₂_eq):  {r_w2eq:.4f}")
        print(f"    Spearman r (MMD²):         {r_mmd:.4f}")

        # Best-layer comparison
        best = {
            "Fisher": {"layer": fisher_result["best_layer"],
                       "certificate": fisher_result["best_J"],
                       "depth": fisher_result["depth_fraction"]},
            "SW2": {"layer": ot_result["best_layer_sw2"],
                    "certificate": ot_result["best_sw2"],
                    "depth": ot_result["best_depth_fraction_sw2"]},
            "Probe": {"layer": int(np.argmax(probe_aurocs)),
                      "auroc": float(np.max(probe_aurocs)),
                      "depth": float(np.argmax(probe_aurocs) / (H.shape[1]-1))},
        }

        print(f"\n  Best-layer summary:")
        print(f"    Fisher best: L{best['Fisher']['layer']} ({best['Fisher']['depth']:.3f})")
        print(f"    SW₂ best:    L{best['SW2']['layer']} ({best['SW2']['depth']:.3f})")
        print(f"    Probe best:  L{best['Probe']['layer']} ({best['Probe']['depth']:.3f})")

        result = {
            "model_name": model_name,
            "n_params": cfg["n_params"],
            "n_layers": H.shape[1],
            "n_samples": H.shape[0],
            "identity_verification": identity_checks,
            "spearman_correlations": {
                "Fisher_J": r_J, "SW2": r_sw2, "MMD2": r_mmd, "W2_eq": r_w2eq
            },
            "best_layer": best,
            "fisher_per_layer": [float(j) for j in J_vals],
            "sw2_per_layer": ot_result["sw2_per_layer"],
            "mmd_per_layer": ot_result["mmd_per_layer"],
            "w2_bures_per_layer": ot_result["w2_bures_per_layer"],
            "w2_eq_per_layer": ot_result["w2_eq_per_layer"],
            "probe_auroc_per_layer": probe_aurocs,
        }
        all_results[model_name] = result

    out = RESULTS_DIR / "08_ot_certificate.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    _plot(all_results)
    _print_winner_table(all_results)
    return all_results


def _print_winner_table(results: dict) -> None:
    print("\n" + "="*70)
    print("CERTIFICATE COMPARISON — Which predicts probe AUROC best?")
    print(f"{'Model':<25} {'Fisher J':>10} {'SW₂':>10} {'MMD²':>10} {'W₂_eq':>10} {'WINNER':>10}")
    print("─"*75)
    for model, r in results.items():
        corrs = r["spearman_correlations"]
        best = max(corrs, key=corrs.get)
        print(f"  {model:<23} "
              f"{corrs['Fisher_J']:>10.4f} "
              f"{corrs['SW2']:>10.4f} "
              f"{corrs['MMD2']:>10.4f} "
              f"{corrs['W2_eq']:>10.4f} "
              f"{best:>10}")


def _plot(results: dict) -> None:
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8*n, 6))
    if n == 1:
        axes = [axes]
    colors = {"Fisher J": "steelblue", "SW₂": "darkorange",
              "MMD²": "green", "Probe AUROC": "black"}

    for ax, (model_name, r) in zip(axes, results.items()):
        n_layers = r["n_layers"]
        nd = [L/(n_layers-1) for L in range(n_layers)]

        # Normalize certificates to [0,1] for comparison
        def norm01(x):
            x = np.array(x, dtype=float)
            return (x - x.min()) / (x.max() - x.min() + 1e-10)

        ax.plot(nd, norm01(r["fisher_per_layer"]),
                label="Fisher J (normalized)", color=colors["Fisher J"], lw=2)
        ax.plot(nd, norm01(r["sw2_per_layer"]),
                label="SW₂ (normalized)", color=colors["SW₂"], lw=2)
        ax.plot(nd, norm01(r["mmd_per_layer"]),
                label="MMD² (normalized)", color=colors["MMD²"], lw=1.5, ls="--")
        ax.plot(nd, r["probe_auroc_per_layer"],
                label="Probe AUROC (OOF)", color=colors["Probe AUROC"], lw=2, ls=":")
        ax.axvline(0.89, color="red", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Normalized Depth")
        ax.set_ylabel("Certificate (normalized) / AUROC")
        ax.set_title(f"{model_name}\nSpearman r: "
                     f"Fisher={r['spearman_correlations']['Fisher_J']:.3f}, "
                     f"SW₂={r['spearman_correlations']['SW2']:.3f}")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.1)

    plt.suptitle("OT Certificate Comparison: Fisher vs Wasserstein vs MMD", fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "08_ot_certificates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()
