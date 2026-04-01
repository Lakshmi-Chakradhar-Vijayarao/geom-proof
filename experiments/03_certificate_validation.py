"""
Exp 03 — Certificate Validation (K-Fold Pre-registration Simulation)

Simulates the pre-registration setting within the HaRP dataset.
For each of 5 CV folds:
  1. Compute J(L) on the TRAINING FOLD only
  2. Predict AUROC_bound = Phi(sqrt(J)/2) — before seeing test fold
  3. Run OOF probe on TEST FOLD → actual AUROC
  4. Compare prediction to measurement

This proves the certificate is useful at test time — it predicts probe AUROC
from geometry alone, before any probe is trained on the test distribution.

Cost: $0 — existing Qwen 2.5 3B hidden states.
Hardware: CPU (~1 hour).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import fisher_ratio, auroc_bound

RESULTS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HS_DIR = ROOT / "results" / "hidden_states"
HIDDEN_STATES_PATH = HS_DIR / "00_halueval_qwen3b.npz"
LABELS_PATH = None  # labels embedded in npz

N_SPLITS = 5
METHOD = "pca"
N_COMPONENTS = 100
RANDOM_STATE = 42


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(HIDDEN_STATES_PATH)
    key = "hidden_states" if "hidden_states" in data else list(data.keys())[0]
    H = data[key].astype(np.float32)
    if LABELS_PATH is not None:
        y = np.load(LABELS_PATH).astype(int)
    else:
        y = data["labels"].astype(int)
    n = min(H.shape[0], y.shape[0])
    return H[:n], y[:n]


def run() -> dict:
    print("=" * 60)
    print("Exp 03 — Certificate Validation (K-fold simulation)")
    print("=" * 60)

    if not HIDDEN_STATES_PATH.exists():
        print(f"Hidden states not found at {HIDDEN_STATES_PATH}")
        print("Run Job A on Kaggle first to generate 00_halueval_qwen3b.npz.")
        return {}

    H, y = load_data()
    n_samples, n_layers, d = H.shape
    print(f"Loaded: {H.shape}, labels: {y.shape}, hall_rate={1-y.mean():.3f}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(H.reshape(n_samples, -1), y)):
        print(f"\n{'─'*40}")
        print(f"Fold {fold_idx + 1}/{N_SPLITS} — train={len(train_idx)}, test={len(test_idx)}")

        H_train = H[train_idx]
        y_train = y[train_idx]
        H_test = H[test_idx]
        y_test = y[test_idx]

        # Step 1: Compute J(L) on training fold only — simulate pre-registration
        print("  Computing Fisher ratios on training fold ...")
        J_per_layer = []
        for L in range(n_layers):
            J_L = fisher_ratio(H_train[:, L, :], y_train, method=METHOD, n_components=N_COMPONENTS)
            J_per_layer.append(J_L)

        # Step 2: Select best layer from training fold Fisher ratio
        predicted_best_layer = int(np.argmax(J_per_layer))
        predicted_J = J_per_layer[predicted_best_layer]
        predicted_auroc_bound = auroc_bound(predicted_J)
        depth_fraction = predicted_best_layer / (n_layers - 1)

        print(f"  Training-fold prediction: best_layer={predicted_best_layer}, "
              f"J={predicted_J:.4f}, AUROC_bound={predicted_auroc_bound:.4f} "
              f"(depth={depth_fraction:.3f})")

        # Step 3: Run probe at the training-fold-selected layer on TEST fold
        H_L_train = H_train[:, predicted_best_layer, :]
        H_L_test = H_test[:, predicted_best_layer, :]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(H_L_train)
        X_test_scaled = scaler.transform(H_L_test)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
        lr.fit(X_train_scaled, y_train)
        test_probs = lr.predict_proba(X_test_scaled)[:, 1]
        actual_auroc = float(roc_auc_score(y_test, test_probs))

        # Also find actual best layer on test fold (oracle — what we'd get with data leakage)
        oracle_aurocs = []
        for L in range(n_layers):
            H_L_train2 = H_train[:, L, :]
            H_L_test2 = H_test[:, L, :]
            scaler2 = StandardScaler()
            X_tr = scaler2.fit_transform(H_L_train2)
            X_te = scaler2.transform(H_L_test2)
            lr2 = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
            lr2.fit(X_tr, y_train)
            preds = lr2.predict_proba(X_te)[:, 1]
            oracle_aurocs.append(float(roc_auc_score(y_test, preds)))

        oracle_best_layer = int(np.argmax(oracle_aurocs))
        oracle_auroc = oracle_aurocs[oracle_best_layer]

        print(f"  Actual AUROC at predicted layer {predicted_best_layer}: {actual_auroc:.4f}")
        print(f"  Oracle best layer: {oracle_best_layer} (AUROC {oracle_auroc:.4f})")
        print(f"  Bound error: {abs(predicted_auroc_bound - actual_auroc):.4f}")
        print(f"  Layer match: {predicted_best_layer == oracle_best_layer}")

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "predicted_best_layer": predicted_best_layer,
            "predicted_J": float(predicted_J),
            "predicted_auroc_bound": float(predicted_auroc_bound),
            "predicted_depth_fraction": float(depth_fraction),
            "actual_auroc_at_predicted_layer": actual_auroc,
            "oracle_best_layer": oracle_best_layer,
            "oracle_auroc": oracle_auroc,
            "oracle_depth_fraction": float(oracle_best_layer / (n_layers - 1)),
            "bound_error": float(abs(predicted_auroc_bound - actual_auroc)),
            "layer_match": predicted_best_layer == oracle_best_layer,
        })

    # Summary statistics
    bound_errors = [r["bound_error"] for r in fold_results]
    actual_aurocs = [r["actual_auroc_at_predicted_layer"] for r in fold_results]
    predicted_bounds = [r["predicted_auroc_bound"] for r in fold_results]
    layer_matches = [r["layer_match"] for r in fold_results]
    depth_fractions = [r["predicted_depth_fraction"] for r in fold_results]

    summary = {
        "mean_bound_error": float(np.mean(bound_errors)),
        "std_bound_error": float(np.std(bound_errors)),
        "max_bound_error": float(np.max(bound_errors)),
        "mean_actual_auroc": float(np.mean(actual_aurocs)),
        "mean_predicted_bound": float(np.mean(predicted_bounds)),
        "layer_match_rate": float(np.mean(layer_matches)),
        "mean_depth_fraction": float(np.mean(depth_fractions)),
        "certificate_valid": float(np.mean(bound_errors)) < 0.05,
    }

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Mean bound error: {summary['mean_bound_error']:.4f} ± {summary['std_bound_error']:.4f}")
    print(f"  Max bound error: {summary['max_bound_error']:.4f}")
    print(f"  Layer match rate: {summary['layer_match_rate']:.2f} ({sum(layer_matches)}/{N_SPLITS} folds)")
    print(f"  Mean predicted bound: {summary['mean_predicted_bound']:.4f}")
    print(f"  Mean actual AUROC: {summary['mean_actual_auroc']:.4f}")
    print(f"  Certificate valid (MAE < 0.05): {summary['certificate_valid']}")

    results = {
        "experiment": "03_certificate_validation",
        "method": METHOD,
        "n_components": N_COMPONENTS,
        "n_splits": N_SPLITS,
        "fold_results": fold_results,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "03_certificate_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    _plot(fold_results, summary)
    return results


def _plot(fold_results: list, summary: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: predicted bound vs actual AUROC
    predicted = [r["predicted_auroc_bound"] for r in fold_results]
    actual = [r["actual_auroc_at_predicted_layer"] for r in fold_results]

    ax1.scatter(predicted, actual, s=100, color="steelblue", zorder=5, edgecolors="white", linewidths=1.5)
    lo, hi = min(predicted + actual) - 0.03, max(predicted + actual) + 0.03
    ax1.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax1.fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05],
                     alpha=0.1, color="green", label="±0.05 band")
    for i, r in enumerate(fold_results):
        ax1.annotate(f"Fold {r['fold']}", (predicted[i], actual[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.set_xlabel("Predicted AUROC (Fisher bound, training fold)")
    ax1.set_ylabel("Actual AUROC (probe on test fold)")
    ax1.set_title(f"Certificate Calibration\nMAE = {summary['mean_bound_error']:.4f}")
    ax1.legend(fontsize=8)

    # Bar: bound error per fold
    folds = [r["fold"] for r in fold_results]
    errors = [r["bound_error"] for r in fold_results]
    colors = ["green" if e < 0.05 else "crimson" for e in errors]
    ax2.bar(folds, errors, color=colors, edgecolor="white", linewidth=1.5)
    ax2.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Pass threshold (0.05)")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("|Bound − Actual|")
    ax2.set_title(f"Bound Error per Fold\nLayer match rate: {summary['layer_match_rate']:.0%}")
    ax2.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "03_certificate_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    run()
