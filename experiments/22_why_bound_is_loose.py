"""
Exp 22 -- decomposes WHY the Fisher-AUROC bound (Prop. 1) is loose, rather
than only reporting that it is (Table certificate-matched, \S{cert-validation}).

Prop. 1's Gaussian-equal-covariance assumption implies that projecting onto
the Fisher/LDA direction w = Sigma_w^{-1}(mu_c - mu_h) should give a
per-class scalar score s(h) = w^T h that is GAUSSIAN. This script tests
that assumption directly, rather than only the marginal consequence
(AUROC), at each of the six models' best-J layer (Table certificate-matched,
reusing the exact PCA(100)+Ledoit-Wolf direction estimation in
src/fisher.py::fisher_ratio(method="pca")).

For each model: project every sample onto its Fisher direction, split by
class, and report (a) skewness and excess kurtosis of each class's
projected-score distribution (0/0 under a true Gaussian), and (b) a
D'Agostino-Pearson omnibus normality test p-value per class (scipy
normaltest, combines skew+kurtosis into one statistic; small p = reject
Gaussian). Then correlates a per-model non-Gaussianity summary against
the already-reported same-layer bound error (Table certificate-matched)
across the six models, to see whether models with less Gaussian
class-conditional geometry also have looser bounds.

Zero new model inference -- reuses only already-cached hidden states
(results/hidden_states/15_hidden_states_*.npz) and already-reported
Table certificate-matched error values (hardcoded below from main.tex,
cited by row).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import normaltest, skew, kurtosis, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
HS_DIR = ROOT / "results" / "hidden_states"
OUT_PATH = ROOT / "results" / "logs" / "22_why_bound_is_loose.json"

# best-J layer per model, from results/logs/15_result_<model>.json's best_J_layer
MODELS = {
    "gpt2_117m":   {"file": "15_hidden_states_gpt2_117m.npz",   "best_j_layer": 8,  "table_error": 0.410},
    "gpt2_medium": {"file": "15_hidden_states_gpt2_medium.npz", "best_j_layer": 20, "table_error": 0.360},
    "qwen_0.5b":   {"file": "15_hidden_states_qwen_0.5b.npz",   "best_j_layer": 24, "table_error": 0.362},
    "qwen_1.5b":   {"file": "15_hidden_states_qwen_1.5b.npz",   "best_j_layer": 22, "table_error": 0.316},
    "qwen_3b":     {"file": "15_hidden_states_qwen_3b.npz",     "best_j_layer": 36, "table_error": 0.337},
    "qwen_7b":     {"file": "15_hidden_states_qwen_7b.npz",     "best_j_layer": 24, "table_error": 0.332},
}

N_COMPONENTS = 100


def fisher_direction_scores(H, y, n_components=N_COMPONENTS):
    """Reproduces src/fisher.py::fisher_ratio(method='pca')'s direction estimation,
    but additionally returns the per-sample projected scalar scores (not just J)."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    mask_c, mask_h = y == 1, y == 0

    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k)
    H_proj = pca.fit_transform(H)

    mu_c_proj = H_proj[mask_c].mean(axis=0)
    mu_h_proj = H_proj[mask_h].mean(axis=0)
    H_c_centered = H_proj[mask_c] - mu_c_proj
    H_h_centered = H_proj[mask_h] - mu_h_proj
    H_centered = np.vstack([H_c_centered, H_h_centered])

    lw = LedoitWolf()
    lw.fit(H_centered)
    Sigma_w_proj = lw.covariance_

    delta_proj = mu_c_proj - mu_h_proj
    try:
        w = np.linalg.solve(Sigma_w_proj, delta_proj)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Sigma_w_proj, delta_proj, rcond=None)[0]
    J = float(delta_proj @ w)

    # per-sample scalar Fisher scores (the quantity Prop. 1 assumes is Gaussian)
    scores = H_proj @ w
    return scores[mask_c], scores[mask_h], J


def analyze_model(model_key, cfg):
    npz_path = HS_DIR / cfg["file"]
    d = np.load(npz_path, allow_pickle=True)
    hidden_states, labels = d["hidden_states"], d["labels"]
    layer = cfg["best_j_layer"]
    H = hidden_states[:, layer, :]

    s_correct, s_hall, J = fisher_direction_scores(H, labels)

    result = {
        "model": model_key,
        "layer": layer,
        "n_correct": int(len(s_correct)),
        "n_hall": int(len(s_hall)),
        "J_recomputed": round(J, 4),
        "table_certificate_matched_error": cfg["table_error"],
    }
    for name, s in [("correct", s_correct), ("hallucinated", s_hall)]:
        skew_v = float(skew(s))
        kurt_v = float(kurtosis(s))  # excess kurtosis (0 = Gaussian)
        stat, p = normaltest(s)  # D'Agostino-Pearson omnibus test
        result[f"{name}_skew"] = round(skew_v, 4)
        result[f"{name}_excess_kurtosis"] = round(kurt_v, 4)
        result[f"{name}_normaltest_stat"] = round(float(stat), 4)
        result[f"{name}_normaltest_p"] = round(float(p), 6)

    # summary non-Gaussianity score: mean |skew| + mean |excess kurtosis| across both classes
    result["non_gaussianity_score"] = round(
        (abs(result["correct_skew"]) + abs(result["hallucinated_skew"])
         + abs(result["correct_excess_kurtosis"]) + abs(result["hallucinated_excess_kurtosis"])) / 4,
        4,
    )
    return result


def main():
    results = {}
    for model_key, cfg in MODELS.items():
        print(f"Analyzing {model_key} at layer {cfg['best_j_layer']}...")
        results[model_key] = analyze_model(model_key, cfg)
        print(json.dumps(results[model_key], indent=2))

    non_gauss = [results[m]["non_gaussianity_score"] for m in MODELS]
    errors = [MODELS[m]["table_error"] for m in MODELS]
    rho, p = spearmanr(non_gauss, errors)

    correlation = {
        "non_gaussianity_scores": dict(zip(MODELS.keys(), non_gauss)),
        "table_errors": dict(zip(MODELS.keys(), errors)),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 4),
        "n_models": len(MODELS),
    }
    results["correlation_non_gaussianity_vs_bound_error"] = correlation
    print("\n=== Correlation: non-Gaussianity score vs. Table certificate-matched bound error ===")
    print(json.dumps(correlation, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
