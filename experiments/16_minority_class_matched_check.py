"""
Paper 4 -- third-round review flagged that the monotonic-J-with-scale
finding (Exp 15) is collinear with minority-class size (n_correct:
24->103 across the 6 models), and Fisher J is known to be upward-biased
at small minority-n. This script directly tests whether the monotonicity
survives when every model's minority (correct) class is subsampled down
to the SAME fixed size (n=24, matching the smallest available, GPT-2
117M), repeated over many random subsamples for a stable estimate.

If monotonicity survives fixed-n subsampling, the scale effect is real
and not an estimation artifact. If it collapses, the "monotonic J
decrease" finding is (at least partly) a small-sample bias artifact.
"""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path
MODELS = ["gpt2_117m", "gpt2_medium", "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_7b"]
PARAMS = [117e6, 345e6, 494e6, 1.54e9, 3.09e9, 7.6e9]
HS_DIR = str(ROOT / "results" / "hidden_states")

N_CORRECT_FIXED = 24   # matches the smallest available (gpt2_117m)
N_HALL_FIXED = 200     # well within range for all models (min available hall = 397)
N_REPEATS = 30
N_COMPONENTS = 100
RANDOM_STATE = 42


def fisher_ratio_pca(H_L, y, k=100, seed=0):
    k = min(k, H_L.shape[1], H_L.shape[0] - 2)
    H_L = H_L.astype(np.float64)
    pca = PCA(n_components=k, random_state=seed)
    H_proj = pca.fit_transform(H_L)
    mu_c = H_proj[y == 1].mean(0)
    mu_h = H_proj[y == 0].mean(0)
    H_c = H_proj[y == 1] - mu_c
    H_h = H_proj[y == 0] - mu_h
    Sigma_w = LedoitWolf().fit(np.vstack([H_c, H_h])).covariance_
    diff = mu_c - mu_h
    try:
        w = np.linalg.solve(Sigma_w, diff)
        J = float(diff @ w)
    except np.linalg.LinAlgError:
        J = float(diff @ np.linalg.lstsq(Sigma_w, diff, rcond=None)[0])
    return max(J, 0.0)


def best_layer_J(H, y, seed):
    n_layers = H.shape[1]
    Js = []
    for L in range(n_layers):
        J = fisher_ratio_pca(H[:, L, :], y, k=N_COMPONENTS, seed=seed)
        Js.append(J)
    return max(Js)


rng_master = np.random.default_rng(RANDOM_STATE)
results = {}

for model_key in MODELS:
    d = np.load(f"{HS_DIR}/15_hidden_states_{model_key}.npz")
    H_full = d["hidden_states"].astype(np.float32)
    y_full = d["labels"]
    idx_correct = np.where(y_full == 1)[0]
    idx_hall = np.where(y_full == 0)[0]

    Js_over_repeats = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(rng_master.integers(0, 1_000_000))
        sub_correct = rng.choice(idx_correct, size=min(N_CORRECT_FIXED, len(idx_correct)), replace=False)
        sub_hall = rng.choice(idx_hall, size=min(N_HALL_FIXED, len(idx_hall)), replace=False)
        sub_idx = np.concatenate([sub_correct, sub_hall])
        H_sub = H_full[sub_idx]
        y_sub = y_full[sub_idx]
        J = best_layer_J(H_sub, y_sub, seed=rep)
        Js_over_repeats.append(J)

    mean_J = float(np.mean(Js_over_repeats))
    std_J = float(np.std(Js_over_repeats))
    results[model_key] = {"mean_J": mean_J, "std_J": std_J, "all_J": Js_over_repeats}
    print(f"{model_key:15s}  n_correct_orig={len(idx_correct):4d}  "
          f"matched-n(24) mean_J={mean_J:.3f}  std_J={std_J:.3f}")

log_params = np.log10(PARAMS)
mean_Js = [results[k]["mean_J"] for k in MODELS]
rho, p = spearmanr(log_params, mean_Js)
print(f"\nSpearman(log_params, matched-n mean_J) = {rho:.4f}, p={p:.4f}")

# Also report original (unmatched) J for comparison
print("\nFor comparison, original (unmatched-n) best_J values were:")
print("7.517, 3.974, 3.692, 2.251, 1.997, 1.633")

# ── Trust-closing check (4th review round): does the surviving "loose
# bound" claim survive the same matched-n scrutiny that killed the
# monotonic-J claim? Recompute bound error using matched-n J paired with
# each model's original best-layer probe AUROC.
from scipy.stats import norm as scipy_norm
PROBE_AUROC = [0.6548, 0.6081, 0.5840, 0.6022, 0.5719, 0.5254]  # from 15_matched_generation.json
matched_bound_errors = []
print("\nTrust-closing check: matched-n=24 bound error vs original probe AUROC")
for k, probe in zip(MODELS, PROBE_AUROC):
    J = results[k]["mean_J"]
    bound = float(scipy_norm.cdf(np.sqrt(J / 2)))
    err = bound - probe
    matched_bound_errors.append(err)
    print(f"  {k:15s} matched-n J={J:.3f} bound={bound:.4f} probe={probe:.4f} error={err:+.4f}")
rho2, p2 = spearmanr(log_params, matched_bound_errors)
print(f"\nSpearman(log_params, matched-n bound error) = {rho2:.4f}, p={p2:.4f}")
print(f"Matched-n error range: {min(matched_bound_errors):.4f} - {max(matched_bound_errors):.4f}, "
      f"mean={np.mean(matched_bound_errors):.4f}")
print("(Original, unmatched-n, same-layer errors were 0.32-0.41, mean 0.353 -- "
      "matched-n errors are LARGER, confirming the certificate is looser, not tighter, "
      "once the minority-class sample-size bias is corrected.)")
