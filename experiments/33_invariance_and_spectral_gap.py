"""
Exp.33 -- GL(d)/O(d) invariance of Fisher separability and the spectral-gap
mechanism behind this paper's own PCA-rank sensitivity finding (Priority-1
reanalysis, Exp. "PCA-rank sensitivity").

MOTIVATION. This paper's Fisher separability J is a Mahalanobis-type
quadratic form; the exact quantity (computed on the true, full-rank
within-class covariance) is provably invariant under any invertible linear
reparametrization of the representation space (GL(d)): substituting
mu' = T mu, Sigma_w' = T Sigma_w T^T into J' = (mu_c'-mu_h')^T Sigma_w'^{-1}
(mu_c'-mu_h') and using Sigma_w'^{-1} = T^{-T} Sigma_w^{-1} T^{-1} shows every
T-factor cancels exactly: J' = J. This is the standard reason Mahalanobis
distances are basis-independent (this is a linear-algebra identity, not an
empirical claim -- verified symbolically before writing this script).

The paper's ACTUAL estimator is not this exact quantity: it is PCA(k=100)
projection followed by Ledoit-Wolf shrinkage (src/fisher.py::fisher_ratio,
method="pca"). Two mechanisms can break exact GL(d)-invariance down to a
narrower group:
  (a) PCA-to-k<d is a rank-deficient projection. Under any ORTHOGONAL
      transform T, covariance transforms by similarity (T Sigma T^T = T
      Sigma T^{-1} since T^{-1}=T^T), which preserves eigenvalues exactly
      and rotates eigenvectors -- so the top-k subspace selection commutes
      with rotation. Under a general invertible T, covariance transforms by
      CONGRUENCE, which does not preserve eigenvalues -- a different
      subspace can be selected, breaking invariance.
  (b) Ledoit-Wolf shrinkage blends toward an ISOTROPIC target (a scaled
      identity matrix). This target transforms correctly (stays a multiple
      of the identity) only under orthogonal (or conformal: orthogonal x
      uniform scaling) T, not under anisotropic invertible T.

Both mechanisms predict: the paper's actual PCA(100)+LW estimator should be
INVARIANT under orthogonal (rotation) transforms of the representation
space, but NOT under general anisotropic invertible transforms -- UNLESS
the covariance spectrum lacks a clean gap at rank k=100, in which case even
orthogonal invariance can fail, because the "top-100" subspace itself is
not a numerically stable/well-defined object when many eigenvalues near the
cutoff are nearly tied (a rotation can shuffle which near-tied component
falls just above/below an arbitrary rank cutoff).

THIS SCRIPT TESTS BOTH PREDICTIONS DIRECTLY ON THE PAPER'S OWN CACHED DATA,
CPU-only, no new GPU inference:
  1. Confirm exact orthogonal invariance holds at NEAR-FULL RANK (k close to
     n-2, i.e. minimal truncation) -- this isolates the "does shrinkage
     itself preserve orthogonal invariance" question from "does truncation
     break it."
  2. Measure the actual instability under orthogonal rotation AT THE PAPER'S
     OWN k=100 CHOICE, across all 9 models this paper now has cached data
     for (6 from Exp.15's matched sweep + 3 from Exp.32's Pythia extension).
  3. Directly check whether the covariance spectrum has a gap at rank 100
     (ratio of eigenvalue[99] to eigenvalue[100]) for every model -- this is
     the mechanistic explanation, not merely a correlate.
  4. Confirm the predicted anisotropic-vs-orthogonal asymmetry: instability
     should grow with how anisotropic an applied invertible transform is.

All data already cached (results/hidden_states/, kaggle/exp32.../output/);
zero new model inference. Full machine-readable output:
results/logs/33_invariance_and_spectral_gap.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fisher import fisher_ratio  # noqa: E402

OUT_PATH = ROOT / "results" / "logs" / "33_invariance_and_spectral_gap.json"
N_ORTHOGONAL_TRIALS = 10
N_ANISOTROPIC_TRIALS_PER_SPREAD = 10
ANISOTROPY_SPREADS = [0.0, 0.5, 1.0, 2.0, 4.0]
SEED = 42

MODELS = {
    "gpt2_117m": ROOT / "results/hidden_states/15_hidden_states_gpt2_117m.npz",
    "gpt2_medium": ROOT / "results/hidden_states/15_hidden_states_gpt2_medium.npz",
    "qwen_0.5b": ROOT / "results/hidden_states/15_hidden_states_qwen_0.5b.npz",
    "qwen_1.5b": ROOT / "results/hidden_states/15_hidden_states_qwen_1.5b.npz",
    "qwen_3b": ROOT / "results/hidden_states/15_hidden_states_qwen_3b.npz",
    "qwen_7b": ROOT / "results/hidden_states/15_hidden_states_qwen_7b.npz",
    "pythia_410m": ROOT / "kaggle/exp32_third_architecture_pythia/output/32_hidden_states_pythia_410m.npz",
    "pythia_1.4b": ROOT / "kaggle/exp32_third_architecture_pythia/output/32_hidden_states_pythia_1.4b.npz",
    "pythia_2.8b": ROOT / "kaggle/exp32_third_architecture_pythia/output/32_hidden_states_pythia_2.8b.npz",
}


def random_orthogonal(d, rng):
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def random_anisotropic(d, rng, spread):
    log_scales = rng.uniform(-spread / 2, spread / 2, size=d)
    scales = 10.0 ** log_scales
    Q = random_orthogonal(d, rng)
    return Q @ np.diag(scales)


def best_layer_J(H_all, y, k=100):
    best_J, best_L = -1.0, -1
    for L in range(H_all.shape[1]):
        J = fisher_ratio(H_all[:, L, :], y, method="pca", n_components=k)
        if J > best_J:
            best_J, best_L = J, L
    return best_L, best_J


def spectral_gap_ratio(H, rank):
    Hc = H - H.mean(axis=0)
    _, s, _ = np.linalg.svd(Hc, full_matrices=False)
    eigvals = s ** 2
    if len(eigvals) <= rank:
        return float("nan")
    return float(eigvals[rank - 1] / eigvals[rank])


def main():
    rng = np.random.default_rng(SEED)
    results = {}

    for name, path in MODELS.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        d = np.load(path)
        H_all, y = d["hidden_states"], d["labels"]
        best_L, J_at_k100 = best_layer_J(H_all, y, k=100)
        H = H_all[:, best_L, :]
        d_dim = H.shape[1]
        n = H.shape[0]
        k_full = min(n, d_dim) - 2

        gap_at_100 = spectral_gap_ratio(H, 100)
        gap_at_25 = spectral_gap_ratio(H, 25)
        gap_at_50 = spectral_gap_ratio(H, 50)
        gap_at_200 = spectral_gap_ratio(H, 200) if d_dim > 200 else float("nan")

        # (1) near-full-rank orthogonal invariance
        J_kfull_base = fisher_ratio(H, y, method="pca", n_components=k_full)
        deltas_kfull = []
        for _ in range(3):
            Q = random_orthogonal(d_dim, rng)
            J_kfull_rot = fisher_ratio(H @ Q.T, y, method="pca", n_components=k_full)
            deltas_kfull.append(abs(J_kfull_rot - J_kfull_base))

        # (2) k=100 orthogonal instability (many trials for mean/std)
        deltas_k100 = []
        for _ in range(N_ORTHOGONAL_TRIALS):
            Q = random_orthogonal(d_dim, rng)
            J_rot = fisher_ratio(H @ Q.T, y, method="pca", n_components=100)
            deltas_k100.append(J_rot - J_at_k100)
        deltas_k100 = np.array(deltas_k100)

        # (4) anisotropic sensitivity vs spread
        aniso_results = {}
        for spread in ANISOTROPY_SPREADS:
            deltas = []
            for _ in range(N_ANISOTROPIC_TRIALS_PER_SPREAD):
                T = random_anisotropic(d_dim, rng, spread)
                J_aniso = fisher_ratio(H @ T.T, y, method="pca", n_components=100)
                deltas.append(J_aniso - J_at_k100)
            deltas = np.array(deltas)
            aniso_results[str(spread)] = {
                "mean_delta": float(deltas.mean()),
                "mean_abs_delta": float(np.abs(deltas).mean()),
                "std_delta": float(deltas.std()),
            }

        results[name] = {
            "best_layer": best_L,
            "d": d_dim,
            "n": n,
            "J_at_k100": J_at_k100,
            "spectral_gap_ratio_at_rank": {
                "25": gap_at_25, "50": gap_at_50, "100": gap_at_100, "200": gap_at_200,
            },
            "near_full_rank_orthogonal_invariance": {
                "k_full": k_full,
                "J_baseline": J_kfull_base,
                "max_abs_delta_over_3_trials": float(max(deltas_kfull)),
            },
            "k100_orthogonal_instability": {
                "n_trials": N_ORTHOGONAL_TRIALS,
                "mean_delta": float(deltas_k100.mean()),
                "mean_abs_delta": float(np.abs(deltas_k100).mean()),
                "std_delta": float(deltas_k100.std()),
            },
            "anisotropic_sensitivity_by_spread": aniso_results,
        }
        print(f"{name}: L={best_L} d={d_dim} gap@100={gap_at_100:.4f} "
              f"k100_orth_meanAbsDelta={np.abs(deltas_k100).mean():.4f} "
              f"kfull_orth_maxAbsDelta={max(deltas_kfull):.2e}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
