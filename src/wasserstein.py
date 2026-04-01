"""
Optimal Transport certificates for hallucination detection.

Core mathematical result (derivable in two lines):
    Fisher ratio J = (μ_c − μ_h)ᵀ Σ_w⁻¹ (μ_c − μ_h)
                   = ‖Σ_w^{-1/2}(μ_c − μ_h)‖²
                   = W₂²(P_c, P_h)  in whitened space (Σ_w = I)

Under equal-covariance Gaussians, Fisher IS the Wasserstein-2 distance in
the Mahalanobis-whitened coordinate system. The existing AUROC bound
Φ(√J/2) is therefore Φ(W₂_whitened / 2).

This module generalizes: what if we drop the Gaussian assumption?
Three non-parametric alternatives are computed and compared:

  (1) Gaussian W₂ (Bures metric) — closed-form, exact under Gaussian assumption
  (2) Sliced Wasserstein (SW₂)   — non-parametric, random projections, no Gaussianity
  (3) Maximum Mean Discrepancy (MMD) — kernel-based, distribution-free two-sample test

For each, we compute a per-layer "separability curve" and compare it against
the Fisher J(L) curve and the actual OOF probe AUROC(L).

Key question: does SW₂ or MMD predict probe AUROC better than J in layers
where the class-conditional distributions are non-Gaussian?
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import rbf_kernel


# ── Gaussian W₂ (Bures metric) ────────────────────────────────────────────────

def bures_w2_squared(H: np.ndarray, y: np.ndarray) -> float:
    """
    W₂² between two Gaussian distributions fit to the class-conditional hidden states.

    W₂²(N(μ_c,Σ_c), N(μ_h,Σ_h)) = ‖μ_c − μ_h‖² + B²(Σ_c, Σ_h)

    where B²(Σ_c, Σ_h) = tr(Σ_c + Σ_h − 2(Σ_h^{1/2} Σ_c Σ_h^{1/2})^{1/2})
    is the Bures metric (Takatsu 2011; Bhatia et al. 2019).

    Under equal-covariance (Σ_c = Σ_h = Σ_w):
      B²(Σ_w, Σ_w) = 0  →  W₂² = ‖μ_c − μ_h‖²  (pure mean difference)

    Parameters
    ----------
    H : ndarray (n_samples, d)
    y : ndarray (n_samples,) — binary labels (1=correct, 0=hallucinated)

    Returns
    -------
    float — W₂² between the two Gaussian approximations
    """
    H = H.astype(np.float64)
    H_c, H_h = H[y == 1], H[y == 0]
    mu_c, mu_h = H_c.mean(0), H_h.mean(0)
    mean_sq = float(np.sum((mu_c - mu_h) ** 2))

    # Estimate covariances with Ledoit-Wolf shrinkage
    lw_c = LedoitWolf().fit(H_c - mu_c)
    lw_h = LedoitWolf().fit(H_h - mu_h)
    Sc, Sh = lw_c.covariance_, lw_h.covariance_

    # Bures term: tr(Sc + Sh - 2·(Sh^{1/2} Sc Sh^{1/2})^{1/2})
    try:
        Sh_sqrt = sqrtm(Sh).real
        M = Sh_sqrt @ Sc @ Sh_sqrt
        M_sqrt = sqrtm(M).real
        bures_sq = float(np.trace(Sc + Sh - 2 * M_sqrt))
        bures_sq = max(0.0, bures_sq)  # numerical floor
    except Exception:
        bures_sq = 0.0

    return mean_sq + bures_sq


def bures_w2_equal_cov(H: np.ndarray, y: np.ndarray) -> float:
    """
    W₂² under the equal-covariance assumption: W₂² = ‖μ_c − μ_h‖²
    This is the Euclidean mean-difference, NOT the whitened Fisher distance.
    Provided for direct comparison with Fisher J.
    """
    mu_c = H[y == 1].mean(0).astype(np.float64)
    mu_h = H[y == 0].mean(0).astype(np.float64)
    return float(np.sum((mu_c - mu_h) ** 2))


# ── Relationship: Fisher = W₂ in whitened space ───────────────────────────────

def fisher_as_whitened_w2(H: np.ndarray, y: np.ndarray, n_components: int = 100) -> dict:
    """
    Verify the core identity: J = W₂²(P_c, P_h) in whitened space.

    Steps:
    1. Compute Fisher J (Mahalanobis mean-difference)
    2. Whiten H using Σ_w^{-1/2} (pooled, Ledoit-Wolf)
    3. Compute W₂² on whitened H under equal-covariance assumption
    4. Verify J ≈ W₂_whitened²  (they should be equal up to numerical precision)

    Returns
    -------
    dict with 'J', 'w2_whitened', 'relative_error', 'identity_holds'
    """
    from src.fisher import fisher_ratio
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)

    # Fisher J (Mahalanobis)
    J = fisher_ratio(H, y, method="pca", n_components=n_components)

    # Whitened W₂
    mu_c = H[y == 1].mean(0)
    mu_h = H[y == 0].mean(0)
    H_c_centered = H[y == 1] - mu_c
    H_h_centered = H[y == 0] - mu_h
    H_centered = np.vstack([H_c_centered, H_h_centered])
    lw = LedoitWolf().fit(H_centered)
    Sigma_w = lw.covariance_

    # Compute Σ_w^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma_w)
    eigvals = np.maximum(eigvals, 1e-10)
    Sigma_w_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Project H into whitened space
    H_white = H @ Sigma_w_inv_sqrt

    # W₂ in whitened space (equal-covariance → pure mean difference)
    mu_c_w = H_white[y == 1].mean(0)
    mu_h_w = H_white[y == 0].mean(0)
    w2_whitened = float(np.sum((mu_c_w - mu_h_w) ** 2))

    rel_error = abs(J - w2_whitened) / max(J, 1e-10)
    return {
        "J": J,
        "w2_whitened": w2_whitened,
        "relative_error": rel_error,
        "identity_holds": rel_error < 0.05,  # within 5% — numerical tolerance
    }


# ── Sliced Wasserstein W₂ (non-parametric) ───────────────────────────────────

def sliced_wasserstein_2(
    H: np.ndarray,
    y: np.ndarray,
    n_projections: int = 200,
    n_components: int = 50,
    random_state: int = 42,
) -> float:
    """
    Sliced Wasserstein-2 distance² between class-conditional hidden-state distributions.

    No Gaussianity assumption. Projects onto random directions and averages
    the 1D Wasserstein distances.

    SW₂²(P, Q) = E_{θ~Uniform(S^{d-1})} [W₁(P_θ, Q_θ)²]

    where P_θ = push-forward of P under the projection x ↦ <x, θ>.

    Parameters
    ----------
    H : ndarray (n_samples, d)
    y : ndarray (n_samples,)
    n_projections : int — number of random directions to average
    n_components : int — PCA projection before slicing (for d >> n)
    random_state : int

    Returns
    -------
    float — SW₂²(P_correct, P_hallucinated)
    """
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(random_state)
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)

    # PCA projection to make SW₂ tractable in high dimensions
    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k, random_state=random_state)
    H_proj = pca.fit_transform(H)

    H_c = H_proj[y == 1]
    H_h = H_proj[y == 0]

    total_w1_sq = 0.0
    for _ in range(n_projections):
        theta = rng.randn(k)
        theta /= np.linalg.norm(theta) + 1e-10
        proj_c = H_c @ theta
        proj_h = H_h @ theta
        # 1D W₁ = mean absolute difference of sorted values (for equal mass)
        # Use interpolation for unequal class sizes
        proj_c_sorted = np.sort(proj_c)
        proj_h_sorted = np.sort(proj_h)
        # Resample to common grid
        n = max(len(proj_c_sorted), len(proj_h_sorted))
        t = np.linspace(0, 1, n)
        c_interp = np.interp(t, np.linspace(0, 1, len(proj_c_sorted)), proj_c_sorted)
        h_interp = np.interp(t, np.linspace(0, 1, len(proj_h_sorted)), proj_h_sorted)
        # W₂ in 1D = L2 distance between quantile functions
        total_w1_sq += float(np.mean((c_interp - h_interp) ** 2))

    return total_w1_sq / n_projections


# ── Maximum Mean Discrepancy (kernel MMD) ────────────────────────────────────

def mmd_squared(
    H: np.ndarray,
    y: np.ndarray,
    gamma: float | None = None,
    n_components: int = 50,
) -> float:
    """
    Unbiased MMD² between class-conditional distributions using RBF kernel.

    MMD²(P,Q) = E[k(X,X')] − 2E[k(X,Y)] + E[k(Y,Y')]
    where X~P (correct), Y~Q (hallucinated), k = RBF kernel.

    Parameters
    ----------
    H : ndarray (n_samples, d)
    y : ndarray (n_samples,)
    gamma : float or None — RBF kernel bandwidth; if None, use median heuristic
    n_components : int — PCA before MMD for numerical stability

    Returns
    -------
    float — MMD²(P_correct, P_hallucinated)
    """
    from sklearn.decomposition import PCA
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)

    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k)
    H_proj = pca.fit_transform(H)

    H_c = H_proj[y == 1]
    H_h = H_proj[y == 0]

    if gamma is None:
        # Median heuristic: γ = 1 / (2 · median_pairwise_distance²)
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(H_proj)
        pos_dists = D[D > 0]
        if len(pos_dists) == 0:
            return 0.0  # degenerate layer — all activations identical
        median_dist = np.median(pos_dists)
        if np.isnan(median_dist) or median_dist < 1e-10:
            return 0.0  # degenerate layer — no pairwise separation
        gamma = 1.0 / (2.0 * median_dist ** 2)

    K_cc = rbf_kernel(H_c, H_c, gamma=gamma)
    K_hh = rbf_kernel(H_h, H_h, gamma=gamma)
    K_ch = rbf_kernel(H_c, H_h, gamma=gamma)

    n_c, n_h = len(H_c), len(H_h)

    # Unbiased estimator (remove diagonal terms)
    np.fill_diagonal(K_cc, 0)
    np.fill_diagonal(K_hh, 0)

    term_cc = K_cc.sum() / (n_c * (n_c - 1)) if n_c > 1 else 0.0
    term_hh = K_hh.sum() / (n_h * (n_h - 1)) if n_h > 1 else 0.0
    term_ch = K_ch.mean()

    return float(term_cc + term_hh - 2.0 * term_ch)


# ── AUROC bound from W₂ / SW₂ / MMD ─────────────────────────────────────────

def auroc_bound_from_w2(sw2: float, scale: float = 1.0) -> float:
    """
    AUROC bound from sliced Wasserstein distance.

    For Gaussian distributions, AUROC_bound = Φ(√J/2) = Φ(W₂_whitened/2).
    For non-Gaussian SW₂, the bound is approximate:
      AUROC_bound_SW₂ ≈ Φ(√(SW₂) · scale / 2)
    where scale is a calibration factor (learned from Exps 01–03).

    The key question is whether SW₂ tracks probe AUROC better than J
    in the non-Gaussian regime (early layers).
    """
    return float(norm.cdf(np.sqrt(max(0.0, sw2)) * scale / 2))


# ── Full OT certificate curve ─────────────────────────────────────────────────

def ot_certificate_curve(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_projections: int = 200,
    n_components: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Compute all OT certificates at every layer.

    Parameters
    ----------
    hidden_states : ndarray (n_samples, n_layers, d)
    labels : ndarray (n_samples,)

    Returns
    -------
    dict with per-layer values for:
        'w2_bures'    : Gaussian W₂² (Bures metric — unequal covariance)
        'w2_eq'       : Gaussian W₂² (equal-covariance — pure mean difference)
        'sw2'         : Sliced Wasserstein-2²
        'mmd'         : MMD²
        'best_layer'  : layer with highest SW₂ (primary non-parametric certificate)
    """
    n_samples, n_layers, d = hidden_states.shape
    w2_bures_per_layer = []
    w2_eq_per_layer = []
    sw2_per_layer = []
    mmd_per_layer = []

    for L in range(n_layers):
        H_L = hidden_states[:, L, :].astype(np.float64)

        w2_b = bures_w2_squared(H_L, labels)
        w2_e = bures_w2_equal_cov(H_L, labels)
        sw = sliced_wasserstein_2(H_L, labels,
                                  n_projections=n_projections,
                                  n_components=n_components)
        mm = mmd_squared(H_L, labels, n_components=n_components)

        w2_bures_per_layer.append(w2_b)
        w2_eq_per_layer.append(w2_e)
        sw2_per_layer.append(sw)
        mmd_per_layer.append(mm)

        if verbose:
            print(f"  Layer {L:3d}: W₂_bures={w2_b:.4f}  "
                  f"SW₂={sw:.4f}  MMD²={mm:.6f}")

    best_layer = int(np.argmax(sw2_per_layer))
    return {
        "w2_bures_per_layer": w2_bures_per_layer,
        "w2_eq_per_layer": w2_eq_per_layer,
        "sw2_per_layer": sw2_per_layer,
        "mmd_per_layer": mmd_per_layer,
        "best_layer_sw2": best_layer,
        "best_sw2": sw2_per_layer[best_layer],
        "best_depth_fraction_sw2": best_layer / max(n_layers - 1, 1),
        "n_layers": n_layers,
    }
