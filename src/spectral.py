"""
Spectral analysis of hidden-state covariance — Direction A.

Uses Random Matrix Theory to characterize when the hallucination signal
is detectable. The core idea:

  Null hypothesis: hidden states have random covariance → eigenvalues follow
  the Marchenko-Pastur (MP) distribution with shape γ = d/n.

  Alternative: a low-rank signal (the "truth direction") creates spike eigenvalues
  above the MP bulk upper edge λ_+ = σ²(1 + √γ)².

  BBP transition (Baik, Ben Arous, Péché 2005):
  A population spike eigenvalue θ causes a sample spike eigenvalue to emerge
  above λ_+ if and only if θ > σ²√γ (the BBP threshold).
  Below this threshold, PCA finds only noise — the truth direction is buried.

  Connection to Fisher J:
  The between-class covariance rank-1 contribution to Σ is Σ_b = p_c·p_h·(μ_c−μ_h)(μ_c−μ_h)^T
  where p_c, p_h are class proportions.
  The signal eigenvalue of Σ_b is ‖μ_c−μ_h‖²·p_c·p_h — proportional to W₂_eq.
  The BBP threshold is σ²√γ = tr(Σ_w)/d · √(d/n).

  This establishes a formal connection: W₂ detectability ↔ BBP threshold crossing.

Key observational hypothesis:
  The KL divergence of the empirical ESD from Marchenko-Pastur at layer L
  should trace the same phase transition as the W₂ curve and the probe AUROC curve.
"""

import numpy as np
from scipy.stats import norm


def marchenko_pastur_pdf(
    x: np.ndarray,
    gamma: float,
    sigma2: float = 1.0,
) -> np.ndarray:
    """
    Marchenko-Pastur density for ratio γ = d/n, noise variance σ².

    λ_± = σ²(1 ± √γ)²

    MP density: ρ(x) = √((λ_+ − x)(x − λ_−)) / (2π·γ·σ²·x)
    for x ∈ [λ_−, λ_+].
    """
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_minus = max(lam_minus, 0.0)

    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    x_m = x[mask]
    pdf[mask] = (
        np.sqrt((lam_plus - x_m) * np.maximum(x_m - lam_minus, 0))
        / (2 * np.pi * gamma * sigma2 * x_m + 1e-15)
    )
    return pdf


def bbp_threshold(d: int, n: int, sigma2: float = 1.0) -> float:
    """
    BBP critical spike eigenvalue: θ* = σ²√(d/n).

    A population spike of strength θ in the covariance matrix produces a
    detectable sample spike above the MP bulk iff θ > θ*.

    For the hallucination signal: the between-class signal eigenvalue must
    exceed θ* for linear probing to reliably find the truth direction.
    """
    return sigma2 * np.sqrt(d / n)


def mp_signal_subspace(
    H: np.ndarray,
    gamma: float | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Project H onto the MP signal subspace — eigenvectors whose eigenvalues
    exceed the Marchenko-Pastur upper edge λ_+ = σ²(1 + √γ)².

    This is the principled, parameter-free dimensionality reduction for
    the d >> n regime: only the 'spike' directions survive, giving an
    effective dimension k_eff << d with γ_eff = k_eff/n << 1.

    No prior LLM probing paper uses this criterion (as of 2026). The standard
    approach is arbitrary PCA k=100. Here, the MP bulk itself sets the cutoff.

    Parameters
    ----------
    H : ndarray (n_samples, d)
    gamma : float or None — d/n ratio (auto-computed if None)

    Returns
    -------
    H_proj : ndarray (n_samples, k_eff) — projected onto spike subspace
    k_eff  : int — number of spike eigenvectors retained
    lam_plus : float — MP upper edge used as cutoff
    """
    H = H.astype(np.float64)
    n, d = H.shape
    if gamma is None:
        gamma = d / n

    H_centered = H - H.mean(0)

    # In d >> n: work in sample-space (n×n) then map back — O(n³) not O(d³)
    if d > n:
        _, s, Vt = np.linalg.svd(H_centered, full_matrices=False)
        eigenvalues = s ** 2 / n
        eigenvectors = Vt.T  # (d, n) — columns are right singular vectors
    else:
        from sklearn.covariance import LedoitWolf
        Sigma = LedoitWolf().fit(H_centered).covariance_
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        # eigh returns ascending order; reverse
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

    sigma2_est = float(np.median(eigenvalues))
    lam_plus = sigma2_est * (1 + np.sqrt(gamma)) ** 2

    spike_mask = eigenvalues > lam_plus
    k_eff = int(spike_mask.sum())

    if k_eff == 0:
        # No spikes: return first eigenvector as a degenerate projection
        k_eff = 1
        spike_mask[0] = True

    H_proj = H_centered @ eigenvectors[:, :k_eff]
    return H_proj, k_eff, float(lam_plus)


def esd_kl_from_mp(
    H: np.ndarray,
    gamma: float | None = None,
    n_bins: int = 100,
) -> dict:
    """
    Compute KL divergence of the empirical spectral distribution (ESD)
    from the Marchenko-Pastur null distribution.

    A large KL divergence means the hidden states have non-random structure —
    there is a detectable signal above the noise floor.

    For d >> n (common in LLM probing), we work in sample-space (n×n) via SVD
    to avoid computing the d×d covariance directly. The resulting n non-zero
    eigenvalues are compared against the MP distribution at γ = d/n.

    Parameters
    ----------
    H : ndarray (n_samples, d) — hidden states at one layer
    gamma : float or None — d/n ratio (auto-computed if None)
    n_bins : int — histogram bins for ESD estimation

    Returns
    -------
    dict with:
        'kl_from_mp'    : float — KL(ESD ‖ MP)
        'eigenvalues'   : ndarray — empirical eigenvalues
        'lam_plus'      : float — MP upper edge (bulk boundary)
        'n_spikes'      : int — eigenvalues above lam_plus (spike count)
        'spike_ratio'   : float — largest eigenvalue / lam_plus
        'bbp_threshold' : float — critical spike strength θ*
        'gamma'         : float — shape parameter d/n
        'k_eff'         : int — spike subspace dimension (MP-guided)
        'gamma_eff'     : float — effective γ after MP-guided projection
    """
    H = H.astype(np.float64)
    n, d = H.shape
    if gamma is None:
        gamma = d / n

    # Always use SVD for eigenvalues — correct for both d>>n and d<<n
    # For d>>n: n non-zero eigenvalues of H H^T / n
    # For d<<n: d eigenvalues via H^T H / n (same non-zero spectrum)
    H_centered = H - H.mean(0)
    _, s, _ = np.linalg.svd(H_centered, full_matrices=False)
    eigenvalues = s ** 2 / n

    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Estimate noise variance from median eigenvalue (robust to spikes)
    sigma2_est = float(np.median(eigenvalues))

    # MP parameters
    lam_plus = sigma2_est * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2_est * max(1 - np.sqrt(gamma), 0) ** 2
    theta_star = bbp_threshold(d, n, sigma2_est)

    # Count spikes above bulk
    n_spikes = int((eigenvalues > lam_plus).sum())
    spike_ratio = float(eigenvalues[0] / lam_plus) if lam_plus > 0 else float("nan")

    # KL divergence of ESD from MP (histogram-based approximation)
    # Only compare the bulk (eigenvalues within [0, 2·lam_plus])
    bulk_mask = eigenvalues <= 2 * lam_plus
    bulk_eigs = eigenvalues[bulk_mask]

    if len(bulk_eigs) < 5:
        kl = float("nan")
    else:
        bins = np.linspace(lam_minus * 0.5, lam_plus * 1.5, n_bins + 1)
        esd_hist, _ = np.histogram(bulk_eigs, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mp_dens = marchenko_pastur_pdf(bin_centers, gamma, sigma2_est)

        # Normalize (add small floor so no bin is exactly zero; then re-normalize)
        # This avoids partial-sum KL issues that produce negative values when the
        # ESD is more concentrated than the MP support.
        epsilon = 1e-8
        esd_reg = (esd_hist + epsilon) / (esd_hist.sum() + epsilon * len(esd_hist))
        mp_reg  = (mp_dens  + epsilon) / (mp_dens.sum()  + epsilon * len(mp_dens))

        # KL(ESD ‖ MP) — always ≥ 0 with the floored distributions
        kl = float(np.sum(esd_reg * np.log(esd_reg / mp_reg)))

    # MP-guided effective dimension
    k_eff = int(n_spikes) if n_spikes > 0 else 1
    gamma_eff = k_eff / n  # effective γ after projecting to spike subspace

    return {
        "kl_from_mp": kl,
        "eigenvalues": eigenvalues.tolist(),
        "lam_plus": float(lam_plus),
        "lam_minus": float(lam_minus),
        "n_spikes": n_spikes,
        "k_eff": k_eff,
        "gamma_eff": float(gamma_eff),
        "spike_ratio": spike_ratio,
        "bbp_threshold": float(theta_star),
        "sigma2_est": float(sigma2_est),
        "gamma": float(gamma),
    }


def spectral_curve(
    hidden_states: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Compute spectral analysis at every layer.

    Handles d >> n correctly by using SVD-based eigenvalue computation
    (O(min(d,n)³) instead of O(d³)) and reporting the MP-guided effective
    dimension k_eff — the number of spike eigenvectors above the bulk edge λ_+.

    This k_eff is the theoretically principled dimensionality for downstream
    Fisher / W₂ computation in the signal subspace, replacing the arbitrary
    k=100 used in prior LLM probing work.

    Parameters
    ----------
    hidden_states : ndarray (n_samples, n_layers, d)

    Returns
    -------
    dict with per-layer spectral statistics and phase transition summary.
    """
    n_samples, n_layers, d = hidden_states.shape
    gamma = d / n_samples

    kl_per_layer = []
    n_spikes_per_layer = []
    spike_ratio_per_layer = []
    k_eff_per_layer = []
    gamma_eff_per_layer = []

    for L in range(n_layers):
        H_L = hidden_states[:, L, :]
        stats = esd_kl_from_mp(H_L, gamma=gamma)
        kl_per_layer.append(stats["kl_from_mp"])
        n_spikes_per_layer.append(stats["n_spikes"])
        spike_ratio_per_layer.append(stats["spike_ratio"])
        k_eff_per_layer.append(stats["k_eff"])
        gamma_eff_per_layer.append(stats["gamma_eff"])

        if verbose:
            print(f"  Layer {L:3d}: KL={stats['kl_from_mp']:.4f}  "
                  f"spikes={stats['n_spikes']}  k_eff={stats['k_eff']}  "
                  f"γ_eff={stats['gamma_eff']:.4f}  "
                  f"spike_ratio={stats['spike_ratio']:.3f}")

    best_layer = int(np.nanargmax(kl_per_layer))
    return {
        "kl_per_layer": kl_per_layer,
        "n_spikes_per_layer": n_spikes_per_layer,
        "spike_ratio_per_layer": spike_ratio_per_layer,
        "k_eff_per_layer": k_eff_per_layer,
        "gamma_eff_per_layer": gamma_eff_per_layer,
        "best_layer": best_layer,
        "best_kl": kl_per_layer[best_layer],
        "best_depth_fraction": best_layer / max(n_layers - 1, 1),
        "n_layers": n_layers,
        "gamma": float(gamma),
        "gamma_label": f"d/n = {d}/{n_samples} = {gamma:.2f} (d>>n regime)",
        "bbp_threshold": bbp_threshold(d, n_samples),
    }
