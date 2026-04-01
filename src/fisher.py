"""
Fisher separability ratio computation for hidden-state geometry.

Given hidden states at a layer and binary labels (correct=1, hallucinated=0),
computes J(L) — the Fisher ratio — and the corresponding AUROC bound.

Standard Fisher (Euclidean):
  J(L) = (mu_c - mu_h)^T · Sigma_w^{-1} · (mu_c - mu_h)

Causal Fisher (Park et al. ICML 2024 — causal inner product):
  J_causal(L) = (mu_c - mu_h)^T · W_U^T W_U · Sigma_w^{-1} · (mu_c - mu_h)
  where W_U is the model's unembedding matrix (vocabulary projection).
  This weights hidden-state dimensions by their actual contribution to output logits —
  the theoretically correct geometry for LLM probing.

AUROC_bound(L) = Phi(sqrt(J(L)) / 2)

Two estimation methods:
  - "lda": Ledoit-Wolf shrinkage on full d-dimensional covariance (more principled)
  - "pca": Project to top-k PCA components first, then compute Fisher ratio (faster)
"""

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fisher_ratio(
    H: np.ndarray,
    y: np.ndarray,
    method: str = "lda",
    n_components: int = 100,
) -> float:
    """
    Compute the Fisher separability ratio J(L).

    Parameters
    ----------
    H : ndarray of shape (n_samples, d)
        Hidden states at a single layer.
    y : ndarray of shape (n_samples,)
        Binary labels — 1 for correct, 0 for hallucinated.
    method : "lda" or "pca"
        Estimation method.
    n_components : int
        PCA components to retain (used only when method="pca").

    Returns
    -------
    float
        Fisher ratio J >= 0.
    """
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    mask_c = y == 1
    mask_h = y == 0

    mu_c = H[mask_c].mean(axis=0)
    mu_h = H[mask_h].mean(axis=0)
    delta = mu_c - mu_h

    if method == "lda":
        # Ledoit-Wolf shrinkage covariance (handles d >> n)
        H_c = H[mask_c] - mu_c
        H_h = H[mask_h] - mu_h
        H_centered = np.vstack([H_c, H_h])
        lw = LedoitWolf()
        lw.fit(H_centered)
        Sigma_w = lw.covariance_

        # Solve Sigma_w · w = delta instead of direct inversion (numerically safer)
        try:
            w = np.linalg.solve(Sigma_w, delta)
            J = float(delta @ w)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            J = float(delta @ np.linalg.lstsq(Sigma_w, delta, rcond=None)[0])

    elif method == "pca":
        k = min(n_components, H.shape[1], H.shape[0] - 2)
        pca = PCA(n_components=k)
        H_proj = pca.fit_transform(H)
        delta_proj = pca.transform(delta.reshape(1, -1)).squeeze()

        H_c_proj = H_proj[mask_c]
        H_h_proj = H_proj[mask_h]
        mu_c_proj = H_c_proj.mean(axis=0)
        mu_h_proj = H_h_proj.mean(axis=0)

        H_c_centered = H_c_proj - mu_c_proj
        H_h_centered = H_h_proj - mu_h_proj
        H_centered = np.vstack([H_c_centered, H_h_centered])

        lw = LedoitWolf()
        lw.fit(H_centered)
        Sigma_w_proj = lw.covariance_

        delta_proj2 = mu_c_proj - mu_h_proj
        try:
            w = np.linalg.solve(Sigma_w_proj, delta_proj2)
            J = float(delta_proj2 @ w)
        except np.linalg.LinAlgError:
            J = float(delta_proj2 @ np.linalg.lstsq(Sigma_w_proj, delta_proj2, rcond=None)[0])

    else:
        raise ValueError(f"method must be 'lda' or 'pca', got '{method}'")

    return max(0.0, J)


def auroc_bound(J: float) -> float:
    """
    Convert Fisher ratio to AUROC bound.

    Under Gaussian equal-covariance assumption:
      AUROC_bound = Phi(sqrt(J) / 2)

    where Phi is the standard normal CDF.
    """
    return float(norm.cdf(np.sqrt(max(0.0, J)) / 2))


def causal_fisher_ratio(
    H: np.ndarray,
    y: np.ndarray,
    W_U: np.ndarray,
    n_components: int = 100,
) -> float:
    """
    Causal Fisher ratio using the unembedding-weighted inner product.

    From Park et al. (ICML 2024) — the theoretically correct geometry for LLM probing
    is not Euclidean but the causal inner product: <u,v>_causal = u^T W_U^T W_U v
    where W_U is the unembedding (lm_head) weight matrix (vocab_size × d).

    This re-weights hidden-state dimensions by their contribution to output logits,
    which is the correct metric for the linear representation hypothesis.

    Parameters
    ----------
    H : ndarray (n_samples, d)
    y : ndarray (n_samples,)
    W_U : ndarray (vocab_size, d) — the unembedding matrix (model.lm_head.weight)
    n_components : int — PCA components applied in the W_U-weighted space

    Returns
    -------
    float — causal Fisher ratio J_causal >= 0
    """
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    W_U = np.asarray(W_U, dtype=np.float64)

    # Project hidden states into the vocabulary output space: H_out = H @ W_U^T
    # This maps (n, d) → (n, vocab_size), but vocab_size is huge.
    # Instead: compute W_U^T W_U (d×d) and use it as the metric tensor.
    # For numerical stability: use top-k SVD of W_U.
    k = min(n_components, W_U.shape[0], W_U.shape[1])
    # Thin SVD of W_U: W_U ≈ U S V^T, so W_U^T W_U ≈ V S^2 V^T
    _, S, Vt = np.linalg.svd(W_U, full_matrices=False)
    S = S[:k]
    Vt = Vt[:k]  # (k, d)
    # Project H into the top-k right singular directions of W_U, scaled by S
    H_proj = H @ Vt.T * S[np.newaxis, :]  # (n, k) — causal metric applied

    # Standard Fisher in the projected (causal) space
    return fisher_ratio(H_proj, y, method="pca", n_components=k)


def fisher_curve(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    method: str = "lda",
    n_components: int = 100,
    verbose: bool = True,
    W_U: np.ndarray | None = None,
) -> dict:
    """
    Compute Fisher ratio and AUROC bound at every layer.

    Parameters
    ----------
    hidden_states : ndarray of shape (n_samples, n_layers, d)
    labels : ndarray of shape (n_samples,)
    method : "lda" or "pca"
    n_components : int
    verbose : bool

    Returns
    -------
    dict with keys:
        'J' : list[float] — Euclidean Fisher ratio per layer
        'J_causal' : list[float] — Causal Fisher ratio per layer (if W_U provided)
        'auroc_bound' : list[float] — AUROC bound (Euclidean) per layer
        'auroc_bound_causal' : list[float] — AUROC bound (causal) per layer
        'best_layer' : int — layer with highest Euclidean J
        'best_J' : float
        'best_auroc_bound' : float
    """
    n_samples, n_layers, d = hidden_states.shape
    J_per_layer = []
    bound_per_layer = []
    J_causal_per_layer = []
    bound_causal_per_layer = []

    for L in range(n_layers):
        H_L = hidden_states[:, L, :]
        J_L = fisher_ratio(H_L, labels, method=method, n_components=n_components)
        bound_L = auroc_bound(J_L)
        J_per_layer.append(J_L)
        bound_per_layer.append(bound_L)

        if W_U is not None:
            J_c = causal_fisher_ratio(H_L, labels, W_U, n_components=n_components)
            bound_c = auroc_bound(J_c)
        else:
            J_c, bound_c = float("nan"), float("nan")
        J_causal_per_layer.append(J_c)
        bound_causal_per_layer.append(bound_c)

        if verbose:
            causal_str = f"  J_causal={J_c:.4f}" if W_U is not None else ""
            print(f"  Layer {L:3d}: J={J_L:.4f}  bound={bound_L:.4f}{causal_str}")

    best_layer = int(np.argmax(J_per_layer))
    return {
        "J": J_per_layer,
        "J_causal": J_causal_per_layer,
        "auroc_bound": bound_per_layer,
        "auroc_bound_causal": bound_causal_per_layer,
        "best_layer": best_layer,
        "best_J": J_per_layer[best_layer],
        "best_auroc_bound": bound_per_layer[best_layer],
        "n_layers": n_layers,
        "depth_fraction": best_layer / (n_layers - 1),
    }
