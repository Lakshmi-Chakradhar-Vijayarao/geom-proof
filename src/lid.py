"""
Local Intrinsic Dimension (LID) as a competing hallucination certificate.

From: "Characterizing Truthfulness in LLM Generations with Local Intrinsic Dimension"
      Yin et al., ICML 2024 (PMLR 235:57069-57084)

LID measures the effective dimensionality of the local activation manifold around
each sample. The hypothesis: truthful and hallucinated responses occupy manifolds of
different intrinsic dimensionality — hallucinations may cluster more tightly (lower LID)
or more diffusely (higher LID) than correct responses.

Used in Exp 01 as a competing certificate against Fisher separability:
  - Fisher certificate: J(L) = (μ_c - μ_h)^T Σ_w^{-1} (μ_c - μ_h) → AUROC bound via Φ
  - LID certificate: |LID_c - LID_h| → AUROC via correlation with actual probe performance

The "LID-AUROC correlation" (r between |ΔS_LID| and probe_AUROC across layers)
is our measure of how useful LID is as a certificate. We compare it to Fisher's correlation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _lid_mle(distances: np.ndarray, k: int) -> float:
    """
    MLE estimator for Local Intrinsic Dimension.

    Amsaleg et al. (2015): LID ≈ -1 / mean(log(r_i / r_k))
    where r_i are the k-NN distances and r_k is the maximum (k-th neighbor distance).

    Parameters
    ----------
    distances : 1D array of k nearest-neighbor distances (sorted ascending)
    k : int — number of neighbors

    Returns
    -------
    float — LID estimate for this sample
    """
    distances = np.asarray(distances[:k], dtype=np.float64)
    r_k = distances[-1]
    if r_k < 1e-10:
        return 0.0
    ratios = np.log(distances / r_k + 1e-10)
    ratios = ratios[ratios < 0]  # only negative ratios (r_i < r_k)
    if len(ratios) == 0:
        return 0.0
    return float(-1.0 / ratios.mean())


def lid_per_sample(H: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Compute LID estimate for each sample in H using k nearest neighbors.

    Parameters
    ----------
    H : ndarray (n_samples, d)
    k : int — number of neighbors (default 20; use min(20, n//4))

    Returns
    -------
    ndarray (n_samples,) — LID estimate per sample
    """
    n = H.shape[0]
    k = min(k, n - 1)
    lids = np.zeros(n)

    # Pairwise distances (efficient for n <= 1000)
    from sklearn.metrics.pairwise import euclidean_distances
    D = euclidean_distances(H)

    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf  # exclude self
        sorted_dists = np.sort(row)[:k]
        lids[i] = _lid_mle(sorted_dists, k)

    return lids


def lid_class_stats(H: np.ndarray, y: np.ndarray, k: int = 20) -> dict:
    """
    Compute mean LID for correct and hallucinated classes at a given layer.

    Returns
    -------
    dict with:
        'lid_correct' : float — mean LID for y=1
        'lid_hall'    : float — mean LID for y=0
        'lid_diff'    : float — |lid_correct - lid_hall|
        'lid_ratio'   : float — lid_correct / lid_hall (or NaN if hall mean is 0)
        'lid_auroc'   : float — AUROC of LID as hallucination detector
                        (higher LID = more hallucinated? or lower? — learn from data)
    """
    lids = lid_per_sample(H, k=k)
    lid_correct = float(lids[y == 1].mean())
    lid_hall = float(lids[y == 0].mean())

    # AUROC: try both directions (higher LID = hall, and lower LID = hall)
    # Use whichever gives AUROC > 0.5
    auroc_pos = float(roc_auc_score(y, -lids))  # higher LID = hallucinated
    auroc_neg = float(roc_auc_score(y, lids))   # lower LID = hallucinated
    lid_auroc = max(auroc_pos, auroc_neg)
    direction = "higher LID → hallucinated" if auroc_pos > auroc_neg else "lower LID → hallucinated"

    return {
        "lid_correct": lid_correct,
        "lid_hall": lid_hall,
        "lid_diff": float(abs(lid_correct - lid_hall)),
        "lid_ratio": float(lid_correct / lid_hall) if lid_hall > 1e-6 else float("nan"),
        "lid_auroc": lid_auroc,
        "lid_direction": direction,
        "lids": lids.tolist(),
    }


def lid_curve(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    k: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Compute LID statistics at every layer.

    Parameters
    ----------
    hidden_states : ndarray (n_samples, n_layers, d)
    labels : ndarray (n_samples,)
    k : int — k-NN neighbors for LID
    verbose : bool

    Returns
    -------
    dict with per-layer LID stats and best-layer summary.
    """
    n_samples, n_layers, d = hidden_states.shape
    results_per_layer = []

    for L in range(n_layers):
        H_L = hidden_states[:, L, :]
        stats = lid_class_stats(H_L, labels, k=k)
        results_per_layer.append(stats)
        if verbose:
            print(f"  Layer {L:3d}: LID_c={stats['lid_correct']:.2f}  "
                  f"LID_h={stats['lid_hall']:.2f}  "
                  f"|Δ|={stats['lid_diff']:.3f}  "
                  f"AUROC={stats['lid_auroc']:.4f}")

    lid_aurocs = [r["lid_auroc"] for r in results_per_layer]
    best_layer = int(np.argmax(lid_aurocs))

    return {
        "per_layer": results_per_layer,
        "lid_auroc_per_layer": lid_aurocs,
        "lid_diff_per_layer": [r["lid_diff"] for r in results_per_layer],
        "best_layer": best_layer,
        "best_lid_auroc": lid_aurocs[best_layer],
        "best_depth_fraction": best_layer / (n_layers - 1),
        "n_layers": n_layers,
    }
