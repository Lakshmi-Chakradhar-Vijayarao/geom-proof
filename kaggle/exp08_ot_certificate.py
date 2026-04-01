"""
GEOM-PROOF — Kaggle Job E: Exp 08 OT Certificate + Exp 09 Spectral Phase Transition
=====================================================================================

What this script does:
  1. Loads HaluEval hidden states (from Job A outputs) for Qwen 2.5 3B and GPT-2 Medium
  2. Runs Exp 08: OT certificate curve (SW₂, Bures W₂, MMD², Fisher J, probe AUROC per layer)
  3. Runs Exp 09: Spectral phase transition (ESD KL divergence from Marchenko-Pastur per layer)
  4. Saves:
       /kaggle/working/08_ot_certificate.json
       /kaggle/working/09_spectral_phase_transition.json
       /kaggle/working/08_ot_certificates.png
       /kaggle/working/09_spectral_phase_transition.png

Input files needed (add as kernel_sources from Job A):
  /kaggle/input/geom-proof-extraction-all-gpu-jobs/00_halueval_qwen3b.npz
  /kaggle/input/geom-proof-extraction-all-gpu-jobs/00_halueval_gpt2med.npz
  /kaggle/input/geom-proof-extraction-all-gpu-jobs/04_mamba_hidden_states.npz

After this job completes, download the two JSONs and one PNG to:
  results/logs/08_ot_certificate.json
  results/logs/09_spectral_phase_transition.json
  results/plots/08_ot_certificates.png
  results/plots/09_spectral_phase_transition.png

Then run locally (fast, ~5 min):
  python experiments/09_spectral_phase_transition.py   # re-checks coincidence
  python experiments/10_conformal_coverage.py          # already done locally

HOW TO USE ON KAGGLE:
  1. Create new notebook, accelerator = GPU T4 x1 (or CPU is fine — no GPU needed here)
  2. In Settings > Data, add previous Job A notebook as a data source:
       chakradharvijayarao/geom-proof-extraction-all-gpu-jobs
  3. Paste this file into a code cell and run
  4. Download output files
"""

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, spearmanr
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FILE PATHS
# Job A outputs are available at /kaggle/input/<kernel-slug>/
# The kernel slug for the extraction job is: geom-proof-extraction-all-gpu-jobs
# ─────────────────────────────────────────────────────────────────────────────

def _find_file(filename):
    """Search all of /kaggle/input/ for a filename, regardless of subfolder."""
    import os
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return Path(root) / filename
    raise FileNotFoundError(f"{filename} not found anywhere under /kaggle/input/")

MODELS = {
    "Qwen 2.5 3B": {
        "hs_path": _find_file("00_halueval_qwen3b.npz"),
        "n_params": 3e9,
    },
    "GPT-2 Medium 345M": {
        "hs_path": _find_file("00_halueval_gpt2med.npz"),
        "n_params": 345e6,
    },
    "Mamba-2 130M": {
        "hs_path": _find_file("04_mamba_hidden_states.npz"),
        "n_params": 130e6,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# INLINE: src/fisher.py (fisher_ratio, auroc_bound, fisher_curve)
# ─────────────────────────────────────────────────────────────────────────────

def fisher_ratio(H, y, method="pca", n_components=100):
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    mu_c = H[y == 1].mean(axis=0)
    mu_h = H[y == 0].mean(axis=0)
    delta = mu_c - mu_h
    if method == "pca":
        k = min(n_components, H.shape[1], H.shape[0] - 2)
        pca = PCA(n_components=k)
        H_proj = pca.fit_transform(H)
        mu_c_proj = H_proj[y == 1].mean(axis=0)
        mu_h_proj = H_proj[y == 0].mean(axis=0)
        delta_proj = mu_c_proj - mu_h_proj
        H_c_c = H_proj[y == 1] - mu_c_proj
        H_h_c = H_proj[y == 0] - mu_h_proj
        lw = LedoitWolf().fit(np.vstack([H_c_c, H_h_c]))
        Sigma_w = lw.covariance_
        try:
            w = np.linalg.solve(Sigma_w, delta_proj)
            J = float(delta_proj @ w)
        except np.linalg.LinAlgError:
            J = float(delta_proj @ np.linalg.lstsq(Sigma_w, delta_proj, rcond=None)[0])
    else:  # lda
        H_c = H[y == 1] - mu_c
        H_h = H[y == 0] - mu_h
        lw = LedoitWolf().fit(np.vstack([H_c, H_h]))
        Sigma_w = lw.covariance_
        try:
            w = np.linalg.solve(Sigma_w, delta)
            J = float(delta @ w)
        except np.linalg.LinAlgError:
            J = float(delta @ np.linalg.lstsq(Sigma_w, delta, rcond=None)[0])
    return max(0.0, J)


def auroc_bound(J):
    return float(norm.cdf(np.sqrt(max(0.0, J)) / 2))


def fisher_curve(H, y, method="pca", n_components=100, verbose=False):
    n_samples, n_layers, d = H.shape
    J_list, bound_list = [], []
    for L in range(n_layers):
        J_L = fisher_ratio(H[:, L, :], y, method=method, n_components=n_components)
        b_L = auroc_bound(J_L)
        J_list.append(J_L)
        bound_list.append(b_L)
        if verbose:
            print(f"  Fisher L{L:3d}: J={J_L:.4f}  bound={b_L:.4f}")
    best = int(np.argmax(J_list))
    return {
        "J": J_list, "auroc_bound": bound_list,
        "best_layer": best, "best_J": J_list[best],
        "depth_fraction": best / max(n_layers - 1, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INLINE: src/wasserstein.py (bures_w2_equal_cov, sliced_wasserstein_2,
#         mmd_squared, fisher_as_whitened_w2, ot_certificate_curve)
# ─────────────────────────────────────────────────────────────────────────────

def bures_w2_squared(H, y):
    H = H.astype(np.float64)
    H_c, H_h = H[y == 1], H[y == 0]
    mu_c, mu_h = H_c.mean(0), H_h.mean(0)
    mean_sq = float(np.sum((mu_c - mu_h) ** 2))
    lw_c = LedoitWolf().fit(H_c - mu_c)
    lw_h = LedoitWolf().fit(H_h - mu_h)
    Sc, Sh = lw_c.covariance_, lw_h.covariance_
    try:
        Sh_sqrt = sqrtm(Sh).real
        M = Sh_sqrt @ Sc @ Sh_sqrt
        M_sqrt = sqrtm(M).real
        bures_sq = max(0.0, float(np.trace(Sc + Sh - 2 * M_sqrt)))
    except Exception:
        bures_sq = 0.0
    return mean_sq + bures_sq


def bures_w2_equal_cov(H, y):
    mu_c = H[y == 1].mean(0).astype(np.float64)
    mu_h = H[y == 0].mean(0).astype(np.float64)
    return float(np.sum((mu_c - mu_h) ** 2))


def fisher_as_whitened_w2(H, y, n_components=100):
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)
    J = fisher_ratio(H, y, method="pca", n_components=n_components)
    mu_c, mu_h = H[y == 1].mean(0), H[y == 0].mean(0)
    H_c_c = H[y == 1] - mu_c
    H_h_c = H[y == 0] - mu_h
    lw = LedoitWolf().fit(np.vstack([H_c_c, H_h_c]))
    Sigma_w = lw.covariance_
    eigvals, eigvecs = np.linalg.eigh(Sigma_w)
    eigvals = np.maximum(eigvals, 1e-10)
    Sigma_w_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    H_white = H @ Sigma_w_inv_sqrt
    mu_c_w = H_white[y == 1].mean(0)
    mu_h_w = H_white[y == 0].mean(0)
    w2_whitened = float(np.sum((mu_c_w - mu_h_w) ** 2))
    rel_error = abs(J - w2_whitened) / max(J, 1e-10)
    return {"J": J, "w2_whitened": w2_whitened,
            "relative_error": rel_error, "identity_holds": rel_error < 0.05}


def sliced_wasserstein_2(H, y, n_projections=100, n_components=50, random_state=42):
    rng = np.random.RandomState(random_state)
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)
    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k, random_state=random_state)
    H_proj = pca.fit_transform(H)
    H_c, H_h = H_proj[y == 1], H_proj[y == 0]
    total_w1_sq = 0.0
    for _ in range(n_projections):
        theta = rng.randn(k)
        theta /= np.linalg.norm(theta) + 1e-10
        proj_c = np.sort(H_c @ theta)
        proj_h = np.sort(H_h @ theta)
        n = max(len(proj_c), len(proj_h))
        t = np.linspace(0, 1, n)
        c_i = np.interp(t, np.linspace(0, 1, len(proj_c)), proj_c)
        h_i = np.interp(t, np.linspace(0, 1, len(proj_h)), proj_h)
        total_w1_sq += float(np.mean((c_i - h_i) ** 2))
    return total_w1_sq / n_projections


def mmd_squared(H, y, gamma=None, n_components=50):
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)
    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k)
    H_proj = pca.fit_transform(H)
    H_c, H_h = H_proj[y == 1], H_proj[y == 0]
    if gamma is None:
        D = euclidean_distances(H_proj)
        pos_dists = D[D > 0]
        if len(pos_dists) == 0:
            return 0.0
        median_dist = np.median(pos_dists)
        if np.isnan(median_dist) or median_dist < 1e-10:
            return 0.0
        gamma = 1.0 / (2.0 * median_dist ** 2)
    K_cc = rbf_kernel(H_c, H_c, gamma=gamma)
    K_hh = rbf_kernel(H_h, H_h, gamma=gamma)
    K_ch = rbf_kernel(H_c, H_h, gamma=gamma)
    n_c, n_h = len(H_c), len(H_h)
    np.fill_diagonal(K_cc, 0)
    np.fill_diagonal(K_hh, 0)
    t_cc = K_cc.sum() / (n_c * (n_c - 1)) if n_c > 1 else 0.0
    t_hh = K_hh.sum() / (n_h * (n_h - 1)) if n_h > 1 else 0.0
    return float(t_cc + t_hh - 2.0 * K_ch.mean())


def ot_certificate_curve(H, y, n_projections=100, n_components=50, verbose=True):
    n_samples, n_layers, d = H.shape
    w2_bures_per_layer, w2_eq_per_layer = [], []
    sw2_per_layer, mmd_per_layer = [], []
    for L in range(n_layers):
        H_L = H[:, L, :].astype(np.float64)
        w2_b = bures_w2_squared(H_L, y)
        w2_e = bures_w2_equal_cov(H_L, y)
        sw = sliced_wasserstein_2(H_L, y, n_projections=n_projections,
                                  n_components=n_components)
        mm = mmd_squared(H_L, y, n_components=n_components)
        w2_bures_per_layer.append(w2_b)
        w2_eq_per_layer.append(w2_e)
        sw2_per_layer.append(sw)
        mmd_per_layer.append(mm)
        if verbose:
            print(f"  Layer {L:3d}: W₂_bures={w2_b:.4f}  SW₂={sw:.4f}  MMD²={mm:.6f}")
    best = int(np.argmax(sw2_per_layer))
    return {
        "w2_bures_per_layer": w2_bures_per_layer,
        "w2_eq_per_layer": w2_eq_per_layer,
        "sw2_per_layer": sw2_per_layer,
        "mmd_per_layer": mmd_per_layer,
        "best_layer_sw2": best,
        "best_sw2": sw2_per_layer[best],
        "best_depth_fraction_sw2": best / max(n_layers - 1, 1),
        "n_layers": n_layers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INLINE: src/spectral.py (marchenko_pastur_pdf, bbp_threshold,
#         esd_kl_from_mp, spectral_curve)
# ─────────────────────────────────────────────────────────────────────────────

def marchenko_pastur_pdf(x, gamma, sigma2=1.0):
    lam_plus  = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2 * max(1 - np.sqrt(gamma), 0.0) ** 2
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    x_m = x[mask]
    pdf[mask] = (np.sqrt((lam_plus - x_m) * np.maximum(x_m - lam_minus, 0))
                 / (2 * np.pi * gamma * sigma2 * x_m + 1e-15))
    return pdf


def bbp_threshold(d, n, sigma2=1.0):
    return sigma2 * np.sqrt(d / n)


def esd_kl_from_mp(H, gamma=None, n_bins=100):
    H = H.astype(np.float64)
    n, d = H.shape
    if gamma is None:
        gamma = d / n
    H_c = H - H.mean(0)
    _, s, _ = np.linalg.svd(H_c, full_matrices=False)
    eigenvalues = s ** 2 / n
    eigenvalues = np.sort(eigenvalues[eigenvalues > 1e-12])[::-1]
    sigma2_est  = float(np.median(eigenvalues))
    lam_plus    = sigma2_est * (1 + np.sqrt(gamma)) ** 2
    lam_minus   = sigma2_est * max(1 - np.sqrt(gamma), 0) ** 2
    theta_star  = bbp_threshold(d, n, sigma2_est)
    n_spikes    = int((eigenvalues > lam_plus).sum())
    spike_ratio = float(eigenvalues[0] / lam_plus) if lam_plus > 0 else float("nan")
    bulk_eigs   = eigenvalues[eigenvalues <= 2 * lam_plus]
    if len(bulk_eigs) < 5:
        kl = float("nan")
    else:
        bins = np.linspace(lam_minus * 0.5, lam_plus * 1.5, n_bins + 1)
        esd_hist, _ = np.histogram(bulk_eigs, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mp_dens = marchenko_pastur_pdf(bin_centers, gamma, sigma2_est)
        eps = 1e-8
        esd_reg = (esd_hist + eps) / (esd_hist.sum() + eps * len(esd_hist))
        mp_reg  = (mp_dens  + eps) / (mp_dens.sum()  + eps * len(mp_dens))
        kl = float(np.sum(esd_reg * np.log(esd_reg / mp_reg)))
    k_eff = int(n_spikes) if n_spikes > 0 else 1
    return {
        "kl_from_mp": kl, "eigenvalues": eigenvalues.tolist(),
        "lam_plus": float(lam_plus), "lam_minus": float(lam_minus),
        "n_spikes": n_spikes, "k_eff": k_eff,
        "gamma_eff": float(k_eff / n),
        "spike_ratio": spike_ratio, "bbp_threshold": float(theta_star),
        "sigma2_est": float(sigma2_est), "gamma": float(gamma),
    }


def spectral_curve(H, verbose=True):
    n_samples, n_layers, d = H.shape
    gamma = d / n_samples
    kl_list, n_spikes_list, spike_ratio_list = [], [], []
    for L in range(n_layers):
        stats = esd_kl_from_mp(H[:, L, :], gamma=gamma)
        kl_list.append(stats["kl_from_mp"])
        n_spikes_list.append(stats["n_spikes"])
        spike_ratio_list.append(stats["spike_ratio"])
        if verbose:
            print(f"  Layer {L:3d}: KL={stats['kl_from_mp']:.4f}  "
                  f"spikes={stats['n_spikes']}  spike_ratio={stats['spike_ratio']:.3f}")
    best = int(np.nanargmax(kl_list))
    return {
        "kl_per_layer": kl_list,
        "n_spikes_per_layer": n_spikes_list,
        "spike_ratio_per_layer": spike_ratio_list,
        "best_layer": best,
        "best_kl": kl_list[best],
        "best_depth_fraction": best / max(n_layers - 1, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_data(cfg):
    path = cfg["hs_path"]
    if not path.exists():
        return None
    data = np.load(path)
    key = "hidden_states" if "hidden_states" in data else list(data.keys())[0]
    H = data[key].astype(np.float32)
    if "labels" not in data:
        return None
    y = data["labels"].astype(int)
    n = min(H.shape[0], y.shape[0])
    return H[:n], y[:n]


# ─────────────────────────────────────────────────────────────────────────────
# PROBE AUROC PER LAYER (OOF)
# ─────────────────────────────────────────────────────────────────────────────

def probe_auroc_per_layer(H, y):
    n_samples, n_layers, d = H.shape
    aurocs = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
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


# ─────────────────────────────────────────────────────────────────────────────
# EXP 08: OT CERTIFICATE CURVE
# ─────────────────────────────────────────────────────────────────────────────

def run_exp08():
    print("=" * 70)
    print("Exp 08 — OT Certificate: Wasserstein Generalization of Fisher")
    print("=" * 70)

    all_results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")

        loaded = load_model_data(cfg)
        if loaded is None:
            print("  Skipping — data not found.")
            continue
        H, y = loaded
        print(f"  Shape: {H.shape}, hall_rate: {1 - y.mean():.3f}")

        if len(np.unique(y)) < 2:
            print(f"  Skipping — single-class labels (hall_rate={1-y.mean():.3f}). "
                  "Cannot compute Fisher certificate or probe AUROC.")
            continue

        # 1. Verify Fisher = W₂ in whitened space
        print("\n  [1] Verifying Fisher = W₂ in whitened space ...")
        check_layers = np.linspace(0, H.shape[1] - 1, 5, dtype=int)
        identity_checks = []
        for L in check_layers:
            chk = fisher_as_whitened_w2(H[:, L, :], y)
            identity_checks.append({
                "layer": int(L), "J": round(chk["J"], 5),
                "w2_whitened": round(chk["w2_whitened"], 5),
                "relative_error": round(chk["relative_error"], 5),
                "identity_holds": chk["identity_holds"],
            })
            print(f"    L{L:3d}: J={chk['J']:.5f}  W₂_white={chk['w2_whitened']:.5f}  "
                  f"err={chk['relative_error']:.4f}  "
                  f"{'✓' if chk['identity_holds'] else '✗'}")

        # 2. Fisher J(L) curve
        print("\n  [2] Fisher J(L) curve ...")
        fisher_result = fisher_curve(H, y, method="pca", n_components=100, verbose=False)

        # 3. OT certificate curve
        print("\n  [3] OT certificate curve (SW₂, MMD, Bures W₂) ...")
        ot_result = ot_certificate_curve(H, y, n_projections=100, n_components=50,
                                         verbose=True)

        # 4. OOF probe AUROC per layer
        print("\n  [4] OOF probe AUROC per layer ...")
        probe_aurocs = probe_auroc_per_layer(H, y)
        print(f"    Best probe AUROC: {max(probe_aurocs):.4f} at L{np.argmax(probe_aurocs)}")

        # 5. Spearman correlations
        J_vals     = np.array(fisher_result["J"])
        sw2_vals   = np.array(ot_result["sw2_per_layer"])
        mmd_vals   = np.array(ot_result["mmd_per_layer"])
        w2eq_vals  = np.array(ot_result["w2_eq_per_layer"])
        probe_vals = np.array(probe_aurocs)

        r_J    = float(spearmanr(J_vals,   probe_vals).statistic)
        r_sw2  = float(spearmanr(sw2_vals, probe_vals).statistic)
        r_mmd  = float(spearmanr(mmd_vals, probe_vals).statistic)
        r_w2eq = float(spearmanr(w2eq_vals, probe_vals).statistic)

        print(f"\n  [5] Correlation analysis ...")
        print(f"    Spearman r (Fisher J):     {r_J:.4f}")
        print(f"    Spearman r (Sliced W₂):    {r_sw2:.4f}")
        print(f"    Spearman r (Bures W₂_eq):  {r_w2eq:.4f}")
        print(f"    Spearman r (MMD²):         {r_mmd:.4f}")

        probe_best_layer = int(np.argmax(probe_aurocs))
        best = {
            "Fisher": {
                "layer": fisher_result["best_layer"],
                "certificate": fisher_result["best_J"],
                "depth": fisher_result["depth_fraction"],
            },
            "SW2": {
                "layer": ot_result["best_layer_sw2"],
                "certificate": ot_result["best_sw2"],
                "depth": ot_result["best_depth_fraction_sw2"],
            },
            "Probe": {
                "layer": probe_best_layer,
                "auroc": float(np.max(probe_aurocs)),
                "depth": float(probe_best_layer / (H.shape[1] - 1)),
            },
        }
        print(f"\n  Best-layer summary:")
        print(f"    Fisher best: L{best['Fisher']['layer']} ({best['Fisher']['depth']:.3f})")
        print(f"    SW₂ best:    L{best['SW2']['layer']} ({best['SW2']['depth']:.3f})")
        print(f"    Probe best:  L{best['Probe']['layer']} ({best['Probe']['depth']:.3f})")

        all_results[model_name] = {
            "model_name": model_name,
            "n_params": cfg["n_params"],
            "n_layers": int(H.shape[1]),
            "n_samples": int(H.shape[0]),
            "identity_verification": identity_checks,
            "spearman_correlations": {
                "Fisher_J": r_J, "SW2": r_sw2, "MMD2": r_mmd, "W2_eq": r_w2eq,
            },
            "best_layer": best,
            "fisher_per_layer": [float(j) for j in J_vals],
            "sw2_per_layer": ot_result["sw2_per_layer"],
            "mmd_per_layer": ot_result["mmd_per_layer"],
            "w2_bures_per_layer": ot_result["w2_bures_per_layer"],
            "w2_eq_per_layer": ot_result["w2_eq_per_layer"],
            "probe_auroc_per_layer": probe_aurocs,
        }

    out = OUT / "08_ot_certificate.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    # Plot
    _plot_exp08(all_results)

    # Winner table
    print("\n" + "=" * 70)
    print("CERTIFICATE COMPARISON — Which predicts probe AUROC best?")
    print(f"{'Model':<25} {'Fisher J':>10} {'SW₂':>10} {'MMD²':>10} {'W₂_eq':>10} {'WINNER':>10}")
    print("─" * 75)
    for model, r in all_results.items():
        corrs = r["spearman_correlations"]
        winner = max(corrs, key=corrs.get)
        print(f"  {model:<23} "
              f"{corrs['Fisher_J']:>10.4f} "
              f"{corrs['SW2']:>10.4f} "
              f"{corrs['MMD2']:>10.4f} "
              f"{corrs['W2_eq']:>10.4f} "
              f"{winner:>10}")

    return all_results


def _plot_exp08(results):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    colors = {"Fisher J": "steelblue", "SW₂": "darkorange",
              "MMD²": "green", "Probe AUROC": "black"}

    def norm01(x):
        x = np.array(x, dtype=float)
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-10)

    for ax, (model_name, r) in zip(axes, results.items()):
        n_layers = r["n_layers"]
        nd = [L / (n_layers - 1) for L in range(n_layers)]
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
        corrs = r["spearman_correlations"]
        ax.set_title(f"{model_name}\nSpearman r: "
                     f"Fisher={corrs['Fisher_J']:.3f}, "
                     f"SW₂={corrs['SW2']:.3f}")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.1)

    plt.suptitle("OT Certificate Comparison: Fisher vs Wasserstein vs MMD", fontsize=13)
    plt.tight_layout()
    out = OUT / "08_ot_certificates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# EXP 09: SPECTRAL PHASE TRANSITION
# ─────────────────────────────────────────────────────────────────────────────

def run_exp09(exp08_results):
    print("\n" + "=" * 70)
    print("Exp 09 — Spectral Phase Transition Analysis (RMT / BBP)")
    print("=" * 70)

    all_results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}")

        loaded = load_model_data(cfg)
        if loaded is None:
            print("  Skipping — data not found.")
            continue
        H, y = loaded
        n, n_layers, d = H.shape
        gamma = d / n
        theta_star = bbp_threshold(d, n)
        print(f"  Shape: {H.shape}, γ=d/n={gamma:.3f}, BBP threshold θ*={theta_star:.4f}")

        print("\n  Computing spectral curve ...")
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

        # Compare spectral peak with SW₂ and probe peaks from Exp 08
        if model_name in exp08_results:
            r08 = exp08_results[model_name]
            sw2_best   = r08["best_layer"]["SW2"]["layer"]
            probe_best = r08["best_layer"]["Probe"]["layer"]
            spec_best  = spec["best_layer"]
            result["spectral_vs_sw2_agreement"]   = abs(spec_best - sw2_best) <= 2
            result["spectral_vs_probe_agreement"] = abs(spec_best - probe_best) <= 2
            print(f"\n  Phase transition coincidence:")
            print(f"    Spectral KL peak: L{spec_best} ({spec['best_depth_fraction']:.3f})")
            print(f"    SW₂ peak:         L{sw2_best}")
            print(f"    Probe AUROC peak: L{probe_best}")
            print(f"    Spectral ↔ probe within ±2: "
                  f"{'YES ✓' if result['spectral_vs_probe_agreement'] else 'NO'}")

        all_results[model_name] = result

    out = OUT / "09_spectral_phase_transition.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out}")

    _plot_exp09(all_results, exp08_results)
    return all_results


def _plot_exp09(results, exp08_results):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    color_map = {"Qwen 2.5 3B": "darkorange",
                 "GPT-2 Medium 345M": "steelblue",
                 "Mamba-2 130M": "green"}

    def norm01(x):
        x = np.array(x, dtype=float)
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return x
        return (x - finite.min()) / (finite.max() - finite.min() + 1e-10)

    for ax, (model_name, r) in zip(axes, results.items()):
        cfg_color = color_map.get(model_name, "black")
        n_layers = r["n_layers"]
        nd = [L / (n_layers - 1) for L in range(n_layers)]

        ax.plot(nd, norm01(r["kl_per_layer"]), color=cfg_color, lw=2.5,
                label="Spectral KL (ESD from MP)")

        if model_name in exp08_results:
            r08 = exp08_results[model_name]
            if len(r08["sw2_per_layer"]) == len(nd):
                ax.plot(nd, norm01(r08["sw2_per_layer"]),
                        color="purple", lw=2, ls="--", label="SW₂ (normalized)")
                ax.plot(nd, r08["probe_auroc_per_layer"],
                        color="black", lw=1.5, ls=":", label="Probe AUROC")

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
    out = OUT / "09_spectral_phase_transition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    # Check input files
    print("Checking input files ...")
    for model_name, cfg in MODELS.items():
        exists = cfg["hs_path"].exists()
        print(f"  {model_name}: {cfg['hs_path']} — {'✓' if exists else '✗ MISSING'}")

    print()
    exp08_results = run_exp08()
    print()
    exp09_results = run_exp09(exp08_results)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    print("\nFiles written to /kaggle/working/:")
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
