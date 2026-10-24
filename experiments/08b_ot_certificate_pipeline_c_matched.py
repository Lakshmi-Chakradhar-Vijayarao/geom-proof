"""
Exp.08b -- OT/Bures certificate comparison, extended to n=3 architectures on
one consistent Pipeline-C (matched, real-generation) methodology, with
bootstrap CIs on every Spearman correlation.

WHY THIS SUPERSEDES EXP.08 FOR THE PAPER'S HEADLINE OT/BURES COMPARISON,
RATHER THAN BEING RUN ALONGSIDE IT. Exp.08's own Qwen-3B/GPT-2-Medium rows
use Pipeline-A (forced-choice) hidden states (00_halueval_qwen3b.npz,
00_halueval_gpt2med.npz) -- the same task-confound this paper's Table
tab:certificate documents and explicitly corrects for the certificate's own
headline result (Exp.15, Pipeline-C matched methodology). Continuing to
report the OT/Bures comparison on Pipeline-A data while the paper's central
Fisher-J result has moved to Pipeline-C would reintroduce exactly the
confound already fixed elsewhere. This script reruns the identical Exp.08
comparison (Fisher/Bures-W2/Sliced-W2/MMD vs. probe AUROC, per layer) on
Pipeline-C data for all three models, and adds Pythia 410M/1.4B/2.8B (Exp.32's
extraction, also Pipeline-C) as a third independent architecture family --
taking the OT/Bures comparison from n=2 to n=3, the review's own recommended
minimum-viable extension given the data was already cached.

BOOTSTRAP CIs (new, not in Exp.08). None of Exp.08's Spearman correlations
carry an uncertainty interval -- flagged directly by an independent review
as this paper's one place where its otherwise-thorough multiplicity/
uncertainty discipline has a gap. Resamples layers with replacement (the
natural resampling unit for a per-layer correlation), refits nothing (the
per-layer certificate/probe values are already computed once and reused),
and reports a percentile bootstrap CI on each Spearman rho -- adapting the
resample-and-percentile pattern already used elsewhere in this project
(experiments/27_bootstrap_ci_nondegenerate_ood.py).

All data already cached; zero new model inference (CPU-only, ~10-20 min,
dominated by Sliced-Wasserstein's random-projection cost, same as Exp.08).
Output: results/logs/08b_ot_certificate_pipeline_c_matched.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.wasserstein import fisher_as_whitened_w2, ot_certificate_curve  # noqa: E402
from src.fisher import fisher_curve  # noqa: E402

RESULTS_DIR = ROOT / "results" / "logs"
HS_DIR = ROOT / "results" / "hidden_states"
PYTHIA_DIR = ROOT / "kaggle" / "exp32_third_architecture_pythia" / "output"

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260812

MODELS = {
    "Qwen 2.5 3B (Pipeline C)": {
        "hs_path": HS_DIR / "15_hidden_states_qwen_3b.npz",
        "n_params": 3e9,
    },
    "GPT-2 Medium (Pipeline C)": {
        "hs_path": HS_DIR / "15_hidden_states_gpt2_medium.npz",
        "n_params": 345e6,
    },
    "Pythia 2.8B (Pipeline C)": {
        "hs_path": PYTHIA_DIR / "32_hidden_states_pythia_2.8b.npz",
        "n_params": 2.8e9,
    },
}


def load(cfg):
    if not cfg["hs_path"].exists():
        return None
    data = np.load(cfg["hs_path"])
    H = data["hidden_states"].astype(np.float32)
    y = data["labels"].astype(int)
    n = min(H.shape[0], y.shape[0])
    return H[:n], y[:n]


def bootstrap_spearman_ci(a, b, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap CI on Spearman rho, resampling layers (the paired
    (certificate, probe_auroc) observations across layers) with replacement."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    rhos = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        rho = spearmanr(a[idx], b[idx]).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi), float(rhos.mean())


def probe_auroc_per_layer(H, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    n_samples, n_layers, _ = H.shape
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


def run():
    print("=" * 70)
    print("Exp 08b -- OT Certificate, Pipeline-C-matched, n=3 architectures, bootstrap CIs")
    print("=" * 70)
    all_results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'-'*60}\nModel: {model_name}")
        loaded = load(cfg)
        if loaded is None:
            print(f"  Skipping -- {cfg['hs_path']} not found.")
            continue
        H, y = loaded
        print(f"  Shape: {H.shape}, hall_rate: {1 - y.mean():.3f}")
        if len(np.unique(y)) < 2:
            print("  Skipping -- single-class labels.")
            continue

        fisher_result = fisher_curve(H, y, method="pca", n_components=100, verbose=False)
        ot_result = ot_certificate_curve(H, y, n_projections=100, n_components=50, verbose=True)
        probe_aurocs = probe_auroc_per_layer(H, y)

        J_vals = np.array(fisher_result["J"])
        sw2_vals = np.array(ot_result["sw2_per_layer"])
        mmd_vals = np.array(ot_result["mmd_per_layer"])
        w2_eq_vals = np.array(ot_result["w2_eq_per_layer"])
        probe_vals = np.array(probe_aurocs)

        correlations = {}
        for cert_name, cert_vals in [("Fisher_J", J_vals), ("SW2", sw2_vals),
                                       ("MMD2", mmd_vals), ("W2_eq_Bures", w2_eq_vals)]:
            r = float(spearmanr(cert_vals, probe_vals).statistic)
            lo, hi, boot_mean = bootstrap_spearman_ci(cert_vals, probe_vals)
            correlations[cert_name] = {
                "spearman_r": r, "bootstrap_ci_95": [lo, hi], "bootstrap_mean": boot_mean,
                "ci_excludes_zero": (lo > 0) or (hi < 0),
            }
            print(f"    {cert_name}: r={r:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'(excludes 0)' if (lo > 0 or hi < 0) else '(includes 0)'}")

        all_results[model_name] = {
            "n_params": cfg["n_params"],
            "n_layers": H.shape[1],
            "n_samples": H.shape[0],
            "spearman_correlations_with_bootstrap_ci": correlations,
            "fisher_per_layer": [float(j) for j in J_vals],
            "sw2_per_layer": ot_result["sw2_per_layer"],
            "w2_eq_per_layer": ot_result["w2_eq_per_layer"],
            "probe_auroc_per_layer": probe_aurocs,
        }

    out = RESULTS_DIR / "08b_ot_certificate_pipeline_c_matched.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
    return all_results


if __name__ == "__main__":
    run()
