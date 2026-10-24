"""
Elite-review follow-up on Exp.26 (non-degenerate OOD transfer, HaluEval ->
NQ-open, both Pipeline C). Bootstraps a 95% percentile CI on the
in-distribution AUROC, the OOD AUROC, and their gap, refitting the
identical probe (PCA(100)+LogisticRegression at the already-selected
best layer, L24) on each bootstrap resample of both the HaluEval
training pool and the NQ-open test set, rather than reporting only a
single point estimate for a result now central to this paper's
qualitative "no collapse" claim.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
HALUEVAL_PATH = ROOT / "results" / "hidden_states" / "15_hidden_states_qwen_3b.npz"
NQ_OPEN_PATH = ROOT / "kaggle" / "exp26_nondegenerate_ood" / "output" / "26_nq_open_qwen3b.npz"
OUT_PATH = ROOT / "results" / "logs" / "27_bootstrap_ci_nondegenerate_ood.json"

BEST_LAYER = 24  # selected in Exp.26 via 5-fold OOF AUROC on HaluEval
N_COMPONENTS = 100
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def fit_and_score(H_train, y_train, H_test, y_test, seed):
    rng = np.random.RandomState(seed)
    sc = StandardScaler()
    X_train = sc.fit_transform(H_train)
    pca = PCA(n_components=min(N_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1]), random_state=seed)
    X_train_p = pca.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(X_train_p, y_train)

    X_test = sc.transform(H_test)
    X_test_p = pca.transform(X_test)
    scores_test = clf.predict_proba(X_test_p)[:, 1]
    try:
        auroc_test = roc_auc_score(y_test, scores_test)
    except ValueError:
        auroc_test = float("nan")
    return auroc_test


def bootstrap_resample(H, y, rng):
    idx = rng.randint(0, len(y), size=len(y))
    return H[idx], y[idx]


def main():
    d_id = np.load(HALUEVAL_PATH)
    H_id_full, y_id_full = d_id["hidden_states"].astype(np.float32), d_id["labels"].astype(int)
    H_id = H_id_full[:, BEST_LAYER, :]

    d_ood = np.load(NQ_OPEN_PATH)
    H_ood_full, y_ood = d_ood["hidden_states"].astype(np.float32), d_ood["labels"].astype(int)
    H_ood = H_ood_full[:, BEST_LAYER, :]

    print(f"HaluEval (in-dist): {H_id.shape}, hall_rate={1-y_id_full.mean():.3f}")
    print(f"NQ-open (OOD): {H_ood.shape}, hall_rate={1-y_ood.mean():.3f}")

    # Point estimate: fit on full in-dist HaluEval, score both in-dist (via a
    # fresh 80/20 split for a fair in-dist test) and OOD.
    rng0 = np.random.RandomState(RANDOM_STATE)
    perm = rng0.permutation(len(y_id_full))
    split = int(0.8 * len(y_id_full))
    tr_idx, te_idx = perm[:split], perm[split:]

    point_id_auroc = fit_and_score(H_id[tr_idx], y_id_full[tr_idx], H_id[te_idx], y_id_full[te_idx], RANDOM_STATE)
    point_ood_auroc = fit_and_score(H_id[tr_idx], y_id_full[tr_idx], H_ood, y_ood, RANDOM_STATE)
    print(f"\nPoint estimate: in-dist AUROC={point_id_auroc:.4f}, OOD AUROC={point_ood_auroc:.4f}, "
          f"gap={point_ood_auroc - point_id_auroc:+.4f}")

    boot_id_aurocs, boot_ood_aurocs, boot_gaps = [], [], []
    rng = np.random.RandomState(RANDOM_STATE)
    for b in range(N_BOOTSTRAP):
        H_tr_b, y_tr_b = bootstrap_resample(H_id[tr_idx], y_id_full[tr_idx], rng)
        if len(set(y_tr_b.tolist())) < 2:
            continue
        H_te_id_b, y_te_id_b = bootstrap_resample(H_id[te_idx], y_id_full[te_idx], rng)
        H_te_ood_b, y_te_ood_b = bootstrap_resample(H_ood, y_ood, rng)
        if len(set(y_te_id_b.tolist())) < 2 or len(set(y_te_ood_b.tolist())) < 2:
            continue

        auroc_id_b = fit_and_score(H_tr_b, y_tr_b, H_te_id_b, y_te_id_b, RANDOM_STATE + b)
        auroc_ood_b = fit_and_score(H_tr_b, y_tr_b, H_te_ood_b, y_te_ood_b, RANDOM_STATE + b)
        if np.isnan(auroc_id_b) or np.isnan(auroc_ood_b):
            continue
        boot_id_aurocs.append(auroc_id_b)
        boot_ood_aurocs.append(auroc_ood_b)
        boot_gaps.append(auroc_ood_b - auroc_id_b)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap [{b+1}/{N_BOOTSTRAP}]", flush=True)

    boot_id_aurocs = np.array(boot_id_aurocs)
    boot_ood_aurocs = np.array(boot_ood_aurocs)
    boot_gaps = np.array(boot_gaps)

    def pct_ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    id_ci = pct_ci(boot_id_aurocs)
    ood_ci = pct_ci(boot_ood_aurocs)
    gap_ci = pct_ci(boot_gaps)

    print(f"\nBootstrap ({len(boot_gaps)} valid resamples of {N_BOOTSTRAP}):")
    print(f"  In-dist AUROC: {boot_id_aurocs.mean():.4f}  95% CI [{id_ci[0]:.4f}, {id_ci[1]:.4f}]")
    print(f"  OOD AUROC:     {boot_ood_aurocs.mean():.4f}  95% CI [{ood_ci[0]:.4f}, {ood_ci[1]:.4f}]")
    print(f"  Gap (OOD-ID):  {boot_gaps.mean():+.4f}  95% CI [{gap_ci[0]:+.4f}, {gap_ci[1]:+.4f}]")
    frac_positive = float((boot_gaps > 0).mean())
    print(f"  Fraction of bootstrap resamples with OOD > in-dist AUROC: {frac_positive:.3f}")

    out = {
        "best_layer": BEST_LAYER, "n_bootstrap_requested": N_BOOTSTRAP,
        "n_bootstrap_valid": len(boot_gaps),
        "point_estimate": {"in_dist_auroc": point_id_auroc, "ood_auroc": point_ood_auroc,
                            "gap": point_ood_auroc - point_id_auroc},
        "bootstrap": {
            "in_dist_auroc_mean": float(boot_id_aurocs.mean()), "in_dist_auroc_ci95": id_ci,
            "ood_auroc_mean": float(boot_ood_aurocs.mean()), "ood_auroc_ci95": ood_ci,
            "gap_mean": float(boot_gaps.mean()), "gap_ci95": gap_ci,
            "fraction_ood_greater_than_indist": frac_positive,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
