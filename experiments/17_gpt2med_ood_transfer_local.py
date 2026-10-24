"""
Closing the "GPT-2 Medium cross-architecture OOD point... left to future
work" gap flagged in round-4 review (main.tex). The Kaggle Exp 14 run
(results/geom-proof-exp14-ood-transfer.log) skipped GPT-2 Medium because
its HaluEval hidden-state cache wasn't attached to that Kaggle session --
but both the in-distribution (HaluEval) and OOD (TruthfulQA) GPT-2
Medium hidden-state tensors are already extracted and sitting locally
(results/hidden_states/00_halueval_gpt2med.npz,
results/hidden_states/14_truthfulqa_gpt2med.npz). This reruns the exact
same protocol as kaggle/exp14_ood_transfer.py's ood_transfer_test()
locally, CPU-only, no new model inference.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path
HS_DIR = ROOT / "results" / "hidden_states"
HALUEVAL_PATH = HS_DIR / "00_halueval_gpt2med.npz"
TRUTHFULQA_PATH = HS_DIR / "14_truthfulqa_gpt2med.npz"

N_COMPONENTS = 100
N_SPLITS = 5
RANDOM_STATE = 42
LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)


def best_layer_probe(H_train, y_train):
    n, n_layers, d = H_train.shape
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_auroc, best_layer = 0.0, 0
    for L in range(n_layers):
        H_L = H_train[:, L, :]
        oof = np.zeros(n)
        for tr, te in skf.split(H_L, y_train):
            sc = StandardScaler()
            pca = PCA(n_components=min(N_COMPONENTS, H_L[tr].shape[0] - 1, H_L.shape[1]),
                      random_state=RANDOM_STATE)
            X_tr = pca.fit_transform(sc.fit_transform(H_L[tr]))
            X_te = pca.transform(sc.transform(H_L[te]))
            clf = LogisticRegression(**LR_KWARGS)
            clf.fit(X_tr, y_train[tr])
            oof[te] = clf.predict_proba(X_te)[:, 1]
        auroc = float(roc_auc_score(y_train, oof))
        if auroc > best_auroc:
            best_auroc, best_layer = auroc, L
    return best_layer, best_auroc


def ood_transfer_test(H_train, y_train, H_ood, y_ood, model_name):
    print(f"\n  OOD transfer: {model_name}")
    n_train = len(y_train)
    n_layers_ood = H_ood.shape[1]

    print(f"    Finding best probe layer on HaluEval ({n_train} samples) ...")
    best_layer, best_id_auroc = best_layer_probe(H_train, y_train)
    print(f"    Best layer: L{best_layer}  in-dist AUROC: {best_id_auroc:.4f}")

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(n_train)
    n_tr = int(0.75 * n_train)
    tr_idx = idx[:n_tr]
    cal_idx = idx[n_tr:]

    H_L_train = H_train[:, best_layer, :]
    sc = StandardScaler()
    pca = PCA(n_components=min(N_COMPONENTS, H_L_train[tr_idx].shape[0] - 1, H_L_train.shape[1]),
              random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(sc.fit_transform(H_L_train[tr_idx]))
    X_cal = pca.transform(sc.transform(H_L_train[cal_idx]))

    clf = LogisticRegression(**LR_KWARGS)
    clf.fit(X_tr, y_train[tr_idx])

    scores_cal = clf.predict_proba(X_cal)[:, 1]
    alpha_star = accept_rate_id = emp_hall_id = thresh_star = None
    for alpha in np.arange(0.01, 0.50, 0.01):
        thresh = np.quantile(scores_cal[y_train[cal_idx] == 0], 1 - alpha)
        accepted = scores_cal > thresh
        if accepted.sum() == 0:
            continue
        hall_rate = float((y_train[cal_idx][accepted] == 0).mean())
        if hall_rate <= alpha:
            alpha_star = float(round(alpha, 2))
            accept_rate_id = float(accepted.mean())
            emp_hall_id = hall_rate
            thresh_star = thresh
            break

    print(f"    Conformal alpha*={alpha_star}  accept_rate={accept_rate_id}  emp_hall={emp_hall_id}")

    ood_layer = min(best_layer, n_layers_ood - 1)
    H_L_ood = H_ood[:, ood_layer, :]
    X_ood = pca.transform(sc.transform(H_L_ood))
    scores_ood = clf.predict_proba(X_ood)[:, 1]

    try:
        ood_auroc = float(roc_auc_score(y_ood, scores_ood))
    except ValueError:
        ood_auroc = float("nan")

    if alpha_star is not None:
        accepted_ood = scores_ood > thresh_star
        n_accepted_ood = accepted_ood.sum()
        if n_accepted_ood > 0:
            emp_hall_ood = float((y_ood[accepted_ood] == 0).mean())
            accept_rate_ood = float(accepted_ood.mean())
            violation_factor = emp_hall_ood / alpha_star if alpha_star > 0 else float("nan")
        else:
            emp_hall_ood = float("nan")
            accept_rate_ood = 0.0
            violation_factor = float("nan")
    else:
        emp_hall_ood = accept_rate_ood = violation_factor = float("nan")

    print(f"    OOD AUROC: {ood_auroc:.4f}  (in-dist: {best_id_auroc:.4f})")
    print(f"    OOD hall rate: {emp_hall_ood}  OOD accept rate: {accept_rate_ood}")
    print(f"    Violation factor: {violation_factor}")

    return {
        "model": model_name,
        "best_layer": best_layer,
        "in_dist_auroc": best_id_auroc,
        "alpha_star": alpha_star,
        "in_dist_accept_rate": accept_rate_id,
        "in_dist_emp_hall": emp_hall_id,
        "ood_auroc": ood_auroc,
        "ood_emp_hall": emp_hall_ood,
        "ood_accept_rate": accept_rate_ood,
        "violation_factor": violation_factor,
        "ood_collapse": violation_factor > 3 if not np.isnan(violation_factor) else None,
        "auroc_drop": float(best_id_auroc - ood_auroc),
        "ood_n_positive": int(y_ood.sum()),
        "ood_n_total": int(len(y_ood)),
    }


def main():
    d_id = np.load(HALUEVAL_PATH)
    H_id = d_id["hidden_states"].astype(np.float32)
    y_id = d_id["labels"].astype(int)
    d_ood = np.load(TRUTHFULQA_PATH)
    H_ood = d_ood["hidden_states"].astype(np.float32)
    y_ood = d_ood["labels"].astype(int)

    print(f"GPT-2 Medium in-dist (HaluEval): {H_id.shape}  hall_rate={1-y_id.mean():.3f}")
    print(f"GPT-2 Medium OOD (TruthfulQA):   {H_ood.shape}  hall_rate={1-y_ood.mean():.3f}  "
          f"n_correct={int(y_ood.sum())}/{len(y_ood)}")

    result = ood_transfer_test(H_id, y_id, H_ood, y_ood, "GPT-2 Medium")

    out_path = ROOT / "results" / "17_gpt2med_ood_transfer.json"
    with open(out_path, "w") as f:
        json.dump({"GPT-2 Medium": result}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
