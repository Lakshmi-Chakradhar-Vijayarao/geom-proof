"""
Closing the round-5 review gap on Pipeline A (HaluEval forced-choice):
"Pipeline A's ~0.99 AUROCs may in part reflect a lexical or
template-identity distinction -- which of two pre-written strings was
fed in -- rather than hallucination detection, with no control that
dissociates the two."

Pipeline A's input construction (00_halueval_extraction.py) is
f"Q: {question}\nA: {answer}" where `answer` is either `right_answer`
(label=1) or `hallucinated_answer` (label=0) for the SAME question. This
script tests whether trivial SURFACE features of the answer string alone
-- with no hidden states, no model forward pass at all -- can separate
the two classes. If a simple surface classifier gets high AUROC, the
near-saturated 0.99 hidden-state AUROC is at serious risk of partly
reading "which pre-written string was this" rather than truthfulness. If
the surface classifier is near chance, that risk is substantially
reduced (though not eliminated -- hidden states could still encode
surface features the shallow classifier below doesn't capture).

No new hidden-state extraction needed -- this operates purely on
re-loaded HaluEval QA text, independent of the cached .npz tensors.
"""
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path

N_SAMPLES = 1000
RANDOM_STATE = 42


def surface_features(question: str, answer: str) -> list:
    q_words = set(question.lower().split())
    a_words = answer.lower().split()
    a_words_set = set(a_words)
    overlap = len(q_words & a_words_set) / max(1, len(q_words | a_words_set))
    return [
        len(answer),                                  # char length
        len(a_words),                                  # word count
        overlap,                                        # Jaccard overlap with question
        answer.count(","),                             # punctuation density proxy
        float(answer[:1].isupper()) if answer else 0.0,  # starts capitalized
        len(question),                                  # question char length (constant per pair, but included)
    ]


def main():
    ds = load_dataset("pminervini/HaluEval", "qa")["data"]
    samples = ds.select(range(min(N_SAMPLES, len(ds))))

    X, y = [], []
    for row in samples:
        X.append(surface_features(row["question"], row["right_answer"]))
        y.append(1)
        X.append(surface_features(row["question"], row["hallucinated_answer"]))
        y.append(0)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=int)
    print(f"N={len(y)}  (correct={int(y.sum())}, hallucinated={int((y==0).sum())})")
    print(f"Feature means (correct): {X[y==1].mean(axis=0).round(2)}")
    print(f"Feature means (hallucinated): {X[y==0].mean(axis=0).round(2)}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
    auc_mean, auc_std = float(cv["test_auroc"].mean()), float(cv["test_auroc"].std())
    print(f"\nSurface-only (6 trivial features, no hidden states) CV AUROC: "
          f"{auc_mean:.4f} +- {auc_std:.4f}")

    # Permutation test for significance
    rng = np.random.default_rng(RANDOM_STATE)
    perm_aucs = []
    for _ in range(500):
        y_perm = rng.permutation(y)
        cv_p = cross_validate(pipe, X, y_perm, cv=skf, scoring=scoring)
        perm_aucs.append(float(cv_p["test_auroc"].mean()))
    perm_aucs = np.array(perm_aucs)
    p_value = float((perm_aucs >= auc_mean).sum() + 1) / (500 + 1)
    print(f"Permutation test (500 shuffles): perm_mean={perm_aucs.mean():.4f}, p={p_value:.4f}")

    import json
    out = {
        "n": len(y), "surface_auroc_mean": auc_mean, "surface_auroc_std": auc_std,
        "permutation_p_value": p_value, "permutation_mean": float(perm_aucs.mean()),
    }
    out_path = ROOT / "results" / "18_pipeline_a_surface_confound_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
