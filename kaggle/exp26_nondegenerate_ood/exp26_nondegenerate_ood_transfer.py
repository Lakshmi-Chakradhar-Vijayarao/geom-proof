"""
GEOM-PROOF -- Exp 26: a non-degenerate OOD transfer test
==========================================================

Tier 2 final-audit pass. Every OOD test this paper has run so far
(Sec. "conformal-ood") calibrates in-distribution on HaluEval and tests
out-of-distribution on TruthfulQA, ROUGE-L-labeled -- and TruthfulQA's
labels are near-degenerate under that scheme (e.g. 4/400 positive for
GPT-2 Medium, 3/400 for Qwen 7B, 7/400 for Qwen 3B). This means the OOD
arm's AUROC is close to undefined in every prior OOD test in this paper:
the collapse could be a real distribution-shift effect, or it could
simply be an artifact of asking for an AUROC on a near-single-class set.
Exp 15 (Pipeline C: SQuAD-style EM/F1 scoring, `label = 1 if EM or F1 >=
0.5 else 0`) already proved non-degenerate on HaluEval itself (24-103
positives per model out of 500, depending on model). This experiment
reuses Pipeline C, unchanged, on a SECOND, different QA dataset --
NQ-open (Natural Questions, open-domain short-answer split) -- as the
OOD arm, so the labeling scheme is held constant across both arms and
only the distribution shifts. This isolates the OOD-collapse question
from the near-degenerate-label confound for the first time in this
paper.

In-distribution arm: reuses ALREADY-CACHED Qwen2.5-3B-Instruct HaluEval
hidden states + Pipeline-C labels from Exp 15
(`results/hidden_states/15_hidden_states_qwen_3b.npz`) -- no
regeneration needed, uploaded here as a Kaggle dataset input.

OOD arm: fresh generation on NQ-open (`google-research-datasets/nq_open`,
validation split) with the SAME model, SAME prompt template
("Q: {question}\\nA:", greedy, max_new_tokens=40) and SAME Pipeline-C
scoring as Exp 15, extracting hidden states at every layer identically.

Analysis: identical procedure to Exp 14's `ood_transfer_test()` --
5-fold-OOF best-layer selection on the in-dist (HaluEval) set, a 75/25
train/calibration split, PCA(100)+LogisticRegression probe, split-
conformal threshold calibrated in-dist, then frozen probe+threshold
applied to the OOD (NQ-open) set: OOD AUROC (now well-defined, unlike
the TruthfulQA arm) and the same conformal-violation-factor metric used
throughout this paper.
"""
import gc
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import norm as scipy_norm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    try:
        _a = torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")
        print(f"GPU sanity check passed: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"GPU sanity check failed ({e}) -- falling back to CPU.")
        DEVICE = "cpu"

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)


def _resolve_input_dir(filename):
    candidates = sorted(Path("/kaggle/input").glob(f"**/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"{filename} not found under /kaggle/input. "
                                 f"Tree: {list(Path('/kaggle/input').rglob('*'))}")
    return candidates[0]


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"   # matches Exp 15's cached HaluEval arm
HALUEVAL_CACHE_NAME = "15_hidden_states_qwen_3b.npz"
N_NQ_OPEN = 500        # matches Exp 15's HaluEval N for comparability
N_COMPONENTS = 100
N_SPLITS = 5
RANDOM_STATE = 42
MAX_NEW_TOKENS = 40
LR_KWARGS = dict(max_iter=1000, C=1.0, random_state=RANDOM_STATE)


# ── Pipeline C scoring, verbatim from exp15_matched_generation.py ───────────

def normalize_answer(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction, reference):
    p, r = normalize_answer(prediction), normalize_answer(reference)
    return bool(r) and (p == r or r in p)


def token_f1(prediction, reference):
    pt, rt = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = Counter(pt) & Counter(rt)
    num_same = sum(common.values())
    if not pt or not rt or num_same == 0:
        return 0.0
    prec, rec = num_same / len(pt), num_same / len(rt)
    return 2 * prec * rec / (prec + rec)


def score_generation(generated, right_answers):
    """right_answers: list of acceptable reference strings (NQ-open provides
    multiple annotated short answers per question) -- score against each and
    take the max, matching this paper's own multi-reference convention
    elsewhere (e.g. TruthfulQA's correct_answers list)."""
    best_em, best_f1 = 0, 0.0
    for ref in right_answers:
        em = exact_match(generated, ref)
        f1 = token_f1(generated, ref)
        best_em = best_em or em
        best_f1 = max(best_f1, f1)
    label = 1 if (best_em or best_f1 >= 0.5) else 0
    return label, max(float(best_em), best_f1)


# ── NQ-open extraction (fresh generation + Pipeline C labeling) ─────────────

def extract_nq_open(model, tokenizer, n):
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    samples = list(ds)[:n]
    print(f"  {len(samples)} NQ-open questions loaded")

    all_hs, all_labels, all_scores, generations = [], [], [], []
    t0 = time.time()
    for i, sample in enumerate(samples):
        question = sample["question"]
        answers = [a for a in sample["answer"] if a.strip()]
        if not answers:
            continue
        prompt = f"Q: {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        label, score = score_generation(generated_text, answers)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])

        all_hs.append(hs)
        all_labels.append(label)
        all_scores.append(score)
        generations.append({"question": question, "generated": generated_text,
                             "answers": answers, "label": label, "score": score})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] elapsed={elapsed:.0f}s "
                  f"n_correct_so_far={sum(all_labels)}", flush=True)

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"NQ-open extraction complete: N={len(y)}, n_correct={int(y.sum())}, "
          f"n_hallucinated={int((y==0).sum())} (hall_rate={1-y.mean():.3f})")
    return H, y, generations


# ── Fisher / probe / conformal, identical to exp14_ood_transfer.py ─────────

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


def fisher_ratio_pca(H_L, y, k=100):
    k = min(k, H_L.shape[1], H_L.shape[0] - 2)
    H_L = H_L.astype(np.float64)
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    H_proj = pca.fit_transform(H_L)
    mu_c = H_proj[y == 1].mean(0)
    mu_h = H_proj[y == 0].mean(0)
    Sigma_w = LedoitWolf().fit(np.vstack([H_proj[y == 1] - mu_c, H_proj[y == 0] - mu_h])).covariance_
    diff = mu_c - mu_h
    try:
        J = float(diff @ np.linalg.solve(Sigma_w, diff))
    except np.linalg.LinAlgError:
        J = 0.0
    return max(J, 0.0)


def ood_transfer_test(H_train, y_train, H_ood, y_ood, model_name):
    print(f"\n  Non-degenerate OOD transfer: {model_name} (HaluEval -> NQ-open, both Pipeline C)")
    n_train = len(y_train)
    n_layers = min(H_train.shape[1], H_ood.shape[1])

    print(f"    Finding best probe layer on HaluEval ({n_train} samples) ...")
    best_layer, best_id_auroc = best_layer_probe(H_train[:, :n_layers, :], y_train)
    print(f"    Best layer: L{best_layer}  in-dist AUROC: {best_id_auroc:.4f}")

    id_J = fisher_ratio_pca(H_train[:, best_layer, :], y_train)
    id_bound = float(scipy_norm.cdf(np.sqrt(max(id_J, 0.0) / 2)))
    print(f"    In-dist Fisher J={id_J:.4f}  AUROC bound={id_bound:.4f}")

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(n_train)
    n_tr = int(0.75 * n_train)
    tr_idx, cal_idx = idx[:n_tr], idx[n_tr:]

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

    ood_layer = min(best_layer, H_ood.shape[1] - 1)
    H_L_ood = H_ood[:, ood_layer, :]
    X_ood = pca.transform(sc.transform(H_L_ood))
    scores_ood = clf.predict_proba(X_ood)[:, 1]

    try:
        ood_auroc = float(roc_auc_score(y_ood, scores_ood))
    except ValueError:
        ood_auroc = float("nan")

    ood_J = fisher_ratio_pca(H_L_ood, y_ood)
    ood_bound = float(scipy_norm.cdf(np.sqrt(max(ood_J, 0.0) / 2)))

    if alpha_star is not None:
        accepted_ood = scores_ood > thresh_star
        n_accepted_ood = int(accepted_ood.sum())
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

    print(f"    OOD AUROC:        {ood_auroc:.4f}  (in-dist: {best_id_auroc:.4f})")
    print(f"    OOD Fisher J:     {ood_J:.4f}  (in-dist: {id_J:.4f})")
    print(f"    OOD hall rate:    {emp_hall_ood}  (alpha*={alpha_star})")
    print(f"    Violation factor: {violation_factor}")

    return {
        "model": model_name, "best_layer": best_layer,
        "in_dist_n": n_train, "in_dist_hall_rate": float(1 - y_train.mean()),
        "in_dist_auroc": best_id_auroc, "in_dist_fisher_J": id_J, "in_dist_auroc_bound": id_bound,
        "ood_n": int(len(y_ood)), "ood_hall_rate": float(1 - y_ood.mean()),
        "ood_n_positive": int(y_ood.sum()), "ood_n_negative": int((y_ood == 0).sum()),
        "alpha_star": alpha_star, "in_dist_accept_rate": accept_rate_id, "in_dist_emp_hall": emp_hall_id,
        "ood_auroc": ood_auroc, "ood_fisher_J": ood_J, "ood_auroc_bound": ood_bound,
        "ood_emp_hall": emp_hall_ood, "ood_accept_rate": accept_rate_ood,
        "violation_factor": violation_factor,
        "auroc_drop": float(best_id_auroc - ood_auroc) if not np.isnan(ood_auroc) else None,
        "fisher_J_drop": float(id_J - ood_J),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("Exp 26: non-degenerate OOD transfer (HaluEval -> NQ-open, Pipeline C both arms)")
    print("=" * 70)

    halueval_path = _resolve_input_dir(HALUEVAL_CACHE_NAME)
    print(f"Loading cached HaluEval hidden states from: {halueval_path}")
    d_id = np.load(halueval_path)
    H_id, y_id = d_id["hidden_states"].astype(np.float32), d_id["labels"].astype(int)
    print(f"HaluEval (in-dist, Exp 15 cache): {H_id.shape}, hall_rate={1-y_id.mean():.3f}, "
          f"n_correct={int(y_id.sum())}")

    print(f"\nLoading {MODEL_ID} for NQ-open extraction...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                                                  device_map="auto" if DEVICE == "cuda" else None)
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()

    H_ood, y_ood, generations = extract_nq_open(model, tokenizer, N_NQ_OPEN)

    nq_open_path = OUT / "26_nq_open_qwen3b.npz"
    np.savez_compressed(nq_open_path, hidden_states=H_ood, labels=y_ood)
    with open(OUT / "26_nq_open_generations.json", "w") as f:
        json.dump(generations, f, indent=2)
    print(f"Saved: {nq_open_path}")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    result = ood_transfer_test(H_id, y_id, H_ood, y_ood, "Qwen 2.5 3B")

    out_path = OUT / "26_nondegenerate_ood_transfer.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
