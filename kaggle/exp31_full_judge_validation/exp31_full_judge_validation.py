"""
GEOM-PROOF -- Exp 31: full-scale judge validation of Pipeline C labels,
resolving Priority-0 item P0.1 from the frontier-lab review
(FRONTIER_LAB_REVIEW_RESULT.md): "Construct validity of the outcome
variable... Until the central result (Table certificate-matched) is
re-run against a validated (judge-based) label set and the 0.32-0.41 gap
is shown to survive, the paper's title-level claim ('hallucination
detectability') is not established."

Exp 30 spot-checked only 100/500 Qwen-3B items. This scales that same
judge protocol to the FULL 500 items for ALL SIX models used in Table
certificate-matched (Exp 15), then recomputes J, the Gaussian bound, and
probe AUROC using the judge's verdict as ground truth instead of
Pipeline C's own SQuAD-style EM/F1 label -- the actual construct-validity
test the review asked for, not just another agreement-rate spot check.

Judge model, prompt, and decision rule are identical to Exp 30 (same
checkpoint, same prompt template, same CORRECT/HALLUCINATED parsing) so
this is a direct scale-up, not a new instrument.
"""
import json
import time
from pathlib import Path

try:
    import bitsandbytes  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device -- Mistral-7B needs a GPU.")
_a = torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")
print(f"GPU sanity check passed: {torch.cuda.get_device_name(0)}")

MODELS = ["gpt2_117m", "gpt2_medium", "qwen_0.5b", "qwen_1.5b", "qwen_3b", "qwen_7b"]


def _resolve_input(filename):
    candidates = sorted(Path("/kaggle/input").glob(f"**/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"{filename} not found under /kaggle/input")
    return candidates[0]


JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

JUDGE_PROMPT = """You are a factuality judge. You will be given a question, the correct reference answer, and a model's answer.
Decide if the model's answer is CORRECT or HALLUCINATED.
A response is CORRECT if its core factual claim matches or is consistent with the reference answer, even if phrased differently.
A response is HALLUCINATED if it asserts something factually wrong, contradicts the reference, or fails to answer.
Ignore phrasing differences and focus only on factual accuracy.

Question: {question}
Reference answer: {reference}
Model's answer: {response}

Reply with exactly one word: CORRECT or HALLUCINATED."""


def query_judge(model, tokenizer, question, reference, response):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, response=response)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    verdict = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    if "CORRECT" in verdict and "HALLUCINAT" not in verdict:
        return 1
    if "HALLUCINAT" in verdict:
        return 0
    return -1


def fisher_ratio_pca(H, y, n_components=100):
    from sklearn.covariance import LedoitWolf
    k = min(n_components, H.shape[1], H.shape[0] - 2)
    pca = PCA(n_components=k)
    H_proj = pca.fit_transform(H)
    mask_c, mask_h = y == 1, y == 0
    mu_c, mu_h = H_proj[mask_c].mean(0), H_proj[mask_h].mean(0)
    Hc_c, Hc_h = H_proj[mask_c] - mu_c, H_proj[mask_h] - mu_h
    lw = LedoitWolf().fit(np.vstack([Hc_c, Hc_h]))
    delta = mu_c - mu_h
    try:
        w = np.linalg.solve(lw.covariance_, delta)
        J = float(delta @ w)
    except np.linalg.LinAlgError:
        J = float(delta @ np.linalg.lstsq(lw.covariance_, delta, rcond=None)[0])
    return max(0.0, J)


def auroc_bound(J):
    return float(norm.cdf(np.sqrt(max(0.0, J) / 2)))


def cv_probe_auroc(H, y, seed=42):
    if len(set(y)) < 2:
        return float("nan")
    k = min(100, H.shape[1], H.shape[0] - 2)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(H, y):
        pca = PCA(n_components=k).fit(H[tr])
        Htr, Hte = pca.transform(H[tr]), pca.transform(H[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Htr, y[tr])
        oof[te] = clf.predict_proba(Hte)[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    print(f"Loading judge {JUDGE_MODEL} (bitsandbytes 4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, device_map="auto", quantization_config=bnb_config)
    model.eval()

    all_results = {}
    t_start = time.time()

    for model_name in MODELS:
        print(f"\n=== {model_name} ===", flush=True)
        gen_path = _resolve_input(f"15_generations_{model_name}.json")
        hs_path = _resolve_input(f"15_hidden_states_{model_name}.npz")

        with open(gen_path) as f:
            data = json.load(f)
        hs = np.load(hs_path)
        H_all, pipeline_c_labels_npz = hs["hidden_states"], hs["labels"]

        assert len(data) == len(pipeline_c_labels_npz), (
            f"{model_name}: generation count {len(data)} != hidden-state label count "
            f"{len(pipeline_c_labels_npz)} -- alignment cannot be assumed, ABORTING this model."
        )
        mismatch = sum(1 for d, l in zip(data, pipeline_c_labels_npz) if d["label"] != int(l))
        print(f"  Alignment check: {mismatch}/{len(data)} label mismatches between "
              f"generations JSON and hidden-state NPZ (expect 0)")
        if mismatch > 0:
            print(f"  WARNING: {model_name} generation/hidden-state ordering may not match index-for-index. "
                  f"Judge labels will still be recorded per-question but the recomputed-table step for "
                  f"this model should be treated as unreliable until this is resolved.")

        judge_verdicts, ambiguous = [], 0
        per_sample = []
        for i, d in enumerate(data):
            v = query_judge(model, tokenizer, d["question"], d["right_answer"], d["generated"])
            per_sample.append({"question": d["question"], "right_answer": d["right_answer"],
                                "generated": d["generated"], "pipeline_c_label": d["label"], "judge_verdict": v})
            judge_verdicts.append(v)
            if v == -1:
                ambiguous += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(data)}]  elapsed={elapsed:.0f}s", flush=True)

        judge_verdicts = np.array(judge_verdicts)
        pipeline_c_labels = np.array([d["label"] for d in data])
        valid = judge_verdicts != -1
        agreement = float((pipeline_c_labels[valid] == judge_verdicts[valid]).mean())
        kappa = (float(cohen_kappa_score(pipeline_c_labels[valid], judge_verdicts[valid]))
                 if len(set(judge_verdicts[valid])) > 1 and len(set(pipeline_c_labels[valid])) > 1 else float("nan"))
        cm = confusion_matrix(pipeline_c_labels[valid], judge_verdicts[valid]).tolist()

        print(f"  n_scored={valid.sum()} (ambiguous={ambiguous})  agreement={agreement:.4f}  kappa={kappa:.4f}")

        # --- Recompute the certificate-matched table under BOTH label sets, best-J layer ---
        recompute = {}
        for label_name, y_full in [("pipeline_c_emf1", pipeline_c_labels), ("judge_verdict", judge_verdicts)]:
            y = y_full.copy()
            keep = y != -1
            if len(set(y[keep])) < 2:
                recompute[label_name] = {"error": "degenerate label set (all one class after dropping ambiguous)"}
                continue
            n_layers = H_all.shape[1]
            best_J, best_layer = -1.0, -1
            for L in range(n_layers):
                J_L = fisher_ratio_pca(H_all[keep, L, :], y[keep])
                if J_L > best_J:
                    best_J, best_layer = J_L, L
            bound = auroc_bound(best_J)
            probe_auroc = cv_probe_auroc(H_all[keep, best_layer, :], y[keep])
            recompute[label_name] = {
                "best_layer": best_layer, "J": best_J, "bound": bound,
                "probe_auroc": probe_auroc, "bound_error": abs(bound - probe_auroc),
                "n": int(keep.sum()), "n_positive": int(y[keep].sum()),
            }
            print(f"  [{label_name}] best_layer={best_layer} J={best_J:.4f} bound={bound:.4f} "
                  f"probe_auroc={probe_auroc:.4f} bound_error={abs(bound-probe_auroc):.4f}")

        all_results[model_name] = {
            "judge_model": JUDGE_MODEL, "n_total": len(data), "n_scored": int(valid.sum()),
            "n_ambiguous": ambiguous, "agreement_with_pipeline_c": agreement, "cohen_kappa": kappa,
            "confusion_matrix_rows_pipelineC_cols_judge": cm,
            "alignment_mismatches": mismatch,
            "recompute_under_each_label_set": recompute,
            "per_sample": per_sample,
        }
        with open(OUT / "31_full_judge_validation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved incremental progress through {model_name} to {OUT}/31_full_judge_validation.json")

    print(f"\nTotal elapsed: {time.time()-t_start:.0f}s")
    print("=== SUMMARY: does the 0.32-0.41 bound-error gap survive under judge labels? ===")
    for m in MODELS:
        r = all_results[m]["recompute_under_each_label_set"]
        emf1 = r.get("pipeline_c_emf1", {})
        judge = r.get("judge_verdict", {})
        print(f"  {m}: EM/F1 bound_error={emf1.get('bound_error','NA')}  "
              f"judge bound_error={judge.get('bound_error','NA')}")


if __name__ == "__main__":
    main()
