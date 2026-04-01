"""
GEOM-PROOF — Kaggle T4 Extraction Notebook
===========================================

Single-session GPU extraction for all four heavy jobs.
Run this on Kaggle T4 (free tier, 9h session limit).
Estimated total time: ~2.5 hours.

After this notebook completes, download from /kaggle/working/outputs/:
  - 00_halueval_qwen3b.npz         → results/hidden_states/
  - 00_halueval_gpt2med.npz        → results/hidden_states/
  - 04_mamba_hidden_states.npz     → results/hidden_states/
  - 04_mamba_transfer.json         → results/logs/
  - 07_judge_labels.npy            → results/logs/
  - 11_qwen05_hidden_states.npz    → results/hidden_states/
  - 11_qwen15_hidden_states.npz    → results/hidden_states/

Then run locally (CPU):
  python experiments/01_fisher_analysis.py
  python experiments/02_scale_curve.py
  ... (see run_all.sh)

HOW TO USE ON KAGGLE:
  1. Create new notebook → set accelerator to GPU T4 x1
  2. Add this file as a script or paste into a code cell
  3. Run all cells
  4. Download outputs/ directory

INSTALL (first cell):
  !pip install -q transformers datasets rouge-score mamba-ssm accelerate
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0: Setup
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, json, gc
import numpy as np
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
      if DEVICE == "cuda" else "")

OUT = Path("/kaggle/working/outputs")
OUT.mkdir(parents=True, exist_ok=True)

ROUGE_THRESHOLD = 0.4
N_TRUTHFULQA = 400
N_HALUEVAL = 1000   # per class → 2000 total

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_lib


def rouge_label(response: str, correct_answers: list[str]) -> int:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return int(max(
        scorer.score(ref, response)["rougeL"].fmeasure
        for ref in correct_answers
    ) >= ROUGE_THRESHOLD)


def load_truthfulqa(n: int) -> list[dict]:
    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for row in ds:
        if len(out) >= n:
            break
        correct = [a for a in row["correct_answers"] if a.strip()]
        if correct:
            out.append({"question": row["question"], "correct_answers": correct})
    return out[:n]


def load_halueval(n: int) -> list[dict]:
    """Load HaluEval QA pairs (right_answer, hallucinated_answer)."""
    ds = None
    for attempt in [
        lambda: load_dataset("HaluEval/halueval", "qa", split="data"),
        lambda: load_dataset("pminervini/HaluEval", "qa"),
        lambda: load_dataset("pminervini/HaluEval", "qa_samples", split="data"),
        lambda: load_dataset("pminervini/HaluEval", "qa_samples", split="train"),
        lambda: load_dataset("pminervini/HaluEval", "qa_samples"),
    ]:
        try:
            ds = attempt()
            break
        except Exception as e:
            print(f"    fallback: {e}")
    if ds is None:
        raise RuntimeError("Could not load HaluEval from any known path.")

    # Unwrap DatasetDict (returned when no split= is specified)
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        split_name = list(ds.keys())[0]
        print(f"    DatasetDict splits: {list(ds.keys())} — using '{split_name}'")
        ds = ds[split_name]

    # Debug: show actual column names so we can fix field mapping if needed
    first = ds[0]
    print(f"    HaluEval columns: {list(first.keys())}")

    # Try all known field name variants
    QUESTION_KEYS  = ["question", "input", "query"]
    RIGHT_KEYS     = ["right_answer", "answer", "correct_answer", "gold_answer"]
    HALL_KEYS      = ["hallucinated_answer", "hallucination", "wrong_answer",
                      "model_answer", "output", "negative"]

    def pick(row, keys):
        for k in keys:
            v = row.get(k, "")
            if v and str(v).strip():
                return str(v).strip()
        return ""

    out = []
    for row in ds:
        if len(out) >= n:
            break
        q     = pick(row, QUESTION_KEYS)
        right = pick(row, RIGHT_KEYS)
        hall  = pick(row, HALL_KEYS)
        if q and right and hall and right != hall:
            out.append({"question": q, "right_answer": right, "hallucinated_answer": hall})

    print(f"    Extracted {len(out)} valid pairs from {ds.num_rows} rows")
    return out[:n]


def extract_last_token_hs(model, tokenizer, text: str, max_len: int = 256) -> np.ndarray | None:
    """Forward pass → (n_layers+1, d) hidden states at last token."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_len, padding=False)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    if out.hidden_states is None:
        return None
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])


def free(model):
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"  VRAM freed. Current: "
          f"{torch.cuda.memory_allocated()/1e9:.2f}GB" if DEVICE == "cuda" else "")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Job A — HaluEval extraction (Qwen 2.5 3B + GPT-2 Medium 345M)
# Estimated time: ~50 min (Qwen 3B ~40min, GPT-2 Med ~10min)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("JOB A: HaluEval extraction")
print("="*60)

try:
  halueval_samples = load_halueval(N_HALUEVAL)
  print(f"  Loaded {len(halueval_samples)} HaluEval QA pairs")
except Exception as e:
  print(f"  ERROR loading HaluEval: {e}")
  print("  Skipping Job A — Jobs B/C/D will still run.")
  halueval_samples = []

halueval_models = {
    "qwen3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "out": OUT / "00_halueval_qwen3b.npz",
    },
    "gpt2med": {
        "model_id": "openai-community/gpt2-medium",
        "out": OUT / "00_halueval_gpt2med.npz",
    },
}

for key, cfg in halueval_models.items():
    if not halueval_samples:
        print(f"  {key}: skipped (no HaluEval data).")
        continue
    if cfg["out"].exists():
        print(f"  {key}: already done, skipping.")
        continue
    print(f"\n  Loading {cfg['model_id']} ...")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  {model.config.num_hidden_layers} layers, d={model.config.hidden_size}")

    all_hs, all_labels, skipped = [], [], 0
    for i, s in enumerate(halueval_samples):
        if i % 200 == 0:
            print(f"    [{i}/{len(halueval_samples)}]")
        # Correct answer → label=1
        hs_c = extract_last_token_hs(model, tok, f"Q: {s['question']}\nA: {s['right_answer']}")
        # Hallucinated answer → label=0
        hs_h = extract_last_token_hs(model, tok, f"Q: {s['question']}\nA: {s['hallucinated_answer']}")
        if hs_c is None or hs_h is None:
            skipped += 1; continue
        all_hs.extend([hs_c, hs_h])
        all_labels.extend([1, 0])

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    np.savez_compressed(cfg["out"], hidden_states=H, labels=y)
    print(f"  Saved: {cfg['out']}  shape={H.shape}  hall_rate={1-y.mean():.3f}  skipped={skipped}")
    free(model)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Job B — LLM-as-Judge relabeling (replaces Ollama, uses transformers)
# Estimated time: ~25 min (700 judgments from Qwen 2.5 3B)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("JOB B: LLM-as-Judge relabeling (Exp 07 — GPU version)")
print("="*60)

JUDGE_OUT = OUT / "07_judge_labels.npy"
JUDGE_LOG = OUT / "07_judge_log.json"

if JUDGE_OUT.exists():
    print("  Already done, skipping.")
else:
    # Load HaRP responses — need to supply them as a text file or regenerate
    # We regenerate responses directly for the 700 TruthfulQA questions

    tqa = load_truthfulqa(700)  # full TruthfulQA used in HaRP
    print(f"  Loaded {len(tqa)} TruthfulQA questions")

    print("  Loading Qwen 2.5 3B for response generation + judgment ...")
    tok_judge = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok_judge.pad_token is None:
        tok_judge.pad_token = tok_judge.eos_token

    model_judge = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model_judge.eval()

    JUDGE_PROMPT = """You are evaluating whether an AI response is factually correct.

Question: {question}
AI Response: {response}

Is this response CORRECT or HALLUCINATED?
Answer with exactly one word: CORRECT or HALLUCINATED"""

    def generate_response(model, tok, question: str, max_new: int = 80) -> str:
        prompt = f"Q: {question}\nA:"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def judge_response(model, tok, question: str, response: str) -> int:
        prompt = JUDGE_PROMPT.format(question=question, response=response)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        verdict = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip().upper()
        return 1 if "CORRECT" in verdict else 0

    judge_labels, rouge_labels, log = [], [], []
    for i, sample in enumerate(tqa):
        if i % 100 == 0:
            print(f"    [{i}/{len(tqa)}]")
        response = generate_response(model_judge, tok_judge, sample["question"])
        rouge_lbl = rouge_label(response, sample["correct_answers"])
        judge_lbl = judge_response(model_judge, tok_judge, sample["question"], response)
        rouge_labels.append(rouge_lbl)
        judge_labels.append(judge_lbl)
        log.append({"q": sample["question"], "response": response,
                    "rouge": rouge_lbl, "judge": judge_lbl})

    judge_arr = np.array(judge_labels, dtype=np.int32)
    rouge_arr = np.array(rouge_labels, dtype=np.int32)
    np.save(JUDGE_OUT, judge_arr)

    # Cohen's kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(rouge_arr, judge_arr))
    print(f"  Cohen's κ (ROUGE vs Judge): {kappa:.4f}")
    print(f"  ROUGE hall_rate={1-rouge_arr.mean():.3f}, Judge hall_rate={1-judge_arr.mean():.3f}")

    with open(JUDGE_LOG, "w") as f:
        json.dump({"kappa": kappa, "n": len(judge_labels),
                   "rouge_hall_rate": float(1-rouge_arr.mean()),
                   "judge_hall_rate": float(1-judge_arr.mean()),
                   "log": log[:10]}, f, indent=2)

    print(f"  Saved: {JUDGE_OUT}")
    free(model_judge)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Job C — Qwen 2.5 family extraction for controlled scale curve (Exp 11)
# Estimated time: ~25 min (0.5B ~5min, 1.5B ~10min, 7B ~30min if VRAM allows)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("JOB C: Qwen 2.5 family — TruthfulQA extraction (Exp 11)")
print("="*60)

tqa_400 = load_truthfulqa(N_TRUTHFULQA)
print(f"  Loaded {len(tqa_400)} TruthfulQA questions")

qwen_family = {
    "Qwen2.5-0.5B": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "out": OUT / "11_qwen05_hidden_states.npz",
        "max_new_tokens": 80,
    },
    "Qwen2.5-1.5B": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "out": OUT / "11_qwen15_hidden_states.npz",
        "max_new_tokens": 80,
    },
    # 7B — only include if VRAM allows (T4 has 15GB; Qwen2.5-7B needs ~14GB fp16)
    "Qwen2.5-7B": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "out": OUT / "11_qwen7b_hidden_states.npz",
        "max_new_tokens": 80,
        "optional": True,   # skip if OOM
    },
}

for name, cfg in qwen_family.items():
    if cfg["out"].exists():
        print(f"  {name}: already done, skipping.")
        continue

    try:
        print(f"\n  Loading {name} ({cfg['model_id']}) ...")
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        d = model.config.hidden_size
        print(f"  {n_layers} layers, d={d}")

        all_hs, all_labels, skipped = [], [], 0
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

        for i, sample in enumerate(tqa_400):
            if i % 100 == 0:
                print(f"    [{i}/{len(tqa_400)}]")
            prompt = f"Q: {sample['question']}\nA:"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                gen_ids = model.generate(**inputs,
                                         max_new_tokens=cfg["max_new_tokens"],
                                         do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            response = tok.decode(gen_ids[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()

            # ROUGE-L label
            best_score = max(
                scorer.score(ref, response)["rougeL"].fmeasure
                for ref in sample["correct_answers"]
            )
            label = int(best_score >= ROUGE_THRESHOLD)

            # Hidden states at last prompt token
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            if out.hidden_states is None:
                skipped += 1; continue

            hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
            all_hs.append(hs)
            all_labels.append(label)

        H = np.stack(all_hs).astype(np.float32)
        y = np.array(all_labels, dtype=np.int32)
        np.savez_compressed(cfg["out"], hidden_states=H, labels=y)
        print(f"  Saved: {cfg['out']}  shape={H.shape}  hall_rate={1-y.mean():.3f}")
        free(model)

    except torch.cuda.OutOfMemoryError:
        if cfg.get("optional"):
            print(f"  {name}: OOM — skipping (optional, run separately if needed).")
            free(model) if "model" in dir() else None
        else:
            raise


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: Job D — Mamba-2 architecture transfer (Exp 04)
# Estimated time: ~40 min
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("JOB D: Mamba-2 130M architecture transfer (Exp 04)")
print("="*60)

MAMBA_HS_OUT = OUT / "04_mamba_hidden_states.npz"
MAMBA_JSON_OUT = OUT / "04_mamba_transfer.json"

if MAMBA_HS_OUT.exists() and MAMBA_JSON_OUT.exists():
    print("  Already done, skipping.")
else:
    try:
        from transformers import Mamba2ForCausalLM
        MambaClass = Mamba2ForCausalLM
        print("  Using Mamba2ForCausalLM")
    except ImportError:
        from transformers import MambaForCausalLM
        MambaClass = MambaForCausalLM
        print("  Falling back to MambaForCausalLM")

    MODEL_ID = "state-spaces/mamba2-130m"
    print(f"  Loading {MODEL_ID} ...")
    tok_mamba = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok_mamba.pad_token is None:
        tok_mamba.pad_token = tok_mamba.eos_token

    model_mamba = MambaClass.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model_mamba.eval()
    n_layers = model_mamba.config.num_hidden_layers
    d = model_mamba.config.hidden_size
    print(f"  {n_layers} layers, d={d}")

    tqa_mamba = load_truthfulqa(N_TRUTHFULQA)
    scorer_m = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

    all_hs, all_labels, responses_out, skipped = [], [], [], 0

    for i, sample in enumerate(tqa_mamba):
        if i % 50 == 0:
            print(f"    [{i}/{len(tqa_mamba)}]")
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tok_mamba(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            gen_ids = model_mamba.generate(**inputs, max_new_tokens=128, do_sample=False,
                                            pad_token_id=tok_mamba.eos_token_id)
        response = tok_mamba.decode(gen_ids[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
        label = rouge_label(response, sample["correct_answers"])

        # Hidden states
        with torch.no_grad():
            out = model_mamba(**inputs, output_hidden_states=True)
        if out.hidden_states is None:
            skipped += 1; continue

        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
        all_hs.append(hs)
        all_labels.append(label)
        responses_out.append(response)

    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"  Shape: {H.shape}, hall_rate={1-y.mean():.3f}, skipped={skipped}")
    np.savez_compressed(MAMBA_HS_OUT, hidden_states=H, labels=y)
    print(f"  Saved: {MAMBA_HS_OUT}")

    # Fisher analysis inline (avoid importing local src on Kaggle)
    from sklearn.decomposition import PCA
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm as scipy_norm

    def fisher_ratio_pca(H_L, y, k=50):
        k = min(k, H_L.shape[1], H_L.shape[0] - 2)
        H_proj = PCA(n_components=k, random_state=42).fit_transform(H_L)
        mu_c = H_proj[y == 1].mean(0)
        mu_h = H_proj[y == 0].mean(0)
        H_c = H_proj[y == 1] - mu_c
        H_h = H_proj[y == 0] - mu_h
        H_pool = np.vstack([H_c, H_h])
        Sigma_w = LedoitWolf().fit(H_pool).covariance_
        diff = mu_c - mu_h
        try:
            Sigma_inv = np.linalg.solve(Sigma_w, diff)
            J = float(diff @ Sigma_inv)
        except np.linalg.LinAlgError:
            J = 0.0
        return max(J, 0.0)

    J_per_layer, bound_per_layer = [], []
    for L in range(H.shape[1]):
        J = fisher_ratio_pca(H[:, L, :], y)
        J_per_layer.append(J)
        bound_per_layer.append(float(scipy_norm.cdf(np.sqrt(J) / 2)))

    best_J_layer = int(np.argmax(J_per_layer))

    # OOF probe
    probe_aurocs = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for L in range(H.shape[1]):
        H_L = H[:, L, :]
        oof = np.zeros(len(y))
        for tr, te in skf.split(H_L, y):
            sc = StandardScaler()
            lr = LogisticRegression(max_iter=500, C=1.0)
            lr.fit(sc.fit_transform(H_L[tr]), y[tr])
            oof[te] = lr.predict_proba(sc.transform(H_L[te]))[:, 1]
        probe_aurocs.append(float(roc_auc_score(y, oof)))
        print(f"  L{L:3d}: J={J_per_layer[L]:.4f}  bound={bound_per_layer[L]:.4f}  probe={probe_aurocs[-1]:.4f}")

    best_probe_layer = int(np.argmax(probe_aurocs))

    mamba_result = {
        "experiment": "04_mamba_transfer",
        "model": MODEL_ID,
        "n_params": 130e6,
        "n_samples": int(H.shape[0]),
        "n_layers": int(H.shape[1]),
        "hidden_size": int(d),
        "hall_rate": float(1 - y.mean()),
        "fisher_best_layer": best_J_layer,
        "fisher_best_J": float(J_per_layer[best_J_layer]),
        "fisher_best_auroc_bound": float(bound_per_layer[best_J_layer]),
        "fisher_depth_fraction": float(best_J_layer / (H.shape[1] - 1)),
        "probe_best_layer": best_probe_layer,
        "probe_best_auroc": float(probe_aurocs[best_probe_layer]),
        "probe_depth_fraction": float(best_probe_layer / (H.shape[1] - 1)),
        "bound_error": float(abs(bound_per_layer[best_J_layer] - probe_aurocs[best_probe_layer])),
        "certificate_validated": float(abs(bound_per_layer[best_J_layer] - probe_aurocs[best_probe_layer])) < 0.05,
        "J_per_layer": [float(j) for j in J_per_layer],
        "auroc_bound_per_layer": [float(b) for b in bound_per_layer],
        "probe_auroc_per_layer": probe_aurocs,
    }

    with open(MAMBA_JSON_OUT, "w") as f:
        json.dump(mamba_result, f, indent=2)
    print(f"  Saved: {MAMBA_JSON_OUT}")
    print(f"\n  MAMBA RESULT:")
    print(f"    Fisher bound: {mamba_result['fisher_best_auroc_bound']:.4f}")
    print(f"    Probe AUROC:  {mamba_result['probe_best_auroc']:.4f}")
    print(f"    Error:        {mamba_result['bound_error']:.4f} "
          f"({'PASS ✓' if mamba_result['certificate_validated'] else 'FAIL ✗'})")
    print(f"    Depth fraction: {mamba_result['probe_depth_fraction']:.3f}")

    free(model_mamba)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: Summary + Download Instructions
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ALL JOBS COMPLETE")
print("="*60)
print("\nFiles saved to /kaggle/working/outputs/:")
for f in sorted(OUT.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:45s}  {size_mb:.1f} MB")

print("""
NEXT STEPS:
1. Download all files from outputs/ (Kaggle sidebar → Output → Download)
2. Copy to local paths:
   *.npz  →  geom-proof/results/hidden_states/
   *.npy  →  geom-proof/results/logs/
   *.json →  geom-proof/results/logs/
3. Run locally:
   python experiments/01_fisher_analysis.py
   python experiments/02_scale_curve.py
   python experiments/03_certificate_validation.py
   python experiments/05_depth_fraction.py
   python experiments/06_boundary_conditions.py
   python experiments/08_ot_certificate.py
   python experiments/09_spectral_phase_transition.py
   python experiments/10_conformal_coverage.py
   python experiments/11_qwen_scale_curve.py
   streamlit run app.py
""")
