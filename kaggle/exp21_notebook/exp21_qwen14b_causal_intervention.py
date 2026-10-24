"""
GEOM-PROOF -- Kaggle Exp 21: Causal Intervention on Qwen2.5-14B-Instruct-AWQ
=============================================================================

Extends Exp 19's causal-intervention test (GPT-2 117M) to the scale-extension
model from Exp 20 (Qwen2.5-14B-Instruct-AWQ). Same protocol:
  1. Fisher-LDA direction w_k at the best-J layer (L30, from Exp 20's cached
     HaluEval hidden states, results/hidden_states/20_halueval_qwen14b.npz),
     PCA(100)+Ledoit-Wolf, lifted to full-d via w_full = P^T w_k.
  2. A random-direction control, orthogonalized against w_full.
  3. Steering added unconditionally to the residual stream (a forward hook
     on model.model.layers[HOOK_BLOCK_IDX]) at generation time on 150
     held-out TruthfulQA prompts, using Qwen's proper chat template (not a
     bare prompt -- this project's own Paper 1 finding is that bare
     templates are an uncontrolled OOD confound for instruction-tuned
     models), at three step sizes.
  4. Flip-to-correct rate + McNemar exact test (found vs random), identical
     methodology to Exp 19 / Paper 1's causal patch, for direct
     cross-scale comparability (117M vs 14.8B).
  5. Capability-preservation check: mean next-token log-likelihood on
     held-out HaluEval `knowledge` fields, under each steering condition.

This directly tests whether Exp 19's causal null (no significant
found-vs-random specificity) holds at 14B scale too, or whether it was a
small-model artifact.

Outputs saved to /kaggle/working/:
  21_qwen14b_causal_intervention.json
"""

import os, sys, json, gc, time, subprocess

try:
    import gptqmodel  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gptqmodel"], check=True)

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
HS_PATH = "/kaggle/input/qwen14b-halueval-hidden-states/20_halueval_qwen14b.npz"
# Fallback if not attached as a Kaggle dataset input -- re-extract inline instead.
HS_PATH_LOCAL = OUT / "20_halueval_qwen14b.npz"

ARRAY_LAYER = 30      # best-J layer index into cached hidden_states (Exp 20)
HOOK_BLOCK_IDX = ARRAY_LAYER - 1  # model.model.layers[29] output == hidden_states[:, 30, :]
N_COMPONENTS = 100
RANDOM_STATE = 42
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12
N_TEST_PROMPTS = 150
ALPHAS = [10.0, 20.0, 40.0]
N_SAMPLES_FOR_DIRECTION = 500  # matches Exp 20's HaluEval extraction size


def load_halueval(n):
    ds = load_dataset("pminervini/HaluEval", "qa")["data"]
    return list(ds.select(range(min(n, len(ds)))))


def get_hidden_states_for_direction(model, tokenizer, device):
    """Reuse Exp 20's cached extraction if attached as a dataset input;
    otherwise re-extract inline (same protocol as Exp 20 Step 1)."""
    if Path(HS_PATH).exists():
        d = np.load(HS_PATH)
        return d["hidden_states"], d["labels"]
    if HS_PATH_LOCAL.exists():
        d = np.load(HS_PATH_LOCAL)
        return d["hidden_states"], d["labels"]

    print("Cached hidden states not found as input -- re-extracting inline (Exp 20 Step 1 protocol)...")
    samples = load_halueval(N_SAMPLES_FOR_DIRECTION)
    all_hs, all_labels = [], []
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  [{i}/{len(samples)}]")
        for text, label in [
            (f"Q: {sample['question']}\nA: {sample['right_answer']}", 1),
            (f"Q: {sample['question']}\nA: {sample['hallucinated_answer']}", 0),
        ]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
            all_hs.append(hs)
            all_labels.append(label)
    H = np.stack(all_hs).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    np.savez_compressed(HS_PATH_LOCAL, hidden_states=H, labels=y)
    return H, y


def fisher_lda_direction(H_layer, y, k=N_COMPONENTS, seed=RANDOM_STATE):
    H = H_layer.astype(np.float64)
    pca = PCA(n_components=k, random_state=seed)
    H_proj = pca.fit_transform(H)
    mu_c = H_proj[y == 1].mean(axis=0)
    mu_h = H_proj[y == 0].mean(axis=0)
    Sigma_w = LedoitWolf().fit(
        np.vstack([H_proj[y == 1] - mu_c, H_proj[y == 0] - mu_h])
    ).covariance_
    diff = mu_c - mu_h
    try:
        w_k = np.linalg.solve(Sigma_w, diff)
    except np.linalg.LinAlgError:
        w_k = np.linalg.lstsq(Sigma_w, diff, rcond=None)[0]
    return w_k, pca


def lift_to_full_dim(w_k, pca):
    P = pca.components_
    w_full = P.T @ w_k
    return w_full / np.linalg.norm(w_full)


def random_orthogonal_direction(reference_unit_vec, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=reference_unit_vec.shape)
    rand = rand - (rand @ reference_unit_vec) * reference_unit_vec
    return rand / np.linalg.norm(rand)


def _word_overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def label_completion(completion, correct_answers, incorrect_answers):
    best_c = max((_word_overlap(completion, a) for a in correct_answers), default=0.0)
    best_i = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)
    if best_c > LABEL_THRESHOLD or best_i > LABEL_THRESHOLD:
        return 1 if best_c >= best_i else 0
    return -1


def make_steer_hook(direction_unit, alpha, device, dtype):
    direction_t = torch.tensor(direction_unit, dtype=dtype, device=device)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h + alpha * direction_t
            return (h,) + out[1:]
        return out + alpha * direction_t
    return hook


def build_chat_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_with_hook(prompt, model, tokenizer, device, layer_idx, direction_unit, alpha):
    handle = None
    if direction_unit is not None and alpha != 0.0:
        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(
            make_steer_hook(direction_unit, alpha, device, model.dtype))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if handle is not None:
            handle.remove()
    return completion


def capability_check(model, tokenizer, device, layer_idx, direction_unit, alpha, texts):
    handle = None
    if direction_unit is not None and alpha != 0.0:
        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(
            make_steer_hook(direction_unit, alpha, device, model.dtype))
    lls = []
    try:
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            lls.append(-out.loss.item())
    finally:
        if handle is not None:
            handle.remove()
    return float(np.mean(lls)) if lls else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
    )
    model.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\nComputing Fisher-LDA direction from cached/re-extracted HaluEval hidden states...")
    H, y = get_hidden_states_for_direction(model, tokenizer, device)
    H_layer = H[:, ARRAY_LAYER, :]
    print(f"H_layer shape: {H_layer.shape}, hall_rate={1-y.mean():.3f}")
    w_k, pca = fisher_lda_direction(H_layer, y)
    w_found = lift_to_full_dim(w_k, pca)
    w_random = random_orthogonal_direction(w_found)
    print(f"Found direction norm check: {np.linalg.norm(w_found):.4f}")

    print("\nLoading TruthfulQA generation prompts...")
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    items = list(tqa)[:N_TEST_PROMPTS]

    print("Loading HaluEval knowledge fields for capability check...")
    halueval = load_dataset("pminervini/HaluEval", "qa")["data"]
    capability_texts = [row["knowledge"] for row in halueval.select(range(50)) if row["knowledge"]]

    print("\nGenerating baseline (unsteered) completions...")
    baseline_labels = []
    for i, item in enumerate(items):
        prompt = build_chat_prompt(tokenizer, item["question"])
        c_base = generate_with_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, None, 0.0)
        baseline_labels.append(label_completion(c_base, item["correct_answers"], item["incorrect_answers"]))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(items)}]", flush=True)
    valid_base = [l for l in baseline_labels if l != -1]
    print(f"Baseline hallucination rate: {1 - np.mean(valid_base):.4f} (n_valid={len(valid_base)})")

    results_per_alpha = []
    for alpha in ALPHAS:
        print(f"\n=== alpha={alpha} ===")
        found_labels, random_labels = [], []
        for i, item in enumerate(items):
            prompt = build_chat_prompt(tokenizer, item["question"])
            c_found = generate_with_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, w_found, alpha)
            c_rand = generate_with_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, w_random, alpha)
            found_labels.append(label_completion(c_found, item["correct_answers"], item["incorrect_answers"]))
            random_labels.append(label_completion(c_rand, item["correct_answers"], item["incorrect_answers"]))
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(items)}]", flush=True)

        valid = [(b, f, r) for b, f, r in zip(baseline_labels, found_labels, random_labels)
                 if b == 0 and f != -1 and r != -1]
        n_valid = len(valid)
        found_flip_rate = float(np.mean([f == 1 for _, f, _ in valid])) if n_valid else float("nan")
        random_flip_rate = float(np.mean([r == 1 for _, _, r in valid])) if n_valid else float("nan")
        b_found_only = sum(1 for _, f, r in valid if f == 1 and r == 0)
        c_random_only = sum(1 for _, f, r in valid if f == 0 and r == 1)
        mcnemar_p = binomtest(b_found_only, b_found_only + c_random_only, 0.5).pvalue if (b_found_only + c_random_only) > 0 else 1.0

        cap_base = capability_check(model, tokenizer, device, HOOK_BLOCK_IDX, None, 0.0, capability_texts)
        cap_found = capability_check(model, tokenizer, device, HOOK_BLOCK_IDX, w_found, alpha, capability_texts)
        cap_random = capability_check(model, tokenizer, device, HOOK_BLOCK_IDX, w_random, alpha, capability_texts)

        print(f"  n_valid={n_valid}  found_flip={found_flip_rate:.4f}  random_flip={random_flip_rate:.4f}  "
              f"McNemar p={mcnemar_p:.4f} (b={b_found_only}, c={c_random_only})")
        print(f"  Capability: baseline={cap_base:.4f}  found={cap_found:.4f}  random={cap_random:.4f}")

        results_per_alpha.append({
            "alpha": alpha, "n_valid": n_valid,
            "found_flip_rate": found_flip_rate, "random_flip_rate": random_flip_rate,
            "mcnemar_b_found_only": b_found_only, "mcnemar_c_random_only": c_random_only,
            "mcnemar_p": float(mcnemar_p),
            "capability_baseline_ll": cap_base, "capability_found_ll": cap_found, "capability_random_ll": cap_random,
            "capability_drop_found": cap_base - cap_found, "capability_drop_random": cap_base - cap_random,
        })

    out = {
        "model": MODEL_ID, "array_layer": ARRAY_LAYER, "hook_block_idx": HOOK_BLOCK_IDX,
        "n_test_prompts": N_TEST_PROMPTS, "alphas": ALPHAS,
        "baseline_hall_rate": float(1 - np.mean(valid_base)),
        "results_per_alpha": results_per_alpha,
    }
    out_path = OUT / "21_qwen14b_causal_intervention.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
