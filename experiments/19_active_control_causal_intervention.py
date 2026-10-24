"""
Exp 19 -- The active-control protocol main.tex specifies but does not run
(Sec "A concrete active-control protocol"): turn the certificate's own
Fisher-LDA direction into a steering vector, test whether it causally
shifts generation toward correctness, and check the three preconditions
the paper itself says none of this has tested:
  (i)   is the Fisher direction causally sufficient, not just correlated?
  (ii)  does a dense edit preserve unrelated capability (fluency/next-token
        prediction), not just move the geometric score?
  (iii) does any effect survive the OOD collapse documented in Sec 5?

Design (GPT-2 117M, switched from GPT-2 Medium after a smoke test found
GPT-2 Medium collapses to fully degenerate repetition -- "ands!!!!..." --
on 5/5 sampled prompts under plain greedy decoding with this exact bare
"Q: ...\nA:" template, making 100% of its baseline completions
unlabelable. GPT-2 117M, confirmed non-degenerate on this template in
Paper 1's own causal-patching experiment (51.9% degenerate rate, not
100%), is used instead, with its own cached hidden states
(results/hidden_states/15_hidden_states_gpt2_117m.npz), so this remains
"this paper's own data," just the 117M cache rather than Medium's):
  1. Fisher-LDA direction w_k computed at array-layer 8 (matching Paper 1's
     independently-converging 7-method finding that L8-9 is where
     hallucination signal localizes on GPT-2 117M) via PCA(100)+Ledoit-Wolf
     on cached hidden states -- the exact methodology already used
     throughout this paper for the certificate itself (fisher_ratio_pca).
     Lifted to full-d via w_full = P^T w_k, per the corrected protocol in
     the main text.
  2. A random-direction control, same construction as Paper 1's causal
     patch script (01_ffn_causal_patch.py), for direct comparability.
  3. Steering tested on FREE GENERATION on TruthfulQA -- i.e. this tests
     whether a direction learned from the cached (real-generation-labelled)
     hidden states transfers causally to changing what the model actually
     generates on held-out prompts.
  4. Flip-to-correct rate + McNemar exact test (found vs random),
     identical methodology to Paper 1 for cross-paper comparability.
  5. NEW: capability-preservation check -- next-token log-likelihood on
     held-out HaluEval `knowledge` fields (factual background text,
     unrelated to the steered question) under each steering condition,
     to check the edit doesn't just break the model generally.

Cost: $0, CPU-only (GPT-2 117M, ~150 generations x few configs).
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path
HS_PATH = str(ROOT / "results" / "hidden_states" / "15_hidden_states_gpt2_117m.npz")
MODEL_ID = "gpt2"
ARRAY_LAYER = 8     # index into cached hidden_states (0=embeddings, i=output of block i-1)
HOOK_BLOCK_IDX = ARRAY_LAYER - 1  # transformer.h[7] output == hidden_states[:, 8, :]
N_COMPONENTS = 100
RANDOM_STATE = 42
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12
N_TEST_PROMPTS = 150
ALPHAS = [10.0, 20.0, 40.0]

OUT_DIR = ROOT / "results"


def fisher_lda_direction(H_layer, y, k=N_COMPONENTS, seed=RANDOM_STATE):
    """Identical methodology to this paper's fisher_ratio_pca: PCA(100) then
    Ledoit-Wolf-regularized LDA direction. Returns (w_k, pca) so w_k can be
    lifted back through pca.components_ to the full residual-stream dim."""
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
    """w_full = P^T w_k, per the corrected protocol in main.tex."""
    P = pca.components_  # (k, d)
    w_full = P.T @ w_k   # (d,)
    return w_full / np.linalg.norm(w_full)


def random_orthogonal_direction(reference_unit_vec, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=reference_unit_vec.shape)
    rand = rand - (rand @ reference_unit_vec) * reference_unit_vec  # orthogonalize
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


def make_steer_hook(direction_unit, alpha, device):
    direction_t = torch.tensor(direction_unit, dtype=torch.float32, device=device)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h + alpha * direction_t
            return (h,) + out[1:]
        else:
            return out + alpha * direction_t
    return hook


def generate_with_hook(prompt, model, tokenizer, device, layer_idx, direction_unit, alpha):
    handle = None
    if direction_unit is not None and alpha != 0.0:
        layer = model.transformer.h[layer_idx]
        handle = layer.register_forward_hook(make_steer_hook(direction_unit, alpha, device))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if handle is not None:
            handle.remove()
    return completion


def capability_check(model, tokenizer, device, layer_idx, direction_unit, alpha, texts):
    """Mean per-token log-likelihood on held-out factual text, under a given
    steering condition -- checks the edit doesn't globally break the model."""
    handle = None
    if direction_unit is not None and alpha != 0.0:
        layer = model.transformer.h[layer_idx]
        handle = layer.register_forward_hook(make_steer_hook(direction_unit, alpha, device))
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
    print("Loading cached GPT-2 117M hidden states (Exp 15, matched generation)...", flush=True)
    d = np.load(HS_PATH)
    H, y = d["hidden_states"], d["labels"]
    H_layer = H[:, ARRAY_LAYER, :]
    print(f"H_layer shape: {H_layer.shape}, hall_rate={1-y.mean():.3f}")

    w_k, pca = fisher_lda_direction(H_layer, y)
    w_found = lift_to_full_dim(w_k, pca)
    w_random = random_orthogonal_direction(w_found)
    print(f"Found direction norm check: {np.linalg.norm(w_found):.4f} (should be 1.0)")

    print(f"\nLoading {MODEL_ID} for generation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.eval()

    print("Loading TruthfulQA generation prompts...")
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    items = list(tqa)[:N_TEST_PROMPTS]

    print("Loading HaluEval knowledge fields for capability check...")
    halueval = load_dataset("pminervini/HaluEval", "qa")["data"]
    capability_texts = [row["knowledge"] for row in halueval.select(range(50)) if row["knowledge"]]

    print("\nGenerating baseline (unsteered) completions once, reused across all alphas...")
    baseline_labels, baseline_completions = [], []
    for i, item in enumerate(items):
        prompt = f"Q: {item['question']}\nA:"
        c_base = generate_with_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, None, 0.0)
        baseline_labels.append(label_completion(c_base, item["correct_answers"], item["incorrect_answers"]))
        baseline_completions.append(c_base)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(items)}]", flush=True)
    print(f"Baseline hallucination rate: {1 - np.mean([l for l in baseline_labels if l != -1]):.4f}")

    results_per_alpha = []
    for alpha in ALPHAS:
        print(f"\n=== alpha={alpha} ===")
        found_labels, random_labels = [], []
        found_completions, random_completions = [], []

        for i, item in enumerate(items):
            q = item["question"]
            prompt = f"Q: {q}\nA:"
            correct_answers = item["correct_answers"]
            incorrect_answers = item["incorrect_answers"]

            c_found = generate_with_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, w_found, alpha)
            c_rand = generate_with_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, w_random, alpha)

            l_found = label_completion(c_found, correct_answers, incorrect_answers)
            l_rand = label_completion(c_rand, correct_answers, incorrect_answers)

            found_labels.append(l_found); random_labels.append(l_rand)
            found_completions.append(c_found); random_completions.append(c_rand)

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(items)}]", flush=True)

        # Only compare on prompts where baseline was hallucinated (label==0) and
        # both found/random gave a valid (non -1) label -- flip-to-correct rate,
        # identical framing to Paper 1's causal patch.
        valid = [(b, f, r) for b, f, r in zip(baseline_labels, found_labels, random_labels)
                 if b == 0 and f != -1 and r != -1]
        n_valid = len(valid)
        found_flip_rate = np.mean([f == 1 for _, f, _ in valid]) if n_valid else float("nan")
        random_flip_rate = np.mean([r == 1 for _, _, r in valid]) if n_valid else float("nan")

        # McNemar exact test: found-flip vs random-flip, discordant pairs only
        b_found_only = sum(1 for _, f, r in valid if f == 1 and r == 0)
        c_random_only = sum(1 for _, f, r in valid if f == 0 and r == 1)
        if b_found_only + c_random_only > 0:
            mcnemar_p = binomtest(b_found_only, b_found_only + c_random_only, 0.5).pvalue
        else:
            mcnemar_p = 1.0

        # Capability check under each condition
        cap_base = capability_check(model, tokenizer, device, HOOK_BLOCK_IDX, None, 0.0, capability_texts)
        cap_found = capability_check(model, tokenizer, device, HOOK_BLOCK_IDX, w_found, alpha, capability_texts)
        cap_random = capability_check(model, tokenizer, device, HOOK_BLOCK_IDX, w_random, alpha, capability_texts)

        print(f"  n_valid(baseline-hallucinated)={n_valid}  found_flip={found_flip_rate:.4f}  "
              f"random_flip={random_flip_rate:.4f}  McNemar p={mcnemar_p:.4f} "
              f"(b={b_found_only}, c={c_random_only})")
        print(f"  Capability (mean token log-lik): baseline={cap_base:.4f}  "
              f"found-steered={cap_found:.4f}  random-steered={cap_random:.4f}")

        results_per_alpha.append({
            "alpha": alpha, "n_valid": n_valid,
            "found_flip_rate": found_flip_rate, "random_flip_rate": random_flip_rate,
            "mcnemar_b_found_only": b_found_only, "mcnemar_c_random_only": c_random_only,
            "mcnemar_p": float(mcnemar_p),
            "capability_baseline_ll": cap_base, "capability_found_ll": cap_found,
            "capability_random_ll": cap_random,
            "capability_drop_found": cap_base - cap_found, "capability_drop_random": cap_base - cap_random,
        })

    out = {
        "model": MODEL_ID, "array_layer": ARRAY_LAYER, "hook_block_idx": HOOK_BLOCK_IDX, "n_test_prompts": N_TEST_PROMPTS,
        "alphas": ALPHAS, "results_per_alpha": results_per_alpha,
    }
    out_path = OUT_DIR / "19_active_control_causal_intervention.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
