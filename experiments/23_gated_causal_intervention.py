"""
Exp 23 -- closes the one piece of the active-control protocol Exp 19/21
explicitly flag as untested: the FULL, GATED protocol (main.tex, "A
concrete active-control protocol" + the Exp 19 correction note), not the
simplified unconditional version already run.

main.tex's spec: "at generation time, project the current hidden state
through P and onto w_k. If the resulting score falls on the hallucination
side of a threshold tau_s, calibrated on that score via the same split
conformal procedure, inject an additive correction... before the residual
stream continues." Exp 19/21 applied the correction unconditionally at
every generation step instead, disclosed explicitly as a simplification.

This script implements the actual gate:
  1. Split GPT-2 117M's cached HaluEval hidden states (Exp 15) into a
     DIRECTION-FIT half and a DISJOINT CALIBRATION half -- calibrating
     tau_s on the same samples used to fit w_k would be exactly the
     per-fold-reuse leakage pattern the companion leakage-audit paper
     documents, so this script deliberately avoids it.
  2. tau_s = the midpoint between the calibration split's correct-class
     and hallucinated-class mean Fisher-LDA projection scores s(h) --
     the Bayes-optimal decision boundary under the same Gaussian
     equal-variance assumption Prop. 1 already makes, so this is not an
     arbitrary threshold choice.
  3. At each generation step (each new token), the steering hook computes
     s(h) from the CURRENT (pre-edit) hidden state at the target layer,
     and injects the additive correction only if s(h) < tau_s ("the
     hallucination side"). If s(h) >= tau_s, the hook passes the
     activation through unmodified -- this is the gate.
  4. Identical found-vs-random, flip-rate, McNemar, and capability-check
     methodology to Exp 19, for direct comparability: the only change is
     unconditional -> gated steering.

Cost: $0, CPU-only (GPT-2 117M), reuses all Exp 19 infrastructure.
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

ROOT = Path(__file__).resolve().parent.parent
HS_PATH = str(ROOT / "results" / "hidden_states" / "15_hidden_states_gpt2_117m.npz")
MODEL_ID = "gpt2"
ARRAY_LAYER = 8
HOOK_BLOCK_IDX = ARRAY_LAYER - 1
N_COMPONENTS = 100
RANDOM_STATE = 42
MAX_NEW_TOKENS = 40
LABEL_THRESHOLD = 0.12
N_TEST_PROMPTS = 150
ALPHAS = [10.0, 20.0, 40.0]
OUT_DIR = ROOT / "results"


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


def make_gated_steer_hook(direction_unit, alpha, tau_s, device, gate_counter):
    direction_t = torch.tensor(direction_unit, dtype=torch.float32, device=device)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        last_h = h[:, -1, :]
        s_h = float((last_h[0] * direction_t).sum().item())
        gate_counter["total"] += 1
        if s_h < tau_s:
            gate_counter["steered"] += 1
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + alpha * direction_t
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


def generate_with_gated_hook(prompt, model, tokenizer, device, layer_idx, direction_unit, alpha, tau_s):
    gate_counter = {"total": 0, "steered": 0}
    handle = None
    if direction_unit is not None and alpha != 0.0:
        layer = model.transformer.h[layer_idx]
        handle = layer.register_forward_hook(
            make_gated_steer_hook(direction_unit, alpha, tau_s, device, gate_counter))
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if handle is not None:
            handle.remove()
    return completion, gate_counter


def capability_check_gated(model, tokenizer, device, layer_idx, direction_unit, alpha, tau_s, texts):
    handle = None
    gate_counter = {"total": 0, "steered": 0}
    if direction_unit is not None and alpha != 0.0:
        layer = model.transformer.h[layer_idx]
        handle = layer.register_forward_hook(
            make_gated_steer_hook(direction_unit, alpha, tau_s, device, gate_counter))
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
    n = len(y)
    print(f"H_layer shape: {H_layer.shape}, hall_rate={1-y.mean():.3f}")

    # Disjoint split: first half fits the direction, second half calibrates tau_s.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(n)
    half = n // 2
    fit_idx, cal_idx = idx[:half], idx[half:]
    print(f"Direction-fit split: {len(fit_idx)}  Calibration split (disjoint): {len(cal_idx)}")

    w_k, pca = fisher_lda_direction(H_layer[fit_idx], y[fit_idx])
    w_found = lift_to_full_dim(w_k, pca)
    w_random = random_orthogonal_direction(w_found)
    print(f"Found direction norm check: {np.linalg.norm(w_found):.4f} (should be 1.0)")

    # Calibrate tau_s on the DISJOINT calibration split: project full-dim
    # cached activations onto w_found, take the midpoint between class means
    # (Bayes-optimal boundary under the paper's own Gaussian-equal-variance
    # assumption, Prop. 1).
    s_cal = H_layer[cal_idx].astype(np.float64) @ w_found
    y_cal = y[cal_idx]
    mean_s_correct = s_cal[y_cal == 1].mean()
    mean_s_hall = s_cal[y_cal == 0].mean()
    tau_s = (mean_s_correct + mean_s_hall) / 2
    print(f"tau_s calibrated on disjoint {len(cal_idx)}-sample split: "
          f"mean_s_correct={mean_s_correct:.4f}, mean_s_hall={mean_s_hall:.4f}, tau_s={tau_s:.4f}")

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
    baseline_labels = []
    for i, item in enumerate(items):
        prompt = f"Q: {item['question']}\nA:"
        c_base, _ = generate_with_gated_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, None, 0.0, tau_s)
        baseline_labels.append(label_completion(c_base, item["correct_answers"], item["incorrect_answers"]))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(items)}]", flush=True)
    print(f"Baseline hallucination rate: {1 - np.mean([l for l in baseline_labels if l != -1]):.4f}")

    results_per_alpha = []
    for alpha in ALPHAS:
        print(f"\n=== alpha={alpha} (gated) ===")
        found_labels, random_labels = [], []
        found_gate_rates, random_gate_rates = [], []

        for i, item in enumerate(items):
            q = item["question"]
            prompt = f"Q: {q}\nA:"
            correct_answers, incorrect_answers = item["correct_answers"], item["incorrect_answers"]

            c_found, gc_found = generate_with_gated_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, w_found, alpha, tau_s)
            c_rand, gc_rand = generate_with_gated_hook(prompt, model, tokenizer, device, HOOK_BLOCK_IDX, w_random, alpha, tau_s)

            found_labels.append(label_completion(c_found, correct_answers, incorrect_answers))
            random_labels.append(label_completion(c_rand, correct_answers, incorrect_answers))
            if gc_found["total"] > 0:
                found_gate_rates.append(gc_found["steered"] / gc_found["total"])
            if gc_rand["total"] > 0:
                random_gate_rates.append(gc_rand["steered"] / gc_rand["total"])

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(items)}]", flush=True)

        valid = [(b, f, r) for b, f, r in zip(baseline_labels, found_labels, random_labels)
                 if b == 0 and f != -1 and r != -1]
        n_valid = len(valid)
        found_flip_rate = np.mean([f == 1 for _, f, _ in valid]) if n_valid else float("nan")
        random_flip_rate = np.mean([r == 1 for _, _, r in valid]) if n_valid else float("nan")

        b_found_only = sum(1 for _, f, r in valid if f == 1 and r == 0)
        c_random_only = sum(1 for _, f, r in valid if f == 0 and r == 1)
        mcnemar_p = binomtest(b_found_only, b_found_only + c_random_only, 0.5).pvalue if (b_found_only + c_random_only) > 0 else 1.0

        cap_base = capability_check_gated(model, tokenizer, device, HOOK_BLOCK_IDX, None, 0.0, tau_s, capability_texts)
        cap_found = capability_check_gated(model, tokenizer, device, HOOK_BLOCK_IDX, w_found, alpha, tau_s, capability_texts)
        cap_random = capability_check_gated(model, tokenizer, device, HOOK_BLOCK_IDX, w_random, alpha, tau_s, capability_texts)

        mean_gate_rate_found = float(np.mean(found_gate_rates)) if found_gate_rates else float("nan")
        mean_gate_rate_random = float(np.mean(random_gate_rates)) if random_gate_rates else float("nan")

        print(f"  n_valid(baseline-hallucinated)={n_valid}  found_flip={found_flip_rate:.4f}  "
              f"random_flip={random_flip_rate:.4f}  McNemar p={mcnemar_p:.4f} "
              f"(b={b_found_only}, c={c_random_only})")
        print(f"  Mean fraction of generation steps actually gated (steered): "
              f"found={mean_gate_rate_found:.3f}  random={mean_gate_rate_random:.3f}")
        print(f"  Capability (mean token log-lik): baseline={cap_base:.4f}  "
              f"found-steered={cap_found:.4f}  random-steered={cap_random:.4f}")

        results_per_alpha.append({
            "alpha": alpha, "n_valid": n_valid,
            "found_flip_rate": found_flip_rate, "random_flip_rate": random_flip_rate,
            "mcnemar_b_found_only": b_found_only, "mcnemar_c_random_only": c_random_only,
            "mcnemar_p": float(mcnemar_p),
            "mean_gate_rate_found": mean_gate_rate_found, "mean_gate_rate_random": mean_gate_rate_random,
            "capability_baseline_ll": cap_base, "capability_found_ll": cap_found,
            "capability_random_ll": cap_random,
            "capability_drop_found": cap_base - cap_found, "capability_drop_random": cap_base - cap_random,
        })

    out = {
        "model": MODEL_ID, "array_layer": ARRAY_LAYER, "hook_block_idx": HOOK_BLOCK_IDX,
        "n_test_prompts": N_TEST_PROMPTS, "alphas": ALPHAS,
        "tau_s": float(tau_s), "mean_s_correct_cal": float(mean_s_correct), "mean_s_hall_cal": float(mean_s_hall),
        "n_direction_fit": int(len(fit_idx)), "n_calibration": int(len(cal_idx)),
        "results_per_alpha": results_per_alpha,
    }
    out_path = OUT_DIR / "logs" / "23_gated_causal_intervention.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
