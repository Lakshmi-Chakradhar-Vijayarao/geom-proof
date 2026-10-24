"""
GEOM-PROOF -- Exp 30: label-quality audit for Pipeline C, mirroring Exp 07's
ROUGE-L-vs-GPT-4o audit (kappa=-0.010, Sec "Label quality" limitation) but
for Pipeline C's SQuAD-style EM/F1 labels, the labeling scheme underlying
this paper's "central result" (Table tab:certificate-matched) -- which,
unlike Pipeline B, was never itself audited against an independent judge.

No GPT-4o API access is available in this environment; substituting an
open, cross-family judge (Mistral-7B-Instruct-v0.2, the same base
checkpoint already validated for this exact judging role in this
project's companion leakage-audit paper, quantized on load via
bitsandbytes 4-bit rather than the pre-quantized AWQ checkpoint used
there) for Qwen2.5-3B's own HaluEval generations, avoiding a
same-family self-judge confound. bitsandbytes was substituted for AWQ
after three separate attempts to load an AWQ checkpoint via its
required gptqmodel backend each failed differently (torch/CUDA-kernel
mismatch, huggingface_hub/transformers version skew, a missing
transitive dependency after pinning) -- all downstream of gptqmodel's
own dependency resolution having drifted since this project last used
it successfully, not a problem with the AWQ-loading pattern itself.
"""
import json
import random
from pathlib import Path

try:
    import bitsandbytes  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device -- Mistral-7B needs a GPU.")
try:
    _a = torch.randn(4, 4, device="cuda") @ torch.randn(4, 4, device="cuda")
    print(f"GPU sanity check passed: {torch.cuda.get_device_name(0)}")
except Exception as e:
    raise RuntimeError(f"GPU sanity check failed: {e}")


def _resolve_input_dir(filename):
    candidates = sorted(Path("/kaggle/input").glob(f"**/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"{filename} not found under /kaggle/input")
    return candidates[0]


JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
RANDOM_STATE = 42
N_SAMPLE = 100

JUDGE_PROMPT = """You are a factuality judge. You will be given a question, the correct reference answer, and a model's answer.
Decide if the model's answer is CORRECT or HALLUCINATED.
A response is CORRECT if its core factual claim matches or is consistent with the reference answer, even if phrased differently.
A response is HALLUCINATED if it asserts something factually wrong, contradicts the reference, or fails to answer.
Ignore phrasing differences and focus only on factual accuracy.

Question: {question}
Reference answer: {reference}
Model's answer: {response}

Reply with exactly one word: CORRECT or HALLUCINATED."""


def stratified_sample(data, n, seed=RANDOM_STATE):
    rng = random.Random(seed)
    correct = [d for d in data if d["label"] == 1]
    wrong = [d for d in data if d["label"] == 0]
    n_correct = round(n * len(correct) / len(data))
    n_wrong = n - n_correct
    n_correct = min(n_correct, len(correct))
    n_wrong = min(n_wrong, len(wrong))
    sample = rng.sample(correct, n_correct) + rng.sample(wrong, n_wrong)
    rng.shuffle(sample)
    return sample


def query_judge(model, tokenizer, device, question, reference, response):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, response=response)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    verdict = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    if "CORRECT" in verdict and "HALLUCINAT" not in verdict:
        return 1
    if "HALLUCINAT" in verdict:
        return 0
    return -1


def main():
    data_path = _resolve_input_dir("15_generations_qwen_3b.json")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} Qwen 3B HaluEval Pipeline-C generations")

    sample = stratified_sample(data, N_SAMPLE)
    n_correct = sum(1 for s in sample if s["label"] == 1)
    print(f"Sampled {len(sample)} ({n_correct} Pipeline-C-correct / {len(sample)-n_correct} Pipeline-C-hallucinated)")

    print(f"Loading judge {JUDGE_MODEL} (bitsandbytes 4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, device_map="auto", quantization_config=bnb_config)
    model.eval()

    pipeline_c_labels, judge_labels, ambiguous = [], [], 0
    per_sample = []
    for i, d in enumerate(sample):
        verdict = query_judge(model, tokenizer, "cuda", d["question"], d["right_answer"], d["generated"])
        per_sample.append({"question": d["question"], "right_answer": d["right_answer"],
                            "generated": d["generated"], "pipeline_c_label": d["label"], "judge_verdict": verdict})
        if verdict == -1:
            ambiguous += 1
            continue
        pipeline_c_labels.append(d["label"])
        judge_labels.append(verdict)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sample)}]", flush=True)

    pipeline_c_labels = np.array(pipeline_c_labels)
    judge_labels = np.array(judge_labels)
    agreement = float((pipeline_c_labels == judge_labels).mean())
    kappa = float(cohen_kappa_score(pipeline_c_labels, judge_labels)) if len(set(pipeline_c_labels)) > 1 and len(set(judge_labels)) > 1 else float("nan")
    cm = confusion_matrix(pipeline_c_labels, judge_labels).tolist()

    print(f"\nn_scored={len(pipeline_c_labels)} (ambiguous={ambiguous})")
    print(f"Agreement={agreement:.4f}  Cohen kappa={kappa:.4f}")
    print(f"Confusion matrix (rows=Pipeline C, cols=judge): {cm}")

    out = {
        "judge_model": JUDGE_MODEL, "judge_quantization": "bitsandbytes-4bit-nf4",
        "generator_model": "Qwen/Qwen2.5-3B-Instruct",
        "n_sampled": len(sample), "n_scored": len(pipeline_c_labels), "n_ambiguous": ambiguous,
        "agreement": agreement, "cohen_kappa": kappa, "confusion_matrix": cm, "per_sample": per_sample,
    }
    with open(OUT / "30_pipeline_c_label_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}/30_pipeline_c_label_audit.json")


if __name__ == "__main__":
    main()
