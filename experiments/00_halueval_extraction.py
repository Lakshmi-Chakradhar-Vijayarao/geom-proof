"""
Exp 00 — HaluEval Hidden State Extraction (Second Dataset)

Extracts hidden states from Qwen 2.5 3B and GPT-2 Medium 345M on the
HaluEval QA subset — a second, structurally different hallucination dataset.

Why HaluEval?
  TruthfulQA: naturally elicited hallucinations (model confidently wrong)
  HaluEval:   adversarially constructed — incorrect answers are plausible
              near-miss substitutions of the correct Wikipedia fact.

This structural difference tests whether the Fisher/W₂ certificate and
the conformal guarantee generalize across hallucination mechanisms,
not just across models.

HaluEval QA format:
  Each sample: {question, right_answer, hallucinated_answer}
  We generate model responses and label them ROUGE-L against right_answer.
  Alternatively: use the pre-labeled (right/hallucinated) pairs directly
  without generation — forward pass only on both answer strings.

We use the NO-GENERATION mode for speed:
  - Forward pass on the CORRECT answer string → extract hidden states → label=1
  - Forward pass on the HALLUCINATED answer string → extract hidden states → label=0
  - This gives perfectly balanced classes (50/50) — label-clean, no ROUGE noise.

Output:
  results/hidden_states/00_halueval_qwen3b.npz   — shape (2*N, n_layers, d)
  results/hidden_states/00_halueval_gpt2med.npz  — shape (2*N, n_layers, d)

Cost: $0 — CPU, ~2–3 hours per model.
N: 1000 samples → 2000 hidden-state vectors (1000 correct, 1000 hallucinated).
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "logs"
HS_DIR = ROOT / "results" / "hidden_states"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HS_DIR.mkdir(parents=True, exist_ok=True)

HARP_DIR = Path("/Users/chakrivijayarao/Desktop/harp")

N_SAMPLES = 1000   # per class → 2000 total (perfectly balanced)

MODELS = {
    "qwen3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "out_path": HS_DIR / "00_halueval_qwen3b.npz",
        "dtype": "float32",
        "device_map": "cpu",
    },
    "gpt2med": {
        "model_id": "openai-community/gpt2-medium",
        "out_path": HS_DIR / "00_halueval_gpt2med.npz",
        "dtype": "float32",
        "device_map": "cpu",
    },
}


def load_halueval(n: int) -> list[dict]:
    """
    Load HaluEval QA subset.
    Returns list of {question, right_answer, hallucinated_answer}.
    """
    from datasets import load_dataset
    # HaluEval QA is at: HaluEval/halueval
    # Split: data/qa_data.json — use the HuggingFace version
    try:
        ds = load_dataset("HaluEval/halueval", "qa", split="data", trust_remote_code=True)
    except Exception:
        # Fallback: try alternate HuggingFace path
        ds = load_dataset("pminervini/HaluEval", split="qa_samples")

    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        question = row.get("question", row.get("input", ""))
        right = row.get("right_answer", row.get("answer", ""))
        hallucinated = row.get("hallucinated_answer", row.get("output", ""))
        if question and right and hallucinated and right != hallucinated:
            samples.append({
                "question": question,
                "right_answer": right,
                "hallucinated_answer": hallucinated,
            })
    return samples[:n]


def extract_hs_for_text(model, tokenizer, text: str, device: str) -> np.ndarray | None:
    """
    Forward pass on `text`, return hidden states at last token, all layers.
    Shape: (n_layers + 1, d)
    """
    import torch
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if out.hidden_states is None:
        return None

    # hidden_states: tuple of (1, seq_len, d) per layer
    # Take last token position across all layers
    hs = np.stack([
        h[0, -1, :].float().cpu().numpy()
        for h in out.hidden_states
    ])  # (n_layers+1, d)
    return hs


def run_model(model_key: str, cfg: dict, samples: list[dict]) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_path = cfg["out_path"]
    if out_path.exists():
        print(f"  {model_key}: already extracted at {out_path}, skipping.")
        return

    print(f"\n{'─'*60}")
    print(f"Extracting: {cfg['model_id']}")
    print(f"  N samples: {len(samples)} pairs → {2*len(samples)} hidden-state vectors")

    device = "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float32,
        device_map=cfg["device_map"],
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  Model: {n_layers} layers, d={d}")

    all_hs = []
    all_labels = []
    skipped = 0

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  [{i}/{len(samples)}] ...")

        # Correct answer → label=1
        prompt_correct = f"Q: {sample['question']}\nA: {sample['right_answer']}"
        hs_c = extract_hs_for_text(model, tok, prompt_correct, device)

        # Hallucinated answer → label=0
        prompt_hall = f"Q: {sample['question']}\nA: {sample['hallucinated_answer']}"
        hs_h = extract_hs_for_text(model, tok, prompt_hall, device)

        if hs_c is None or hs_h is None:
            skipped += 1
            continue

        all_hs.append(hs_c)
        all_labels.append(1)
        all_hs.append(hs_h)
        all_labels.append(0)

    H = np.stack(all_hs).astype(np.float32)   # (2N, n_layers+1, d)
    y = np.array(all_labels, dtype=np.int32)

    print(f"  Done. Shape: {H.shape}, hall_rate={1-y.mean():.3f}, skipped={skipped}")
    np.savez_compressed(out_path, hidden_states=H, labels=y)
    print(f"  Saved: {out_path}")

    # Free memory
    del model
    import gc; gc.collect()


def run() -> None:
    print("=" * 70)
    print("Exp 00 — HaluEval Hidden State Extraction")
    print("Mode: NO-GENERATION (forward pass on pre-labeled answer pairs)")
    print("This gives perfectly balanced classes, no ROUGE noise.")
    print("=" * 70)

    print(f"\nLoading HaluEval QA ({N_SAMPLES} samples) ...")
    samples = load_halueval(N_SAMPLES)
    print(f"  Loaded {len(samples)} QA pairs.")

    if not samples:
        print("ERROR: Could not load HaluEval. Check dataset name/access.")
        return

    for model_key, cfg in MODELS.items():
        run_model(model_key, cfg, samples)

    print("\n" + "=" * 70)
    print("HaluEval extraction complete.")
    print("Next: run experiments 07 → 01 → 08 → 09 → 10 (will auto-detect HaluEval data)")
    print("=" * 70)


if __name__ == "__main__":
    run()
