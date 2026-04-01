"""
Exp 04 — Architecture Transfer: Mamba 130M (SSM)

Tests whether the Fisher separability certificate transfers to a qualitatively
different architecture — Mamba, a State Space Model with no attention mechanism.

Pre-registration: Run Exps 01–03 FIRST. Commit results/preregistration.md
with git BEFORE running this experiment. The commit timestamp is the evidence.

Hardware: Kaggle T4 (free tier) — the mamba-ssm library requires CUDA.
Cost: $0

Steps:
  1. Load TruthfulQA questions
  2. Generate responses with state-spaces/mamba-130m
  3. Label with ROUGE-L (threshold 0.4)
  4. Extract block outputs at each of 24 Mamba layers
  5. Compute Fisher ratio J(L) at each layer → AUROC_bound
  6. Run OOF linear probe at each layer → actual AUROC
  7. Compare prediction to pre-registered value

NOTE: This script is designed for Kaggle T4 execution.
      Set KAGGLE = True when uploading to Kaggle.
      The data paths will differ from local paths.
"""

import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import sys

# ── Environment detection ─────────────────────────────────────────────────────
KAGGLE = Path("/kaggle").exists()

if KAGGLE:
    ROOT = Path("/kaggle/working/geom-proof")
    ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
else:
    ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT))

from src.fisher import fisher_curve

RESULTS_DIR = ROOT / "results" / "logs"
HIDDEN_STATES_DIR = ROOT / "results" / "hidden_states"
PLOTS_DIR = ROOT / "results" / "plots"
for d in [RESULTS_DIR, HIDDEN_STATES_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "state-spaces/mamba2-130m"   # Mamba-2 (ACL 2025: different factual flow from Mamba-1)
N_SAMPLES = 400         # TruthfulQA subset — sufficient for proof-of-concept
MAX_NEW_TOKENS = 128
ROUGE_THRESHOLD = 0.4   # Same as HaRP
METHOD = "pca"
N_COMPONENTS = 50       # Mamba hidden dim = 768; 50 components is generous
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_truthfulqa(n: int) -> list[dict]:
    """Load TruthfulQA validation split, first n questions."""
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        correct_answers = [a for a in row["correct_answers"] if a.strip()]
        if not correct_answers:
            continue
        samples.append({
            "question": row["question"],
            "best_answer": row["best_answer"],
            "correct_answers": correct_answers,
        })
    return samples[:n]


def label_response(response: str, correct_answers: list[str]) -> int:
    """Return 1 (correct) if ROUGE-L >= threshold vs any correct answer, else 0."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, response)["rougeL"].fmeasure for ref in correct_answers]
    return int(max(scores) >= ROUGE_THRESHOLD)


def generate_and_extract(model, tokenizer, question: str) -> tuple[str, np.ndarray | None]:
    """
    Generate a response and extract block outputs (hidden states) at all layers.

    Returns:
        response: str
        hidden_states: ndarray (n_layers, d) — last-token representation at each block
    """
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generation (no hidden states needed here — Mamba doesn't expose them during generation easily)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Hidden state extraction: forward pass with output_hidden_states=True
    # For Mamba, this returns the output of each Mamba block
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if out.hidden_states is None:
        return response, None

    # Stack: (n_layers+1, seq_len, d) → take last token position, all layers
    hs = torch.stack(out.hidden_states, dim=0)  # (n_layers+1, seq_len, d)
    last_token_hs = hs[:, -1, :].float().cpu().numpy()  # (n_layers+1, d)
    return response, last_token_hs


def run() -> dict:
    print("=" * 60)
    print("Exp 04 — Mamba 130M Architecture Transfer")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    if DEVICE == "cpu":
        print("WARNING: mamba-ssm CUDA kernels may not be available on CPU.")
        print("This experiment is designed for Kaggle T4. If running locally, results may vary.")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    # Mamba-2 uses Mamba2ForCausalLM in transformers >= 4.45
    try:
        from transformers import Mamba2ForCausalLM, Mamba2Config
        ModelClass = Mamba2ForCausalLM
    except ImportError:
        from transformers import MambaForCausalLM
        ModelClass = MambaForCausalLM
        print("  Warning: Mamba2ForCausalLM not found — falling back to MambaForCausalLM")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ModelClass.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"Model loaded: Mamba-2 {MODEL_ID}, {n_layers} layers, hidden_size={hidden_size}")

    # ── Load TruthfulQA ────────────────────────────────────────────────────────
    print(f"\nLoading {N_SAMPLES} TruthfulQA questions ...")
    samples = load_truthfulqa(N_SAMPLES)
    print(f"Loaded {len(samples)} samples.")

    # ── Generate responses and extract hidden states ───────────────────────────
    all_hidden_states = []
    labels = []
    responses = []

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  [{i}/{len(samples)}] generating ...")
        response, hs = generate_and_extract(model, tokenizer, sample["question"])
        if hs is None:
            print(f"  Warning: no hidden states for sample {i}, skipping.")
            continue
        label = label_response(response, sample["correct_answers"])
        all_hidden_states.append(hs)
        labels.append(label)
        responses.append(response)

    H = np.stack(all_hidden_states, axis=0)  # (n_valid, n_layers+1, d)
    y = np.array(labels, dtype=int)
    print(f"\nHidden states shape: {H.shape}")
    print(f"Hall rate: {1-y.mean():.3f} ({(1-y).sum()} hallucinated / {len(y)} total)")

    # Save hidden states
    hs_path = HIDDEN_STATES_DIR / "04_mamba_hidden_states.npz"
    np.savez_compressed(hs_path, hidden_states=H, labels=y)
    print(f"Hidden states saved: {hs_path}")

    # ── Fisher analysis ────────────────────────────────────────────────────────
    print("\nComputing Fisher curves ...")
    curve = fisher_curve(H, y, method=METHOD, n_components=N_COMPONENTS, verbose=True)

    # ── OOF probe at each layer ────────────────────────────────────────────────
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    n_samples = H.shape[0]
    n_l = H.shape[1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probe_aurocs = []

    print("\nRunning OOF probes ...")
    for L in range(n_l):
        H_L = H[:, L, :]
        oof_preds = np.zeros(n_samples)
        for train_idx, test_idx in skf.split(H_L, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(H_L[train_idx])
            X_te = scaler.transform(H_L[test_idx])
            lr = LogisticRegression(max_iter=500, C=1.0)
            lr.fit(X_tr, y[train_idx])
            oof_preds[test_idx] = lr.predict_proba(X_te)[:, 1]
        auroc = float(roc_auc_score(y, oof_preds))
        probe_aurocs.append(auroc)
        print(f"  Layer {L:3d}: J={curve['J'][L]:.4f}, bound={curve['auroc_bound'][L]:.4f}, probe={auroc:.4f}")

    best_probe_layer = int(np.argmax(probe_aurocs))
    best_probe_auroc = float(np.max(probe_aurocs))

    # ── Load pre-registered prediction ────────────────────────────────────────
    prereg_path = ROOT / "results" / "preregistration.md"
    prereg_content = prereg_path.read_text() if prereg_path.exists() else "Not found"

    results = {
        "experiment": "04_mamba_transfer",
        "model": MODEL_ID,
        "n_params": 130e6,
        "n_samples": int(n_samples),
        "n_layers": int(n_l),
        "hidden_size": int(hidden_size),
        "hall_rate": float(1 - y.mean()),
        "fisher_best_layer": int(curve["best_layer"]),
        "fisher_best_J": float(curve["best_J"]),
        "fisher_best_auroc_bound": float(curve["best_auroc_bound"]),
        "fisher_depth_fraction": float(curve["depth_fraction"]),
        "probe_best_layer": int(best_probe_layer),
        "probe_best_auroc": float(best_probe_auroc),
        "probe_depth_fraction": float(best_probe_layer / (n_l - 1)),
        "bound_error": float(abs(curve["best_auroc_bound"] - best_probe_auroc)),
        "certificate_validated": float(abs(curve["best_auroc_bound"] - best_probe_auroc)) < 0.05,
        "J_per_layer": [float(j) for j in curve["J"]],
        "auroc_bound_per_layer": [float(b) for b in curve["auroc_bound"]],
        "probe_auroc_per_layer": probe_aurocs,
    }

    out_path = RESULTS_DIR / "04_mamba_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n" + "=" * 60)
    print("RESULT SUMMARY:")
    print(f"  Fisher best layer: {results['fisher_best_layer']} "
          f"({results['fisher_depth_fraction']:.3f} depth)")
    print(f"  Fisher AUROC bound: {results['fisher_best_auroc_bound']:.4f}")
    print(f"  Probe best AUROC:   {results['probe_best_auroc']:.4f}")
    print(f"  Bound error:        {results['bound_error']:.4f} "
          f"({'PASS ✓' if results['certificate_validated'] else 'FAIL ✗'})")
    print(f"  Depth fraction (probe): {results['probe_depth_fraction']:.3f} "
          f"({'≈89%' if abs(results['probe_depth_fraction'] - 0.89) < 0.05 else 'different from 89%'})")

    return results


if __name__ == "__main__":
    run()
