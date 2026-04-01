"""
GEOM-PROOF — Kaggle GPU Job: Causal Fisher + TruthfulQA OOD Extraction
=======================================================================

What this script does:
  1. Loads Qwen 2.5 3B-Instruct model (from /kaggle/input dataset)
  2. Extracts lm_head weight W_U → computes Causal Fisher (Certificate B) per layer
     using HaluEval hidden states already in the dataset
  3. Runs Qwen on 400 TruthfulQA questions → extracts hidden states at each layer
     (for real OOD test in Exp 10)

Inputs needed in dataset:
  - 00_halueval_qwen3b.npz  (hidden states + labels)
  - truthfulqa_400.json     (400 questions pre-extracted locally)

Outputs written to /kaggle/working/:
  - causal_fisher_qwen3b.json   (J_causal per layer, best layer, depth)
  - qwen3b_truthfulqa_ood.npz   (hidden states + labels for TruthfulQA OOD)
"""

import json, numpy as np, os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from rouge_score import rouge_scorer as rs_module
from scipy.special import ndtr as phi

OUT = Path("/kaggle/working")

def _find_file(filename):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return Path(root) / filename
    raise FileNotFoundError(f"{filename} not found under /kaggle/input/")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading HaluEval hidden states...")
data = np.load(_find_file("00_halueval_qwen3b.npz"))
H_halueval = data["hidden_states"].astype(np.float32)
y_halueval  = data["labels"].astype(int)
n_samples, n_layers, d = H_halueval.shape
print(f"  Shape: {H_halueval.shape}, hall_rate: {1-y_halueval.mean():.3f}")

print("Loading TruthfulQA questions...")
try:
    with open(_find_file("truthfulqa_400.json")) as f:
        questions = json.load(f)
    print(f"  Loaded from dataset: {len(questions)} questions")
except FileNotFoundError:
    print("  truthfulqa_400.json not found — downloading from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    ds = ds.shuffle(seed=42).select(range(400))
    questions = [{"question": ex["question"],
                  "best_answer": ex["best_answer"] if ex["best_answer"] else ex["correct_answers"][0]}
                 for ex in ds]
    print(f"  Downloaded: {len(questions)} questions")
print(f"  {len(questions)} questions ready")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"\nLoading model: {MODEL_ID}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).to(device)
model.eval()
print("  Model loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: CAUSAL FISHER (Certificate B)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Part 1 — Causal Fisher (Certificate B)")
print("="*60)

# Extract lm_head weight matrix W_U: shape (vocab_size, d)
W_U = model.lm_head.weight.detach().float().cpu().numpy()
print(f"  W_U shape: {W_U.shape}")

def causal_fisher(H, y, W_U, n_components=100):
    H = H.astype(np.float64)
    y = np.asarray(y, dtype=int)
    # Project hidden dim through top-k principal directions of W_U (vocab x d)
    # PCA on W_U: rows=vocab tokens, cols=hidden_dim → components shape (k, d) → W_proj (d, k)
    k = min(n_components, W_U.shape[1], H.shape[0]-2)
    pca = PCA(n_components=k, random_state=42)
    W_proj = pca.fit(W_U).components_.T  # (d, k)
    H_c = H @ W_proj  # (n, k)
    mu_c, mu_h = H_c[y==1].mean(0), H_c[y==0].mean(0)
    delta = mu_c - mu_h
    lw = LedoitWolf().fit(np.vstack([H_c[y==1]-mu_c, H_c[y==0]-mu_h]))
    try:
        J = float(delta @ np.linalg.solve(lw.covariance_, delta))
    except:
        J = float(delta @ np.linalg.lstsq(lw.covariance_, delta, rcond=None)[0])
    return max(0.0, J)

J_causal = []
for L in range(n_layers):
    J = causal_fisher(H_halueval[:, L, :], y_halueval, W_U, n_components=100)
    J_causal.append(J)
    if L % 5 == 0 or L == n_layers - 1:
        print(f"  L{L:3d}: J_causal={J:.4f}, bound={phi(np.sqrt(max(0,J))/2):.6f}")

best_L = int(np.argmax(J_causal))
causal_result = {
    "model": "Qwen 2.5 3B",
    "n_samples": int(n_samples),
    "n_layers": int(n_layers),
    "J_causal_per_layer": J_causal,
    "auroc_bound_causal_per_layer": [float(phi(np.sqrt(max(0,J))/2)) for J in J_causal],
    "best_causal_layer": best_L,
    "best_causal_J": float(J_causal[best_L]),
    "best_causal_depth_fraction": float(best_L / max(n_layers-1, 1)),
    "W_U_shape": list(W_U.shape),
    "note": "Causal Fisher projects hidden states via unembedding W_U PCA subspace before Fisher ratio"
}
print(f"\n  Best causal layer: L{best_L} (depth={best_L/max(n_layers-1,1):.3f}), J={J_causal[best_L]:.4f}")

out_causal = OUT / "causal_fisher_qwen3b.json"
with open(out_causal, "w") as f:
    json.dump(causal_result, f, indent=2)
print(f"  Saved: {out_causal}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: TruthfulQA OOD EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Part 2 — TruthfulQA OOD Extraction")
print("="*60)

scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
ROUGE_THRESHOLD = 0.4
MAX_NEW_TOKENS = 64

hidden_states_list = []
labels_list = []

for i, ex in enumerate(questions):
    prompt = f"Question: {ex['question']}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Forward pass for hidden states at input last token
        fwd = model(**inputs, output_hidden_states=True)
        all_hs = fwd.hidden_states  # tuple: (n_layers+1,) each (1, seq, d)
        hs = torch.stack([all_hs[l][0, -1, :].float().cpu() for l in range(1, len(all_hs))], dim=0)

        # Generate response for ROUGE label
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True).strip()
    score = scorer.score(ex["best_answer"].lower(), generated.lower())["rougeL"].fmeasure
    label = 1 if score >= ROUGE_THRESHOLD else 0

    hidden_states_list.append(hs.numpy())
    labels_list.append(label)

    if (i+1) % 50 == 0:
        hr = 1 - sum(labels_list)/(i+1)
        print(f"  [{i+1}/400] hall_rate={hr:.3f}")

H_ood = np.stack(hidden_states_list, axis=0).astype(np.float32)
y_ood = np.array(labels_list, dtype=np.int32)
print(f"\n  Final: shape={H_ood.shape}, hall_rate={1-y_ood.mean():.3f}, n_correct={y_ood.sum()}")

out_ood = OUT / "qwen3b_truthfulqa_ood.npz"
np.savez_compressed(out_ood, hidden_states=H_ood, labels=y_ood)
print(f"  Saved: {out_ood}")

print("\n=== DONE ===")
print(f"Outputs in /kaggle/working/:")
print(f"  causal_fisher_qwen3b.json")
print(f"  qwen3b_truthfulqa_ood.npz")
