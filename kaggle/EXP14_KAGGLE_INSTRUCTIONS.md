# Exp 14 — OOD Transfer (Qwen 7B + GPT-2 Medium on TruthfulQA)

## What this runs
- Loads Qwen 7B HaluEval hidden states from Exp 13 (already on disk)
- Extracts GPT-2 Medium hidden states on HaluEval (fast, ~5 min)
- Extracts TruthfulQA hidden states for both models
- Trains probe on HaluEval (in-distribution), tests on TruthfulQA (OOD)
- Reports: in-dist AUROC, OOD AUROC, collapse magnitude, conformal violation factor
- Outputs: `14_ood_transfer.json`, `14_ood_transfer.png`, `14_truthfulqa_qwen7b.npz`, `14_truthfulqa_gpt2med.npz`

**Estimated runtime: ~25–35 min on T4**

---

## IMPORTANT: Kaggle API Token

The token from the previous session has been invalidated. You MUST:
1. Go to kaggle.com → Account → API → Create New Token
2. Download the new `kaggle.json` and place at `~/.kaggle/kaggle.json`
3. Run: `chmod 600 ~/.kaggle/kaggle.json`
4. Test: `kaggle kernels list --mine`

---

## Step-by-step setup

### 1. Push the notebook via CLI (after token renewal)
```bash
kaggle kernels push -p /Users/chakrivijayarao/Desktop/geom-proof/kaggle/exp14_notebook/
```

### 2. Monitor status
```bash
kaggle kernels status chakradharvijayarao/geom-proof-exp14-ood-transfer
```

### 3. Download outputs when complete
```bash
kaggle kernels output chakradharvijayarao/geom-proof-exp14-ood-transfer -p /Users/chakrivijayarao/Desktop/geom-proof/results/
```

---

## Alternative: manual Kaggle UI setup

### 1. Create notebook
- Go to kaggle.com → New Notebook
- Title: `geom-proof-exp14-ood-transfer`
- **Accelerator: GPU T4 x1**
- Session: maximum (9 hours)

### 2. Add dataset source
- Click **Add Data** → search for `geom-proof-exp13-qwen7b` (your own dataset)
- This makes `13_halueval_qwen7b.npz` available at `/kaggle/input/geom-proof-exp13-qwen7b/`

### 3. Add code
- First cell: `!pip install -q transformers datasets accelerate scikit-learn matplotlib scipy`
- Second cell: paste entire contents of `exp14_ood_transfer.py`

### 4. Run All

---

## Download outputs
After completion, download these files:

| File | Size | Local destination |
|------|------|-------------------|
| `14_ood_transfer.json` | ~50 KB | `results/logs/` |
| `14_ood_transfer.png` | ~200 KB | `results/plots/` |
| `14_truthfulqa_qwen7b.npz` | ~300 MB | `results/hidden_states/` |
| `14_truthfulqa_gpt2med.npz` | ~50 MB | `results/hidden_states/` |

---

## After download — paper updates

Open `14_ood_transfer.json` and look for:
```json
{
  "qwen7b": {
    "indist_auroc": X.XXXX,
    "ood_auroc": X.XXXX,
    "auroc_collapse": X.XXXX,
    "conformal_alpha_star": X.XX,
    "conformal_violation_factor": X.X
  },
  "gpt2med": {
    "indist_auroc": X.XXXX,
    "ood_auroc": X.XXXX,
    "auroc_collapse": X.XXXX,
    "conformal_alpha_star": X.XX,
    "conformal_violation_factor": X.X
  }
}
```

Then update `paper/main.tex` Section ~8.2 (sec:conformal-ood):
- Expand the OOD collapse paragraph to cover both models
- Note: collapse is now "architecture- and scale-independent" with data from GPT-2 Med + Qwen 7B

Also add `14_ood_transfer.png` to Overleaf upload list in `OVERLEAF_INSTRUCTIONS.md`.

---

## Expected results

Based on the GPT-2 finding (AUROC→0.485, 14× violation):
- Qwen 7B OOD AUROC will likely be near chance (0.45–0.55)
- Violation factor likely 10–20×
- The finding strengthens: "OOD collapse is consistent across architectures (GPT-2) and scales (0.5B–7.6B)"
