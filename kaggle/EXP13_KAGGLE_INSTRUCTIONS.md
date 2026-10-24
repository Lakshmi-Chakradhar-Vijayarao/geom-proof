# Exp 13 — Qwen 2.5 7B on Kaggle T4

## What this runs
Full certificate validation on Qwen 2.5 7B:
- Balanced HaluEval extraction (1000 QA pairs → 2000 vectors, 50/50)
- Fisher certificate Φ(√J/2) at all 29 layers
- Linear probe (5-fold OOF AUROC)
- MLP probe (64 hidden, ReLU, Adam, 5-fold OOF AUROC)
- Conformal calibration (α* search)

**Estimated runtime: 2.5–3.5 hours on T4**

---

## Step-by-step Kaggle setup

### 1. Create notebook
- Go to kaggle.com → New Notebook
- Title: `geom-proof-exp13-qwen7b`
- **Accelerator: GPU T4 x1** (required — 7B model needs ~14GB VRAM)
- Session: set to maximum (9 hours)

### 2. First cell — install packages
```python
!pip install -q transformers datasets accelerate scikit-learn matplotlib scipy
```
Run this cell first, wait for it to complete (~2 min).

### 3. Second cell — paste the script
Copy the entire contents of `kaggle/exp13_qwen7b_full_analysis.py` and paste into a single code cell.

### 4. Run
Click Run All. The notebook will:
- Download Qwen 2.5 7B from HuggingFace automatically (~14GB, ~5 min)
- Extract hidden states (~2–2.5 hours)
- Run all analysis (~30–45 min)
- Save 3 output files

### 5. Monitor progress
The script prints progress every 100 samples with elapsed time and ETA.
You can watch the output in real time.

---

## If you run out of VRAM (OOM error)

Edit line in the script:
```python
N_SAMPLES = 1000   # change to 500
```
This gives 1000 vectors instead of 2000 — still valid for all analyses.

---

## Download outputs

After the notebook finishes:
1. Click **Output** in the left panel (or go to the notebook's Output tab)
2. Download these 3 files:

| File | Size (approx) | Local destination |
|------|---------------|-------------------|
| `13_halueval_qwen7b.npz` | ~600 MB | `geom-proof/results/hidden_states/` |
| `13_qwen7b_analysis.json` | ~1 MB | `geom-proof/results/logs/` |
| `13_qwen7b_certificate.png` | ~200 KB | `geom-proof/results/plots/` |

---

## After download — update the paper

Open `13_qwen7b_analysis.json` and find these values:

```json
{
  "best_linear_auroc": X.XXXX,
  "best_linear_layer": XX,
  "best_mlp_auroc":    X.XXXX,
  "best_mlp_layer":    XX,
  "mlp_gain":          X.XXXX,
  "mlp_violations":    X,
  "linear_violations": X,
  "best_bound":        X.XXXX,
  "linear_bound_error_max":  X.XXXX,
  "linear_bound_error_mean": X.XXXX,
  "conformal": {"alpha_star": X.XX, "accept_rate": X.XXX}
}
```

Then fill in the `---` cells in `paper/main.tex`:

**Table 2 (tab:certificate) — add to Qwen 7B row:**
```
Qwen 2.5 7B  | best_J_layer | best_J_layer/28 depth% | best_J | best_bound | best_linear_auroc | error
```

**Table 6 (tab:mlp) — add to Qwen 7B row:**
```
Qwen 2.5 7B  | best_bound | best_linear_auroc (Lxx) | best_mlp_auroc (Lxx) | +mlp_gain | 0 / 29
```

**Table 7 (tab:scale) — fill in Qwen 7B row:**
```
Qwen 2.5 7B  | 7.6B | best_linear_auroc | Lxx | xx%
```

Recompile on Overleaf. Done.

---

## Expected results (based on pattern from 3B)

The Fisher certificate should hold (0 violations) since:
- At 3B: 0/37 violations, mean error 1.3%
- 7B has higher d (3584 vs 2048) but same balanced HaluEval setup

If the bound holds at 7B, you have:
> "Zero violations across all 91 layer-model combinations (37 + 25 + 29)"

That completely eliminates the small-model objection.
