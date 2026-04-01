# Pre-Registration: Mamba 130M Architecture Transfer

**Project:** GEOM-PROOF
**Experiment:** 04 — Mamba 130M (SSM) architecture transfer
**Status:** PRE-REGISTERED — committed before running Exp 04
**Date:** 2026-03-27

---

## Basis Experiments (run before this commit)

| Exp | Result |
|-----|--------|
| 01 Fisher Analysis | Qwen 3B: probe AUROC=0.6537, Fisher bound=0.9485, depth fraction=0.472; GPT-2 Med: probe AUROC=0.4752, Fisher bound=0.9534 |
| 02 Scale Curve | AUROC = sigmoid(0.4896 × log₁₀(params) − 4.0821), R²=0.7954 |
| 03 Certificate Validation | Mean bound error=0.0913 ± 0.0453; certificate NOT validated on Qwen 3B (MAE > 0.05) |

---

## Pre-Registered Predictions for Mamba 130M

**Model:** `state-spaces/mamba-130m-hf`
**Parameters:** 130M
**Architecture:** SSM (Mamba-1, State Space Model — no attention mechanism)
**Task:** TruthfulQA (400 questions subset)
**Label method:** ROUGE-L ≥ 0.4

### Predicted AUROC

| Quantity | Predicted Value | Basis |
|---|---|---|
| Best-layer probe AUROC | **0.47** | Log-linear scale curve extrapolation to 130M (below training range — high uncertainty) |
| AUROC 90% interval | **[0.35, 0.62]** | Wide CI; extrapolating below smallest data point (GPT-2 117M at 0.50) |
| Fisher bound at best layer | **~0.82** | Expected from pattern in Exp 01; transformer trend extrapolated to SSM |

*Caveat: the scale curve was fit on transformer architectures (GPT-2 117M, GPT-2 Med 345M, Qwen 2.5 3B). Mamba-1 uses SSM layers (no attention, no MLP). Whether Fisher separability scales identically across architectural families is the research question this experiment tests. The prediction is hypothesis-derived, not architecture-aware.*

### Predicted Depth Fraction

| Quantity | Predicted Value | Basis |
|---|---|---|
| Best-layer depth fraction | **0.89** | MECH-INT + HaRP convergent finding; transformers peak at ~89% depth |
| Best layer index | **~Layer 21 of 24** | 0.89 × 24 = 21.4 |

### Pass/Fail Criterion

- **PASS:** Actual probe AUROC falls within ±0.05 of predicted AUROC (i.e., within [0.42, 0.52])
- **FAIL:** Actual probe AUROC outside ±0.05 of predicted AUROC

*Note: A FAIL is also informative. It would indicate either (a) SSM architectures encode factual uncertainty differently than transformers, (b) the Fisher certificate does not transfer across architectural families, or (c) the depth-fraction hypothesis is transformer-specific.*

---

## Commit Information

*(Filled in automatically by git)*

- **Commit hash:** [filled by git]
- **Commit timestamp:** [filled by git]
- **Committed by:** Lakshmi Chakradhar Vijayarao

---

## Outcome (Fill in after Exp 04)

*(Do not fill this in before running Exp 04)*

- **Actual AUROC:** [TO FILL]
- **Actual best-layer depth fraction:** [TO FILL]
- **Bound error:** [TO FILL]
- **Verdict:** PASS / FAIL
- **Notes:** [Any unexpected findings]
