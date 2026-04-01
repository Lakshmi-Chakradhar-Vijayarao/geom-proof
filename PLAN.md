# GEOM-PROOF — Complete Project Plan

**From Fisher to Wasserstein: A Formal Optimal Transport Certificate for Hallucination Detection**

---

## Mission Statement

GEOM-PROOF earns one claim: **the hallucination signal is formally bounded — and the bound is a Wasserstein distance.**

Fisher separability is the starting point. Optimal Transport is the destination.

The central insight: **Fisher ratio J = W₂² in whitened space.** The AUROC bound Φ(√J/2) is therefore Φ(W₂_whitened/2) — Fisher is the Gaussian special case of Optimal Transport. This lets us generalize beyond Gaussianity, connect to spectral RMT theory, and wrap the detector in a distribution-free conformal guarantee.

Three directions, one unified arc:
- **Direction B (OT — Major):** Generalize Fisher to Wasserstein. Compare Bures W₂, Sliced W₂, and MMD as competing certificates. J is the Gaussian special case; SW₂ makes no distributional assumption.
- **Direction A (RMT/Spectral — Supporting theory):** The BBP phase transition in the hidden-state covariance predicts when the signal becomes detectable. The spectral KL departure from Marchenko-Pastur mirrors the SW₂ curve.
- **Direction C (Conformal — Application layer):** Wrap the OT score in split conformal prediction. Formal hallucination rate bound: P(hall | ACCEPT) ≤ α with probability ≥ 1−δ.

This is the transition from MECH-INT + HaRP (empirical observation) to a predictive, mathematically grounded theory.

---

## What We Already Know (Free — No New Experiments Needed)

From MECH-INT and HaRP, we have three confirmed data points:

| Model | Params | Best AUROC | Depth Fraction | Data Location |
|---|---|---|---|---|
| GPT-2 (117M) | 117M | ~0.500 (null) | L9/13 ≈ 69% | `/Desktop/MECH-INT/` |
| GPT-2 Medium (345M) | 345M | 0.579 | ~88% | `/Desktop/harp/results/hidden_states/25_gpt2_medium_hidden_states.npz` |
| Qwen 2.5 3B | 3B | 0.775 | L32/36 ≈ 89% | `/Desktop/harp/results/hidden_states/hidden_states.npz` |

We also have:
- Labels for all 700 TruthfulQA samples (HaRP ROUGE-L labels)
- Labels for 534 TruthfulQA samples (MECH-INT Jaccard labels)
- All OOF probe results from HaRP Exp 06 (layer-wise AUROC curve)
- All hidden-state files — no re-extraction needed for Exps 01–03

**Experiments 01–03 cost exactly $0.00.**

---

## The Fisher Separability Certificate — Core Theory

### Why Fisher?

A linear logistic regression probe on hidden states is asking: can I find a hyperplane in ℝᵈ that separates correct from hallucinated representations?

The theoretical maximum AUROC achievable by any linear classifier on two Gaussian-distributed classes is determined entirely by how well-separated those classes are in the hidden-state space — the Fisher ratio.

### The Derivation

Let:
- μ_c = centroid of correct responses at layer L (shape: d)
- μ_h = centroid of hallucinated responses at layer L (shape: d)
- Σ_w = pooled within-class covariance matrix (shape: d × d)

**Fisher ratio:**
```
J(L) = (μ_c - μ_h)ᵀ · Σ_w⁻¹ · (μ_c - μ_h)
```

Under the Gaussian equal-covariance assumption, the Bayes-optimal linear classifier has:
```
Mahalanobis distance: d_M = √J(L)
AUROC_bound(L) = Φ(√J(L) / 2)
```

where Φ is the standard normal CDF.

**This is the certificate:** compute J(L) from the hidden-state distribution → predict AUROC before running any probe.

### Practical Computation

Since d = 2048 (Qwen) or 768 (GPT-2), direct inversion of Σ_w is unstable. We use:
1. **Ledoit-Wolf shrinkage** estimator (sklearn, no GPU needed)
2. **Top-k PCA projection** (k=100 captures 95%+ variance in practice)

Both approaches give J(L) efficiently on CPU. We validate that both estimates agree.

### Expected Result

| Model | Predicted J at best layer | Predicted AUROC | Actual AUROC | Match? |
|---|---|---|---|---|
| GPT-2 117M | Low | ~0.50 | 0.500 | → Verify |
| GPT-2 Medium 345M | Medium | ~0.57 | 0.579 | → Verify |
| Qwen 2.5 3B | High | ~0.77 | 0.775 | → Verify |
| Mamba (new) | Pre-registered | Pre-published | → Run after | → Report |

---

## Experiments

### Exp 07 — LLM-as-Judge Re-evaluation of HaRP Labels (RUN FIRST)
**Cost: $0 | Hardware: CPU / Ollama | Time: ~2–3 hours | Data: HaRP responses**

**Why first:** "The Illusion of Progress" (EMNLP 2025) shows ROUGE evaluation can inflate AUROC by up to 45.9 points. Our AUROC 0.775 was measured with ROUGE-L labels. This experiment defends against that critique.

**What it does:** Re-labels 700 HaRP responses using Qwen 2.5 3B as LLM judge (already installed via Ollama). Measures label agreement with ROUGE (Cohen's κ). Saves judge labels — Exp 01 uses them automatically.

**Pass/fail:** κ ≥ 0.70 → labels consistent, AUROC robust. κ < 0.70 → re-report AUROC under judge labels. Either outcome is publishable.

---

### Exp 01 — Three-Certificate Comparison on All Existing Hidden States
**Cost: $0 | Hardware: CPU | Time: ~3–4 hours | Data: existing + Exp 07 output**

**Upgraded to three certificates:**

**(A) Euclidean Fisher** — our primary certificate
  J(L) = (μ_c - μ_h)^T Σ_w^{-1} (μ_c - μ_h),  AUROC_bound = Φ(√J / 2)

**(B) Causal Fisher** — Park et al. ICML 2024
  Re-weights by W_U^T W_U (unembedding matrix) — the theoretically correct geometry for transformer probing. Tests whether the causal metric gives a tighter bound.

**(C) Local Intrinsic Dimension (LID)** — Yin et al. ICML 2024
  Measures effective manifold dimensionality per class — a model-free alternative certificate. Compared against Fisher as the competing method.

**Key outputs:**
- `results/logs/01_fisher_analysis.json` — all three certificate values per layer, per model
- `results/plots/01_three_certificates.png` — all three curves + probe AUROC across normalized depth

**What it proves:** Which certificate best predicts probe AUROC? Does the causal metric improve on Euclidean? Is LID competitive or complementary?

---

### Exp 02 — Scale Curve Formalization
**Cost: $0 | Hardware: CPU | Time: ~30 minutes | Data: Exp 01 outputs**

**What it does:**
Fit a functional relationship between model scale (log params) and best-layer Fisher ratio J*. Use the three confirmed data points. Extrapolate to 7B.

**Fitting procedure:**
```
log(J*) = a · log(params) + b   [log-log linear]
```
Also try:
```
AUROC* = Φ(c · log(params) + d)   [direct AUROC fit on logit scale]
```

Use scipy.optimize.curve_fit. Report:
- Fit coefficients a, b (or c, d)
- 95% prediction interval at 7B (bootstrapped from 3 data points — wide, honest)
- Point estimate: predicted AUROC for Mistral 7B / LLaMA 3 8B

**Key outputs:**
- `results/logs/02_scale_curve.json` — fit coefficients, 7B prediction with CI
- `results/plots/02_scale_curve.png` — three-point fit + extrapolation + CI band

**Expected 7B prediction:**
Based on the 345M→3B jump (0.579→0.775), the 3B→7B increment should be smaller (diminishing returns). Point estimate: **AUROC ≈ 0.80–0.83** at 7B. CI will be wide but the direction is clear.

**This is the number we pre-register for GUARDIAN.**

---

### Exp 03 — Certificate Validation (K-Fold Pre-registration Simulation)
**Cost: $0 | Hardware: CPU | Time: ~1 hour | Data: existing HaRP hidden states**

**What it does:**
Simulate the pre-registration setting entirely within the HaRP dataset. For each of 5 CV folds:
1. Compute J(L) on the training fold only
2. Predict AUROC_bound = Φ(√J / 2) before seeing the test fold
3. Run the probe on the test fold → get actual AUROC
4. Compare prediction to measurement

**What it proves:**
The Fisher certificate is useful at test time — it predicts probe AUROC from geometry alone, before any probe is trained on the test distribution. This is the core claim: **reliability is predictable before measurement.**

**Key metrics:**
- Mean absolute error: |AUROC_bound - AUROC_actual|
- Calibration: does the bound consistently over- or under-predict?
- Layer selection: does the layer with highest J(L) on the training fold match the layer selected by CV probe?

**Key outputs:**
- `results/logs/03_certificate_validation.json` — fold-by-fold predictions vs actuals
- `results/plots/03_certificate_calibration.png` — predicted vs actual AUROC (45° calibration line)

---

### Exp 04 — Architecture Transfer: Mamba SSM
**Cost: $0 (Kaggle T4 free tier) | Hardware: Kaggle T4 GPU | Time: ~3–4 hours**

**The architecture transfer test.** This is the new compute in the project.

**Why Mamba?**
Mamba is a State Space Model (SSM) — no attention mechanism, no key-value cache, no transformer blocks. If the Fisher separability certificate predicts its probe AUROC correctly, it suggests the bound is architecture-agnostic, not just a transformer property.

**Model choice: state-spaces/mamba2-130m** (Mamba-2, not Mamba-1)
- Mamba-2 uses a different SSM variant (structured state-space duality, SSD)
- ACL 2025 (Mamba Knockout) shows Mamba-1 and Mamba-2 have different factual flow patterns — makes our architecture transfer test stronger
- 130M parameters — Kaggle T4 handles this easily
- Hidden dim 768, 24 layers
- Available on HuggingFace (no download restrictions)
- No instruction tuning needed — we label by ROUGE-L (+ judge labels)

**Pre-registration sequence (this order is CRITICAL):**
1. Run Exps 01–03 first (free, existing data)
2. Compute Fisher bound prediction for Mamba-130M (using Exp 01 model of J vs AUROC)
3. **Publish the predicted AUROC range** (add to RESULTS.md with commit timestamp — before running Exp 04)
4. Run Exp 04 on Kaggle
5. Report whether prediction holds

**What Exp 04 does:**
- Download TruthfulQA (400 questions — sufficient for proof-of-concept)
- Generate responses with Mamba-130M
- Label with ROUGE-L (threshold 0.4, same as HaRP)
- Extract block outputs at each of 24 layers (analogue of transformer hidden states)
- Compute J(L) at each layer → AUROC_bound
- Run OOF linear probe at each layer → actual AUROC
- Compare prediction to measurement

**Key outputs:**
- `results/hidden_states/04_mamba_hidden_states.npz`
- `results/logs/04_mamba_transfer.json`
- `results/plots/04_mamba_fisher_curve.png`

**Depth fraction check:**
Does the best layer occur at ≈89% depth for Mamba? If L_best ≈ 0.89 × 24 ≈ 21, the depth-fraction hypothesis holds across architectures.

---

### Exp 05 — Depth Fraction Universality
**Cost: $0 | Hardware: CPU | Time: ~1 hour | Data: all prior outputs**

**What it does:**
Across all four models (GPT-2 117M, GPT-2 Medium 345M, Qwen 2.5 3B, Mamba 130M), plot:
- J(L) / J_max as a function of L / L_total (normalized depth)
- Actual probe AUROC as a function of L / L_total

**Key question:** Do all models peak at ≈89% normalized depth?

| Model | Architecture | Peak Layer | Peak Depth Fraction |
|---|---|---|---|
| GPT-2 117M | Transformer | L9 | 9/13 = 69% | ← different
| GPT-2 Medium 345M | Transformer | ~L21 | ~88% |
| Qwen 2.5 3B | Transformer | L32 | 32/36 = 89% |
| Mamba 130M | SSM | ? | ? (pre-register: ≈89%) |

Note: GPT-2 117M peaks at 69% — but the signal is null there (AUROC ~0.50). The 89% pattern may require sufficient scale to emerge. This is a finding, not a failure.

**Key outputs:**
- `results/plots/05_depth_fraction_overlay.png` — normalized depth vs Fisher ratio, all models

---

### Exp 06 — Boundary Condition: When Does the Certificate Fail?
**Cost: $0 | Hardware: CPU | Time: ~1 hour | Data: Exp 01–05 outputs**

**What it does:**
Systematically test the limits of the Fisher certificate:

1. **Null regime:** At GPT-2 117M scale, J is near zero. Does the bound correctly predict AUROC ≈ 0.50? (It should — this is the null boundary condition.)
2. **Weak-signal regime:** At 345M scale, J is small but nonzero. Does the bound correctly predict weak but above-chance AUROC?
3. **Strong-signal regime:** At 3B, J is large. Does the bound correctly predict governance-grade AUROC?
4. **OOD regime:** What happens if we compute J on the wrong layer (not the best layer)? Does the certificate degrade gracefully?

**Key outputs:**
- `results/logs/06_boundary_conditions.json`
- `results/plots/06_boundary_scatter.png` — Fisher ratio on x-axis, actual AUROC on y-axis, all layers all models

This plot is the project's central figure: it shows the mapping from geometry to predictability across models, architectures, and layers.

---

---

### Exp 08 — OT Certificate: Wasserstein Generalization of Fisher
**Cost: $0 | Hardware: CPU (~3–4 hours) | Data: existing hidden states**

**Direction B — the mathematical heart of the project.**

**Core identity to verify:**
```
J = W₂²(P_c, P_h)  in whitened space (Σ_w = I)
→ AUROC_bound = Φ(√J / 2) = Φ(W₂_whitened / 2)
```
Fisher is Optimal Transport in disguise — the Gaussian special case.

**What it does:**
1. Verifies identity J ≈ W₂_whitened² at 5 representative layers (relative error < 5%)
2. Computes four competing certificates at every layer:
   - Fisher J (existing)
   - Bures W₂ (Gaussian, unequal covariance — Bures metric)
   - Sliced W₂ (non-parametric, no Gaussianity assumption)
   - MMD² (kernel-based, distribution-free)
3. Measures Spearman correlation of each certificate vs probe AUROC across layers
4. Prints winner table: which certificate best predicts probe AUROC?

**Key question:** In early layers where distributions are non-Gaussian, does SW₂ predict probe AUROC better than J?

**Models:** Qwen 2.5 3B, GPT-2 Medium 345M, Mamba-2 (if Exp 04 ran)

---

### Exp 09 — Spectral Phase Transition: BBP Transition in Hidden-State Covariance
**Cost: $0 | Hardware: CPU (~2–3 hours) | Data: existing hidden states**

**Direction A — the RMT supporting theory.**

**Core hypothesis:** The emergence of the hallucination signal at layer ~89% depth is a Baik-Ben Arous-Péché (BBP) phase transition in the spectral structure of the hidden-state covariance.

Below the BBP threshold θ* = σ²√(d/n), the class-mean difference is buried in the noise bulk and no linear probe can detect it. Above the threshold, a spike eigenvalue emerges and probing works.

**What it does:**
1. Computes ESD at every layer for all models
2. Measures KL divergence from Marchenko-Pastur (departure from pure noise)
3. Tracks spike count and spike ratio above the bulk upper edge λ_+
4. Tests whether spectral KL, SW₂, and probe AUROC curves show a coincident phase transition

**Key prediction:**
- GPT-2 117M: KL ≈ 0 at all layers (no signal above bulk) → null AUROC
- Qwen 2.5 3B: sharp KL rise at L28–L32 → AUROC lifts from chance

---

### Exp 10 — Conformal Coverage Guarantee
**Cost: $0 | Hardware: CPU (~10 minutes) | Data: Exp 08 probe scores**

**Direction C — the application layer.**

**Core guarantee:**
```
P(hallucinated | detector ACCEPTS) ≤ α,  with probability ≥ 1 − δ
```
where ACCEPT = probe score ≤ conformal threshold τ.

**What it does:**
1. Split conformal calibration: calibrates τ at α=10%, δ=5% on held-out set
2. Mondrian CP per quadrant (epistemic, aleatoric, confident hallucination, confident correct)
3. Coverage vs α curve: shows the guarantee holds and where it softens
4. OOD shift simulation: demonstrates coverage degrades under distribution shift
5. Compares conformal τ to HaRP's governance α* = 0.15

**Key result:** The formal guarantee holds on i.i.d. TruthfulQA; breaks under simulated OOD shift — demarcating the scope conditions of the certificate.

---

## Cost Summary

| Experiment | What | Hardware | Cost |
|---|---|---|---|
| Exp 07 | LLM-as-Judge re-labeling | CPU / Ollama | **$0** |
| Exp 01 | Three-certificate analysis | CPU | **$0** |
| Exp 02 | Scale curve fit | CPU | **$0** |
| Exp 03 | Certificate validation (K-fold) | CPU | **$0** |
| Exp 04 | Mamba architecture transfer | Kaggle T4 (free) | **$0** |
| Exp 05 | Depth fraction universality | CPU | **$0** |
| Exp 06 | Boundary conditions | CPU | **$0** |
| Exp 08 | OT certificate (W₂ vs Fisher) | CPU | **$0** |
| Exp 09 | Spectral phase transition (BBP) | CPU | **$0** |
| Exp 10 | Conformal coverage guarantee | CPU | **$0** |
| **Exp 11** | **Qwen 2.5 controlled scale curve (0.5B + 1.5B extraction)** | **CPU / Kaggle T4** | **$0** |
| **TOTAL** | | | **$0** |

---

### Exp 11 — Controlled Scale Curve: Qwen 2.5 Family
**Cost: $0 | Hardware: CPU (~4–6 hours for 0.5B+1.5B) | New data: Qwen2.5-0.5B, 0.5B**

**Why this experiment exists:** The original scale curve (Exp 02) uses three heterogeneous models (GPT-2 117M, GPT-2 Medium 345M, Qwen 2.5 3B) — different architectures, tokenizers, training data. Scale is confounded with everything else. That is a trend observation, not a scaling law.

**The fix:** Use the Qwen 2.5 family exclusively. Same architecture, same tokenizer, same training data distribution, same RLHF procedure. Only scale changes.

| Model | Params | Status |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 494M | New extraction (CPU ~2h) |
| Qwen2.5-1.5B-Instruct | 1.54B | New extraction (CPU ~4h) |
| Qwen2.5-3B-Instruct | 3.09B | Existing HaRP data |
| Qwen2.5-7B-Instruct | 7B | Kaggle T4 if needed |

The GPT-2 and Mamba-2 results become **cross-family** validation points — not on the controlled curve.

Also tests: does the 89% depth fraction hold consistently within the Qwen 2.5 family?

**Novel contribution:** The first controlled within-family scaling curve for hallucination probe AUROC in the literature. Marks & Tegmark (COLM 2024) report LLaMA-2 7B/13B/70B probe accuracy but do not fit a curve. No paper fits AUROC vs. log(params) within a single controlled family.

---

## Pre-Registration Protocol

Pre-registration is what makes GUARDIAN meaningful and what separates this project from post-hoc rationalization.

**What to pre-register (after Exp 01–03, before Exp 04):**
1. Predicted AUROC for Mamba-130M (point estimate + 90% interval)
2. Predicted best-layer depth fraction for Mamba-130M (≈89% or different?)
3. Predicted Fisher ratio range at the best layer

**Where to pre-register:**
- Commit to `results/preregistration.md` in git with a timestamp (the commit hash serves as evidence of priority)
- Optionally: OSF (osf.io) pre-registration (free)

**Pass/fail criterion for architecture transfer:**
- The bound predicts AUROC within ±0.05 → certificate validates
- The bound misses by >0.10 → certificate fails; report why (non-Gaussian distributions? asymmetric covariance?)

---

## Data Flow

```
MECH-INT activations (GPT-2 117M)
    └── Exp 01 → J(L) curve, GPT-2 117M

HaRP hidden_states.npz (Qwen 2.5 3B, 700×37×2048)
    ├── Exp 01 → J(L) curve, Qwen 2.5 3B
    └── Exp 03 → K-fold certificate validation

HaRP 25_gpt2_medium_hidden_states.npz (345M)
    └── Exp 01 → J(L) curve, GPT-2 Medium

Exp 01 outputs (J curves, all models)
    ├── Exp 02 → Scale curve fit → 7B prediction
    ├── Exp 05 → Depth fraction overlay
    └── Exp 06 → Boundary conditions

Mamba-130M (Kaggle T4) [new data]
    └── Exp 04 → hidden_states, J(L), probe AUROC

All outputs → Exp 06 → Central figure (Fisher ratio vs AUROC, all models)
```

---

## Claim Ladder

| Claim | Experiment | Evidence required |
|---|---|---|
| Fisher ratio correlates with probe AUROC across layers | Exp 01 | Pearson r > 0.8 within each model |
| Fisher ratio predicts probe AUROC quantitatively | Exp 01 | AUROC_bound within ±0.05 of actual at best layer |
| Certificate predicts AUROC before seeing test data | Exp 03 | MAE < 0.05 across K-fold splits |
| Scale curve formalizes monotonic relationship | Exp 02 | Fit R² > 0.95 on log-log scale |
| 7B AUROC is predictable | Exp 02 | CI published before any 7B experiment |
| Certificate transfers to SSM architecture | Exp 04 | Prediction within ±0.05, published before experiment |
| ≈89% depth holds (or has a principled exception) | Exp 05 | Best-layer depth fraction reported for all 4 models |
| Certificate degrades gracefully at null-signal scale | Exp 06 | GPT-2 117M J predicts ≈0.50 correctly |

---

## Execution Order

```
Week 1 — Label defense + free experiments (existing data):
  Day 1: Run Exp 07 — LLM-as-Judge re-labeling (overnight Ollama job)
  Day 2: Run Exp 01 — Three-certificate analysis (uses judge labels automatically)
  Day 3: Run Exp 02 — Scale curve + depth-fraction-vs-scale curve
  Day 4: Run Exp 03 — Certificate validation (K-fold)
  Day 5: Fill preregistration.md from Exp 01–03 outputs. Commit with git (timestamp evidence).

Week 2 — Mamba-2 transfer + synthesis:
  Day 6: Kaggle T4 — Exp 04, Mamba-2 130M hidden-state extraction + probe
  Day 7: Exp 05 — Depth fraction overlay (all four models)
  Day 8: Exp 06 — Boundary conditions + central figure
  Day 9: Write RESULTS.md, update README
  Day 10: Streamlit app
```

---

## Deliverables

1. **Ten experiment scripts** — `experiments/01_*.py` through `10_*.py`
2. **`src/fisher.py`** — Fisher ratio computation (Ledoit-Wolf + PCA, causal variant)
3. **`src/wasserstein.py`** — OT certificates: Bures W₂, Sliced W₂, MMD, identity verification
4. **`src/spectral.py`** — RMT spectral analysis: Marchenko-Pastur, BBP threshold, ESD KL
5. **`src/conformal.py`** — Conformal prediction: split CP, Mondrian CP, OOD shift
6. **`src/scale_curve.py`** — Scale curve fitting and prediction
7. **`src/certificate.py`** — Certificate interface (compute J → predict AUROC)
8. **`src/lid.py`** — Local intrinsic dimension certificate
9. **`results/preregistration.md`** — Pre-registered Mamba prediction (committed before Exp 04)
10. **`RESULTS.md`** — Full numerical results, all 10 experiments
11. **`app.py`** — Streamlit dashboard (follows MECH-INT / HaRP pattern)
12. **Central figure:** SW₂ certificate vs actual AUROC, all layers, all models, all architectures

---

## What GEOM-PROOF Contributes to the Arc

MECH-INT: signal is real, causal, located at L8–L9
HaRP: signal converts to governance; token signals insufficient
**GEOM-PROOF: signal is formally bounded; reliability is predictable before measurement; bound transfers to SSM architecture**
JEPA-PROBE: signal is paradigm-agnostic
GUARDIAN: system generalizes with formal pre-registered prediction

The Fisher separability certificate is what lets GUARDIAN make a formal pre-registered prediction about Mistral 7B without any prior experiments on it. Without GEOM-PROOF, GUARDIAN's prediction is just intuition. With GEOM-PROOF, it's a bound.

---

## Open Design Decisions

These need to be resolved before starting Exp 01:

1. **PCA projection dimension k for Fisher computation.** Default: k=100. Should we tune this or fix it a priori? **Recommendation:** fix at k=100 (same as MECH-INT's 100/768 active dimensions — a principled choice with prior evidence).

2. **Label source for GPT-2 117M and GPT-2 Medium 345M.** MECH-INT uses Jaccard labels; HaRP uses ROUGE-L. These are different label sets for the same TruthfulQA questions. **Recommendation:** use ROUGE-L throughout for comparability (re-label MECH-INT hidden states with ROUGE-L threshold = 0.4, using the HaRP labeling code).

3. **Mamba hidden-state definition.** Mamba block outputs (after normalization) are the direct analogue of transformer layer outputs. We use these, not the SSM recurrent state. **Fixed.**

4. **Pre-registration platform.** Git commit timestamp (free, immediate) is sufficient. OSF optional. **Recommendation:** git commit for now; add OSF if submitting to a venue.
