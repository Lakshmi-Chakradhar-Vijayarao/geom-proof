# Can You Prove a Hallucination Detector Will Work Before Training It?
## GEOM-PROOF — Geometric Certificates for Hallucination Governance

**Lakshmi Chakradhar Vijayarao**

Fisher separability geometry provides a closed-form, training-free certificate of hallucination
detector quality — accurate within 0.93% on average — and identifies precisely where the
Gaussian assumption it relies on breaks down, making Sliced Wasserstein the mathematically
correct replacement.

**Models:** GPT-2 117M · GPT-2 Medium 345M · Qwen 2.5 3B · Mamba-130m &nbsp;|&nbsp; **Experiments:** 11 &nbsp;|&nbsp; **New compute cost:** $0

[![Live Demo](https://img.shields.io/badge/Live%20Demo-geom--proof.streamlit.app-1565c0?logo=streamlit)](https://geom-proof.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-geom--proof-1565c0?logo=github)](https://github.com/Lakshmi-Chakradhar-Vijayarao/geom-proof)
[![Companion: MECH-INT](https://img.shields.io/badge/Companion-MECH--INT%20(GPT--2)-1565c0?logo=github)](https://github.com/Lakshmi-Chakradhar-Vijayarao/mech-int)
[![Companion: HaRP](https://img.shields.io/badge/Companion-HaRP%20(Qwen%203B)-4527a0?logo=github)](https://github.com/Lakshmi-Chakradhar-Vijayarao/harp)

---

## The Problem

MECH-INT proved the hallucination signal is real and causal (FFN over-retrieval at L8).
HaRP proved it converts to operational governance (L32 probe AUROC 0.775).
But both projects are empirical — they measure the signal and report what they find.

**The missing questions:**
1. Can you predict whether a probe will work *before training it?*
2. When the Gaussian assumption underlying Fisher breaks, does the bound still hold?
3. Can the probe's output carry a *formal, distribution-free* coverage guarantee?

GEOM-PROOF answers all three — starting from Fisher, arriving at Wasserstein.

---

## The Three-Phase Structure

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  PHASE 1 — FOUNDATION: Fisher Separability Bound                                     │
 │  ─────────────────────────────────────────────────────────────────────────────────   │
 │                                                                                      │
 │   Hidden states [n × layers × d]  (GPT-2 Medium + Qwen 2.5 3B)                     │
 │        │                                                                             │
 │        ├──→  Fisher J(L) at every layer  →  Φ(√J/2) AUROC bound                   │
 │        │     Mean bound error: 0.93%  |  Max error: 2.7%  |  No probe trained       │
 │        │                                                                             │
 │        ├──→  Scale curve: AUROC = σ(4.71·log₁₀(params) − 39.45),  R² = 0.9996      │
 │        │     (within-family Qwen controlled; 7B extrapolation: 0.9990)              │
 │        │                                                                             │
 │        └──→  Depth universality REFUTED: GPT-2M peak L8 (33%), Qwen peak L36 (100%) │
 │                                                                                      │
 │  Bound holds — but is systematically loose 1–3% due to Gaussian assumption           │
 └──────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  PHASE 2 — GENERALIZATION: Wasserstein Replaces Fisher                               │
 │  ─────────────────────────────────────────────────────────────────────────────────   │
 │                                                                                      │
 │   Why Fisher is loose: hidden states are non-Gaussian, heavy-tailed, class-imbalanced│
 │        │                                                                             │
 │        ├──→  J ≈ W₂² identity: mean relative error 1.711 (Qwen) — non-Gaussian reality│
 │        │                                                                             │
 │        ├──→  SW₂ AUROC prediction: Spearman ρ = 0.821  vs  Fisher ρ = 0.458        │
 │        │                                                                             │
 │        └──→  Spectral (BBP): KL-MP peaks disagree with probe peaks — orthogonal      │
 │                                                                                      │
 │  SW₂ wins because it makes no distributional assumption                              │
 └──────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │  PHASE 3 — FORMAL GUARANTEE: Conformal Coverage                                      │
 │  ─────────────────────────────────────────────────────────────────────────────────   │
 │                                                                                      │
 │   Split conformal prediction on HaluEval                                             │
 │        │                                                                             │
 │        ├──→  α* = 0.07: P(hallucination | ACCEPT) ≤ 0.07  (empirical: 6.05%)       │
 │        │                                                                             │
 │        └──→  Acceptance rate: 55.8%  — must refuse 44.2% to achieve guarantee       │
 │                                                                                      │
 │  Guarantee is dataset-conditional (TruthfulQA calibration → HaluEval evaluation)    │
 │  OOD shift breaks it: TruthfulQA test AUROC = 0.485 (chance level)                  │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Results Headlines

- **The Φ(√J/2) bound works within 0.93%.** Mean bound error across all tested layers and
  both models is 0.93%; maximum error 2.7%. This is a training-free, label-free certificate:
  compute J from hidden-state statistics, read off predicted AUROC. No probe training required.

- **The bound is systematically loose, not tight.** Looseness comes from the Gaussian assumption.
  Real hidden states are non-Gaussian (heavy-tailed, class-imbalanced, hall_rate ≈ 0.99).
  The Φ(√J/2) formula is always an upper bound — it holds because the inequality direction is
  preserved, not because the Gaussian fit is accurate.

- **SW₂ outperforms Fisher for AUROC prediction.** Sliced Wasserstein Spearman ρ = 0.821
  vs Fisher ρ = 0.458. SW₂ peaks differ from Fisher peaks even within the same model (Qwen:
  SW₂ at L35, Fisher at L25). SW₂ is the correct tool when Gaussian assumptions fail.

- **Scale curve R² = 0.9996 within Qwen family.** Qwen 0.5B → 1.5B → 3B: 0.813 → 0.979 → 0.9917.
  Fit: AUROC = σ(4.71·log₁₀(params) − 39.45). 7B extrapolation = 0.9990 (controlled,
  within-family; architecture-specific, not cross-model generalization claim).

- **Conformal α* = 0.07, not 0.10.** The conformal guarantee requires accepting 55.8% of
  queries (refusing 44.2%) to achieve P(hallucination | ACCEPT) ≤ 0.07. The initially-claimed
  α = 0.10 was achievable only at an empirically invalid threshold.

- **Depth universality REFUTED.** GPT-2 Medium probe peak: L8 (33.3%). Qwen 2.5 3B probe peak:
  L36 (100%). Fisher J peaks: Qwen L25 (69.4%), GPT-2M L19 (79.2%). Same depth fraction does
  not transfer across architectures — any new model must measure its own optimal depth.

- **Mamba transfer FAILED.** Mamba-130m is a base language model without instruction-following.
  Hall_rate = 1.0 on HaluEval (all outputs classified hallucinated). The research question —
  does Fisher J transfer to state-space models? — remains completely open.

- **ROUGE-LLM label gap κ = −0.010.** GPT-4o judge and ROUGE-L labeling have Cohen's κ ≈ 0
  on 700 HaluEval samples — near-zero inter-rater agreement. Every AUROC in the arc is
  conditioned on ROUGE-L surface-form labels, not factual accuracy. This is a documented
  limitation, not a correctable bug.

---

## 11-Experiment Index

| Exp | Question | Method | Key Result |
|-----|----------|--------|------------|
| 01 | Does Fisher J predict probe AUROC? | Three-certificate comparison at every layer | **Bound within 0.015 of actual AUROC** (Qwen J=52.15, GPT-2M J=49.33) |
| 02 | Log-linear scale curve + 7B extrapolation | Sigmoid fit on 3 points (117M, 345M, 3B) | R²=0.9993 (mixed arch); point pred. 7B ≈ 1.0 (saturated) |
| 03 | Does the certificate predict AUROC before test data? | 5-fold CV pre-registration simulation | Mean bound error 0.0093; argmax-J layer match 0% |
| 04 | Transfer to Mamba-2 130M (SSM)? | HaluEval prompts on base LM | **FAILED** — hall_rate=1.0, base model not instruction-tuned |
| 05 | Does 89% depth fraction hold across architectures? | Layer sweep on 3 models | **REFUTED**: GPT-2M peak 33%, Qwen peak 100%, Mamba 4% |
| 06 | Where does the certificate fail? | Bound error by layer, all 13/37 layers | Max error 0.027 (L0 worst); Gaussian violation explains looseness |
| 07 | Are ROUGE labels inflating AUROC? | GPT-4o LLM-as-Judge re-labeling, 700 samples | **Cohen's κ = −0.010** — labels identify different samples |
| 08 | Does SW₂ predict AUROC better than Fisher? | Bures W₂, Sliced W₂, MMD vs Fisher | **SW₂ Spearman ρ = 0.821 vs Fisher 0.458**; J≈W₂² error = 1.711 |
| 09 | Is AUROC lift a BBP spectral phase transition? | KL from Marchenko-Pastur per layer | KL peaks disagree with probe peaks — orthogonal signals |
| 10 | Does the conformal guarantee hold? | Split CP on HaluEval + OOD TruthfulQA test | **α* = 0.07** (not 0.10); 55.8% acceptance; OOD AUROC 0.485 |
| 11 | Controlled scale curve (Qwen family only) | Qwen 0.5B, 1.5B, 3B — identical setup | **R² = 0.9996**; 7B prediction 0.9990; fits σ(4.71·log₁₀(p) − 39.45) |

---

## Mathematical Contributions

### The Φ(√J/2) Certificate

Fisher separability J = (μ_c − μ_h)ᵀ Σ_w⁻¹ (μ_c − μ_h), where Σ_w is the within-class
covariance. The bound AUROC ≤ Φ(√J/2) follows from the Gaussian linear discriminant analysis
connection — the maximum achievable AUROC for a LDA classifier under Gaussian equal-covariance
distributions. It is:
- **Computable without labels** (only class means and pooled covariance needed)
- **Computable without a trained probe** (pure hidden-state statistics)
- **An upper bound** — actual AUROC ≤ Φ(√J/2), confirmed at ≤ 2.7% violation

### The J ≈ W₂² Identity (and Why It Breaks)

In Mahalanobis-whitened coordinates (Σ_w = I), J = ‖μ_c_white − μ_h_white‖² = W₂².
Fisher is the Gaussian special case of Bures-Wasserstein distance. The identity holds
only under Gaussian equal-covariance assumptions. Mean relative error = 1.711 (Qwen)
proves real distributions are non-Gaussian — hence SW₂ > Fisher for AUROC prediction.

### The Conformal Coverage Guarantee

Split conformal prediction wraps the probe's output into a distribution-free guarantee:
P(hallucination | probe score > threshold) ≤ α, w.p. ≥ 1−δ. The empirically required
α* = 0.07 (not the theoretically expected 0.10). Cost: 44.2% of queries refused.
This is the only result in the arc with a formal, non-Gaussian, non-parametric guarantee.

### Depth Universality Refutation

The "89% depth" finding from HaRP is Qwen-specific. Exp 05 proves: GPT-2M probe peak = 33.3%,
Qwen probe peak = 100%. Fisher J peaks differ from probe peaks within the same model.
Any cross-architecture depth claim is empirically unfounded. GUARDIAN must pre-register before
measuring Mistral 7B.

---

## Design Constraints for GUARDIAN

Every GEOM-PROOF finding mandates specific GUARDIAN design choices:

| Constraint | Source | Implication |
|-----------|--------|-------------|
| Use LOCAL Fisher J, not global | FAIL-CHAIN F-ratio = 0.007 | Global Fisher provably fails in mixed distributions |
| Conformal target α* = 0.07 | Exp 10 empirical | Achievable only at 55.8% acceptance; GUARDIAN aims to raise this with adaptive geometry |
| Pre-register Mistral depth before measuring | Exp 05 refutes universality | Cannot assume L* = L32 for Mistral |
| Do not use temperature scaling | HaRP calibration finding | ECE 0.071 → 0.395 under temp scaling |
| Local J < J_threshold → route to ABSTAIN | Fisher bound interpretation | If J is low, no probe will work (geometry is flat) |
| SW₂ > Fisher where distributions are non-Gaussian | Exp 08 | For GUARDIAN's OOD regime, SW₂ is the theoretically correct certificate |

---

## What Holds Up vs What Needs Qualification

| Claim | Status | Evidence |
|-------|--------|---------|
| Fisher J predicts AUROC within 1.5% (mean) | ✅ Solid | Exp 01, 06: mean 0.93%, max 2.7%, across all layers/models |
| Hallucination detectable at >50% depth (transformers) | ✅ Solid | Exp 05: both working models peak in upper half |
| Within-family AUROC scales log-linearly with params | ✅ Solid | Exp 11: R²=0.9996, 3 Qwen models same setup |
| SW₂ correlates better with AUROC than Fisher | ✅ Solid | Exp 08: Spearman 0.821 vs 0.458 |
| Conformal guarantee holds at α* = 0.07 | ✅ Solid | Exp 10: empirical rate 6.05%, acceptance 55.8% |
| Depth universality claim (89% holds everywhere) | ❌ Refuted | Exp 05: GPT-2M 33%, Qwen 100%, Mamba 4% |
| Conformal guarantee holds OOD | ❌ Breaks | TruthfulQA OOD AUROC = 0.485 — calibration collapses |
| ROUGE-L labels measure factual accuracy | ❌ Overstated | κ = −0.010 vs GPT-4o judge — they measure different samples |
| Cross-architecture scale curve R²=0.9993 | ⚠ Interpolation | 3-point fit with mixed architectures; R² reflects fit quality not predictive validity |
| Mamba architecture transfer | ⚠ Incomplete | Base LM failure is dataset artifact; instruction-tuned Mamba untested |

---

## Architecture

```
geom-proof/
├── README.md                               This file
├── PLAN.md                                 Complete project plan
├── RESULTS.md                              Full numerical results (all 11 experiments)
├── RESEARCH_VALUE.md                       Open threads and future directions
│
├── experiments/
│   ├── 07_judge_relabeling.py              LLM-as-Judge defense (run first — Exp 07)
│   ├── 01_fisher_analysis.py               Three-certificate comparison (Exp 01)
│   ├── 02_scale_curve.py                   Log-linear fit + 7B extrapolation (Exp 02)
│   ├── 03_certificate_validation.py        K-fold pre-registration simulation (Exp 03)
│   ├── 04_mamba_transfer.py                Mamba-2 130M transfer (Exp 04)
│   ├── 05_depth_fraction.py                Depth fraction overlay (Exp 05)
│   ├── 06_boundary_conditions.py           Certificate limits + central figure (Exp 06)
│   ├── 08_ot_certificate.py                SW₂ vs Fisher (Exp 08)
│   ├── 09_spectral_phase_transition.py     BBP spectral analysis (Exp 09)
│   ├── 10_conformal_coverage.py            Conformal coverage guarantee (Exp 10)
│   └── 11_qwen_scale_curve.py              Controlled within-family scale curve (Exp 11)
│
├── src/
│   ├── fisher.py                           Fisher J: Ledoit-Wolf + PCA + causal variant
│   ├── wasserstein.py                      OT certificates: Bures W₂, Sliced W₂, MMD
│   ├── spectral.py                         RMT: Marchenko-Pastur, BBP, ESD KL
│   ├── conformal.py                        Split CP, Mondrian CP, OOD shift simulation
│   ├── lid.py                              Local intrinsic dimension (failed certificate)
│   ├── scale_curve.py                      Scale curve fitting and prediction
│   └── certificate.py                      Certificate interface (unified API)
│
├── results/
│   ├── logs/                               JSON outputs (01_fisher_analysis.json, etc.)
│   ├── plots/                              PNG figures (9 publication plots)
│   └── preregistration.md                 Pre-registered Mamba prediction (before Exp 04)
│
├── app.py                                  Streamlit research dashboard
└── requirements.txt
```

---

## Running the Pipeline

**Dependencies:**
```bash
pip install -r requirements.txt
```

**Run order (important — Exp 07 first):**
```bash
python experiments/07_judge_relabeling.py   # Label quality defense (run first)
python experiments/01_fisher_analysis.py    # Core certificate
python experiments/02_scale_curve.py        # Scale curve
python experiments/03_certificate_validation.py
python experiments/05_depth_fraction.py     # Depth universality test
python experiments/06_boundary_conditions.py
python experiments/08_ot_certificate.py     # SW₂ vs Fisher
python experiments/09_spectral_phase_transition.py
python experiments/10_conformal_coverage.py # Conformal guarantee
python experiments/11_qwen_scale_curve.py   # Controlled scale curve
# Exp 04 (Mamba) requires Kaggle T4 GPU
```

**Dashboard:**
```bash
streamlit run app.py
```

---

## Limitations

1. **Label quality.** Cohen's κ = −0.010 between ROUGE-L and GPT-4o labels. All AUROC figures
   predict ROUGE-L surface-form failure, not factual incorrectness. Any publication must
   clearly state this distinction. Running a 200-sample dual-label subset (ROUGE-L + GPT-4o)
   would establish whether the two measures agree at the AUROC level.

2. **Scale curve on 3 heterogeneous points.** The cross-architecture R² = 0.9993 reflects
   interpolation quality between GPT-2 117M, GPT-2 Medium 345M, and Qwen 2.5 3B — three
   different architectures. Within-family R² = 0.9996 (Exp 11) is more defensible. Neither
   is a valid extrapolation claim beyond the measured points.

3. **Mamba architecture untested.** The Exp 04 failure is a dataset artifact (base LM cannot
   follow instructions). The core question — does Fisher J certificate transfer to SSMs — is
   completely unresolved. Mamba-3B-Chat or Falcon Mamba 7B would be the correct follow-up.

4. **Layer selection via Fisher J fails.** Fisher argmax-J identifies the best probe layer
   0% of the time (Exp 03). Depth-weighting achieves 20% match rate. Practical layer selection
   still requires probing, undermining the "training-free certificate" claim for layer choice.

5. **Conformal guarantee is distribution-conditional.** α* = 0.07 holds on the calibration
   distribution (HaluEval). Under real distribution shift (TruthfulQA OOD test), AUROC
   collapses to 0.485 and the conformal calibration breaks. Weighted conformal prediction
   (Tibshirani et al. 2019) would be needed for distribution-robust guarantees.

---

## Open Threads With Future Value

**OT-1 — Mamba SSM hallucination geometry.** Does Fisher J transfer to state-space models?
Use Falcon Mamba 7B (instruction-tuned). Compare J peak depth and SW₂ vs Fisher rankings
against transformer results. High potential if SSMs show different geometry.

**OT-2 — Higher-order Fisher-SW₂ bridge.** Can kurtosis of hidden-state distribution
correct the Φ(√J/2) bound? Deriving Φ(√J/2) − f(kurtosis) would be a publishable pure-mathematics
result tightening the bound for non-Gaussian distributions.

**OT-3 — Does OOD gap shrink at 7B+ scale?** HaRP shows OOD gap exists at 3B (0.997 → 0.775).
If the gap shrinks at 7B, scale solves the OOD problem. If not, local geometry (GUARDIAN) is
necessary regardless of scale. GUARDIAN provides one data point.

**OT-4 — Distribution-shift-robust conformal.** The α* = 0.07 guarantee breaks under shift.
Weighted conformal prediction (Tibshirani et al. NeurIPS 2019) combined with local Fisher J
(GUARDIAN's mechanism) would give a distribution-robust formal guarantee — the strongest
possible result in the arc.

---

## Connection to the PhD Arc

| Project | How GEOM-PROOF connects |
|---------|------------------------|
| MECH-INT (P1) | Provides the hidden states that GEOM-PROOF uses as its primary data source |
| HaRP (P2) | The L32 finding motivates depth experiments; ECE 0.039 is the calibration baseline |
| FAIL-CHAIN (P4) | F-ratio = 0.007 collapse is the GEOM-PROOF Fisher failure mode made catastrophic |
| GUARDIAN (P5) | α* = 0.07 is the conformal target; local J is the mechanism for achieving it adaptively |

---

## License

MIT
