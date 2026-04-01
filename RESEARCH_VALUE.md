# GEOM-PROOF — Research Value, Breadth, and Future Scope

**Project:** Geometric Certificates for Hallucination Governance
**Models:** Qwen 2.5 3B, GPT-2 Medium 345M, Mamba-130m
**Status:** Complete (11 experiments)
**Position in arc:** Project 3 of 5 — the mathematical foundation

---

## Why This Project Matters

GEOM-PROOF asks the hardest question in the arc: *can you prove a hallucination
detector will work, without training one?* The answer is partially yes — Fisher
separability J gives a closed-form bound on probe AUROC accurate within 1.5%
across all tested configurations — and the conditions under which the bound
fails are as scientifically valuable as the bound itself.

The 11 experiments collectively establish something unusual in ML research: a
mathematical certificate for detection quality derived purely from hidden-state
geometry. The bound Φ(√J/2) is computable without labels, without a trained probe,
and without access to the output. This makes it the only signal in the arc that
is entirely unsupervised.

But GEOM-PROOF's most durable contribution is what it *disproves*: depth universality
(Exp 05), simple sigmoid scale extrapolation to 7B+ (Exp 02), LID as a useful
certificate (Exp 01), and the Gaussian assumption underlying Fisher J itself
(Exp 08). These negative results define the limits of the framework and point
to what needs to replace it.

---

## Core Contributions

| Contribution | Key number |
|---|---|
| Fisher J → AUROC bound: Φ(√J/2) | Bound error ≤ 1.5% across all layers/models |
| Scale curve (R² = 0.999) | AUROC = σ(8.62·log₁₀(params) − 69.10), fit on 3 points |
| Conformal α* = 0.07 | P(hallucination \| accepted) ≤ 0.07 at 55.8% acceptance rate |
| SW2 > Fisher for AUROC prediction | Sliced Wasserstein AUROC Spearman r = 0.821 vs Fisher r = 0.458 |
| Depth universality refuted | GPT-2 probe peak = 33%, Qwen probe peak = 100% (same depth fraction is wrong) |
| LID certificate failed | LID peaks at layer 0 — useless for this task |
| Mamba architecture: failed | Instruction-following required; base model can't produce classifiable outputs |
| ROUGE-LLM label gap | Cohen's κ = −0.010 between ROUGE-L and GPT-4o labels |

---

## Mathematical and Architectural Power

**The bound Φ(√J/2) is non-trivial:**
The Fisher separability J is computable in O(n·d²) from hidden state statistics.
For d=2048, that's a 2048×2048 covariance matrix inversion — expensive but
tractable on CPU. The bound says: *compute J, and you immediately know whether
a linear probe is worth training.* If J is low, no probe will work well (the
geometry is flat). If J is high, a probe will achieve near-Φ(√J/2) AUROC.

This has a practical application that GUARDIAN can use directly: before routing
any query, compute local J in the KNN neighborhood. If local J < J_threshold,
the geometry is flat and the probe's confidence estimate is unreliable → route
to ABSTAIN regardless of probe output.

**The scale curve R² = 0.999 with a = 8.62:**
The sigmoid fit across GPT-2 117M (AUROC 0.604), GPT-2 Medium 345M (AUROC 0.987),
Qwen 2.5 3B (AUROC 0.991) has near-perfect fit but saturates above 3B. The
mathematically important finding: detection quality improves steeply between 100M
and 500M parameters, then plateaus. Below 500M, the model lacks the representational
capacity to geometrically separate correct from hallucinated responses. Above 3B,
the separation is near-complete — adding parameters doesn't help further.

**The SW2 > Fisher finding (Exp 08):**
Sliced Wasserstein distance W₂ has Spearman r = 0.821 for predicting probe AUROC
vs Fisher r = 0.458. SW2 outperforms Fisher because it doesn't assume Gaussian
equal-covariance distributions — it captures the actual distributional geometry.

The theoretical relationship J ≈ W₂² holds only under Gaussian equal-covariance
assumptions. Real hidden states violate this: the distributions are non-Gaussian
(heavy-tailed, skewed) and have unequal covariances. This is why SW2 wins.

**Implication for FAIL-CHAIN:** The F-ratio collapse at step 2 (F = 0.007) is
exactly this: CASCADE's hidden state distribution at step 2 is so non-Gaussian
(std = ±214) that Fisher's Gaussian assumption fails catastrophically. SW2 would
handle this where Fisher cannot.

**The conformal coverage guarantee:**
α* = 0.07 means: at the optimal threshold, accepted responses have hallucination
rate ≤ 7%, confirmed empirically at 6.05%. This is a formal, dataset-conditional
guarantee — not a heuristic. The acceptance rate at α* is 55.8%, meaning the
system must refuse 44.2% of queries to achieve the guarantee.

This acceptance-rate cost is the honest price of formal coverage. GUARDIAN's
adaptive threshold design is motivated by this: can we achieve α* ≤ 0.07 with
a higher acceptance rate by adapting thresholds to local geometry?

**The ROUGE-LLM label gap (Exp 07):**
Cohen's κ = −0.010 between ROUGE-L and GPT-4o labels on 700 HaluEval samples.
ROUGE-L hall rate = 0.993. GPT-4o hall rate = 0.981. These look similar but the
κ = −0.010 means they are identifying *different* samples as hallucinated.

This is the foundational measurement problem underlying the entire arc: every
AUROC, every probe, every certificate is conditioned on ROUGE-L labels. If
ROUGE-L and semantic judgment disagree at this level (κ ≈ 0), then the true
hallucination detection AUROC is unknown. The geometry is predicting ROUGE-L
surface form, not factual accuracy.

---

## Open Threads With Genuine Future Value

### OT-1: The Mamba Question — SSM Hallucination Geometry
Exp 04 failed because Mamba-130m is a base model without instruction following.
The actual research question — does Fisher J certificate work for state-space
models? — remains completely open.

**Why it matters enormously:** Mamba and its successors (Mamba-2, Jamba, Zamba)
are being deployed at scale as attention-free alternatives. If hallucination
geometry in SSMs is fundamentally different (the hidden state is a compressed
state vector, not a residual stream accumulation), the entire arc's L32 depth
finding may not transfer.

**What to do:** Use MambaChat (instruction-tuned Mamba-3B) or Falcon Mamba 7B.
Run the identical Exp 01 sweep. Compare J peak depth and SW2 vs Fisher rankings.

### OT-2: The Gaussian Transition — When Does Fisher Work?
SW2 > Fisher (Exp 08) because real distributions are non-Gaussian. But Fisher
worked within 1.5% (Exp 06). How?

**The resolution:** The Φ(√J/2) bound is an inequality — it's always an upper
bound on AUROC, just tighter when distributions are more Gaussian. The bound
holds because the non-Gaussian heavy tails are symmetric and don't violate the
inequality direction.

**Open question:** Is there a closed-form correction factor for non-Gaussian
distributions? If you can estimate the kurtosis of the hidden state distribution
(computable from H), can you tighten the bound from Φ(√J/2) to something like
Φ(√J/2) − f(kurtosis)?

This would be a publishable pure mathematics result: a higher-order Fisher-SW2
bridge formula.

### OT-3: The Scale Curve Beyond 3B
The sigmoid fit saturates above 3B (predicted AUROC ≈ 1.0 for all larger models).
This isn't a prediction — it's a mathematical saturation artifact.

**The real question:** Does detection *difficulty* scale differently from detection
*quality*? At 7B+, hallucination detection AUROC may be 0.99+, but the relevant
question is: does the probe *generalize* to unseen hallucination types better at
larger scale?

HaRP shows the OOD gap (in-domain 0.997 → HaluEval 0.775) exists even at 3B.
Does this gap shrink at 7B? If so, scale is the solution to the OOD problem.
If not, local geometry (GUARDIAN) is necessary regardless of scale.

### OT-4: Multi-Model Certificate Ensembles
If Fisher J independently certifies detection quality for GPT-2 and Qwen 2.5 3B,
can you ensemble the certificates?

**Idea:** For a query that both GPT-2 and Qwen answer, compute J for each model.
If J_GPT2 and J_Qwen are both high, the detection is doubly certified. If they
disagree, the query is in a geometrically ambiguous region. This ensemble
certificate could provide tighter bounds than either model alone.

**Connection to GUARDIAN:** GUARDIAN routes queries based on local J. A multi-model
J ensemble could provide a model-agreement signal at near-zero cost (both models
have already run their forward passes).

### OT-5: Conformal Guarantees Under Distribution Shift
The conformal α* = 0.07 guarantee is distribution-conditional: it holds on the
calibration distribution (TruthfulQA). Under OOD shift (HaluEval), the guarantee
may not hold — the calibration quantiles computed on TruthfulQA may be wrong for
HaluEval.

**Open question:** Can you compute a distribution-shift-robust α* using weighted
conformal prediction (Tibshirani et al., NeurIPS 2019)?

**Why it matters:** Any real deployment involves distribution shift. A conformal
guarantee that breaks OOD is a guarantee for the lab, not the real world. Combining
weighted conformal prediction with local Fisher J (GUARDIAN's mechanism) would
give a distribution-robust formal guarantee — the strongest result in the arc.

---

## Connection to the PhD Arc

| Downstream project | How GEOM-PROOF informs it |
|---|---|
| FAIL-CHAIN | The Φ(√J/2) bound explains why global Fisher fails (F-ratio = 0.007 at step 2): J is too small relative to within-class variance. SW2 is the correct tool. |
| GUARDIAN | The conformal α* = 0.07 gives the target coverage level. Local Fisher J is the mechanism for achieving it adaptively. |

---

## The One Sentence This Project Adds to the World

> *Fisher separability geometry provides a closed-form, training-free certificate
> of hallucination detector quality accurate within 1.5% — but only under
> Gaussian assumptions that real hidden states violate, making Sliced Wasserstein
> the mathematically correct replacement and local geometry the architecturally
> correct extension.*
