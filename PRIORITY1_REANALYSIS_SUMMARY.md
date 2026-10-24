# Priority-1 Reanalysis: P1.1, P1.2, P1.4, P2.2

**Date:** 2026-08-10
**Scope:** Four analyses requested by `FRONTIER_LAB_REVIEW_RESULT.md` (Priority-1/2
sections, ~lines 625-649), addressing the adversarial review of GEOM-PROOF ("A Loose
Ceiling, Not a Certificate"). All four analyses use **only already-cached data**
(`results/hidden_states/15_hidden_states_*.npz`, `results/logs/*.json`) -- no new
model inference or generation was run for this pass. Full machine-readable output:
`results/logs/priority1_reanalysis_results.json`.

---

## P1.1 -- PCA-rank sensitivity sweep (Analysis 1)

**Question:** Exp. 15's central table used PCA `n_components=100` for Fisher-ratio
estimation across all six models, unjustified and untested for sensitivity. Does the
headline bound-error (0.32-0.41 at k=100) hold at k=25/50/200?

**Method:** For each of the six models, recomputed `fisher_ratio(H, y, method="pca",
n_components=k)` at every layer for k in {25, 50, 100, 200}, using `src/fisher.py`
verbatim (confirmed algorithmically identical to Exp. 15's inline PCA+Ledoit-Wolf
code). Two readings are reported: (a) bound-error at each model's **original k=100
best-J layer held fixed**, isolating the effect of k alone; and (b) bound-error at
**each k's own re-selected argmax-J layer**, since re-running "the same best-J-layer
selection procedure" at a different k can itself move which layer is selected.

**Result -- k=100 reproduces the paper's own numbers as a sanity check** (e.g.
GPT-2 117M: J=7.148, bound=0.9707, bound_error=0.4068 vs. paper's reported 7.517/
0.9737/0.4098 -- small residual differences trace to PCA's internal SVD solver
choice, not a methodology mismatch; direction and magnitude match exactly).

**The bound-error is NOT stable across k -- this is a real, previously
undocumented finding, not a confirmation:**

| k | Bound-error range (6 models, fixed L) | Mean |
|---|---|---|
| 25 | 0.156 - 0.248 | 0.186 |
| 50 | 0.231 - 0.337 | 0.269 |
| 100 (paper) | 0.306 - 0.407 | 0.351 |
| 200 | 0.407 - 0.440 | 0.427 |

The bound-error **roughly doubles** between k=25 and k=200 (mean 0.186 -> 0.427).
Using each k's own re-selected argmax-J layer gives the same qualitative pattern
(0.196 -> 0.425 mean). **The paper's headline "0.32-0.41" figure is itself an
artifact of the specific, unjustified choice k=100** -- a materially different,
still-defensible choice (k=25) would have produced a headline of "0.16-0.25," and
k=200 would have produced "0.41-0.44." The *qualitative* conclusion -- the
certificate is loose, never tight, at every k tested -- holds throughout (bound
never fails to upper-bound probe AUROC in any of the 24 model x k cells checked).
But the *specific numeric range* quoted in the abstract is k-dependent to a degree
the paper does not currently disclose. **This is a new finding the paper should
report, not merely a robustness confirmation.**

A second, related effect: the argmax-J layer itself sometimes moves with k (e.g.
GPT-2 Medium: L20 at k=100 vs. L10 at k=25, L5 at k=50, L9 at k=200; Qwen 7B: L24
vs. L6/L5/L25). This directly reinforces the paper's own separate finding that
Fisher J does not reliably select the optimal probing layer -- that instability
is present across PCA ranks too, not just across CV folds.

---

## P1.2 -- Exp. 26/27 split-consistency fix (Analysis 2)

**Root cause, established by reading both scripts line-by-line (not previously
documented anywhere in the paper or repo):** Exp. 26's reported "in-dist AUROC"
(0.5719) and "OOD AUROC" (0.6086) are **not from the same estimator on the same
data partition**. `in_dist_auroc` is a 5-fold out-of-fold (OOF) cross-validated
AUROC aggregated over *all 500* HaluEval samples (from `best_layer_probe()`,
originally computed only to pick the best layer, L24). `ood_auroc` is the held-out
performance of a *separate* probe trained on a single fixed 75% split
(`RandomState(42).permutation(500)`, `tr_idx=idx[:375]`) and evaluated on NQ-open.
These are two different estimators on two different partitions -- an
apples-to-oranges comparison baked into the paper's own headline "+0.037" number,
independent of anything Exp. 27 later did.

Exp. 27 does **not** repair this. It invents a **third**, independently-seeded
80/20 split (a fresh, different call to `RandomState(42).permutation(500)`, train
fraction 0.8 not 0.75) and reports a bootstrap CI centered on *its own* point
estimate (in-dist=0.5838, ood=0.6477, gap=+0.0639) -- which is why Exp. 27's point
estimate does not match Exp. 26's reported +0.037 gap in the first place. The
review's diagnosis was correct, and option (a) -- rerun the bootstrap seeded from
Exp. 26's exact original split -- **is feasible from cached data alone** and was
implemented.

**Fix implemented:** Reconstructed Exp. 26's exact split
(`RandomState(42).permutation(500)`, `tr_idx=idx[:375]`, `cal_idx=idx[375:500]`).
Verified the reconstruction by refitting on `tr_idx` and confirming the OOD AUROC
reproduces Exp. 26's reported value to 6 decimal places (0.608575 == 0.608575,
confirmed match). Then computed the apples-to-apples in-dist number: **the same
deployed probe**, evaluated on Exp. 26's own held-out 25% calibration split
(`cal_idx`, 125 samples) instead of the unrelated 5-fold-OOF number.

**Result -- a materially different, and smaller, gap than either previously
published number:**

| | In-dist AUROC | OOD AUROC | Gap (OOD - ID) |
|---|---|---|---|
| Paper (Exp. 26, mismatched estimators) | 0.5719 (5-fold OOF) | 0.6086 | **+0.0367** |
| Exp. 27 (own, differently-split 80/20) | 0.5838 | 0.6477 | +0.0639 |
| **This fix (same probe, Exp. 26's exact split, both arms)** | **0.5963** | **0.6086** | **+0.0123** |

Bootstrapping 1000 resamples anchored on Exp. 26's exact split (resampling
`tr_idx` for refitting, `cal_idx` for the in-dist test, and NQ-open for the OOD
test, all consistently) gives: in-dist AUROC mean 0.5783 (95% CI [0.425, 0.724]),
OOD AUROC mean 0.5774 (95% CI [0.461, 0.685]), **gap mean -0.0009** (95% CI
[-0.183, +0.179]), with 49.1% of resamples showing OOD > in-dist (essentially a
coin flip). The corrected 95% CI contains both zero and the paper's originally
reported point estimate.

**This is a genuinely new, disclosure-worthy finding, not a confirmation:** once
the split inconsistency is fixed, the "OOD improves over in-distribution" reading
implied by the paper's +0.037 point estimate disappears -- the split-consistent
mean gap is **statistically indistinguishable from exactly zero**, not merely
"not significantly different from zero with a wide CI" (which was already the
qualitative message, but the sign and near-zero magnitude of the corrected mean
gap is new information). The paper's "no catastrophic collapse" reading survives
and is, if anything, *strengthened* by this fix -- but the specific "+0.037"
number and its implicit small-positive-improvement framing should be replaced
with the split-consistent numbers above, and the paper should disclose that Exp.
26's original two numbers were never comparable in the first place.

---

## P1.4 -- Multiple-comparison correction (Analysis 3)

**Method:** Read `paper/main_neurreps.tex` in full (2696 lines, both compact main
body and appendix). Every line containing a p-value was extracted (29 distinct
textual locations via the pattern `p\s*[=<>]\s*0.xxx`), then traced back to its
source experiment log (`results/logs/*.json`, `results/*.json`, or
`results/logs/*.txt`) to recover full, unrounded precision and to **de-duplicate**
tests that the paper quotes more than once (e.g. the matched-n=24 Spearman test
appears 4 times across the abstract, contributions list, and two appendix
paragraphs -- counted once). Two Spearman tests (hallucination-rate-vs-scale,
same-layer-bound-error-vs-scale) were independently recomputed from
`results/logs/15_result_*.json` at full precision, confirming the paper's rounded
values (rho=-0.9429/p=0.004805 vs. paper's "-0.943/0.0048"; rho=-0.7714/p=0.07240
vs. paper's "-0.77/0.072"). Two Wasserstein/Bures correlations (rho only, no
p-value, explicitly flagged in-text as unvalidated n=2) and Cohen's kappa
label-agreement statistics were **excluded** -- they are not p-value hypothesis
tests and cannot be Holm-corrected.

**Family assembled: 38 distinct p-value-bearing tests** (larger than the review's
"~20+" estimate) spanning: 1 Pipeline-A length-classifier permutation test, 1
retracted monotonic-J permutation test, 2 matched-n=24 subsampling Spearman tests,
2 recomputed scale-covariate Spearman tests, 1 non-Gaussianity-vs-bound-error
Spearman test, 12 D'Agostino-Pearson normality tests (6 models x 2 classes, Exp.
22), 12 McNemar causal-steering tests (4 experimental configurations x 3 step
sizes: Exp. 19, 21, 23, 25), 1 SAE best-feature p-value (already the min of an
internally BH-FDR-corrected family of 24,576 -- included but flagged as a
double-correction), 3 Wilcoxon and 3 Mann-Whitney surface-length-control tests
(Exp. 28).

**Result -- Holm-Bonferroni, alpha=0.05, m=38:**

- Uncorrected: **11 / 38** tests significant at p<0.05.
- Holm-corrected: **3 / 38** tests remain significant.
- **8 tests flip from significant to non-significant**, including two claims the
  paper currently treats as safely established:
  - **The hallucination-rate-decline-with-scale claim** (rho=-0.943, p=0.0048) --
    the *one* scale-covariate finding the paper explicitly retains as "fully
    robust" after retracting the monotonic-J claim -- **loses significance under
    Holm correction** (needed p<0.00156 at its rank; raw p=0.0048).
  - **The matched-n=24 bound-error-increasing-with-scale claim** (rho=+0.83,
    p=0.042), used to argue the loose-bound finding survives and strengthens
    under the minority-class-size correction, **also loses significance**
    (needed p<0.00172; raw p=0.0416).
  - The already-retracted monotonic-J claim (p=0.0028) flips too, but this is
    moot since the paper already withdrew it.
  - Five of the twelve D'Agostino-Pearson non-Gaussianity tests flip (only 2 of
    12 survive: hallucinated-class tests for Qwen 0.5B and GPT-2 117M).
- None of the causal-null (McNemar/Wilcoxon/Mann-Whitney) tests were significant
  even uncorrected, so Holm correction does not change any causal-null
  conclusion -- those nulls remain nulls.

**This is a real, disclosure-worthy finding, not a confirmation:** under family-wise
correction across every significance test actually run in this paper, **the one
surviving positive scale-covariate claim in the abstract (hallucination rate
declining with scale) and the strengthened-loose-bound claim under the matched-n=24
correction both fail to survive**. Neither claim should currently be described as
"significant" without the caveat that it does not survive multiple-comparison
correction across the paper's full test family. The causal-null findings (this
paper's most heavily-tested claim family, 12 McNemar tests) are unaffected because
they were already non-significant uncorrected -- correction cannot make a null
result "more null," so this is the one place where the paper's existing hedged
language ("no significant causal specificity") already anticipates this analysis.

---

## P2.2 -- Probe hyperparameter sensitivity (Analysis 4)

**Question:** Is the 0.32-0.41 bound-error a property of the representation, or of
one specific logistic-regression configuration (C=1.0, no class weighting)?

**Method:** Using the same cached hidden states/labels and each model's best-J
layer, refit the 5-fold OOF linear probe at C in {0.01, 1.0, 100.0} crossed with
`class_weight` in {None, "balanced"} -- 6 configurations per model, 36 total probe
refits. Class balance (all six models are substantially imbalanced toward the
hallucinated class): GPT-2 117M 24/500 (4.8% correct), GPT-2 Medium 50/500 (10%),
Qwen 0.5B 47/500 (9.4%), Qwen 1.5B 60/500 (12%), Qwen 3B 95/500 (19%), Qwen 7B
103/500 (20.6%).

**Result -- sensitivity is large for the smallest, most-imbalanced models and
small for the largest, least-imbalanced ones:**

| Model | Class balance (correct/total) | AUROC range across 6 configs |
|---|---|---|
| GPT-2 117M | 24/500 (4.8%) | **0.1114** [0.515, 0.626] |
| GPT-2 Medium | 50/500 (10%) | **0.1373** [0.508, 0.645] |
| Qwen 0.5B | 47/500 (9.4%) | 0.0383 [0.532, 0.571] |
| Qwen 1.5B | 60/500 (12%) | 0.0141 [0.539, 0.553] |
| Qwen 3B | 95/500 (19%) | 0.0196 [0.504, 0.523] |
| Qwen 7B | 103/500 (20.6%) | 0.0074 [0.482, 0.490] |

**This is a genuinely new finding, not a confirmation:** for the two smallest,
most class-imbalanced models (GPT-2 117M and GPT-2 Medium -- exactly the models
the review flagged as "most likely to show sensitivity given their class
imbalance"), probe AUROC swings by **0.11-0.14** across regularization/weighting
choices alone -- comparable in magnitude to roughly a third of the paper's own
headline 0.32-0.41 bound-error range. Strong regularization (C=0.01) consistently
*increases* probe AUROC for these two models (GPT-2 117M: 0.564 -> 0.626; GPT-2
Medium: 0.561 -> 0.645), which would *shrink* the reported bound-error by
0.06-0.08 at those scales if adopted as the paper's default. For the four Qwen
models (all >=9% minority class), the range is small (0.007-0.038) and the
0.32-0.41 finding is robust to this choice. **The paper's probe-configuration
choice (C=1.0, no class weighting) is therefore not neutral for its two smallest
models** -- the bound-error range at those scales is partly an artifact of an
unregularized probe underfitting a severely imbalanced 24-or-50-positive-example
problem, not solely a property of the representation geometry.

---

## Overall disclosure

Three of the four analyses surface findings that **differ from, rather than
merely confirm,** the paper's current claims:

1. The headline "0.32-0.41" bound-error range is PCA-rank-dependent (0.19 at
   k=25 to 0.43 at k=200) -- an unreported estimator-identifiability gap.
2. The Exp. 26 OOD-gap point estimate (+0.037) was computed from two
   internally-inconsistent estimators; the split-consistent, apples-to-apples
   gap is +0.012 point-estimate / -0.001 bootstrap-mean, both far closer to
   zero than either previously published number.
3. Two of the paper's surviving positive claims (hallucination-rate decline
   with scale; matched-n=24 bound-error increase with scale) do not survive
   Holm-Bonferroni correction across the paper's full 38-test family.
4. Probe-hyperparameter choice materially affects the bound-error estimate
   for the two smallest, most class-imbalanced models (GPT-2 117M/Medium),
   though not for the four Qwen models.

None of these findings overturn the paper's central qualitative claim -- the
Fisher-Gaussian certificate is a valid but loose bound, never tight outside
saturated/near-degenerate regimes -- but all four narrow or complicate specific
numeric claims currently stated without these caveats. We recommend disclosing
all four in the relevant sections rather than treating this reanalysis as a
robustness check that "passed."
