# Frontier-Lab Popperian Review — GEOM-PROOF
### "A Loose Ceiling, Not a Certificate: Auditing Fisher-Separability Bounds on Hallucination Detectability Across Architectures"
### Reviewed against `/Users/chakrivijayarao/Desktop/geom-proof/FRONTIER_LAB_REVIEW_FRAMEWORK.md`, verbatim, 18 steps, no step skipped.

Sources checked: `paper/main_neurreps.tex` (full, 2694 lines), `results/logs/*.json`, `results/*.json`,
`experiments/*.py`, `kaggle/exp{08,15,19,20,21,26,27,29,30}*`, `references.bib` (2 concurrent-work
citations independently verified against live arXiv listings: `ricco2026geometric` = arXiv:2602.14778,
accepted ICML 2026; `ying2026comparing` = arXiv:2606.19603). Numeric spot-checks below quote the exact
JSON fields used to reproduce each paper number.

---

## STEP 1 — UNDERSTAND THE PAPER

**One sentence:** This paper tries to establish that a classical closed-form statistic (Fisher/binormal
separability, $J$) computed from LLM hidden-state class means and covariance, while a mathematically valid
never-violated upper bound on linear-probe AUROC for hallucination detection, is **empirically far too
loose to be useful as a predictive certificate**, and that several previously reported "positive" findings
built on the same pipeline (tightness, monotone scaling with model size, architecture-invariant OOD
collapse) do not survive matched-methodology and confound testing.

This sentence is writable, which by the framework's own diagnostic means the paper does not have a Step-1
structural failure — but writing it required discarding the paper's own title-level framing ("certificate")
in favor of what the results actually show ("we tried to build a certificate and mostly broke it"). The
paper is, functionally, a negative/cautionary result dressed in the vocabulary of a positive one. That
tension recurs through every step below.

---

## STEP 2 — TRUE CONTRIBUTION

Ignoring the authors' framing and reconstructing independently:

- **Already existed:** The binormal ROC identity $\mathrm{AUROC}=\Phi(\sqrt{J/2})$ is a 1970s signal-detection
  result (Green & Swets 1966; Bamber 1975; Hanley & McNeil 1982). Fisher LDA is 1936. PCA+Ledoit-Wolf
  covariance shrinkage for high-dimensional discriminant estimation is standard (Ledoit-Wolf 2004). Using
  hidden-state geometry to detect hallucination is a crowded, active area (Azaria & Mitchell 2023, Burns et
  al. 2022, Chen et al. 2024, Marks & Tegmark 2024). A near-simultaneous paper (Ricco et al. 2026, ICML,
  verified) already applies the *same* Fisher-separability quantity to hallucination detection in LLMs,
  albeit operatively (as a classifier) rather than diagnostically (as a ceiling) — a real but narrow
  distinction the authors correctly identify and cite honestly.
- **Genuinely new:** (a) the specific empirical claim that the Gaussian binormal ceiling, when tested under
  a real-generation, non-saturated, matched-across-scale methodology, over-predicts probe AUROC by
  0.32–0.41 rather than the ≤2.7% the paper's own earlier (superseded) pipeline reported; (b) the explicit,
  well-instrumented demonstration that this earlier tightness result was a saturation artifact (bound
  pinned near 1.0 trivially matches AUROC pinned near 1.0); (c) the discovery and retraction, in-paper, of
  an apparent monotone $J$-vs-scale law, shown via a direct fixed-$n$ subsampling test to be a
  minority-class covariance-estimation artifact; (d) the length-confound audit of HaluEval's forced-choice
  pipeline (a 6-feature, no-hidden-state classifier matches near-saturated hidden-state probes); (e) a
  reasonably serious, appropriately powered-and-disclosed multi-granularity causal null for Fisher-LDA
  steering vectors (dense, gated, sparse-SAE, 117M→14.8B).
- **Incremental:** the OT/Wasserstein comparison (Fisher vs. Bures/SW2/MMD) is a two-architecture
  correlational table with explicitly no CI — a pilot observation, correctly self-labeled as such in the
  final revision, not a new method.
- **Would the field lose anything if this paper never existed?** It would lose a well-documented cautionary
  tale about saturation artifacts, minority-class bias in Fisher estimators, and length confounds in
  HaluEval — a genuinely useful methodological corrective, of the kind field practitioners cite in a
  "gotchas" section rather than build research programs on. It would **not** lose a certificate, a theorem,
  a new estimator, or a result that changes what anyone would build next. The corrective value is real but
  narrow: it is primarily a lesson in *how not to validate a geometric bound*, not a new bound, new theory,
  or new detector.

---

## STEP 3 — SCIENTIFIC NECESSITY

- **Why must "certificate" exist as a concept?** It doesn't, strictly — "upper bound on achievable linear
  AUROC from a Gaussian-approximated Mahalanobis distance" is a complete description without the word
  "certificate." The term borrows credibility connotations (formal verification, guarantee) that the
  paper's own results (a bound loose by 32-46 points) do not support. The paper's own mid-text terminology
  note concedes this ("read certificate below as shorthand... not a claim it is tight, causal, or safe to
  deploy") — which is an admission that the chosen name is doing rhetorical work the content does not
  license, corrected mid-document rather than by simply renaming the paper. Renaming (something like "an
  audit of the Fisher-Gaussian AUROC bound") would have been the more scientifically honest resolution;
  keeping the branded term and repeatedly caveating it is not equivalent.
- **Could the field continue without it?** Trivially yes — this is a diagnostic re-application of a known
  identity, not a new capability.
- **Does it simplify or merely rename?** For the *bound* itself: pure re-application, no renaming risk (the
  paper is explicit that $J$ and the identity are not novel). For "certificate": renaming, and a
  contestable one, given the result.

---

## STEP 4 — DEFINITION AUDIT

For each definition: precision / circularity / operationalization / testability / independence /
minimality / hidden assumptions / ambiguity / missing edge cases.

| Definition | Rank | Justification |
|---|---|---|
| **Fisher separability $J$** — $(\mu_c-\mu_h)^\top\Sigma_w^{-1}(\mu_c-\mu_h)$ | **Good** | Precise, standard, operational (given data, computable). Hidden assumption not stated at definition time (only later, §Background): requires $\Sigma_w$ invertible / well-conditioned, which fails at $d=2048$+ without PCA+shrinkage — the *actual* estimator used is not $J$ but a regularized, dimensionality-reduced proxy for $J$, and the two are silently treated as interchangeable throughout. This is a real gap: no result in the paper reports how sensitive $J$ is to the PCA rank ($k=100$, chosen without a stated criterion beyond ">95% variance in practice," not validated against $k=50$ or $k=200$). |
| **"Certificate"** | **Weak** | Ambiguous by the paper's own admission (mid-document terminology-correction). It connotes a formal guarantee; the object delivered is an empirical, sometimes-30-46-point-loose, non-causal, non-theoretically-guaranteed inequality that requires the same labels a probe needs. Three successive review rounds required the authors to walk back what "certificate" implies (training-free→not label-free; tight→loose; universal→saturated-regime-only). A definition needing three corrective addenda to not overclaim is not a stable definition — it is a moving target stabilized only by heavy annotation. |
| **"Training-free"** | **Good, post-correction** | After the round-2 fix distinguishing it from "label-free," this is precise and testable: no classifier fitting/CV is needed, only class means and covariance. The correction history itself, though, is evidence the *original* draft's definition was Broken (conflated with label-free), which matters for Step 13 (reproducibility of the writing/claims process, not just the code). |
| **Empirical vs. theoretical bound distinction** (§Background, "We distinguish two senses throughout") | **Good** | This is the paper's most careful piece of conceptual bookkeeping: theoretically $\Phi(\sqrt{J/2})$ is an *approximation* (could go either direction under a violated assumption); empirically it is observed to only ever over-predict. This distinction is well-drawn, consistently used, and does real work (it lets the paper report "zero violations across 91 pairs" without claiming a proof of that direction). One residual issue: no argument, only an empirical regularity across a specific data regime, is offered for *why* the over-prediction direction should generalize outside the tested pipelines/datasets — the paper does not claim it does, but the absence of a mechanism means "zero violations" is an inductive claim on a sample size of 6 models / 3 pipelines, not a property with any argued robustness. |
| **"Diagnostic" vs. "operative" use of $J$** | **Good** | Cleanly operationalized (certificate reads a ceiling; classifier commits to a decision rule) and correctly used to distinguish this paper from the concurrent Ricco et al. work. This is the cleanest definitional contribution in the paper. |
| **"Loose ceiling" (title)** | **Good** | Matches the measured result (0.32–0.41 over-prediction) and is the most defensible piece of framing in the whole paper — arguably more defensible than "certificate" survives to be, which raises the question of why the title keeps "certificate" in the same breath as "not a Certificate." |
| **Pipelines A/B/C** | **Good, but late** | Once introduced they are precise and consistently annotated per-table. But the paper's own text states the *first two review rounds* were needed before these were disentangled at all, and an earlier draft conflated all results under a single "ROUGE-L" label. This is a construct-validity failure that existed in the object of study, not just its description — see Step 16 "most dangerous assumption." |
| **Hallucination label itself (ROUGE-L≥0.4 / EM-F1≥0.5 / forced-choice)** | **Broken** | None of the three pipelines' labels are argued, only assumed, to track factual correctness. The paper's own audits show $\kappa=-0.010$ (Pipeline B vs. GPT-4o) and $\kappa=0.141$ (Pipeline C vs. Mistral-7B judge) — both near chance-level agreement with an independent semantic judge. A definition of the target variable that a semantic judge agrees with at $\kappa\approx0$–$0.14$ is not a workable operationalization of "hallucination"; it is an operationalization of "surface-form/length/template divergence," which the paper itself states in the Pipeline-A/B scope conditions but does not carry through consistently into how strongly later sections (e.g., "the certificate is loose," "probes are near-chance") are stated. |

---

## STEP 5 — LOGICAL STRUCTURE

Dependency chain for the **central claim** ("the certificate is a loose, not tight, bound"):

`Gaussian equal-covariance assumption (known false) → J computed via PCA(100)+Ledoit-Wolf proxy (approximation of an approximation) → Φ(√(J/2)) computed on this proxy → compared against a 5-fold logistic-regression probe AUROC on labels with κ≈0–0.14 semantic validity → same-layer AUROC subtracted → "bound error 0.32-0.41" reported as a property of "the certificate" rather than of the whole measurement chain.`

**Leap 1 (unsupported transition):** From "$\Phi(\sqrt{J/2})$ over-predicts a specific logistic-regression
probe's cross-validated AUROC" to "the certificate over-predicts *linear-probe* AUROC" (unqualified, as in
the abstract's boldface central-result sentence). Only one probe family (PCA(100)+LogisticRegression,
$C=1.0$, default regularization) is tested. No sensitivity check varies the probe's regularization, feature
count, or class-weighting (relevant given hallucination-rate imbalance up to 95:5). The claim as stated
generalizes over "any linear classifier"; the evidence is one hyperparameter setting of one linear
classifier family, on one probe per model per layer.

**Leap 2:** From "bound error is stable across HaluEval-based pipelines" to using this as license to also
draw a scale-covariate conclusion ("hallucination rate declines with scale, ρ=-0.943, survives"). This
specific claim is on much firmer ground than the withdrawn $J$-scale claim (it's a simple proportion, not a
covariance estimate) — correctly distinguished by the authors — but it still inherits the label-validity
problem above: what "declines with scale" is the rate of SQuAD-EM/F1 mismatch with a reference string, which
the paper's own $\kappa=0.141$ judge-audit shows is a weak proxy for actual factual accuracy. The claim
"hallucination rate declines with scale" is stated in the abstract without this qualifier; "surface-form
mismatch rate declines with scale" is the claim the evidence actually supports.

**Leap 3:** From "OOD AUROC does not collapse on one Pipeline-C/NQ-open run, Qwen-3B only" (Exp. 26/27) to
treating this as informative evidence at all about generalization, given (a) in-distribution AUROC itself is
statistically indistinguishable from chance (bootstrap CI [0.386, 0.747]), and (b) $n=1$ architecture. The
authors do flag the floor-effect caveat explicitly and do not overclaim in the final text — this is a
place where the self-correction process closed a leap that would otherwise be fatal. Credit where due (see
Step 16, "most convincing evidence" is *not* this, but this is one of the paper's better-guarded claims).

**Leap 4 (unresolved):** The paper repeatedly uses "we ran the control / we tested this directly" language
(surface-length control, minority-class matched-$n$ test, bootstrap CI) as evidence of rigor, then in
several places (Exp. 26 vs. Exp. 27, see Step 8) uses numbers from two non-identical experimental
protocols (different train/test splits) side-by-side as if they were one point estimate plus its own CI.
The paper *discloses* this ("close to but not identical... under a different split") but does not resolve
it: presenting a CI computed around split-B's point estimate immediately next to split-A's point estimate,
in the same sentence, without re-deriving a CI around split-A's actual number, is a logical seam that a
reader skimming the abstract (which quotes 0.609/0.572 with a CI computed elsewhere) will not detect.

---

## STEP 6 — MATHEMATICAL REVIEW

**Proposition 1 / Proof re-derivation (independently redone, not merely read):**

Given $w=\Sigma_w^{-1}(\mu_c-\mu_h)$, $s(h)=w^\top h$:

- Variance under either class: $w^\top\Sigma_w w = (\mu_c-\mu_h)^\top\Sigma_w^{-1}\Sigma_w\Sigma_w^{-1}(\mu_c-\mu_h) = (\mu_c-\mu_h)^\top\Sigma_w^{-1}(\mu_c-\mu_h) = J$. **Confirmed correct.**
- Mean gap: $w^\top\mu_c - w^\top\mu_h = w^\top(\mu_c-\mu_h) = (\mu_c-\mu_h)^\top\Sigma_w^{-1}(\mu_c-\mu_h) = J$. **Confirmed correct** (mean gap equals $J$, not $\sqrt J$ — a detail an earlier paper draft evidently got wrong, per the in-text correction note, and which I verified independently rather than took on faith).
- $d' = J/\sqrt{J} = \sqrt J$. **Confirmed.**
- $D = s_c - s_h \sim \mathcal N(J, 2J)$ for independent $s_c,s_h$ (variances add). $\mathrm{AUROC}=P(D>0)=\Phi(J/\sqrt{2J})=\Phi(\sqrt{J/2})$. **Confirmed correct** and matches the standard binormal-model identity $\mathrm{AUROC}=\Phi(d'/\sqrt2)$ for equal-variance Gaussians (Simpson & Fitter 1973 form). **Verdict: the derivation is mathematically sound as stated; I could not break it.** This is a real point in the paper's favor — the in-text note about a previous, incorrect $\Phi(\sqrt J/2)$ vs. $\Phi(\sqrt{J/2})$ confusion (conflating AUROC with equal-prior Bayes accuracy) is consistent with an error that *did* need catching, and the caught/corrected version is the textbook-correct one.

**Identifiability / estimator dependence (the actual weak point, not the proof):** The theorem is proved for
population $\mu_c,\mu_h,\Sigma_w$. Every reported $J$ is a plug-in estimate from (i) a PCA(100) projection
whose principal-component basis is *itself estimated from the same 500–2000 samples used for the class
statistics*, and (ii) Ledoit-Wolf shrinkage. Projecting to a 100-dim subspace before computing $J$ does not
compute $J$ in $\mathbb R^d$; it computes $J$ restricted to a data-dependent, unpenalized-for-degrees-of-
freedom 100-dimensional subspace, which is generally *larger or smaller* than the true $\Sigma_w^{-1}$-
weighted separation in $\mathbb R^d$ depending on whether the top 100 PCs happen to align with the
discriminative direction. No bound or bias direction for this approximation is derived or empirically
characterized (e.g., by varying $k$). This matters most exactly where the paper's headline number lives:
the minority class has as few as 24–103 examples in $d=768$–$3584$ ambient dimension even after PCA to
$k=100$ — an aspect ratio the paper itself calls out for Qwen-0.5B/1.5B ($\gamma>1$) but does not
re-flag for the Exp. 15 central table, where GPT-2-117M's 24-example minority class in a 100-dim PCA space
has $\gamma=100/24\approx4.2$, worse than the flagged Qwen cases. The minority-class subsampling test (Step
7, Exp. 16) partially addresses this for the *scale* claim but does not by itself validate the PCA-rank
choice for the *bound-error* claim built on the same estimator.

**Can the mathematics be simplified?** Yes — nothing forces the specific $k=100$ PCA + Ledoit-Wolf combination;
a Bayesian/hierarchical shrinkage estimator of $\Sigma_w^{-1}$ with a data-driven degrees-of-freedom
penalty would be the standard alternative and would let the paper report a *bias-corrected* $J$ with a
defensible confidence interval instead of a bare plug-in point estimate. **Does the math explain anything,
or merely describe?** It describes: nothing in the paper derives *why* real hidden-state geometry deviates
from Gaussian equal-covariance in the specific asymmetric way Exp. 22 finds (hallucinated class
non-Gaussian in 4/6 models, correct class Gaussian-consistent in 4/6) beyond noting the correlation with
Table certificate-matched is itself non-significant ($\rho=0.429$, $p=0.397$, $n=6$). The paper is honest
that this is unresolved (§Limitations item 2) — that honesty does not convert "describes" into "explains."

**Missing theorem:** No result anywhere bounds *how loose* $\Phi(\sqrt{J/2})$ can be as a function of a
measurable non-Gaussianity statistic (e.g., a Wasserstein distance to the nearest Gaussian with matched
first two moments). This would be the natural theoretical payoff of an "auditing" paper and is not
attempted; the paper stays at the level of "we observe X is loose empirically" throughout, with zero
theoretical account of the looseness's own scaling behavior.

---

## STEP 7 — EXPERIMENT REVIEW

Format per experiment: hypothesis tested / competing hypothesis / rating with justification.

**Exp. 01/06/13/20 (Table `certificate`, saturated-regime validation).**
Hypothesis: bound tightly predicts probe AUROC. Competing explanation actually applies here and is the
correct one, per the paper's own subsequent analysis: a bound saturating near 1.0 trivially matches an
AUROC saturating near 1.0 (ceiling-effect confound). Prompt-engineering/hyperparameter confounds: not
directly applicable (forced-choice task, no generation). Dataset artifact: **yes, decisively** — HaluEval
Pipeline-A's ≈5× answer-length gap between classes (Exp. 18) plausibly explains most of the reported
≈0.99 AUROC these rows treat as ground truth to bound. **Rating: Fatal**, as originally presented (the
paper itself now agrees, hence the retraction language) — retained in the paper only as a documented
artifact, which is the right editorial call, but the raw numbers (≤2.7% error, "MLP zero violations across
91 pairs") are still quoted at full strength in the contribution list and abstract-adjacent text as though
independently meaningful mathematical facts, when they are artifacts of a saturating, near-length-only
target.

**Exp. 15 (Table `certificate-matched`, central result).**
Hypothesis: under one real-generation, non-saturated, matched pipeline, does the bound stay tight?
Competing explanations checked directly by the authors: minority-class sample-size bias (checked, Exp. 16,
and found to *worsen* rather than explain away the result — good practice); cross-pipeline confound
(explicitly the reason this experiment exists, controlled by construction). Not checked: whether the same
6-feature length/surface classifier that explains Pipeline-A (Exp. 18) also explains Pipeline-C's residual
signal — the paper explicitly flags this exact gap as open ("Elite-review follow-up... this same
surface-only confound check has never been run on Pipeline~C") rather than hiding it, which is unusual and
commendable, but it means the paper's own central table has a disclosed, unresolved, plausible confound at
publication time. Selection bias: $n=500$ per model, single dataset (HaluEval), single generation run
(greedy, presumably — not stated whether temperature=0 or sampled, and if sampled whether seeded), no
repeated-generation variance estimate. **Rating: Strong, not Excellent.** Strong because it is the paper's
best-controlled, most self-audited experiment and the numbers reproduce exactly against the underlying
JSON logs I read directly (`results/logs/15_matched_generation.json`). Not Excellent because of the
disclosed-but-unresolved length-confound gap on Pipeline C and the absence of a probe-hyperparameter or
repeated-run sensitivity check.

**Exp. 16 (minority-class matched-$n$ subsampling).**
Hypothesis: is the monotone-$J$-with-scale trend a small-$n$ estimator artifact? This is a genuinely good,
falsification-oriented experiment — it is designed to kill the paper's own prior finding, and it succeeds
(sign reverses, $p=0.27$). **Rating: Strong.** This is the single best piece of methodological practice in
the paper: a positive finding was retracted because the authors built the specific test capable of killing
it, rather than defending it. The only weakness: 30 resamples per model at $n=24$ is a modest resample
count for a Spearman test on $k=6$ models; the reported $p=0.27$ (null) and $p=0.042$ (bound-error trend)
are both computed on $n=6$ data points, which is underpowered for *any* directional claim in either
direction — a coin flip on rank correlation with $n=6$ has wide sampling variance regardless of which way
it lands. The paper does not caveat the newly *significant* $p=0.042$ finding (bound error increasing with
scale) with the same $n=6$-is-tiny skepticism it correctly applies to the retracted claim — a mild
asymmetry in how skeptically the authors treat findings that support versus undercut the paper's surviving
thesis.

**Exp. 18 (Pipeline-A surface-length confound).**
Hypothesis: does a trivial surface classifier match Pipeline-A hidden-state probes? Alternative explanation
this experiment is specifically designed to test *for* (not against): dataset construction artifact. Result
directly confirms it (AUROC 0.9773±0.0060 vs. probes at 0.986–00.9998, permutation $p=0.002$).
**Rating: Excellent** as an isolated falsification experiment — clean, well-specified 6-feature baseline,
proper permutation test, decisive result. Its consequence is not fully metabolized elsewhere in the paper,
however: Tables `certificate`, `cv`, and `mlp` still report the confounded numbers as headline evidence for
contributions 3, 4, 8 in the contribution list, softened only by inline flags rather than removal or
re-scoping of the contribution claims themselves.

**Exp. 08 (Fisher vs. Bures/SW2/MMD, $n=2$ architectures).**
Hypothesis: is Fisher the best certificate, or do distribution-free alternatives dominate? Competing
explanation: with $n=2$ architectures and a target (probe AUROC) with ≈1-2% dynamic range across layers on
Qwen (Table `ot`: 0.973–0.992), a Spearman $\rho$ computed over ~30 near-tied values is highly sensitive to
noise in the AUROC estimates themselves — the "reversal" could be rank-order noise on a near-flat curve
rather than a real architecture effect. The paper's own round-2 correction explicitly downgrades this from
"central finding" to "suggestive $n=2$ observation" — appropriate, but the number still appears, unhedged,
in the abstract's third paragraph and the contribution list's item 5 ("architecture-dependent certificate
selection... established"), where "established" is not the register the authors use in the body text two
pages later. **Rating: Weak** (as a claim about architecture-dependence); **Acceptable** (as a documented,
narrow, two-point pilot observation, correctly caveated in the section that reports it).

**Exp. 10/14/17/26/27 (conformal coverage + OOD collapse family).**
Hypothesis: does conformal coverage collapse OOD, and is that scale/architecture-invariant? The dominant
alternative explanation — near-degenerate OOD label rates (3–7/400) making AUROC and violation-factor
numerically unstable — is identified and disclosed by the authors themselves, more thoroughly than most
published OOD-collapse claims in this literature bother to. The one non-degenerate test (Exp. 26/NQ-open)
does not replicate collapse, but from a near-chance in-distribution baseline (floor effect, also disclosed).
**Rating: Acceptable.** The qualitative claim ("guarantees void OOD") is credible and consistent with prior
literature (Dubanowska et al. 2025, independently verified as a real EMNLP 2025 paper via topic match).
The quantitative claim ("14.0×≈14.2×, near scale-invariant") is **Weak-to-Fatal as originally framed** and
is correctly withdrawn by the authors down to a qualitative-only claim. The Exp.26/27 point-estimate vs.
CI mismatch (Step 5, Leap 4; Step 8 below) is a genuine, if minor, defect that survives to the final text.

**Exp. 19/21/23/25/28 (causal steering family: dense/gated/SAE, 117M/14.8B).**
Hypothesis: is the Fisher-LDA direction causally sufficient? This is the most methodologically serious part
of the paper: a real intervention (not just a passive probe), a matched random-direction control, a
capability-preservation check, a length-confound check on the *causal* outcome (not just the passive
probe), and — most unusually for this literature — a minimum-detectable-effect analysis computed *after*
the fact showing several configurations (3–5 discordant pairs) could not have reached significance under
any possible outcome. Rating each sub-experiment: Exp. 19 (117M, dense, unconditional): **Acceptable**
(honestly null, honestly underpowered, correctly flagged). Exp. 21 (14.8B replication): **Acceptable**, same
caveats, sign reversal from Exp. 19 noted rather than cherry-picked. Exp. 23 (gated): **Acceptable**, though
the capability check is explicitly disclosed as "uninformative by construction" — an honest but also fairly
weak null result to report as a completed test. Exp. 25 (sparse SAE feature): **Acceptable**, extends
granularity meaningfully. **Aggregate rating across the family: Acceptable, not Strong,** because the
authors' own MDE analysis shows the family's evidentiary weight is closer to "we ran five variously
underpowered tests that all happen to point the same qualitative direction" than to "we ran one adequately
powered causal test." The text is unusually candid about this ("qualitative, not statistical, aggregation")
— candor that improves the paper's honesty score without improving the underlying evidence's strength.

---

## STEP 8 — STATISTICS REVIEW

- **Power:** Nearly every headline comparison in this paper operates at $n=6$ models (Table
  `certificate-matched`) or $n=2$ architectures (OT comparison). Spearman correlations at $n=6$ have such
  wide sampling distributions that a "significant" $p=0.042$ (Exp. 16 bound-error trend) and a
  "non-significant" $p=0.27$ (Exp. 16 monotone-$J$ retraction) are not qualitatively very different results
  — both are draws from a rank-correlation statistic with 6 data points. The paper treats them as
  qualitatively different (one retracted, one retained) using $p<0.05$ as a bright line that its own
  small-$n$ context does not support treating as bright.
- **Multiple comparisons:** No correction anywhere (Bonferroni, BH-FDR, or otherwise) across the very large
  number of Spearman/Pearson correlations, McNemar tests, and D'Agostino-Pearson normality tests reported
  across the appendix (my count: at least 20 distinct hypothesis tests with unadjusted $p$-values quoted).
  The one place FDR correction *is* used (Exp. 25, SAE feature search, BH $q=0.05$ across 24,576 features)
  is appropriately rigorous — showing the authors know how to do this — which makes its absence everywhere
  else in the paper a choice, not an oversight born of unfamiliarity.
- **Confidence intervals:** Present for some results (Exp. 27 bootstrap, the 7B sigmoid-fit prediction
  interval) and absent for others that carry more argumentative weight (the central Table
  `certificate-matched` bound-error values themselves have no CI — only a single point estimate per model
  from a single train/test configuration; the OT Spearman $\rho$ values in Table `ot` have no CI, which the
  authors themselves flag as a needed addition without adding it).
- **Point-estimate/CI mismatch (Exp. 26 vs. 27):** Documented above (Step 5). Concretely: the abstract
  quotes point estimates 0.609/0.572 (gap +0.037, from Exp. 26's 5-fold-OOF train/test split) immediately
  followed by a CI of $[-0.181,+0.244]$ that was actually computed by bootstrapping around a *different*
  split's point estimate (+0.064, Exp. 27's fresh 80/20 split). Both are disclosed as different splits in
  the appendix, but the abstract-level compression loses that distinction and presents them as if one
  procedure produced both numbers. This is a **real, findable statistical inconsistency**, not fatal (the
  qualitative "CI includes zero" conclusion is likely robust to which split's estimate you center it on),
  but exactly the kind of thing this framework's Popperian standard exists to catch, and it survived what
  the paper calls its "elite-review" and "final-audit" passes.
- **Sample adequacy at the estimator level:** Fisher $J$ estimated from as few as 24 minority-class examples
  in a 100-dimensional PCA space (effective $\gamma\approx4.2$, worse than the two cases the paper flags for
  $\gamma>1$) is not adequately powered for a stable covariance estimate even with Ledoit-Wolf shrinkage;
  shrinkage reduces variance at the cost of bias, and no bias-correction or de-shrinkage adjustment is
  applied before reporting $J$ as if it were a clean plug-in estimate.
- **Practical vs. statistical significance:** Generally handled well and is a genuine strength — the paper
  repeatedly asks "is 0.9998 vs 0.9994 practically distinguishable" (e.g., the MLP-vs-linear gain of
  ≤0.09%) rather than relying on significance stars, and explicitly reads its own near-chance probe AUROCs
  (0.52–0.65) as *practically* uninformative rather than merely "significant but small." This is one of the
  paper's genuinely strong statistical instincts, applied inconsistently (see the $n=6$ point above) but
  present and real.

---

## STEP 9 — CAUSAL REVIEW

Separating registers explicitly:

- **Correlation:** $J$ correlates (loosely, per-model, no CI) with probe AUROC across layers; Bures $W_2$/SW2
  correlate with probe AUROC across layers, architecture-dependently.
- **Prediction:** $\Phi(\sqrt{J/2})$ is used as a predictor of probe AUROC and is shown, descriptively, to
  systematically over-predict it — this is a legitimate predictive (not causal) claim, correctly scoped as
  such throughout.
- **Association, not mechanism:** The claim that non-Gaussianity "explains" bound looseness is tested
  (Exp. 22) and explicitly found *not* to hold at the cross-model level ($\rho=0.429$, $p=0.397$) — a rare
  case of a paper testing its own proposed mechanism and reporting the negative result plainly rather than
  quietly dropping the analysis.
- **Causality, tested directly:** Only the steering-vector family (Exp. 19/21/23/25/28) makes a causal
  claim, and it is a causal-*null* claim (the Fisher direction is not causally sufficient), tested with an
  actual intervention rather than inferred from correlational geometry — methodologically the most
  causally-literate part of the paper.
- **Where the paper exceeds its evidence:** The abstract's framing "detection performance is a property of
  representation geometry, not of the probe" (repeated three times, including as the paper's stated
  "central thesis") is a causal/structural claim about what *binds* detection performance. The actual
  evidence supporting it is (a) an MLP not beating a linear probe by more than 0.09% (consistent with, but
  not proof of, "geometry is the bottleneck" — an MLP with 64 hidden units and early stopping is also just a
  weak nonlinear function class, and its failure to improve on the linear probe is equally consistent with
  "this specific shallow MLP is a weak learner" as with "the representation caps all classifiers alike");
  and (b) the Gaussian bound never being violated by either probe (which, given how loose the bound already
  is at 32-46 points, is an extremely low bar — a bound that loose would absorb almost any classifier's
  performance, so "never violated" is close to guaranteed by construction once bound error is this large,
  not strong evidence that geometry specifically is the binding constraint as opposed to the bound simply
  being uninformatively slack). This is the paper's single largest logical overreach: the central thesis is
  stated with more confidence than the MLP-margin evidence, on its own, supports.

---

## STEP 10 — ALTERNATIVE THEORIES

Constructed independently, not merely gestured at:

**Theory A — Label-noise ceiling, not geometry ceining.** Every AUROC in this paper (probe *and* the
target the certificate is bounding) is computed against labels the paper's own judge-audits show agree with
an independent semantic judge at $\kappa\in[-0.010,0.141]$. Under this theory, the observed near-chance
probe AUROCs (0.52–0.65) and the large bound-error are not telling us anything about *representation
geometry's* capacity to support hallucination detection — they are telling us that a well-separated,
geometrically clean signal cannot produce a high AUROC against a target variable that is itself close to
unpredictable noise relative to ground truth. **How well it explains the observations:** very well — it
predicts near-chance AUROC regardless of representation quality, exactly what Exp. 15 shows.
**Distinguishing experiment:** compute probe AUROC and $J$ against the GPT-4o/Mistral-judge labels directly
(not ROUGE-L/EM-F1) on the same hidden states. **Does current evidence reject Theory A?** No — the paper
never runs this experiment. It runs the $\kappa$ audit (showing the labels disagree with a judge) but never
re-computes its central Table `certificate-matched` numbers *using the judge labels as ground truth*. This
is the single most important missing experiment in the paper (see Step 17).

**Theory B — Minority-class estimator bias explains not just the withdrawn monotone-$J$ claim but also
part of the surviving bound-error trend.** The paper convincingly shows (Exp. 16) that $J$ is biased upward
at small minority-class $n$. The surviving claim "bound error increases with scale, $\rho=+0.83$,
$p=0.042$" is computed from the *same* estimator family, on minority classes ranging 24→103 as scale
increases — i.e., minority-class $n$ and scale are exactly as collinear for this trend as they were for the
retracted one. The paper does test a matched-$n=24$ version of this and finds the trend *survives and
strengthens* (0.32-0.46 vs 0.32-0.41) — genuine evidence against Theory B for this specific claim, correctly
presented as such. Theory B is therefore a **live theory the paper's own matched-$n$ counter-test does
directly address and substantially weaken (though not eliminate, since 30 resamples per model at $n=24$ is
still a modest correction for a bias whose magnitude as a function of $n$ is never itself characterized).**

**Theory C — The causal-null result is a power artifact of using McNemar on a hallucination rate that is
already high (leaving few "wrong" baseline items to flip), not evidence the Fisher direction lacks causal
content.** The paper's own MDE analysis is close to conceding this theory outright for several
configurations (3-5 discordant pairs, cannot reach $p<0.05$ regardless of outcome). **Does the evidence
reject Theory C?** No — the paper explicitly does not claim it does ("we have not run... a formal
equivalence test"). This is a case where the paper's own honesty (Step 16 "most dangerous assumption"
territory) already does the alternative-theory work the framework asks a reviewer to do independently;
credit given, but the underlying causal-null conclusion in the abstract ("no significant causal
specificity") reads more conclusively than the underpowered evidence, taken alone, supports.

**Theory D — Answer-length/verbosity, not "representation quality," explains the Pipeline-C scale trend as
well as the Pipeline-A one.** Not tested (explicitly disclosed as an open gap by the authors). Given that
Theory D is confirmed for Pipeline A (Exp. 18) and structurally similar length/verbosity confounds are
known to correlate with model scale (larger/better-instruction-tuned models often produce shorter, more
calibrated answers), this is a live, untested, and highly plausible alternative to "detection performance is
a property of representation geometry" for the entire central-result table.

---

## STEP 11 — FAILURE ANALYSIS

Assume the paper's surviving central claim (bound is loose, ~32-41 points, at real-generation scale) is
wrong. Three most likely reasons, with confidence:

1. **Label-validity confound (Theory A above) — 45% confidence this substantially or entirely explains the
   result.** If the true factual-correctness signal is only weakly captured by EM/F1-vs-reference scoring,
   both the probe AUROC and the geometric class means used to compute $J$ are anchored to a noisy target;
   the "looseness" would then reflect noisy-label ceiling effects rather than a property of Gaussian
   approximation error specifically. What would overturn this: re-running Table `certificate-matched` with
   GPT-4o/Mistral-judge labels as ground truth and finding the same 30-40 point gap persists.
2. **Answer-length/verbosity confound on Pipeline C (Theory D) — 30% confidence it explains a meaningful
   fraction (though probably not all) of the result.** What would overturn this: the length-only baseline
   experiment the authors themselves flag as not yet run (analogous to Exp. 18, applied to Pipeline C).
3. **PCA-rank/estimator artifact in the $J$ computation itself — 20% confidence it explains a smaller,
   secondary fraction.** What would overturn this: recomputing $J$ across $k\in\{25,50,100,200,400\}$ and
   showing the bound-error magnitude is stable to this choice (not attempted in the paper).

These are not mutually exclusive, and the paper's own honesty (disclosing all three as open, to varying
degrees) is precisely why this step can be executed concretely rather than as generic reviewer speculation.

---

## STEP 12 — GENERALIZATION

- **Architectures:** GPT-2 (117M, 345M — encoder-decoder-free, non-instruction-tuned) and Qwen 2.5
  (0.5B–14.8B, instruction-tuned, AWQ-quantized at 14.8B). Two architecture families. Mamba/SSM transfer
  attempted and failed for an unrelated reason (all-hallucinated degenerate labels from a non-instruction-
  tuned model) — an honest "could not test" rather than a silently dropped result, but it means the paper's
  generalization claim is empirically two-family, not architecture-general, despite section titles like
  "Across Architectures."
- **Datasets:** HaluEval (in-distribution throughout) and TruthfulQA/NQ-open (OOD arms only). No
  multi-hop, no code-generation, no long-form generation, no non-English data.
- **Training objectives:** All decoder-only causal LM, no RLHF-vs-non-RLHF comparison, no base-vs-instruct
  ablation beyond the confound the paper itself finds in a companion study (prompt-template sensitivity).
- **Scale:** 117M–14.8B, the top end at 4-bit AWQ quantization only, explicitly and honestly flagged as
  confounding scale with quantization at the largest tested size.
- **Reasoning systems / modalities / deployment settings:** Not addressed; out of scope and not claimed.

**Explicit limit, stated plainly:** this paper's evidence generalizes, at best, to short-form
factoid-QA hallucination detection via linear/shallow-MLP probes on last-token hidden states, in two decoder-only
transformer families under 15B parameters, under three specific (weakly label-valid) scoring pipelines. The
title's "Across Architectures" is doing more work than the two-family, one-SSM-failure evidence base
supports; a more accurate title would say "Across Two Transformer Families."

---

## STEP 13 — REPRODUCIBILITY

**Strong, and this is the paper's clearest genuine strength.** Every numeric table I spot-checked against
its underlying JSON log reproduced exactly (Table `certificate-matched` against
`results/logs/15_matched_generation.json`; Exp. 16 minority-class numbers against
`16_minority_class_matched_check.txt`; causal McNemar $p$-values against
`19_active_control_causal_intervention.json` and `mde_mcnemar_summary.json`). Code is organized
per-experiment (`experiments/NN_*.py`, `kaggle/expNN_*`), scripts contain inline provenance comments
pointing to the exact JSON keys quoted in text, and random seeds are fixed and stated
(`RANDOM_STATE=42` throughout). Model checkpoints are named precisely (including the specific AWQ quant for
14B). Weaknesses: (a) no `requirements.txt`/environment lockfile found alongside the experiment scripts in
this review's read (dependency versions like `transformers`/`gptqmodel` are referenced only inside failure-
postmortem prose, not pinned); (b) generation sampling parameters (temperature/top-p/seed for the
free-generation pipelines) are not uniformly stated in the excerpts read; (c) the Exp. 26/27 split
inconsistency (Step 8) is itself a reproducibility problem — a third party re-running "the bootstrap CI
experiment" would not reproduce the abstract's quoted point estimate without knowing to use a *different*
script's split. **Verdict: another lab could substantially reproduce this paper's numbers from the provided
artifacts, with moderate effort and some environment-reconstruction work.** This is well above the norm for
a workshop extended abstract and should be recorded as a real point in the paper's favor, independent of
whether its scientific conclusions hold up.

---

## STEP 14 — FIELD IMPACT

Assuming every result here is correct as stated:

- **Would researchers build on this?** Some — primarily as a citable warning ("see [this paper] for why
  Fisher/Gaussian AUROC bounds on HaluEval-style pipelines are not tight") rather than as a foundation for
  new methods. It does not supply a replacement certificate, so it closes a door more than it opens one.
- **New benchmarks?** Unlikely directly, though the length-confound and label-$\kappa$ findings could
  motivate a cleaned relabeling of HaluEval (which the paper explicitly calls for but does not build).
- **New research directions?** The depth-fraction non-universality finding and the causal read/write null
  are the two most generative threads — both point toward "where and why does discriminative geometry form"
  as a mechanistic question, which the paper itself names as future work rather than answering.
- **Would companies care?** Marginally, and mostly as risk-mitigation guidance ("do not treat a Fisher-
  Gaussian bound as a deployment gate") rather than as a capability to adopt.
- **Would safety/interpretability research change?** The causal-null result (Fisher-LDA direction is not a
  causally sufficient steering vector) is the most safety-relevant finding and is consistent with, and adds
  one more architecture/scale/granularity data point to, an existing read/write-gap literature the paper's
  own companion work already establishes — incremental confirmation, not a new phenomenon.
- **Cited in ten years?** Possibly as a methodological cautionary citation in a related-work paragraph about
  saturation artifacts and label-validity checks for hidden-state probing — the way a well-executed null
  result sometimes gets a long tail of citations as "see X for why this doesn't work as cleanly as it looks."
  Unlikely to be cited for its own positive contribution, because the paper does not leave behind an
  artifact (method, benchmark, dataset, theorem) that later work needs to build on; it leaves behind a
  warning.

---

## STEP 15 — LAB DIRECTOR REVIEW

Would a Director of Research at a frontier lab fund a five-year program around this idea? **No, not as
framed.** The tested idea — "compute a closed-form Gaussian statistic from hidden-state class means and use
it as a training-free ceiling on probe performance" — has now been run to its logical endpoint by this
paper's own hand: the bound is loose exactly where it would need to be tight to be useful (non-saturated,
realistic regimes), it cannot select a deployment layer, its distribution-free competitors are architecture-
inconsistent at $n=2$, and its diagnostic direction has no demonstrated causal content across five
independent tests. A program built on *this specific object* ($J$/the binormal identity) has been
falsified by its own authors within the scope of one paper — that is Popperian success for the paper as
science, and simultaneously a reason not to fund a follow-on program on the same object.

**What would be worth funding instead, visible in this paper's own wreckage:** (1) the label-validity
problem — a systematic study of what a "ground-truth" hallucination label even is, decoupled from any
particular geometric method, given $\kappa\approx0$ agreement between three cheap proxies and an LLM judge;
this is the highest-leverage, least-glamorous, most field-wide-relevant thread the paper touches and
immediately drops. (2) The depth-fraction non-universality finding, pursued mechanistically (why does
discriminative geometry peak at different relative depths across architectures) rather than catalogued.

**Highest-risk assumption for any such program:** that hidden-state-geometry-based detection signal is
separable from surface-form/length/prompt-template artifacts at all, in currently-available QA
hallucination benchmarks — a concern the paper's own citations (the "recurring meta-pattern across this
project's companion studies" paragraph) suggest is a *general* problem across this entire research program's
outputs, not specific to this one paper.

**Milestone that would restore confidence:** a single experiment computing the central Table
`certificate-matched` result against LLM-judge (not string-match) labels, with a length/verbosity-only
baseline run alongside it, on at least one additional architecture family beyond GPT-2/Qwen. This is
concretely achievable with the infrastructure already built in this repository and is the most obvious next
step the paper itself stops just short of.

---

## STEP 16 — AREA CHAIR SYNTHESIS

- **Top strengths:** (1) An unusually thorough, independently-reproducible self-correction trail — I could
  verify essentially every quoted number against raw JSON, and the correction history (formula fix, pipeline
  disentanglement, minority-class retraction, length-confound discovery) reflects real adversarial process,
  not cosmetic hedging. (2) The causal-null experiment family is genuinely well-designed relative to the
  rest of this literature (matched controls, capability checks, MDE analysis, multi-granularity replication).
  (3) The core mathematical derivation is correct (independently re-verified, Step 6).
- **Top weaknesses:** (1) The target labels every quantitative claim in the paper depends on are
  weakly-to-not valid proxies for hallucination ($\kappa\in[-0.010,0.141]$), and this is never propagated
  into how confidently the surviving claims are stated. (2) The paper's confounded, saturated-regime
  numbers (Table `certificate`, the ≤2.7% headline, the 91-pair MLP result) remain in the paper as
  full-strength contribution-list items despite being shown, by the paper's own Exp. 18, to likely be
  substantially explained by answer length rather than the certificate's mathematics. (3) Nearly every
  cross-model or cross-architecture statistical claim operates at $n=2$–$6$, with $p$-values used as a
  bright line the sample sizes cannot support in either direction.
- **Most dangerous assumption:** That the three hallucination-labeling pipelines (forced-choice
  correct/incorrect string, ROUGE-L≥0.4, EM/F1≥0.5) are close enough to ground-truth factual correctness that
  AUROC computed against them — for the probe *and* for the geometric quantity the certificate is trying to
  bound — is a meaningful measurement of "hallucination detectability" rather than "surface-form/template/
  length divergence detectability." The paper's own two independent judge-audits put agreement at
  $\kappa\approx0$ and $\kappa=0.14$ — both close to chance. Every other finding in the paper (the bound is
  loose, probes are near-chance, hallucination rate declines with scale, the certificate cannot select a
  layer) is a second-order fact *about* a first-order measurement whose own construct validity the paper's
  own evidence puts in serious doubt. This is more dangerous than any single statistical or estimator issue
  above because it is not fixable by better statistics on the same data — it requires better labels.
- **Most convincing evidence:** The minority-class subsampling retraction (Exp. 16) — because it is the one
  result in the paper where the authors built the specific instrument capable of killing their own finding,
  ran it, and reported the kill, with the numbers matching independently-verified logs.
- **Weakest experiment:** The saturated-regime validation (Table `certificate`, Exp. 01/06/13) as originally
  reported — a ceiling-effect artifact now known, by the paper's own later work, to be substantially
  confounded by answer length, yet still quoted as a headline "≤2.7% error" fact in the contribution list.
- **Strongest experiment:** Exp. 16 (minority-class matched-$n$ subsampling test) for the reason above, with
  Exp. 18 (surface-length confound) a close second for decisiveness (a clean, well-specified, permutation-
  tested falsification of a load-bearing assumption).
- **Biggest missing theorem:** A bound on bound-looseness as a function of a measurable non-Gaussianity
  statistic — the paper measures that the bound is loose and that hidden states are non-Gaussian, but never
  connects the two quantitatively (Exp. 22 explicitly tests and fails to find this connection at the
  cross-model level).
- **Biggest missing experiment:** Recomputing the central result (Table `certificate-matched`) against
  independent LLM-judge labels instead of string-match labels — the single experiment that would most
  directly test Theory A (Step 10) and is well within the demonstrated capabilities of this project's own
  tooling (an LLM-judge pipeline already exists and is used for the $\kappa$ audits).
- **Biggest overclaim:** The repeated "central thesis" framing — "detection performance is a property of
  representation geometry, not of the probe" — stated with more structural/causal confidence than the
  supporting evidence (a weak MLP's small margin over a linear probe, and a bound so loose that near-any
  classifier would satisfy it) actually licenses (Step 9).
- **Biggest underclaim:** The depth-fraction non-universality finding (§Scale Behavior, Table `depth`) is
  presented as a minor limitations-adjacent note, but is arguably the paper's most robust and least
  confound-prone result (it is a within-model relative-position comparison, immune to both the
  cross-pipeline and minority-class-size confounds that the paper had to retract its two more prominently
  marketed findings over) and is undersold relative to its actual evidentiary robustness.

---

## STEP 17 — REVISION ROADMAP

**Priority 0 — Fatal issues that prevent acceptance at a general-scientific-rigor bar (not the NeurReps
Extended-Abstract bar already cleared):**

- **P0.1 — Construct validity of the outcome variable.** Every quantitative claim in the paper is
  conditioned on labels shown, by the paper's own audits, to agree with an independent semantic judge at
  $\kappa\approx0$–$0.14$. Until the central result (Table `certificate-matched`) is re-run against a
  validated (judge-based or human-adjudicated) label set and the 0.32-0.41 gap is shown to survive, the
  paper's title-level claim ("hallucination detectability") is not established — only "surface-form/
  template/length-mismatch detectability" is. *Expected scientific gain:* resolves the paper's single
  largest open threat to validity (Step 16). *Effort:* moderate — the judge-labeling pipeline already
  exists (Exp. 07, 30) and would need to be scaled from ~100-700 items to the full 500-per-model Pipeline-C
  set. *Impact on acceptance probability at this bar:* without it, the paper's contribution reduces to "an
  audit of an unvalidated pipeline," which is not fatal for an *extended abstract* but is fatal for the
  general-rigor bar this review applies.
- **P0.2 — Retire, not merely re-flag, the saturated-regime headline numbers.** Table `certificate`'s
  "≤2.7% error" and the 91-pair MLP zero-violation result are quoted at full strength in the abstract-
  adjacent contribution list despite Exp. 18 showing the underlying AUROC these numbers bound is
  substantially explainable by answer length alone. *Fix:* either run the Pipeline-C-equivalent length
  baseline the paper itself calls for and report the corrected numbers, or demote these results out of the
  contribution list entirely into a "historical artifact, superseded" appendix note (which the paper already
  does narratively but not structurally — the numbers still appear boldfaced in the main list). *Effort:*
  low (editorial) to moderate (if the missing length-baseline experiment is also run). *Impact:* removes the
  paper's most exploitable inconsistency between its own stated epistemic status and its structural
  presentation.

**Priority 1 — Major scientific weaknesses:**

- **P1.1** Recompute $J$ across multiple PCA ranks ($k\in\{25,50,100,200\}$) and report bound-error
  sensitivity to this choice (Step 6). *Gain:* closes the estimator-identifiability gap. *Effort:* low
  (rerun existing pipeline with different $k$). 
- **P1.2** Resolve the Exp. 26/27 split inconsistency by computing the bootstrap CI around the *same*
  train/test split used for the reported point estimate, not a separate one (Step 8). *Effort:* low
  (re-run Exp. 27 seeded from Exp. 26's exact split).
- **P1.3** Report the abstract's "hallucination rate declines with scale" and OOD-collapse claims with the
  same $n$-is-small skepticism applied throughout to the retracted monotone-$J$ claim, rather than
  asymmetrically. *Effort:* low (framing only).
- **P1.4** Apply multiple-comparison correction (or at minimum an explicit acknowledgment) across the ~20+
  unadjusted hypothesis tests scattered through the appendix. *Effort:* low.

**Priority 2 — Important improvements:**

- **P2.1** Add a third architecture family (Llama or Pythia, as the paper itself names as the needed next
  step for the OT comparison) to move the $n=2$ architecture claims to $n=3$. *Effort:* moderate
  (compute-bound, similar to existing Kaggle pipeline).
- **P2.2** Vary the probe (regularization strength, class weighting given imbalance) to test whether the
  0.32-0.41 bound-error is a property of the representation or of one specific logistic-regression
  configuration (Step 5, Leap 1). *Effort:* low.
- **P2.3** State generation sampling parameters (temperature, seeding) explicitly for every free-generation
  pipeline, and report a repeated-run variance estimate for at least one model. *Effort:* low.

**Priority 3 — Minor revisions:**

- **P3.1** Retitle or reframe the persistent use of "certificate" given the paper's own terminology note
  already concedes the term overclaims; consider "audit of the Fisher-Gaussian AUROC bound" framing
  throughout rather than repeated inline caveats on a retained brand name.
- **P3.2** Elevate the depth-fraction non-universality finding (Step 16, "biggest underclaim") out of a
  limitations-adjacent subsection into the main contribution list at a prominence matching its actual
  confound-robustness.
- **P3.3** Add a dependency/environment lockfile alongside the experiment scripts for full third-party
  reproducibility (Step 13).

---

## STEP 18 — FINAL DECISION

| Axis | Score | Justification (one line each) |
|---|---|---|
| Scientific Originality | **3/10** | Textbook 1970s identity re-applied to a crowded sub-field, with a concurrent paper (Ricco et al. 2026) applying the identical quantity to the identical problem; originality is concentrated in the audit methodology, not the object of study. |
| Scientific Importance | **4/10** | A real, useful negative/cautionary result for a specific measurement practice, but does not change what anyone should build; importance is capped by the paper's own admission that it establishes mainly what does *not* work. |
| Technical Quality | **6/10** | Math correct (independently re-verified), numbers reproduce exactly against logs, but core estimator (PCA-rank choice) never sensitivity-tested and one internal split-inconsistency (Exp. 26/27) survives to final text. |
| Experimental Rigor | **6/10** | Genuinely unusual amount of self-directed falsification (Exp. 16, 18, causal MDE analysis) pulls this up; near-universal $n=2$–$6$ statistics with unadjusted multiple comparisons and an unresolved, disclosed-but-live label-validity confound on the central table pull it back down. |
| Theoretical Strength | **3/10** | One correctly-proved but non-novel proposition; no new theorem connecting non-Gaussianity to bound looseness despite directly testing for (and failing to find) that connection. |
| Statistical Quality | **4/10** | Good practical-significance instincts and one properly FDR-corrected test, undermined by pervasive small-$n$ correlation claims, no CIs on the central table, and the point-estimate/CI mismatch in the OOD-gap result. |
| Clarity | **5/10** | The correction-annotation style, while evidentially valuable for this review, makes the paper extremely difficult to read as a coherent scientific narrative — a reader cannot easily tell, without cross-referencing, which of a dozen boldfaced claims are live versus superseded versus partially-retracted-but-still-quoted. |
| Long-Term Impact | **3/10** | Plausible as a citable cautionary footnote in future related-work paragraphs; no artifact (method, dataset, theorem, benchmark) left behind for others to build on. |
| Overall Confidence | **7/10** | High confidence in this assessment: numbers were independently reproduced against raw logs, the core proof was independently re-derived, and two external citations were independently verified — the review's factual basis is solid even where its judgments are severe. |

**Acceptance Recommendation: Weak Reject** (at the general-scientific-rigor bar this review applies — not
the NeurReps Extended-Abstract bar already cleared). The paper's own falsification work is its most valuable
content and is executed with real rigor in several places (Exp. 16, 18, the causal family); but the central
table the paper's revised headline claim rests on inherits an undischarged, paper-acknowledged label-validity
confound (P0.1) and sits alongside retained, now-known-confounded headline numbers from the paper's earlier
draft (P0.2) that a reader skimming the contribution list would still take at face value. A version that
executed P0.1 and P0.2 — both concretely achievable with tooling already built in this repository — would
plausibly clear a Weak Accept to Accept bar even under this framework's Popperian standard, because the
paper's process (falsify aggressively, report the kills) is exactly the process this framework asks for; it
has simply not yet been pointed at the one assumption (label validity) that would matter most if it broke.
