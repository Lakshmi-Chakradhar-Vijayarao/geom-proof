# NeurReps 2026 Proceedings Review — GEOM-PROOF
### "Geometric Certificates for Hallucination Governance"
Review executed verbatim against `NEURREPS_REVIEW_FRAMEWORK.md`, 2026-08-10.

---

## PRELIMINARY NOTE — which draft is actually "current" (read this first)

The task named two drafts to compare: `arxiv_submission/main.tex` (~1218 lines,
dated April 2026) and `ICML_2026_workshop/main.tex` (~356 lines, dated April
2026). **Both are stale.** Git and filesystem evidence show a third, far more
developed draft that supersedes both and was not named in the task:

- `/Users/chakrivijayarao/Desktop/geom-proof/paper/main.tex` — 2,562 lines,
  last modified **2026-07-28**, tracked in git, with a compiled `main.pdf`
  (same date) and a `main.bbl`. Its most recent commit
  (`225765a "Popperian review fix: scope Bures/SW2-vs-Fisher abstract claim
  to match its n=2 evidentiary weight"`) postdates six other revision commits
  (`086c9c27`, `199c566`, `1f326fa`, `6f52228`, `b29e254`, `083d3a1`) that are
  not reflected in either `arxiv_submission/` or `ICML_2026_workshop/`, which
  are **untracked** (`git status` shows both as `??`) one-shot exports frozen
  at early April.
- `paper/main.tex` retitles the paper **"A Loose Ceiling, Not a Certificate:
  Auditing Fisher-Separability Bounds on Hallucination Detectability Across
  Architectures"** and explicitly withdraws or heavily qualifies most of the
  headline claims the other two drafts still assert as clean positive
  results (bound tightness, monotone-$J$-with-scale, quantitative OOD
  scale-invariance). It incorporates the July 20–23 experiment logs
  (`16_minority_class_matched_check.txt`, `22_why_bound_is_loose.json`,
  `26/27_nondegenerate_ood_*.json`, `20/21/23/25_*causal*.json`,
  `mde_mcnemar_summary.json`) that neither `arxiv_submission/` nor
  `ICML_2026_workshop/` cites at all.
- **This review is executed against `paper/main.tex` as the true current
  draft**, since it is the only one of the three that reflects the project's
  actual state of knowledge as of the most recent commits. Findings specific
  to the two stale drafts are called out explicitly wherever they diverge in
  a way that matters for a submission decision.

This distinction is the single most consequential finding of this review
before any phase of the framework is even applied: **if either
`arxiv_submission/main.tex` or `ICML_2026_workshop/main.tex` were submitted
to NeurReps as-is, the submission would contain a paper whose own author has
since retracted its central quantitative claim, in a tracked, dated commit,
five weeks after both files were last touched.**

---

## Do the two named drafts diverge from each other, and from `paper/main.tex`?

**Between `arxiv_submission/` and `ICML_2026_workshop/`:** no important
divergence. The ICML file is a 4-page compression of the arXiv file with the
same numbers throughout ($\kappa=-0.010$, bound errors $\le 0.027$/mean
$0.012$–$0.014$, $R^2=0.9993$–$0.9996$, $\alpha^*=0.04$–$0.07$, SW$_2$/Bures
$\rho=0.821$/$0.828$ vs. Fisher $\rho=0.458$/$0.395$). They are two lengths
of the same April snapshot, not competing analyses.

**Between the April drafts and `paper/main.tex`:** the divergence is not
incremental, it is structural. Four things changed:

1. **A corrected formula.** Both April drafts write the certificate as
   $\Phi(\sqrt{J}/2)$. The July draft proves this was wrong — the correct
   binormal-model identity is $\Phi(\sqrt{J/2})$ — and documents that the
   original formula silently computed the equal-prior *Bayes accuracy*,
   $1-\Phi(-d'/2)$, not the *AUROC*, $\Phi(d'/\sqrt2)$. These coincide only if
   $2=\sqrt2$. Verified independently below (Phase 4).
2. **A task confound identified and named.** The April tables mix three
   methodologically distinct labeling/task pipelines (forced-choice
   "Pipeline A," ROUGE-L generation "Pipeline B," EM/F1 generation "Pipeline
   C") under a single "ROUGE-L labels" umbrella. The July draft shows the
   near-saturated ($\approx$0.99) AUROCs driving the "$\le2.7\%$ bound error"
   headline come entirely from Pipeline A, an easier forced-choice task the
   April drafts do not disclose as distinct from real generation.
3. **A withdrawn headline claim.** Under one matched real-generation
   pipeline across all six models (Exp. 15), bound error is $0.32$–$0.41$,
   not $\le2.7\%$. The July draft's abstract states this plainly: *"We
   therefore withdraw the paper's original headline claim that the
   Fisher-Gaussian certificate is empirically tight."* Neither April draft
   contains this sentence or anything like it.
4. **A withdrawn secondary claim plus a new causal-null result.** The
   apparent monotone $J$-vs-scale trend (visible in the July draft's own
   first pass, $\rho=-1.0$) is shown to be a minority-class sample-size
   artifact once resampled at fixed $n$ ($\rho=+0.54$, $p=0.27$, sign
   reversal) — withdrawn. Four new causal-intervention experiments
   (steering along the Fisher direction, GPT-2/Qwen-14B/SAE-feature
   granularities) all return null McNemar results. None of this exists in
   either April draft.

In short: the two files the user named are internally consistent with each
other but represent a scientific position the project has since abandoned in
its own tracked history. Any review of "the current draft" must be a review
of `paper/main.tex`.

---

## PHASE 1 — SCIENTIFIC IDENTITY

**One-sentence statement of the central theorem/principle, as it stands in
`paper/main.tex`:**

> *Fisher separability $J$, via the classical (non-novel) binormal-model
> identity $\Phi(\sqrt{J/2})$, is a mathematically valid but empirically
> loose upper bound on the AUROC any linear probe can achieve on LLM
> hidden-state hallucination detection — and essentially every stronger,
> more interesting claim initially built on top of that bound (that it is
> numerically tight, that separability decreases predictably with scale,
> that a universal distribution-free replacement exists, that the bound's
> own diagnostic direction is causally load-bearing) fails to survive the
> paper's own subsequent robustness checks.*

**Source of ambiguity, stated explicitly per the framework's instruction:**
This one sentence is actually two different claims wearing one title, and
the paper does not fully disambiguate which one it is proposing as its
contribution:

- **Claim A (mathematical):** the binormal AUROC identity, applied to
  Fisher-LDA-projected hidden states, is a valid upper bound. This is true,
  proved correctly (Phase 4), and is not new — it is 1930s discriminant
  analysis composed with a 1966/1975/1982 signal-detection identity
  (Green & Swets; Bamber; Hanley & McNeil), which the paper itself
  disclaims: *"We make no claim to originality for the identity itself."*
- **Claim B (empirical/diagnostic):** this bound is a *useful* certificate
  for practitioners deciding whether to train a hallucination probe. This is
  the claim with actual novelty-potential, and it is the claim the paper's
  own Exp. 15/16/22/26/27 data mostly falsifies or substantially weakens:
  the bound is loose by 30–40 points in the one non-degenerate, matched
  setting tested; the probes it bounds are themselves near-chance; the
  layer it identifies is not the layer a probe would use; and the
  distribution-free alternative (SW$_2$/Bures) that looked like it might
  rescue the framework reverses sign on a second architecture with $n=2$
  architectures tested.

A reader has to reach the withdrawal sentences in the conclusion to learn
that Claim B — the interesting one — has been walked back to "the certificate
is a passive, honestly-scoped upper bound with a wide, non-shrinking gap,"
which is closer to *"we checked, and this classical bound doesn't tell
practitioners much they can act on"* than to a new geometric principle.
That is a legitimate scientific contribution (a well-executed negative
result with an unusually thorough confound audit), but it is not the
contribution the paper's own remaining rhetoric (title aside) still
sometimes gestures at in Sections 4–5 before the corrections arrive.

---

## PHASE 2 — DEFINITION AUDIT

For every genuinely new (not merely imported) definition/quantity:

**$J(L)$, the Fisher separability ratio (Eq. 1).**
- *Necessity*: Necessary as the object of study — it is literally what the
  paper measures. Not novel as a definition (classical Fisher LDA
  discriminant, Fisher 1936, cited correctly).
- *Precision*: Precise and unambiguous given $\mu_c,\mu_h,\Sigma_w$.
- *Minimality*: Minimal — nothing removable.
- *Operationalization*: Concretely operationalized via PCA(100) + Ledoit–Wolf
  shrinkage, with the $\gamma=d/n$ aspect-ratio caveat disclosed for the
  small-scale Qwen models. This is good practice.
- *Testability*: Directly testable and tested (Table `certificate-matched`).
- *Relationship to existing concepts*: Explicitly related to $W_2^2$
  (Gaussian special case) and to classical LDA. No overclaim of originality
  here — correctly scoped.
- **Verdict: passes.** This is a well-behaved, honestly-labeled import, not
  a renamed known idea dressed up as new.

**"Certificate" (the paper's central noun).**
- *Necessity*: The paper needs a word for "$\Phi(\sqrt{J/2})$ used
  diagnostically rather than as a classifier." Fine in principle.
- *Precision*: Materially imprecise across the paper's own revision history.
  The introduction contains a standalone corrected paragraph: *"An earlier
  draft described the certificate as usable 'without ground-truth labels.'
  This contradicts its own definition... this is training-free, not
  label-free."* A second added paragraph instructs the reader to *"read
  certificate below as shorthand for this bound — not as a claim that it is
  tight, causal, or safe to deploy as an operational gate."* A definition
  that requires two after-the-fact disclaimers to stop readers from
  over-reading it is not yet a stable, precise definition; it is a moving
  target the paper is actively fencing in, revision by revision.
- *Minimality*: The paper's own title change ("Loose Ceiling, Not a
  Certificate") signals the authors themselves now doubt the word is the
  right frame.
- *Testability*: The bound-holds/violates test is well defined and run
  (zero violations across every configuration tested — the one part of the
  definition that has stayed stable).
- **Verdict: partially passes.** The underlying quantity is fine; the
  packaging word "certificate" is doing more rhetorical work than the
  paper's own data now supports, and the paper is aware of this (a point in
  its favor for honesty, a point against for Phase 2 precision).

**Pipelines A/B/C (task/labeling protocols).**
- Not present at all as a distinguished concept in the April drafts (which
  refer to "ROUGE-L labels" uniformly); introduced only in `paper/main.tex`
  as a direct response to a reviewer catching the confound. Once introduced,
  the definitions are precise, operationalized, and testable (each pipeline
  is a concrete scoring function over concrete generation protocols), and
  they do real explanatory work — most of Phase 6 below routes through this
  distinction. **Verdict: passes, and is this revision's single best
  methodological addition.**

**"Non-Gaussianity score" (Exp. 22).**
- Defined as mean $|\text{skew}|+|\text{excess kurtosis}|$ across both
  classes. Reasonable as an exploratory summary statistic, but under-precise
  as a certificate-diagnosis tool: the paper itself reports it does not
  significantly correlate with bound error ($\rho=0.429$, $p=0.397$, $n=6$)
  and explicitly declines to claim this settles anything. **Verdict:
  passes as an honestly-caveated exploratory measure; would not pass as a
  load-bearing definition if the paper leaned on it more than it does.**

---

## PHASE 3 — THEOREM AUDIT

The paper contains exactly one theorem-like object: **Proposition (Gaussian
AUROC Certificate)**, "restated for our setting" per the paper's own
sub-heading (correctly disclaiming novelty).

- **Hidden assumptions.** (i) Equal covariance $\Sigma_w$ across both
  classes — stated. (ii) Gaussianity of $h\mid y$ — stated. (iii) A more
  subtle, *unstated* assumption: independence of the draws $s_c,s_h$ used to
  compute $D=s_c-s_h\sim\mathcal N(J,2J)$. This is standard for the AUROC
  definition (comparing an independently drawn positive and negative
  instance) and is almost certainly intended, but the current proof text
  (paper/main.tex lines ~693–700) does not say "independent" explicitly at
  the point it adds the two variances — a reader who does not already know
  the identity could reasonably ask why $\mathrm{Var}(D)=2\sigma^2$ rather
  than $2\sigma^2(1-\rho)$ for some correlation $\rho$. This is a missing
  half-sentence, not a substantive gap, but it is the kind of thing a
  sophisticated referee will flag as underspecified.
- **Boundary conditions.** $\Sigma_w$ invertible (handled via PCA(100) +
  Ledoit–Wolf in practice, correctly disclosed as an approximation, not part
  of the theorem itself).
- **Logical dependence.** The proposition depends on nothing else proved in
  the paper; it is a standalone import.
- **Counterexamples.** None to the proposition itself (it is a correct,
  classical fact once assumptions hold). Counterexamples *to its practical
  relevance* are the paper's own main empirical content (Table
  `certificate-matched`): every one of six models' projected scores violates
  Gaussianity in at least one class (Exp. 22), yet the direction of the
  inequality (bound $\ge$ actual) never flips — consistent with, though not
  fully explained by, the paper's own non-Gaussianity measurements.
- **Scope.** Strictly linear classifiers under the stated Gaussian model.
  The paper is explicit and correct that the MLP result (91/91 non-
  violations) is empirical, not covered by the proposition — this is one of
  the better-scoped claims in the paper.
- **Necessity — could weaker assumptions suffice?** Yes, and the paper
  implicitly knows this: the proposition only needs the *scalar projection*
  $s=w^\top h$ to be Gaussian-with-equal-variance per class, not the full
  joint distribution of $h\in\mathbb R^d$ to be multivariate Gaussian. Exp.
  22's own D'Agostino–Pearson test is run on exactly this weaker, correct
  object (the 1-D projected score), which is methodologically the right
  move — but the *proposition statement* itself (Prop. `cert`) still states
  the assumption at the stronger, joint-Gaussian level ($P_c=\mathcal
  N(\mu_c,\Sigma_w)$). Tightening the stated hypothesis to the actually-used
  weaker one is a one-sentence fix that would make the theorem statement
  match what the paper actually tests.
- **Would the conclusion still hold under the weaker hypothesis?** Yes —
  the derivation only ever uses the projected scalar's distribution.

**Verdict:** The theorem is correctly scoped in substance, slightly
over-stated in its formal hypothesis relative to what is actually needed
and actually tested, and its main scientific weight in this paper is carried
not by the theorem itself (a 90-year-old fact) but by the empirical
looseness measurements built around it — which is a defensible thing for a
paper to do, provided it is honest that the theorem is not the contribution.
`paper/main.tex` mostly is honest about this; the two April drafts are not
(they present the derivation without the "restated, not novel" framing).

---

## PHASE 4 — PROOF AUDIT

**Attempt to invalidate the proof, per instruction.**

The proof (paper/main.tex, `sec:cert-derivation`) proceeds:
$w=\Sigma_w^{-1}(\mu_c-\mu_h)$; $s\mid P_c\sim\mathcal N(\mu_c^\top w,
w^\top\Sigma_w w)$, similarly for $P_h$; substituting $w$ gives
$\sigma^2=w^\top\Sigma_w w=J$ and mean gap $=J$, so $d'=J/\sqrt J=\sqrt J$;
$D=s_c-s_h\sim\mathcal N(J,2J)$; $\mathrm{AUROC}=P(D>0)=\Phi(J/\sqrt{2J})
=\Phi(\sqrt J/\sqrt2)=\Phi(\sqrt{J/2})$.

I re-derived this independently rather than trust-checking the LaTeX:
- $\mathrm{Var}(s\mid P_c) = w^\top \Sigma_w w = (\mu_c-\mu_h)^\top
  \Sigma_w^{-1}\Sigma_w\Sigma_w^{-1}(\mu_c-\mu_h) = (\mu_c-\mu_h)^\top
  \Sigma_w^{-1}(\mu_c-\mu_h) = J$. **Confirmed.**
- Mean gap $\mu_c^\top w - \mu_h^\top w = (\mu_c-\mu_h)^\top w =
  (\mu_c-\mu_h)^\top\Sigma_w^{-1}(\mu_c-\mu_h) = J$. **Confirmed.**
- $D=s_c-s_h$ for *independent* $s_c,s_h$: $\mathrm{Var}(D)=J+J=2J$,
  $\mathbb E[D]=J$. **Confirmed** (modulo the unstated independence
  assumption noted in Phase 3).
- $\mathrm{AUROC}=P(D>0)=P\!\left(Z>\frac{-J}{\sqrt{2J}}\right)
  =\Phi\!\left(\frac{J}{\sqrt{2J}}\right)=\Phi\!\left(\sqrt{\frac J2}\right)$.
  **Confirmed** — this matches the standard signal-detection relationship
  $\mathrm{AUC}=\Phi(d'/\sqrt2)$ for equal-variance binormal ROC models,
  where $d'=\sqrt J$ here. This is the textbook Bamber (1975) / Hanley &
  McNeil (1982) identity.

**Conclusion of independent re-derivation: the proof in `paper/main.tex` is
correct.** $\Phi(\sqrt{J/2})$, not $\Phi(\sqrt J/2)$, is the right formula.

**This makes the two April drafts independently falsifiable on pure
mathematical grounds, not merely superseded on empirical grounds.** Both
`arxiv_submission/main.tex` and `ICML_2026_workshop/main.tex` state
$\mathrm{AUROC}=\Phi(\sqrt J/2)$ (e.g., arxiv `Eq. (2)`, ICML `Eq. (2)` and
`Prop. cert`) and derive it with the same steps up through $d'=\sqrt J$, then
divide by 2 instead of $\sqrt2$ at the final step. This is exactly the
"equal-prior Bayes-accuracy vs. AUROC" conflation `paper/main.tex` itself
now names and corrects. **Any reviewer who reproduces the derivation from
either April draft, as this framework's Phase 4 instructs, will independently
find this error without needing the July draft as a hint** — it is not a
subtle numerical discrepancy, it is a wrong constant in the final line of an
otherwise-correct four-line derivation. Numerically, in the paper's own
saturated regime ($J\approx52$), $\Phi(\sqrt{52}/2)=\Phi(3.606)\approx
0.99984$ vs. the correct $\Phi(\sqrt{52/2})=\Phi(5.099)\approx0.9999998$ —
close enough in this particular saturated regime that it does not visibly
break the April drafts' reported bound values (both round to "$\approx1$"
at these $J$ magnitudes), which is presumably why it went unnoticed through
two rounds of that draft's own polishing. It would **not** stay invisible at
smaller $J$ — e.g. at $J=1$, $\Phi(\sqrt1/2)=\Phi(0.5)=0.691$ vs. the correct
$\Phi(\sqrt{0.5})=\Phi(0.707)=0.760$, a 7-point discrepancy — which is
exactly the regime the July draft's Exp. 15 (non-saturated, $J=1.6$–$7.5$)
newly probes, and is presumably how the error was eventually caught.

**Missing lemmas / dimension mismatches / circular reasoning:** none found.
**Notation inconsistency:** none material once the corrected formula is used
consistently (verified: `paper/main.tex` uses $\Phi(\sqrt{J/2})$ uniformly
in all 14 occurrences found by search — Table captions, remarks, and
equations agree).

**Why the corrected proof survives scrutiny:** it is a direct, checkable
four-line algebraic derivation from a stated model, matching a well-known
published identity by three independent citations (Green & Swets 1966;
Bamber 1975; Hanley & McNeil 1982) the paper itself cites for exactly this
purpose. There is nothing left to attack in the proof itself; all remaining
risk in the paper is in the *empirical* application of a correct formula
to real, non-Gaussian data, which is precisely what Phases 5–6 assess.

---

## PHASE 5 — GEOMETRIC EXPLANATION

Does the framework *explain* model behavior, *predict* new phenomena, or
*describe* observations?

- **Explains?** Weakly, and only at the level of "linear separability is
  bounded by covariance-weighted mean separation," which is a property of
  linear classifiers in general, not of LLM representations specifically.
  The paper does not explain *why* discriminative geometry for
  hallucination concentrates at particular depths, *why* it is
  architecture-dependent, or *why* the Wasserstein/Bures ranking reverses
  between Qwen and GPT-2 — all three of these are reported as unexplained
  empirical patterns ("mechanistically unexplained," in the paper's own
  words for the (withdrawn) 3B contraction; "no principled criterion yet"
  for the Wasserstein reversal).
- **Predicts new phenomena?** One genuine, falsifiable, and pre-specified
  prediction exists: the Mamba-130M architecture-transfer test (Exp. 04).
  It failed for an uninteresting reason (base model, not instruction-tuned,
  hall rate saturates to 1.0) rather than confirming or falsifying the
  geometric prediction, so this is a null result on the one place the paper
  tried to predict rather than describe.
- **Describes observed behavior?** Yes, predominantly. The paper is, at
  its core, a large, careful, honestly-audited *description* of where a
  classical bound holds, how loose it is, and what confounds explain away
  several of its more interesting-sounding secondary patterns. This is
  valuable science, but per the framework's explicit instruction — "if the
  paper only visualizes/describes representations, it is weak; if it
  explains why geometric structure constrains model behavior, it becomes
  scientifically significant" — this paper sits closer to the "describes"
  end than the "explains" end, and its own revision history moves it
  further in that direction over time (each withdrawal converts a would-be
  explanation into a disclosed non-finding).
- **Could the same observations arise from a different mechanism?** Yes,
  repeatedly, and the paper itself is the one demonstrating this: the
  original "monotone $J$ with scale" pattern arose from a sample-size
  mechanism, not a capability mechanism (Exp. 16). The original "OOD
  collapse is scale-invariant" pattern arose from a shared near-degenerate
  labeling artifact, not a shared geometric collapse mechanism (Exp.
  26/27). The original "certificate becomes exact at scale" pattern arose
  from a saturated-task artifact, not from representations becoming more
  separable (Table `certificate-matched`). This is exactly the diagnostic
  question Phase 5 asks reviewers to apply, and the paper has already,
  admirably, applied it to itself three times.

**Verdict:** by the framework's own stated standard, this paper does not
yet establish a necessary geometric principle. It establishes, with unusual
rigor, that several previously-plausible geometric principles do not
survive scrutiny, and that one classical, non-geometric-in-any-novel-sense
identity (Fisher/binormal AUROC) holds as a correct but uninformative-in-
practice upper bound. That is a legitimate contribution to a methods/
evaluation venue. It is a weak contribution to a venue whose primary
question is whether "geometric structure constrains model behavior" in an
explanatory sense.

---

## PHASE 6 — EMPIRICAL CONNECTION

For each experiment: which principle does it test, does it genuinely test
it, and could the same observation arise from a different mechanism?

| Exp. | Tests | Genuine test? | Alternative mechanism ruled out? |
|---|---|---|---|
| 01/06 (Fisher/GPT-2/Qwen 3B saturated validation) | Bound validity | Only within Pipeline A (forced-choice) | No — later shown to be a saturated/easy-task artifact (Exp. 15/18) |
| 03 (5-fold CV) | Predictive utility before training | Yes, on Pipeline A | Same saturation caveat applies |
| 07 (LLM-judge relabeling) | Label validity | Yes — real, appropriately-caveated audit ($\kappa=-0.010$, correctly flagged as unstable at 99% prevalence) | N/A — this is itself the alternative-mechanism check |
| 08 (Wasserstein/Bures/SW$_2$) | Non-Gaussian geometry | Yes, but $n=2$ architectures only, no bootstrap CI, target range <2% AUROC — paper now explicitly flags this itself | Not ruled out; paper correctly downgrades this to "suggestive," not "validated" |
| 10/14 (conformal, OOD) | Guarantee collapse | Confounded by near-degenerate TruthfulQA labels (3–7/400 positives) and a Pipeline-A-vs-B task switch — paper discloses this itself at length | Directly re-tested and *not* replicated once the confound is removed (Exp. 26/27, NQ-open, non-degenerate labels): AUROC rises OOD, not falls; bootstrap CI on the gap includes zero |
| 15 (six-model matched pipeline) | Bound tightness at scale | Yes — the paper's best-designed experiment; single consistent methodology across six models | This is the test that overturns the two April drafts' headline result |
| 16 (matched-$n$ minority class) | Whether monotone-$J$ is a sample-size artifact | Yes, directly, via subsampling | Confirms it is an artifact; correctly reported as such |
| 18 (surface-length confound) | Whether Pipeline-A AUROC is genuinely about hallucination | Yes, a 6-feature length classifier vs. hidden-state probes | Confirms answer length alone matches probe AUROC on Pipeline A ($0.977$ vs. $0.986$–$1.000$) |
| 20/21/23/25/28 (14B extension, causal steering, SAE feature, length control) | Whether the certificate's direction is causally load-bearing | Yes — the one place the paper runs a write-side, not read-side, test | McNemar null at every strength/architecture/granularity tested; MDE analysis shows several configurations are underpowered by design (as few as 3 discordant pairs), disclosed rather than hidden |
| 22 (why the bound is loose) | Gaussianity of the projected score | Yes, correctly re-specified at the scalar-projection level (Phase 3) | Explored, not confirmed: non-Gaussianity score does not significantly correlate with bound error ($p=0.397$, $n=6$) — paper does not oversell this |

**Overall Phase 6 finding:** this is the review's strongest section for the
paper. Nearly every experiment that looked, on first pass, like it supported
an interesting positive geometric claim was subsequently re-tested by the
same authors for an alternative, non-geometric mechanism (sample size, task
difficulty, label degeneracy, answer length, causal irrelevance), and in
four of five cases (Exp. 16, 18, 22, 26/27, causal nulls) the alternative
mechanism was found to explain some or all of the original pattern. This is
exactly the kind of adversarial self-testing the framework's Phase 6
instructs a reviewer to attempt — the paper has, in large part, already done
the reviewer's job for them. The cost is that very little "clean" positive
geometric finding survives to build a proceedings paper's contribution
around.

---

## PHASE 7 — LONG-TERM VALUE

Assuming the mathematics is correct (it is, Phase 4), would this framework
become a reusable analytical tool, a standard representation viewpoint, or
an isolated result?

- **As a reusable analytical tool:** unlikely in its current form. A bound
  that (a) requires labels anyway (not label-free, as the paper itself now
  clarifies), (b) is loose by 30–40 AUROC points in the one non-degenerate
  regime tested, and (c) cannot identify the layer at which it is tightest,
  does not save a practitioner meaningful effort over simply training the
  probe it was meant to substitute for. The paper's own Limitations section
  states this directly ("No inference-economy claim... does not propose or
  validate a deployable compute-saving mechanism").
- **As a standard representation viewpoint:** the Wasserstein/Bures-vs-
  Fisher comparison (Section `ot`) is the part of the paper closest to a
  durable viewpoint — "which geometric divergence best tracks linear
  separability is architecture-dependent" is a claim that could generalize
  and is worth someone extending past $n=2$ architectures with bootstrap
  CIs. But the paper's own abstract now scopes this to a "suggestive
  $n=2$-architecture observation, not a validated claim," which is honest
  but also an explicit statement that this is not yet a standard viewpoint.
- **As an isolated result:** this is the most likely long-term home for the
  paper's mathematical content specifically — the corrected binormal
  identity applied to hidden states is not, on the paper's own account, a
  new idea (a concurrent paper, `ying2026comparing`, independently applies
  the identical classical identity to a different but related question
  about probe-to-probe OOD transfer, and the July draft's Related Work
  section handles this candidly and well).
- **Where the paper's long-term value most plausibly lies** is not in the
  certificate itself but in its **negative-result methodology**: the
  minority-class-size confound (Exp. 16), the answer-length confound
  (Exp. 18), and the near-degenerate-OOD-label confound (Exp. 26/27) are
  all generalizable cautionary patterns for *any* future hidden-state
  probing paper on HaluEval/TruthfulQA, independent of Fisher geometry
  specifically. The paper's own closing "recurring meta-pattern" paragraph
  makes exactly this case, and undersells it by filing it as a paragraph in
  the conclusion rather than treating it as a candidate framing for the
  paper as a whole.

---

## PHASE 8 — REVIEWER SIMULATION

### Reviewer #1 (Representation Geometry / Information Geometry)

*Summary:* The paper applies Fisher-LDA and its associated binormal AUROC
identity to LLM hidden states, benchmarks it against Wasserstein/Bures/MMD
alternatives, and documents extensive failure modes.

*Strengths:* The identity is correctly re-derived after an initial error was
caught (I checked the algebra myself; the corrected version in the current
draft is right). The Wasserstein-identity-violation analysis ($J\neq W_2^2$
empirically, mean relative error up to 1.71) is a genuine, useful empirical
fact about non-Gaussianity in hidden-state space, and is the closest thing
in the paper to novel geometric content.

*Weaknesses:* I do not see a geometric or symmetry-related *principle* here
in the sense this venue expects. Fisher LDA is 90 years old; the binormal
AUROC identity is 50+ years old (properly cited as such). The paper's
contribution is an empirical audit of where a classical tool holds and
where it doesn't — useful, but it is an evaluation/robustness paper wearing
a geometry paper's vocabulary. The $n=2$-architecture Wasserstein-reversal
finding, which is the one piece of content that might interest this
audience specifically, is explicitly disclaimed by the authors themselves
as underpowered. I would want to see this generalized to $\ge5$
architectures with bootstrap CIs before treating it as a geometric claim
about architecture-dependence, and the authors already say as much.

*Recommendation:* Weak reject for this venue on fit and thinness of the
geometric contribution, notwithstanding the work's evident rigor.

### Reviewer #2 (Statistical Learning Theory / Evaluation Methodology)

*Summary:* A six-experiment audit of a classical separability bound applied
to hallucination probing, with an unusually extensive confound-hunting
program (task-pipeline confound, sample-size confound, label-degeneracy
confound, answer-length confound, causal-irrelevance test).

*Strengths:* This is, candidly, one of the more rigorous self-audits I have
refereed. The authors caught their own algebra error, caught their own
task-pipeline confound, caught their own sample-size artifact, ran a direct
causal test rather than assuming correlational geometry implies causal
control, computed a minimum-detectable-effect analysis for their own
underpowered McNemar tests rather than reporting bare non-significance, and
computed a bootstrap CI on a central OOD-gap claim rather than a point
estimate. The honesty is close to exemplary for a solo-authored preprint.

*Weaknesses:* The density of retractions is itself a problem for a
proceedings submission, independent of how well each retraction is handled.
By my count, the current draft withdraws or substantially downgrades: (1)
bound tightness, (2) monotone-$J$-with-scale, (3) quantitative OOD scale-
invariance, (4) the causal relevance of the certificate's own direction, and
qualifies (5) label validity twice over (both Pipeline B and Pipeline C).
What remains as *positive, unretracted* content is: the bound is a valid
(if loose) upper bound; the certificate cannot select the best layer; depth
fraction is architecture-dependent; and hallucination rate declines with
scale (a simple proportion, not a geometric finding). That is a thin
positive residue for a proceedings-length paper, however well-earned the
thinness is.

*Recommendation:* Borderline. I would want to see this reframed explicitly
as a negative/audit result — which is a legitimate paper type — rather than
retain a title and structure built around a certificate whose defining
property (usefulness) the paper's own data undercuts.

### Reviewer #3 (Mechanistic Interpretability)

*Summary:* Beyond the certificate, the paper's most interesting content to
me is the causal-intervention section: steering along the Fisher-LDA
direction fails to produce a significant hallucination-correction effect
at three strengths, two architectures (117M, 14B), and two intervention
granularities (dense direction, single SAE feature).

*Strengths:* This is a real, if modest, contribution independent of the
certificate framing — it is evidence that a geometrically-defined
"hallucination direction" derived exactly the way this literature commonly
derives such directions (LDA on class means) is not obviously steerable,
consistent with a broader pattern the authors say recurs across their other
projects (FFN-patching null, cited but not detailed here). The
minimum-detectable-effect disclosure is good practice I do not see often
enough in this literature.

*Weaknesses:* Every one of these causal tests is underpowered by
construction (as few as 3 discordant pairs at some settings), and the
authors' own MDE analysis says so. A null result under an analysis that
could not have detected a true effect at this sample size is weak evidence
of "no causal effect" and should be read as "this test was not capable of
finding an effect regardless of ground truth," which the paper does state,
but the abstract's framing ("no significant causal specificity ... shows no
significant causal specificity") risks being read by a less careful reader
as stronger evidence of a genuine null than the power analysis supports.
I would ask for either larger $n$ or a formal equivalence/TOST test with a
pre-specified effect-size bound, not just a non-significant p-value plus a
post-hoc MDE note.

*Recommendation:* Borderline, contingent on reframing the causal section's
evidentiary strength downward to match its own power analysis.

### Proceedings Chair

I read this alongside the two "stale" drafts the authors' own repository
still contains (`arxiv_submission/`, `ICML_2026_workshop/`) to understand
what changed. The difference is stark and, on balance, reassuring about the
authors' scientific process and concerning about submission readiness: the
version in `paper/` is scientifically far more trustworthy than either
named draft, precisely because it retracts most of what made the named
drafts sound exciting. A Proceedings decision has to be made about the
paper that would actually be submitted, and right now that is ambiguous —
three drafts exist, two of them contain a derivation error and an
unretracted, since-falsified headline claim, and the third has not been
formatted for the NeurReps Proceedings template at all (it is a generic
10pt two-column article, not verified against `venue_template_neurreps/`).

Two decisions are bundled here and should be separated. First: is the
underlying science, as it now stands in `paper/main.tex`, honest and
carefully done? Yes, unusually so. Second: is a paper whose own abstract is
substantially organized around walking back its own prior claims, applied
to a classical (not novel) identity, with a thin geometric contribution by
the framework's own explanatory-power standard, and with no NeurReps-
specific framing (no symmetry, no equivariance, no group structure, no
manifold-topology argument connecting to the venue's actual scope) a good
fit for a Proceedings track with a hard page budget and a symmetry/geometry
mandate? Less clearly.

*Recommendation:* Weak Reject at NeurReps specifically, with an explicit
suggestion to the authors that this paper's actual home is an evaluation/
robustness/trustworthy-ML venue (its content — confound-hunting in
hidden-state probing, label degeneracy, causal null-testing of steering
vectors — is a stronger fit there than at a geometry-and-symmetry workshop),
and a hard requirement, if resubmitted here or anywhere, to submit only the
`paper/main.tex` line of development, not the two retracted April drafts.

---

## FINAL DECISION

Scored 1 (lowest) – 5 (highest), evaluated against `paper/main.tex`
specifically (the two April drafts would score materially lower on
Mathematical Correctness and Empirical Support, per Phase 4 and Phase 6
above, due to the uncorrected formula and the un-disclosed pipeline
confound):

| Dimension | Score (1–5) | Basis |
|---|---|---|
| Mathematical Correctness | 4 | Corrected derivation independently re-verified (Phase 4); minor unstated independence assumption; theorem statement slightly over-strong relative to what's actually used/tested (Phase 3) |
| Theoretical Originality | 2 | Explicitly, correctly disclaimed by the authors as a non-novel classical identity; the PCA/shrinkage estimation pipeline is engineering, not theory |
| Proof Quality | 4 | Short, correct, checkable; missing only an explicit independence clause |
| Conceptual Clarity | 3 | Much improved over the April drafts via the Pipeline A/B/C distinction, but "certificate" itself remains an unstable, multiply-caveated term across the paper's own revision history |
| Geometric Insight | 2 | The one candidate geometric insight (architecture-dependent Wasserstein-vs-Fisher ranking) is explicitly self-scoped as $n=2$, underpowered, not a validated claim |
| Empirical Support | 3 | Exceptionally thorough and self-adversarial (Phase 6), but the bulk of what it supports is the *absence* of the paper's original claims, not new positive findings |
| Explanatory Power | 2 | Predominantly descriptive/diagnostic rather than explanatory, by the framework's own definition (Phase 5); the one falsifiable prediction (Mamba transfer) returned an uninformative null |
| Long-Term Research Value | 3 | Low as a reusable certificate (Phase 7); moderate-to-good as a methodological cautionary case study for hidden-state probing generally |
| Venue Fit (NeurReps) | 2 | No symmetry/equivariance/manifold-topology content; "geometry" here means classical multivariate statistics (covariance, Mahalanobis distance, Wasserstein), not the representation-symmetry structure this workshop centers |

**Acceptance Recommendation: Weak Reject.**

Rationale, applying the framework's Final Requirement directly: the paper
should be rewarded for necessity, not complexity, and by that standard its
best material — the confound audits (Exp. 16, 18, 22, 26/27) and the causal
null (Exp. 19/21/23/25/28) — is genuinely necessary, well-executed science.
But the object those audits are performed *on* (a classical Fisher/binormal
bound, explicitly disclaimed as non-novel) does not constitute a geometric
principle that materially improves scientific understanding of *why*
geometry constrains model behavior; it is, on the paper's own final
accounting, a description of where a known bound is loose, largely for
reasons (label degeneracy, task-pipeline confounds, sample-size bias) that
turn out to be about experimental design rather than about representation
geometry per se. That is publishable, valuable work — but for a Proceedings
track at a venue whose mandate is symmetry and geometry in neural
representations, it falls short of Accept, and the unresolved three-draft
situation (two of which contain a derivation error and a since-retracted
headline claim) makes it not yet submission-ready in any of its current
forms.

---

*Review complete. All 8 phases and the Final Decision section executed per
`NEURREPS_REVIEW_FRAMEWORK.md`, in full, against `paper/main.tex` as the
verified current draft, with cross-references to `arxiv_submission/main.tex`,
`ICML_2026_workshop/main.tex`, `RESULTS.md`, `RESEARCH_VALUE.md`, `PLAN.md`,
`REFERENCES.md`, the project's `.bib` files, `git log`/`git status`, and the
following `results/logs/` artifacts read directly:
`01_fisher_analysis.json`, `02_scale_curve.json`, `03_certificate_validation.json`,
`07_judge_log.json`, `08_ot_certificate.json`, `09_spectral_phase_transition.json`,
`10_conformal_coverage.json`, `11_qwen_scale_curve.json`, `12_mlp_probe.json`,
`13_qwen7b_analysis.json`, `14_ood_transfer.json`,
`16_minority_class_matched_check.txt`, `20_qwen14b_analysis.json`,
`21_qwen14b_causal_intervention.json`, `22_why_bound_is_loose.json`,
`23_gated_causal_intervention.json`, `25_sae_feature_causal_clamp.json`,
`26_nondegenerate_ood_transfer.json`, `27_bootstrap_ci_nondegenerate_ood.json`,
`28_surface_length_control_causal_steering.json`, `mde_mcnemar_summary.json`.*
