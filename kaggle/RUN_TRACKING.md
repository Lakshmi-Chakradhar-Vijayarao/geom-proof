# GEOM-PROOF Kaggle Run Tracking

## Exp 15 — Matched-methodology generation certificate (all 6 models)
- Kernel: anonymous-author/geom-proof-exp15-matched-gen
- Kernel version 1 (2026-07-20): FAILED — Kaggle assigned a Tesla P100
  (sm_60), incompatible with the pre-installed PyTorch build (supports
  sm_70+ only) → "CUDA error: no kernel image is available for execution
  on the device" on all 6 models within the first ~4 minutes. No usable
  output.
- **Kernel version 2 (current, tracked): scriptVersionId=336623111**
  — pushed with `--accelerator nvidiaTeslaT4` to force a compatible GPU.
  URL: https://www.kaggle.com/code/anonymous-author/geom-proof-exp15-matched-gen?scriptVersionId=336623111
- Status: **COMPLETE** (98.5 min total runtime). Output downloaded to
  `kaggle_output/exp15/`, copied into `results/logs/15_matched_generation.json`
  + `15_result_*.json` and `results/hidden_states/15_hidden_states_*.npz`.
- Result summary (n=500 per model, all real generation + SQuAD EM/F1):
  hall_rate and best-layer J both decrease monotonically with scale
  (J: 7.52 -> 3.97 -> 3.69 -> 2.25 -> 2.00 -> 1.63 across
  117M/345M/494M/1.54B/3.09B/7.6B). Certificate bound error is 0.25-0.33 at
  every scale (not <=2.7% as the old cross-pipeline validation claimed).
  Fully written into `paper/main.tex`: new Table `tab:certificate-matched`
  is now the paper's primary certificate-validation result; abstract,
  contributions, `tab:scope`, the depth-fraction/non-monotonicity section,
  and the conclusion were all rewritten to lead with this finding and
  correctly scope the old (now-secondary) saturated-regime numbers.
  Paper compiles cleanly (verified locally with a stubbed nicefrac.sty and
  microtype expansion disabled, both pre-existing environment quirks
  unrelated to content).

## Exp 16 — Minority-class-matched subsampling check (local, no Kaggle needed)
- Script: `experiments/16_minority_class_matched_check.py`
- Output: `results/logs/16_minority_class_matched_check.txt`
- Third-round review of the Exp 15 "J decreases perfectly monotonically
  with scale" finding flagged that minority-class size (n_correct: 24,
  50, 47, 60, 95, 103 across the 6 models) is itself collinear with
  scale, and Fisher J is known to be upward-biased at small minority-n --
  meaning the monotonicity could be a sample-size artifact, not a
  capability effect.
- Ran directly on the already-downloaded Exp 15 hidden states: subsampled
  every model's minority (correct) class to a FIXED n=24 (matching the
  smallest available), 30 resamples per model, same Fisher/PCA/Ledoit-Wolf
  estimator, then checked Spearman correlation between log(params) and
  matched-n mean J across the 6 models.
- **Result: the monotonic trend completely disappears once minority-class
  size is held fixed** (Spearman rho = -1.0 unmatched -> rho = +0.54, p=0.27
  matched -- not significant, and the point estimate even reverses sign).
  This confirms the "J decreases monotonically with scale" finding was a
  minority-class sample-size estimation artifact, not a genuine
  capability-dependent geometric phenomenon.
- Fully written into `paper/main.tex`: the monotonic-J-with-scale claim is
  withdrawn everywhere it was previously asserted (abstract, Table
  tab:certificate-matched discussion, the "Depth Fraction" resolution
  section, contributions list, conclusion). Hallucination rate's own
  decline with scale (a simple proportion, unaffected by this estimation
  issue) is retained as the one surviving scale-covariate finding from
  Exp 15. The bound-error metric in tab:certificate-matched was also
  corrected (same-layer error, 0.32-0.41, not the previously reported
  cross-layer 0.25-0.33 which mixed two different layers) -- this makes
  the surviving "certificate is loose, not tight" finding stronger, not
  weaker.
- Purpose: fixes the IID-vs-OOD task confound found in second-round adversarial
  review — Exp 00's HaluEval numbers used no-generation forced-choice labeling
  (balanced 50/50, no ROUGE-L at all) while Exp 11's Qwen 0.5B/1.5B numbers used
  real generation + ROUGE-L on TruthfulQA, and Exp 10's OOD collapse numbers also
  used real generation + ROUGE-L. Comparing these as "in-distribution vs OOD" or
  as points on one "controlled" scale curve conflated task difficulty with
  distribution shift / scale.
- Exp 15 runs ONE consistent pipeline (real generation given only the question,
  SQuAD-style EM/F1 scoring against right_answer, same methodology validated for
  papers-2026 Paper 3) across all 6 models used in the paper: GPT-2 117M,
  GPT-2 Medium 345M, Qwen 2.5 0.5B/1.5B/3B/7B.
- N=500 HaluEval QA pairs per model, single real generation each (not the old
  forced-choice pair-of-forward-passes design).
- Outputs (download to results/logs/ + results/hidden_states/):
  15_matched_generation.json (consolidated, all 6 models)
  15_hidden_states_<model_key>.npz (per model)
  15_generations_<model_key>.json (raw generations, for spot-checking label quality)
- Auth note: this Kaggle CLI version (2.2.2) requires `export KAGGLE_API_TOKEN=<token>`
  as an environment variable per push/status call — writing to ~/.kaggle/access_token
  or the legacy kaggle.json format both failed with "Authentication required" despite
  `kaggle config view` showing the right username. Re-export before every kaggle CLI
  call in a fresh shell.
