# References — GEOM-PROOF

*Papers cited or directly compared against in this project.*

---

## Geometric & Fisher-Based Methods

**[1] Park et al. — Causal Fisher Information for Feature Attribution**
*ICML*, 2024.
> Causal Fisher as a competitor to our global Fisher J. Our experiments show Causal Fisher does not outperform standard Fisher J for the AUROC-bounding task.

**[2] Yin et al. — Local Intrinsic Dimension (LID) for Representation Analysis**
*ICML*, 2024.
> Local Intrinsic Dimension as an alternative geometric descriptor. Benchmarked against Fisher J in our scaling experiments.

**[3] Sugiyama — Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction (LFDA)**
Masashi Sugiyama.
*ICML*, 2006. Extended in *JMLR*, 2007.
> Conceptual ancestor of local Fisher separability at training time. Our local J uses KNN neighborhoods at inference time — key distinction.

---

## Conformal Prediction

**[4] Tibshirani et al. — Conformal Prediction Under Covariate Shift**
Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel Candes, Aaditya Ramdas.
*NeurIPS*, 2019.
> Weighted conformal prediction under distribution shift. Theoretical anchor for our α* = 0.07 guarantee and its scope limitation (in-distribution only).

**[5] Angelopoulos & Bates — A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification**
Anastasios N. Angelopoulos, Stephen Bates.
*arXiv:2107.07511*, 2022.
> Conformal prediction theory — basis for our coverage guarantee derivation and α* computation.

---

## Evaluation & Label Quality

**[6] "The Illusion of Progress" — ROUGE Evaluation Inflation**
*EMNLP*, 2025.
> Shows ROUGE evaluation can inflate AUROC by up to 45.9 points in some settings. Motivates our dual-label experiment (Exp 07) showing ROUGE-L vs GPT-4o-mini labels have κ = −0.010.

---

## Scaling Laws

**[7] Marks & Tegmark — The Geometry of Truth**
Samuel Marks, Max Tegmark.
*COLM*, 2024.
> LLaMA-2 7B/13B/70B probe accuracy reported but no fitted curve. We are first to fit AUROC = σ(8.62·log₁₀(params) − 69.10) with R² = 0.9993 within a controlled model family.

---

## Fisher Information for LLMs

**[8] Fisher Information Metric for Large Language Models**
*arXiv:2506.15830*, 2025.
> Fisher Information applied to LLM representations — related theoretical work motivating our Φ(√J/2) bound.

---

## Datasets

**[9] Lin et al. — TruthfulQA: Measuring How Models Mimic Human Falsehoods**
Stephanie Lin, Jacob Hilton, Owain Evans.
*ACL*, 2022.
> Primary labeled dataset for all probing and bounding experiments.

**[10] Li et al. — HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models**
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen.
*EMNLP*, 2023.
> OOD validation dataset. Conformal coverage does not hold under OOD shift (Exp 10 result).

---

*Total: 10 references*
