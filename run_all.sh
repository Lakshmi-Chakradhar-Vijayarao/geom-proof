#!/bin/bash
# GEOM-PROOF — Full Experiment Execution Order
#
# Run from the project root: bash run_all.sh
# Or run individual experiments: python experiments/07_judge_relabeling.py
#
# CRITICAL ORDER — do not skip steps.

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "GEOM-PROOF — Testing Phase Execution"
echo "============================================================"
echo ""

# ── Phase 0: Second dataset extraction ───────────────────────────────────────
echo "[Phase 0] HaluEval extraction (second dataset)"
echo "  python experiments/00_halueval_extraction.py"
echo "  Expected: ~2-3h per model on CPU"
echo ""

# ── Phase 1: Label defense (run first, overnight) ────────────────────────────
echo "[Phase 1] LLM-as-Judge re-labeling"
echo "  python experiments/07_judge_relabeling.py"
echo "  Expected: ~2-3h (Ollama, Qwen 2.5 3B local)"
echo ""

# ── Phase 2: Core certificates (free, existing data) ─────────────────────────
echo "[Phase 2] Three-certificate analysis"
echo "  python experiments/01_fisher_analysis.py"
echo "  Expected: ~3-4h (Fisher + LID + probe on all layers)"
echo ""

echo "[Phase 2] Scale curve + extrapolation"
echo "  python experiments/02_scale_curve.py"
echo "  Expected: ~30min"
echo ""

echo "[Phase 2] Certificate validation (K-fold)"
echo "  python experiments/03_certificate_validation.py"
echo "  Expected: ~1h"
echo ""

# ── Phase 3: Pre-registration (COMMIT BEFORE EXP 04) ─────────────────────────
echo "[Phase 3] *** STOP HERE ***"
echo "  Fill results/preregistration.md with Mamba-2 prediction."
echo "  Then: git add results/preregistration.md && git commit -m 'Pre-register Mamba-2 prediction'"
echo "  The commit hash is your evidence of priority."
echo ""

# ── Phase 4: New compute (Kaggle T4) ─────────────────────────────────────────
echo "[Phase 4] Mamba-2 architecture transfer — RUN ON KAGGLE T4"
echo "  Upload experiments/04_mamba_transfer.py to Kaggle."
echo "  Download 04_mamba_hidden_states.npz → results/hidden_states/"
echo "  Download 04_mamba_transfer.json → results/logs/"
echo ""

# ── Phase 5: Depth + boundary ─────────────────────────────────────────────────
echo "[Phase 5] Depth fraction universality"
echo "  python experiments/05_depth_fraction.py"
echo "  Requires: Exp 01 output (and Exp 04 if available)"
echo ""

echo "[Phase 5] Boundary conditions + central figure"
echo "  python experiments/06_boundary_conditions.py"
echo "  Requires: Exp 01 output"
echo ""

# ── Phase 6: OT + Spectral + Conformal ───────────────────────────────────────
echo "[Phase 6] OT certificate (Direction B — ~3-4h)"
echo "  python experiments/08_ot_certificate.py"
echo ""

echo "[Phase 6] Spectral phase transition (Direction A — ~2-3h)"
echo "  python experiments/09_spectral_phase_transition.py"
echo ""

echo "[Phase 6] Conformal coverage guarantee (Direction C — ~15min)"
echo "  python experiments/10_conformal_coverage.py"
echo ""

# ── Phase 7: Qwen scale curve ─────────────────────────────────────────────────
echo "[Phase 7] Qwen 2.5 controlled scale curve"
echo "  python experiments/11_qwen_scale_curve.py"
echo "  Requires: Qwen2.5-0.5B and 1.5B extraction (~4-6h CPU)"
echo "  Or: run on Kaggle T4 for speed"
echo ""

# ── Dashboard ─────────────────────────────────────────────────────────────────
echo "[Final] Launch dashboard"
echo "  streamlit run app.py"
echo ""

echo "============================================================"
echo "EXECUTION ORDER SUMMARY"
echo "============================================================"
echo ""
echo "  PHASE 0:  Exp 00  — HaluEval extraction (CPU, ~3h/model)"
echo "  PHASE 1:  Exp 07  — LLM-as-Judge re-labeling (CPU + Ollama)"
echo "  PHASE 2:  Exp 01  — Three certificates (CPU, ~4h)"
echo "            Exp 02  — Scale curve (CPU, ~30min)"
echo "            Exp 03  — K-fold validation (CPU, ~1h)"
echo "  PHASE 3:  *** PRE-REGISTER Mamba-2 prediction (git commit) ***"
echo "  PHASE 4:  Exp 04  — Mamba-2 transfer (Kaggle T4)"
echo "  PHASE 5:  Exp 05  — Depth fraction (CPU, ~1h)"
echo "            Exp 06  — Boundary conditions (CPU, ~1h)"
echo "  PHASE 6:  Exp 08  — OT certificate (CPU, ~4h)"
echo "            Exp 09  — Spectral (CPU, ~3h)"
echo "            Exp 10  — Conformal (CPU, ~15min)"
echo "  PHASE 7:  Exp 11  — Qwen scale curve (CPU or Kaggle)"
echo ""
echo "  All experiments: \$0 cost."
echo "  Total estimated time: ~2 weeks running in parallel where possible."
echo "============================================================"
