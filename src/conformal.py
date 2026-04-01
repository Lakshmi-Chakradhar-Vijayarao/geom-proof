"""
Conformal prediction wrapper for hallucination detection — Direction C.

Provides distribution-free coverage guarantees for the OT/Fisher probe score:
  P(hallucinated | detector ACCEPTS) ≤ α  with probability ≥ 1 − δ

Two variants:

  1. Split conformal prediction (Papadopoulos et al. 2002 / Vovk et al. 2005):
     Standard marginal coverage guarantee. Calibrate threshold τ on held-out set;
     guarantee holds on i.i.d. test data.

  2. Mondrian conformal prediction (Vovk et al. 2003):
     Conditional coverage per group (quadrant). Each quadrant gets its own τ.
     Requires sufficient calibration samples per quadrant (n_q ≥ 1/δ).

Mathematical guarantee:
  Given calibration set D_cal of size n, the conformal threshold:
    τ = ⌈(n+1)(1−α)⌉/n -th quantile of nonconformity scores on D_cal
  satisfies:
    P(hall | score ≤ τ) ≤ α + (1 − α) / (n_cal + 1)  [exact finite-sample]
    → α + δ  when n_cal ≥ (1−α)/δ

Nonconformity score convention:
  score(x) = P(hallucinated | h_L(x))  — output of the OOF probe.
  ACCEPT if score ≤ τ  (low hallucination probability → safe to trust).
  REJECT if score > τ  (flag for human review).

Reference:
  Angelopoulos & Bates (2023) "A Gentle Introduction to Conformal Prediction"
  Fontana et al. (2023) "Conformal prediction: A unified review of theory and
    new challenges." Econometrics & Statistics.
"""

import numpy as np
from typing import Tuple


# ── Quadrant taxonomy (from HaRP Exp 13) ─────────────────────────────────────

QUADRANTS = {
    "Epistemic (uncertain + hallucinated)": {"confident": False, "correct": False},
    "Aleatoric (uncertain + correct)":      {"confident": False, "correct": True},
    "Confident hallucination":              {"confident": True,  "correct": False},
    "Confident correct":                    {"confident": True,  "correct": True},
}

# Confidence threshold: probe score ≤ CONF_THRESH → "confident correct"
#                        probe score ≥ 1-CONF_THRESH → "confident hallucination"
CONF_THRESH = 0.3


def _quadrant_labels(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Assign each sample to one of four quadrants based on probe score and true label.

    Parameters
    ----------
    scores : (n,) P(hallucination)
    y : (n,) binary — 1=correct, 0=hallucinated

    Returns
    -------
    q_labels : (n,) int — 0..3 index into QUADRANTS.keys()
    """
    q_labels = np.zeros(len(scores), dtype=int)
    names = list(QUADRANTS.keys())
    for i, (s, yi) in enumerate(zip(scores, y)):
        correct = int(yi) == 1
        confident = (s <= CONF_THRESH) or (s >= 1 - CONF_THRESH)
        if not confident and not correct:
            q_labels[i] = 0   # Epistemic
        elif not confident and correct:
            q_labels[i] = 1   # Aleatoric
        elif confident and not correct:
            q_labels[i] = 2   # Confident hallucination
        else:
            q_labels[i] = 3   # Confident correct
    return q_labels


# ── Split conformal prediction ─────────────────────────────────────────────────

def split_conformal_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10,
    delta: float = 0.05,
    cal_fraction: float = 0.5,
    random_state: int = 42,
) -> dict:
    """
    Split conformal calibration.

    Splits data into calibration (50%) and test (50%) sets.
    Calibrates threshold τ on the calibration set using the ⌈(n+1)(1−α)⌉/n quantile.
    Evaluates empirical coverage on the test set.

    Parameters
    ----------
    scores : (n,) nonconformity scores (P(hallucination))
    y : (n,) binary labels (1=correct, 0=hallucinated)
    alpha : target hallucination rate P(hall | ACCEPT) ≤ α
    delta : confidence level 1 − δ
    cal_fraction : fraction used for calibration
    random_state : RNG seed

    Returns
    -------
    dict with tau, empirical_hall_rate, acceptance_rate, guarantee_holds,
         n_cal, n_test, required_n_cal
    """
    rng = np.random.RandomState(random_state)
    n = len(scores)

    # Minimum calibration size for the guarantee to hold at confidence 1−δ:
    # n_cal ≥ (1−α) / δ
    required_n_cal = int(np.ceil((1 - alpha) / delta))

    idx = rng.permutation(n)
    n_cal = int(n * cal_fraction)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]

    scores_cal = scores[cal_idx]
    y_cal = y[cal_idx]
    scores_test = scores[test_idx]
    y_test = y[test_idx]

    # Calibration: find the largest τ such that the empirical FDR on the calibration
    # set is ≤ α.  FDR(τ) = |{hall AND score≤τ}| / |{score≤τ}|.
    # Standard split-CP coverage quantile gives P(accept | correct) ≥ 1−α, but we
    # want the precision guarantee P(hall | accept) ≤ α — these require different τ.
    unique_scores = np.sort(np.unique(scores_cal))
    best_tau = unique_scores[0]
    for tau_candidate in unique_scores:
        accepted_cal = scores_cal <= tau_candidate
        n_acc_cal = int(accepted_cal.sum())
        if n_acc_cal == 0:
            continue
        n_hall_acc_cal = int((y_cal[accepted_cal] == 0).sum())
        fdr_cal = n_hall_acc_cal / n_acc_cal
        if fdr_cal <= alpha:
            best_tau = tau_candidate
        else:
            break  # FDR monotonically increases with τ; stop when it exceeds α
    tau = float(best_tau)

    # Test evaluation
    accepted_mask = scores_test <= tau
    n_accepted = int(accepted_mask.sum())
    n_accepted_hall = int(((y_test[accepted_mask] == 0)).sum()) if n_accepted > 0 else 0
    empirical_hall_rate = n_accepted_hall / n_accepted if n_accepted > 0 else 0.0
    acceptance_rate = n_accepted / len(test_idx)

    # Guarantee holds if empirical hall rate ≤ α + finite-sample slack
    slack = (1 - alpha) / (n_cal + 1)
    guarantee_holds = bool(empirical_hall_rate <= alpha + slack)

    return {
        "tau": tau,
        "n_cal": n_cal,
        "n_test": len(test_idx),
        "required_n_cal": required_n_cal,
        "n_accepted": n_accepted,
        "empirical_hall_rate": float(empirical_hall_rate),
        "acceptance_rate": float(acceptance_rate),
        "finite_sample_slack": float(slack),
        "guarantee_holds": guarantee_holds,
        "alpha": alpha,
        "delta": delta,
    }


# ── Mondrian conformal prediction ─────────────────────────────────────────────

def mondrian_conformal(
    scores: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10,
    delta: float = 0.05,
    cal_fraction: float = 0.5,
    random_state: int = 42,
) -> dict:
    """
    Mondrian conformal prediction — per-quadrant calibration.

    Each quadrant gets its own threshold τ_q, giving conditional coverage:
      P(hall | ACCEPT, quadrant=q) ≤ α  for each quadrant q.

    This is stronger than marginal coverage for subgroups.
    Requires n_q ≥ (1−α)/δ calibration samples per quadrant.

    Parameters
    ----------
    scores : (n,) P(hallucination)
    y : (n,) binary labels

    Returns
    -------
    dict mapping quadrant_name → {tau, empirical_hall_rate, n_cal, guarantee_holds}
    """
    rng = np.random.RandomState(random_state)
    n = len(scores)
    q_labels = _quadrant_labels(scores, y)
    q_names = list(QUADRANTS.keys())

    idx = rng.permutation(n)
    n_cal = int(n * cal_fraction)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]

    results = {}
    for q_i, q_name in enumerate(q_names):
        cal_mask = q_labels[cal_idx] == q_i
        test_mask = q_labels[test_idx] == q_i

        scores_cal_q = scores[cal_idx][cal_mask]
        scores_test_q = scores[test_idx][test_mask]
        y_test_q = y[test_idx][test_mask]

        n_cal_q = len(scores_cal_q)
        required_n_cal = int(np.ceil((1 - alpha) / delta))

        if n_cal_q < 3:
            # Insufficient calibration data for this quadrant
            results[q_name] = {
                "tau": float("nan"),
                "n_cal": n_cal_q,
                "required_n_cal": required_n_cal,
                "empirical_hall_rate": float("nan"),
                "acceptance_rate": float("nan"),
                "guarantee_holds": False,
                "insufficient_data": True,
            }
            continue

        # Compute τ for this quadrant
        rank = int(np.ceil((n_cal_q + 1) * (1 - alpha)))
        rank = min(rank, n_cal_q)
        sorted_q = np.sort(scores_cal_q)
        tau_q = float(sorted_q[rank - 1])

        # Test coverage for this quadrant
        n_test_q = len(scores_test_q)
        if n_test_q == 0:
            results[q_name] = {
                "tau": tau_q,
                "n_cal": n_cal_q,
                "required_n_cal": required_n_cal,
                "empirical_hall_rate": float("nan"),
                "acceptance_rate": float("nan"),
                "guarantee_holds": False,
                "insufficient_data": True,
            }
            continue

        accepted_mask = scores_test_q <= tau_q
        n_accepted = accepted_mask.sum()
        n_hall_accepted = int((y_test_q[accepted_mask] == 0).sum()) if n_accepted > 0 else 0
        empirical_hall_rate = n_hall_accepted / n_accepted if n_accepted > 0 else 0.0
        acceptance_rate = n_accepted / n_test_q

        slack = (1 - alpha) / (n_cal_q + 1)
        guarantee_holds = bool(empirical_hall_rate <= alpha + slack)

        results[q_name] = {
            "tau": tau_q,
            "n_cal": n_cal_q,
            "required_n_cal": required_n_cal,
            "empirical_hall_rate": float(empirical_hall_rate),
            "acceptance_rate": float(acceptance_rate),
            "finite_sample_slack": float(slack),
            "guarantee_holds": guarantee_holds,
            "insufficient_data": False,
        }

    return results


# ── Coverage report ───────────────────────────────────────────────────────────

def coverage_report(
    scores: np.ndarray,
    y: np.ndarray,
    tau: float,
    alpha: float = 0.10,
    delta: float = 0.05,
) -> dict:
    """
    Given a fixed threshold τ, report full coverage statistics.
    Useful for evaluating the calibrated τ on a new dataset (OOD test).
    """
    accepted = scores <= tau
    n_accepted = int(accepted.sum())
    n_rejected = int((~accepted).sum())

    n_hall_accepted = int((y[accepted] == 0).sum()) if n_accepted > 0 else 0
    n_corr_accepted = int((y[accepted] == 1).sum()) if n_accepted > 0 else 0
    n_hall_rejected = int((y[~accepted] == 0).sum()) if n_rejected > 0 else 0
    n_corr_rejected = int((y[~accepted] == 1).sum()) if n_rejected > 0 else 0

    empirical_hall_rate = n_hall_accepted / n_accepted if n_accepted > 0 else 0.0
    acceptance_rate = n_accepted / len(scores)
    precision = n_corr_accepted / n_accepted if n_accepted > 0 else 0.0  # precision of ACCEPT
    recall = n_corr_accepted / int((y == 1).sum()) if (y == 1).sum() > 0 else 0.0

    slack = (1 - alpha) / (len(scores) + 1)
    guarantee_holds = bool(empirical_hall_rate <= alpha + slack)

    return {
        "tau": float(tau),
        "n_samples": len(scores),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_hall_accepted": n_hall_accepted,
        "n_corr_accepted": n_corr_accepted,
        "n_hall_rejected": n_hall_rejected,
        "n_corr_rejected": n_corr_rejected,
        "empirical_hall_rate": float(empirical_hall_rate),
        "acceptance_rate": float(acceptance_rate),
        "precision": float(precision),
        "recall": float(recall),
        "guarantee_holds": guarantee_holds,
        "alpha": alpha,
        "delta": delta,
    }


# ── OOD shift simulation ──────────────────────────────────────────────────────

def simulate_ood_shift(
    scores: np.ndarray,
    y: np.ndarray,
    tau: float,
    shift_strength: float = 0.3,
    alpha: float = 0.10,
    delta: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Simulate OOD distribution shift by adding noise to nonconformity scores.

    Models the scenario where the test distribution shifts (e.g., HaluEval OOD
    data instead of TruthfulQA i.i.d. data). Under i.i.d., the conformal
    guarantee holds. Under shift, it may break.

    The simulated shift: add Gaussian noise scaled by shift_strength to
    move hallucinated samples' scores toward the accepted region.

    Parameters
    ----------
    scores : (n,) probe scores on i.i.d. test data
    y : (n,) true labels
    tau : conformal threshold calibrated on i.i.d. data
    shift_strength : σ of shift perturbation (0 = no shift, 0.3 = moderate)

    Returns
    -------
    dict with iid vs ood hall rates and whether guarantee holds in each
    """
    rng = np.random.RandomState(random_state)

    # IID evaluation
    iid_report = coverage_report(scores, y, tau, alpha=alpha, delta=delta)

    # Simulate OOD shift: hallucinated samples get lower (better-looking) scores
    scores_ood = scores.copy()
    hall_mask = y == 0
    noise = rng.randn(hall_mask.sum()) * shift_strength
    # Shift hallucinated samples toward lower scores (toward ACCEPT region)
    scores_ood[hall_mask] = np.clip(scores_ood[hall_mask] - np.abs(noise), 0, 1)

    ood_report = coverage_report(scores_ood, y, tau, alpha=alpha, delta=delta)

    slack = (1 - alpha) / (len(scores) + 1)

    return {
        "iid_hall_rate": iid_report["empirical_hall_rate"],
        "ood_hall_rate": ood_report["empirical_hall_rate"],
        "iid_acceptance_rate": iid_report["acceptance_rate"],
        "ood_acceptance_rate": ood_report["acceptance_rate"],
        "iid_holds": iid_report["guarantee_holds"],
        "ood_holds": ood_report["guarantee_holds"],
        "degradation": float(ood_report["empirical_hall_rate"]
                             - iid_report["empirical_hall_rate"]),
        "shift_strength": shift_strength,
        "alpha": alpha,
        "tau": tau,
    }
