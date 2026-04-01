"""
Scale curve fitting: log(params) → Fisher ratio and AUROC.

Fits a log-linear relationship between model size and best-layer Fisher ratio J*.
Uses three confirmed data points (GPT-2 117M, GPT-2 Medium 345M, Qwen 2.5 3B).
Extrapolates to 7B with bootstrapped confidence intervals.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm


# ── Confirmed data points ──────────────────────────────────────────────────────

SCALE_DATA = {
    "GPT-2 117M":        {"params": 117e6,  "auroc": 0.604, "source": "MECH-INT Step 4C"},
    "GPT-2 Medium 345M": {"params": 345e6,  "auroc": 0.579, "source": "HaluEval Job A"},
    "Qwen 2.5 3B":       {"params": 3e9,    "auroc": 0.775, "source": "HaluEval Job A"},
}


def _logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def _log_linear_auroc(log_params, a, b):
    """AUROC = sigmoid(a * log10(params) + b)."""
    return _sigmoid(a * log_params + b)


def fit_scale_curve(data: dict | None = None) -> dict:
    """
    Fit a log-linear relationship: AUROC = sigmoid(a * log10(params) + b).

    Parameters
    ----------
    data : dict or None
        Dict of {name: {"params": float, "auroc": float}}.
        Defaults to SCALE_DATA (the three confirmed data points).

    Returns
    -------
    dict with keys:
        'a', 'b' : float — fit coefficients
        'r_squared' : float — goodness of fit
        'predict' : callable — f(params) -> predicted AUROC
        'residuals' : dict — per-model residual
    """
    if data is None:
        data = SCALE_DATA

    log_params = np.array([np.log10(v["params"]) for v in data.values()])
    aurocs = np.array([v["auroc"] for v in data.values()])

    (a, b), _ = curve_fit(_log_linear_auroc, log_params, aurocs, p0=[0.5, -4.0])

    predicted = _log_linear_auroc(log_params, a, b)
    ss_res = np.sum((aurocs - predicted) ** 2)
    ss_tot = np.sum((aurocs - aurocs.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    residuals = {name: float(aurocs[i] - predicted[i]) for i, name in enumerate(data)}

    def predict(params: float) -> float:
        return float(_log_linear_auroc(np.log10(params), a, b))

    return {
        "a": float(a),
        "b": float(b),
        "r_squared": float(r_squared),
        "predict": predict,
        "residuals": residuals,
        "fitted_values": {name: float(predicted[i]) for i, name in enumerate(data)},
    }


def bootstrap_prediction(
    params_target: float,
    data: dict | None = None,
    n_bootstrap: int = 10_000,
    ci: float = 0.90,
) -> dict:
    """
    Bootstrap confidence interval for the predicted AUROC at params_target.

    With only 3 data points, the CI will be wide — that is honest.

    Parameters
    ----------
    params_target : float
        Number of parameters for the target model (e.g., 7e9 for Mistral 7B).
    data : dict or None
    n_bootstrap : int
    ci : float — confidence level (0.90 = 90% CI)

    Returns
    -------
    dict with keys:
        'point_estimate' : float
        'ci_lower', 'ci_upper' : float
        'ci_level' : float
    """
    if data is None:
        data = SCALE_DATA

    names = list(data.keys())
    log_params = np.array([np.log10(v["params"]) for v in data.values()])
    aurocs = np.array([v["auroc"] for v in data.values()])
    n = len(names)
    log_target = np.log10(params_target)

    fit = fit_scale_curve(data)
    point_estimate = fit["predict"](params_target)

    # Residual bootstrap: resample residuals and add to fitted values
    residuals = aurocs - _log_linear_auroc(log_params, fit["a"], fit["b"])
    bootstrap_predictions = []

    for _ in range(n_bootstrap):
        resampled_residuals = np.random.choice(residuals, size=n, replace=True)
        aurocs_boot = np.clip(
            _log_linear_auroc(log_params, fit["a"], fit["b"]) + resampled_residuals,
            0.0, 1.0,
        )
        try:
            (a_b, b_b), _ = curve_fit(
                _log_linear_auroc, log_params, aurocs_boot, p0=[fit["a"], fit["b"]]
            )
            pred = float(_log_linear_auroc(log_target, a_b, b_b))
            bootstrap_predictions.append(np.clip(pred, 0.0, 1.0))
        except RuntimeError:
            pass

    alpha = (1 - ci) / 2
    ci_lower = float(np.quantile(bootstrap_predictions, alpha))
    ci_upper = float(np.quantile(bootstrap_predictions, 1 - alpha))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "n_bootstrap": len(bootstrap_predictions),
    }
