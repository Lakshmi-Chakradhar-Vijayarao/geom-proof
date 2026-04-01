"""
Fisher Separability Certificate.

The certificate interface: given a model's hidden-state distribution at a layer,
predict whether a linear hallucination probe will achieve above-chance AUROC —
before fitting any probe.

Usage:
    cert = Certificate.from_hidden_states(H, labels)
    print(cert.predict_auroc())     # predicted AUROC before any probe
    print(cert.is_governance_grade())  # True if predicted AUROC > 0.70
"""

import numpy as np
from dataclasses import dataclass
from src.fisher import fisher_ratio, auroc_bound, fisher_curve


# Threshold for "governance-grade" signal (from HaRP: ECE = 0.039 at AUROC 0.775)
GOVERNANCE_THRESHOLD = 0.70

# Threshold for "above chance" (weak but detectable signal)
ABOVE_CHANCE_THRESHOLD = 0.55


@dataclass
class Certificate:
    """
    A pre-probe certificate for a single (model, layer) configuration.

    Fields
    ------
    model_name : str
    n_params : int or None
    layer : int
    n_layers : int
    J : float — Fisher ratio
    auroc_predicted : float — Phi(sqrt(J)/2)
    method : str — "lda" or "pca"
    depth_fraction : float — layer / (n_layers - 1)
    """

    model_name: str
    n_params: int | None
    layer: int
    n_layers: int
    J: float
    auroc_predicted: float
    method: str
    depth_fraction: float

    @classmethod
    def from_hidden_states(
        cls,
        H: np.ndarray,
        labels: np.ndarray,
        layer: int,
        n_layers: int,
        model_name: str = "unknown",
        n_params: int | None = None,
        method: str = "lda",
        n_components: int = 100,
    ) -> "Certificate":
        """
        Compute certificate from hidden states at a single layer.

        Parameters
        ----------
        H : ndarray (n_samples, d) — hidden states at one layer
        labels : ndarray (n_samples,) — binary labels
        layer : int — layer index (0-based)
        n_layers : int — total layers in model
        """
        J = fisher_ratio(H, labels, method=method, n_components=n_components)
        predicted = auroc_bound(J)
        depth_frac = layer / max(n_layers - 1, 1)
        return cls(
            model_name=model_name,
            n_params=n_params,
            layer=layer,
            n_layers=n_layers,
            J=J,
            auroc_predicted=predicted,
            method=method,
            depth_fraction=depth_frac,
        )

    def predict_auroc(self) -> float:
        """Return predicted AUROC from Fisher bound."""
        return self.auroc_predicted

    def is_governance_grade(self) -> bool:
        """True if predicted AUROC >= 0.70 (governance-worthy signal)."""
        return self.auroc_predicted >= GOVERNANCE_THRESHOLD

    def is_above_chance(self) -> bool:
        """True if predicted AUROC >= 0.55 (detectable but weak signal)."""
        return self.auroc_predicted >= ABOVE_CHANCE_THRESHOLD

    def verdict(self) -> str:
        if self.is_governance_grade():
            return "GOVERNANCE_GRADE"
        elif self.is_above_chance():
            return "WEAK_SIGNAL"
        else:
            return "NULL"

    def summary(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_params": self.n_params,
            "layer": self.layer,
            "n_layers": self.n_layers,
            "depth_fraction": round(self.depth_fraction, 4),
            "J": round(self.J, 6),
            "auroc_predicted": round(self.auroc_predicted, 4),
            "verdict": self.verdict(),
            "method": self.method,
        }


def best_layer_certificate(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    model_name: str = "unknown",
    n_params: int | None = None,
    method: str = "lda",
    n_components: int = 100,
    verbose: bool = True,
) -> Certificate:
    """
    Compute certificates at all layers and return the one with the highest J.

    This is the primary interface for Exps 01, 03, 04, 05, 06.

    Parameters
    ----------
    hidden_states : ndarray (n_samples, n_layers, d)
    labels : ndarray (n_samples,)
    """
    if verbose:
        print(f"Computing Fisher curves for {model_name} ...")
    curve = fisher_curve(
        hidden_states, labels, method=method, n_components=n_components, verbose=verbose
    )
    best_L = curve["best_layer"]
    n_layers = curve["n_layers"]
    H_best = hidden_states[:, best_L, :]
    return Certificate.from_hidden_states(
        H=H_best,
        labels=labels,
        layer=best_L,
        n_layers=n_layers,
        model_name=model_name,
        n_params=n_params,
        method=method,
        n_components=n_components,
    )
