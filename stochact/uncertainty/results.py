"""Results returned by Monte Carlo prediction. No training/evaluation dependency."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictionResult:
    """Statistics of Monte Carlo stochastic-activation inference.

    All tensors remain on the model's output device, with gradients disabled.

    ``mean`` / ``probabilities`` are class probabilities for classification and
    segmentation, or raw continuous outputs for regression. ``variance`` is
    population variance (divide by T), elementwise over stochastic passes.

    ``predictions`` are integer labels for classification and segmentation,
    or the continuous predictive mean for regression. ``uncertainty`` is one
    scalar per image/example, obtained by averaging predictive variance over
    all non-batch dimensions. ``uncertainty_map`` is returned for segmentation:
    for multiclass predictions it is the *mean* class-wise variance; for binary
    predictions it is the positive-class variance. This is NOT a calibrated
    probability of segmentation error.

    For classification and segmentation, entropy and mutual information are
    computed from categorical or Bernoulli predictions. Mutual information is
    entropy(mean probability) minus mean entropy(probability per pass), a
    measure of stochastic prediction disagreement, not a guaranteed full
    Bayesian epistemic-uncertainty decomposition.

    ``samples`` is None by default to avoid storing T large prediction tensors.
    If requested, it is shaped [T, *output_shape] and contains probabilities
    (or continuous outputs for regression).
    """

    task: str
    mode: str | None
    n_samples: int
    mean: torch.Tensor
    variance: torch.Tensor
    predictions: torch.Tensor
    uncertainty: torch.Tensor
    predictive_entropy: torch.Tensor | None = None
    expected_entropy: torch.Tensor | None = None
    mutual_information: torch.Tensor | None = None
    uncertainty_map: torch.Tensor | None = None
    samples: torch.Tensor | None = None

    @property
    def probabilities(self) -> torch.Tensor:
        """Alias for mean; only defined for classification and segmentation."""
        if self.task == "regression":
            raise AttributeError("Regression outputs are continuous; access result.mean instead")
        return self.mean

    @property
    def classes(self) -> torch.Tensor:
        """Predicted class indices, for classification only."""
        if self.task != "classification":
            raise AttributeError("result.classes is only for classification")
        return self.predictions

    @property
    def mask(self) -> torch.Tensor:
        """Predicted segmentation mask, for segmentation only."""
        if self.task != "segmentation":
            raise AttributeError("result.mask is only for segmentation")
        return self.predictions
