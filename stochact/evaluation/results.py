"""Optional rich evaluation results for plotting and further analysis."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class EvaluationResult:
    task: str
    metrics: dict[str, float]
    reliability: dict[str, torch.Tensor]
    uncertainty: torch.Tensor
    per_example_error: torch.Tensor
    confusion_matrix: torch.Tensor

    def __getitem__(self, metric: str) -> float:
        return self.metrics[metric]


@dataclass(frozen=True)
class OODResult:
    metrics: dict[str, float]
    id_scores: torch.Tensor
    ood_scores: torch.Tensor
    curves: dict[str, torch.Tensor | float]
    score: str

    def __getitem__(self, metric: str) -> float:
        return self.metrics[metric]
