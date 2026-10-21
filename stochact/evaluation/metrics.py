"""Task-independent metrics for probabilistic predictions, without sklearn.

Metric conventions:
- Multiclass Brier = sum_c (p_c - 1[y=c])^2 per sample/pixel.
- Binary Brier = (p - y)^2 per sample/pixel.
- ECE = top-label (or binary decision) confidence ECE, equal-width bins.
- OOD positive class = out of distribution; higher scores signal OOD.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_valid(probabilities: torch.Tensor, targets: torch.Tensor, *,
                   mode: str, task: str, ignore_index: int | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N,C] multiclass or [N] binary probs and [N] target labels."""
    if mode not in ("binary", "multiclass") or task not in ("classification", "segmentation"):
        raise ValueError("Expected a classification/segmentation task and binary/multiclass mode")
    p = probabilities.detach()
    y = targets.detach()
    if p.ndim < 1 or p.shape[0] == 0:
        raise ValueError("Probabilities must have a nonempty batch axis")
    if mode == "binary":
        if task == "classification":
            if p.ndim == 2 and p.shape[1] == 1:
                p = p[:, 0]
            if p.ndim != 1:
                raise ValueError("Binary classification requires probabilities [B] or [B,1]")
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
        else:
            if p.ndim >= 4 and p.shape[1] == 1:
                p = p[:, 0]
            if y.ndim == p.ndim + 1 and y.shape[1] == 1:
                y = y[:, 0]
        if p.shape != y.shape:
            raise ValueError(f"Probability/target shape mismatch: {tuple(p.shape)}, {tuple(y.shape)}")
        p = p.reshape(-1)
    else:
        if p.ndim < 2 or p.shape[1] < 2:
            raise ValueError("Multiclass probabilities require [B,C,...] with C>=2")
        if y.ndim == p.ndim and y.shape[1] == 1:
            y = y[:, 0]
        if y.shape != (p.shape[0], *p.shape[2:]):
            raise ValueError("Multiclass targets must have shape [B,*spatial]")
        p = p.movedim(1, -1).reshape(-1, p.shape[1])
    y = y.reshape(-1).long().to(p.device)
    mask = torch.ones_like(y, dtype=torch.bool) if ignore_index is None else y != ignore_index
    p = p[mask]
    y = y[mask]
    if not p.numel():
        return p, y
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if mode == "binary":
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("Binary targets must be 0/1 outside ignore_index")
    else:
        if (y < 0).any() or (y >= p.shape[1]).any():
            raise ValueError("Multiclass target outside class range")
        if not torch.allclose(p.sum(-1), torch.ones_like(y, dtype=p.dtype), atol=1e-4, rtol=1e-4):
            raise ValueError("Multiclass probabilities must sum to 1")
    return p, y


def calibration_metrics(probabilities: torch.Tensor, targets: torch.Tensor, *, mode: str,
                        task: str = "classification", n_bins: int = 15,
                        threshold: float = 0.5, ignore_index: int | None = None,
                        ) -> dict[str, float]:
    """Calculate ECE, Brier score and NLL from normalized probabilities."""
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    p, y = _flatten_valid(probabilities, targets, mode=mode, task=task, ignore_index=ignore_index)
    if y.numel() == 0:
        raise ValueError("No valid targets")
    confidence, correct, nll, brier = _probabilistic_terms(p, y, mode, threshold)
    ids = torch.clamp((confidence * n_bins).long(), max=n_bins - 1)
    counts = torch.bincount(ids, minlength=n_bins)
    ece = torch.zeros((), dtype=torch.float64, device=p.device)
    for i in range(n_bins):
        valid = ids == i
        if counts[i]:
            ece = ece + (counts[i] / len(ids)) * (correct[valid].float().mean() - confidence[valid].mean()).abs()
    return {"ece": float(ece), "brier": float(brier.mean()), "nll": float(nll.mean())}


def _probabilistic_terms(p: torch.Tensor, y: torch.Tensor, mode: str, threshold: float):
    eps = 1e-7
    if mode == "binary":
        confidence = torch.where(p >= threshold, p, 1 - p)
        correct = (p >= threshold) == (y == 1)
        nll = F.binary_cross_entropy(p.float().clamp(eps, 1-eps), y.float(), reduction="none")
        brier = (p.float() - y.float()).square()
    else:
        confidence, predicted = p.max(dim=1)
        correct = predicted == y
        nll = -p.float().clamp_min(eps).log().gather(1, y[:, None]).squeeze(1)
        one_hot = F.one_hot(y, p.shape[1]).float()
        brier = (p.float() - one_hot).square().sum(dim=1)
    return confidence, correct, nll, brier


def _ccq_counts(predictions: torch.Tensor, targets: torch.Tensor, *, tolerance: int = 0,
                ignore_index: int | None = None) -> tuple[int, int, int, int]:
    """Number of matched prediction/target positives and their respective totals."""
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative integer")
    p, y = predictions.detach(), targets.detach()
    if p.ndim == 4 and p.shape[1] == 1:
        p = p[:, 0]
    if y.ndim == 4 and y.shape[1] == 1:
        y = y[:, 0]
    if p.shape != y.shape or p.ndim != 3:
        raise ValueError("binary_ccq expects matching [B,H,W] 2D binary masks")
    valid = torch.ones_like(y, dtype=torch.bool) if ignore_index is None else y != ignore_index
    pred = (p != 0) & valid
    gt = (y != 0) & valid
    if tolerance:
        k = 2*tolerance + 1
        expanded_gt = F.max_pool2d(gt[:, None].float(), k, stride=1, padding=tolerance)[:, 0].bool()
        expanded_pred = F.max_pool2d(pred[:, None].float(), k, stride=1, padding=tolerance)[:, 0].bool()
    else:
        expanded_gt, expanded_pred = gt, pred
    return (int((pred & expanded_gt).sum()), int(pred.sum()),
            int((gt & expanded_pred).sum()), int(gt.sum()))


def binary_ccq(predictions: torch.Tensor, targets: torch.Tensor, *, tolerance: int = 0,
               ignore_index: int | None = None) -> dict[str, float]:
    """Spatial-tolerance correctness/completeness/quality for 2D binary masks.

    Correctness: fraction of predicted foreground within `tolerance` pixels of GT.
    Completeness: fraction of GT foreground within `tolerance` pixels of prediction.
    Quality: C*R / (C + R - C*R), zero when no matching foreground exists.
    For tolerance=0, these reduce to precision, recall, IoU.
    This is a morphological-tolerance formulation, not a replacement for
    benchmark-specific centerline matching protocols.
    """
    matched_pred, n_pred, matched_gt, n_gt = _ccq_counts(
        predictions, targets, tolerance=tolerance, ignore_index=ignore_index)
    correct = matched_pred / n_pred if n_pred else (1.0 if not n_gt else 0.0)
    complete = matched_gt / n_gt if n_gt else (1.0 if not n_pred else 0.0)
    denom = correct + complete - correct * complete
    quality = correct * complete / denom if denom > 0 else 0.0
    return {"correctness": correct, "completeness": complete, "quality": quality}


def binary_curves(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """Tie-aware binary ROC/PR curves, OOD positives=1; no sklearn dependency.

    AUROC is trapezoidal ROC area; AUPR is average-precision (stepwise PR area).
    FPR95 is the lowest achievable FPR with TPR >= 0.95.
    """
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    labels = torch.as_tensor(labels).flatten().long().cpu()
    if scores.numel() != labels.numel() or scores.numel() < 2:
        raise ValueError("scores/labels require matching lengths >= 2")
    if not torch.isfinite(scores).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("scores must be finite and labels must be 0/1")
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Both positive and negative examples are required")
    idx = torch.argsort(scores, descending=True, stable=True)
    sorted_scores, sorted_labels = scores[idx], labels[idx]
    ends = torch.cat((torch.nonzero(sorted_scores[1:] != sorted_scores[:-1]).flatten(),
                      torch.tensor([scores.numel()-1])))
    tp = torch.cat((torch.zeros(1, dtype=torch.float64), sorted_labels.cumsum(0)[ends].double()))
    fp = torch.cat((torch.zeros(1, dtype=torch.float64), (1-sorted_labels).cumsum(0)[ends].double()))
    tpr, fpr = tp/positives, fp/negatives
    precision = torch.cat((torch.ones(1, dtype=torch.float64), tp[1:]/(tp[1:]+fp[1:])))
    recall = tpr
    auroc = float(torch.trapz(tpr, fpr))
    aupr = float(((recall[1:] - recall[:-1]) * precision[1:]).sum())
    fpr95 = float(fpr[tpr >= 0.95].min())
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall,
            "auroc": auroc, "aupr": aupr, "fpr95": fpr95}
