"""General-purpose, task-configured Monte Carlo evaluation without task wrappers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math
import torch
from torch import nn

from ..training.trainer import _batch_parts
from ..uncertainty import predict_mc
from .metrics import _flatten_valid, _probabilistic_terms, binary_curves, _ccq_counts
from .results import EvaluationResult, OODResult


class Evaluator:
    """Evaluate classification, segmentation or regression from MC predictions.

    Classification returns accuracy, macro precision/recall/F1, NLL, Brier, ECE.
    Segmentation returns pixel accuracy, foreground micro Dice/IoU/precision/recall,
    macro foreground Dice/IoU, NLL, Brier, ECE, and variance/error correlation.
    Regression returns MAE, MSE and RMSE. All reductions are sample/pixel-weighted,
    not unweighted averages of batch metrics. Background=class 0 for segmentation;
    absent foreground classes are omitted from macro foreground scores.
    """

    def __init__(self, model: nn.Module, *, task: str = "classification",
                 mode: str = "auto", n_samples: int = 5,
                 from_logits: bool = True, threshold: float = 0.5,
                 ignore_index: int | None = None, n_bins: int = 15,
                 ccq_tolerance: int = 0,
                 device: str | torch.device | None = None,
                 output_key: str | None = None, output_index: int | None = None):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")
        if task not in ("classification", "segmentation", "regression"):
            raise ValueError("Unsupported task")
        if mode not in ("auto", "binary", "multiclass"):
            raise ValueError("mode must be auto/binary/multiclass")
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        if not isinstance(ccq_tolerance, int) or isinstance(ccq_tolerance, bool) or ccq_tolerance < 0:
            raise ValueError("ccq_tolerance must be a nonnegative integer")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        if output_key is not None and output_index is not None:
            raise ValueError("Set only one of output_key or output_index")
        self.model, self.task, self.mode = model, task, mode
        self.n_samples, self.from_logits = n_samples, from_logits
        self.threshold, self.ignore_index, self.n_bins = threshold, ignore_index, n_bins
        self.ccq_tolerance = ccq_tolerance
        self.output_key, self.output_index = output_key, output_index
        try:
            inferred_device = next(model.parameters()).device
        except StopIteration:
            inferred_device = torch.device("cpu")
        self.device = torch.device(device) if device is not None else inferred_device

    def _predict(self, x: torch.Tensor, *, n_samples: int | None = None):
        return predict_mc(self.model, x.to(self.device), task=self.task,
                          mode=self.mode, n_samples=self.n_samples if n_samples is None else n_samples,
                          from_logits=self.from_logits, threshold=self.threshold,
                          output_key=self.output_key, output_index=self.output_index)

    def evaluate(self, loader: Any, *, return_details: bool = False) -> dict[str, float] | EvaluationResult:
        """Evaluate labeled loader of (inputs, targets) or recognized mapping batches.

        `return_details=True` returns bin statistics and per-example uncertainty
        for reliability and risk–coverage visualizations (O(N_examples) memory).
        """
        confusion = None
        sums = dict(nll=0.0, brier=0.0, absolute=0.0, squared=0.0)
        n_valid = 0
        bin_count = torch.zeros(self.n_bins, dtype=torch.float64)
        bin_conf = torch.zeros_like(bin_count)
        bin_acc = torch.zeros_like(bin_count)
        ccq_counts = [0, 0, 0, 0]
        corr_sums = torch.zeros(6, dtype=torch.float64)  # n, sumx, sumy, sumxx, sumyy, sumxy
        uncertainty_chunks, error_chunks = [], []
        for batch in loader:
            x, targets = _batch_parts(batch)
            if not torch.is_tensor(x):
                raise TypeError("Evaluator currently requires Tensor inputs for MC prediction")
            targets = targets.to(self.device)
            result = self._predict(x)
            if self.task == "regression":
                if result.mean.shape != targets.shape:
                    raise ValueError("Regression targets must match model output shape")
                if not torch.isfinite(targets).all():
                    raise ValueError("Regression targets must be finite")
                error = result.mean.float() - targets.float()
                sums["absolute"] += float(error.abs().sum())
                sums["squared"] += float(error.square().sum())
                n_valid += error.numel()
                per_error = error.square().flatten(1).mean(1).detach().cpu()
            else:
                p, y = _flatten_valid(result.mean, targets, mode=result.mode, task=self.task,
                                      ignore_index=self.ignore_index)
                if y.numel() == 0:
                    continue
                n_valid += y.numel()
                conf, correct, nll, brier = _probabilistic_terms(p, y, result.mode, self.threshold)
                sums["nll"] += float(nll.double().sum())
                sums["brier"] += float(brier.double().sum())
                bins = torch.clamp((conf.double() * self.n_bins).long(), max=self.n_bins-1)
                bin_count += torch.bincount(bins.cpu(), minlength=self.n_bins).double()
                bin_conf += torch.bincount(bins.cpu(), weights=conf.double().cpu(), minlength=self.n_bins)
                bin_acc += torch.bincount(bins.cpu(), weights=correct.double().cpu(), minlength=self.n_bins)
                labels = (p >= self.threshold).long() if result.mode == "binary" else p.argmax(1)
                k = 2 if result.mode == "binary" else p.shape[1]
                cm = torch.bincount((y*k + labels).cpu(), minlength=k*k).view(k, k).double()
                if confusion is None:
                    confusion = cm
                else:
                    if cm.shape != confusion.shape:
                        raise ValueError("Class count changed between evaluation batches")
                    confusion += cm
                if self.task == "classification":
                    # One example corresponds to one class prediction.
                    per_error = (~correct).float().cpu()
                else:
                    truth = targets
                    predicted = result.predictions
                    if truth.ndim == predicted.ndim+1 and truth.shape[1] == 1:
                        truth = truth[:, 0]
                    if truth.shape != predicted.shape:
                        raise ValueError("Segmentation prediction/target shapes differ")
                    if result.mode == "binary":
                        counts = _ccq_counts(predicted, truth, tolerance=self.ccq_tolerance,
                                              ignore_index=self.ignore_index)
                        ccq_counts = [a+b for a,b in zip(ccq_counts, counts)]
                    valid = torch.ones_like(truth, dtype=torch.bool) if self.ignore_index is None else truth != self.ignore_index
                    wrong = ((predicted != truth) & valid)
                    count = valid.flatten(1).sum(1)
                    per_error = (wrong.flatten(1).sum(1).float() / count.clamp_min(1)).cpu()
                    if result.uncertainty_map is not None:
                        umap = result.uncertainty_map[valid].double().cpu()
                        missed = wrong[valid].double().cpu()
                        if umap.numel():
                            corr_sums += torch.stack([torch.tensor(float(umap.numel()), dtype=torch.float64),
                                                       umap.sum(), missed.sum(), (umap**2).sum(),
                                                       (missed**2).sum(), (umap*missed).sum()])
            if return_details:
                uncertainty_chunks.append(result.uncertainty.detach().float().cpu())
                error_chunks.append(per_error)
        if n_valid == 0:
            raise ValueError("Evaluation loader contains no valid targets")
        if self.task == "regression":
            mse = sums["squared"] / n_valid
            metrics = {"mae": sums["absolute"]/n_valid, "mse": mse, "rmse": math.sqrt(mse)}
        else:
            assert confusion is not None
            tp = confusion.diag()
            support = confusion.sum(1)
            predicted_support = confusion.sum(0)
            fp, fn = predicted_support - tp, support - tp
            precision = tp / (tp + fp).clamp_min(1)
            recall = tp / (tp + fn).clamp_min(1)
            f1 = 2*tp / (2*tp + fp + fn).clamp_min(1)
            ece = float(((bin_acc / bin_count.clamp_min(1) - bin_conf / bin_count.clamp_min(1)).abs()
                         * bin_count).sum()/n_valid)
            metrics = {"ece": ece, "brier": sums["brier"]/n_valid, "nll": sums["nll"]/n_valid}
            if self.task == "classification":
                metrics.update(accuracy=float(tp.sum()/confusion.sum()),
                               precision_macro=float(precision.mean()), recall_macro=float(recall.mean()),
                               f1_macro=float(f1.mean()), f1_weighted=float((f1*support).sum()/support.sum()))
            else:
                fg_tp, fg_fp, fg_fn = tp[1:].sum(), fp[1:].sum(), fn[1:].sum()
                fg_dice = 2*fg_tp / (2*fg_tp + fg_fp + fg_fn).clamp_min(1)
                fg_iou = fg_tp / (fg_tp + fg_fp + fg_fn).clamp_min(1)
                fg_present = (support[1:]+predicted_support[1:]) > 0
                metrics.update(pixel_accuracy=float(tp.sum()/confusion.sum()),
                               foreground_dice=float(fg_dice), foreground_iou=float(fg_iou),
                               foreground_precision=float(fg_tp/(fg_tp+fg_fp).clamp_min(1)),
                               foreground_recall=float(fg_tp/(fg_tp+fg_fn).clamp_min(1)),
                               dice_macro_fg=float(f1[1:][fg_present].mean()) if fg_present.any() else 1.0,
                               iou_macro_fg=float((tp[1:]/(tp[1:]+fp[1:]+fn[1:]).clamp_min(1))[fg_present].mean())
                               if fg_present.any() else 1.0)
                if confusion.shape[0] == 2:
                    matched_pred, total_pred, matched_gt, total_gt = ccq_counts
                    correctness = matched_pred / total_pred if total_pred else (1. if not total_gt else 0.)
                    completeness = matched_gt / total_gt if total_gt else (1. if not total_pred else 0.)
                    denom_ccq = correctness + completeness - correctness*completeness
                    quality = correctness * completeness / denom_ccq if denom_ccq else 0.
                    metrics.update(correctness=correctness, completeness=completeness, quality=quality)
                n, sx, sy, sxx, syy, sxy = [float(v) for v in corr_sums]
                cov = n*sxy - sx*sy
                denom = (n*sxx - sx*sx)*(n*syy - sy*sy)
                metrics["uncertainty_error_corr"] = cov/math.sqrt(denom) if denom > 1e-24 else float("nan")
        if not return_details:
            return metrics
        reliability = {"count": bin_count, "mean_confidence": bin_conf / bin_count.clamp_min(1),
                       "accuracy": bin_acc / bin_count.clamp_min(1),
                       "bin_edges": torch.linspace(0, 1, self.n_bins+1)}
        return EvaluationResult(self.task, metrics, reliability,
                                torch.cat(uncertainty_chunks) if uncertainty_chunks else torch.empty(0),
                                torch.cat(error_chunks) if error_chunks else torch.empty(0),
                                confusion if confusion is not None else torch.empty(0, 0))

    def evaluate_ood(self, id_loader: Any, ood_loader: Any, *,
                     score: str = "variance", return_details: bool = False
                     ) -> dict[str, float] | OODResult:
        """Evaluate binary OOD detection. Positive labels correspond to OOD examples.

        Supported scores: variance, predictive_entropy, mutual_information,
        max_probability (negative maximum predicted class probability).
        The score is aggregated to one value per input image/volume.
        """
        if score not in ("variance", "predictive_entropy", "mutual_information", "max_probability"):
            raise ValueError("score must be variance, predictive_entropy, mutual_information or max_probability")
        if self.task == "regression" and score != "variance":
            raise ValueError("Regression supports only variance OOD score")

        def collect(loader):
            parts = []
            for batch in loader:
                if torch.is_tensor(batch):
                    x = batch
                elif isinstance(batch, Mapping):
                    key = next((k for k in ("inputs", "input", "x", "images", "image") if k in batch), None)
                    if key is None:
                        raise ValueError("No input tensor found in batch mapping")
                    x = batch[key]
                elif isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    raise TypeError("Unsupported OOD batch format")
                if not torch.is_tensor(x):
                    raise TypeError("OOD evaluation requires Tensor inputs")
                r = self._predict(x)
                if score == "variance":
                    values = r.uncertainty
                elif score == "max_probability":
                    if r.mode == "binary":
                        values = -torch.maximum(r.mean, 1-r.mean).reshape(r.mean.shape[0], -1).mean(1)
                    else:
                        values = -r.mean.max(dim=1).values.reshape(r.mean.shape[0], -1).mean(1)
                elif score == "predictive_entropy":
                    values = r.predictive_entropy.reshape(r.mean.shape[0], -1).mean(1)
                else:
                    values = r.mutual_information.reshape(r.mean.shape[0], -1).mean(1)
                parts.append(values.detach().float().cpu())
            if not parts:
                raise ValueError("OOD loader is empty")
            return torch.cat(parts)

        in_scores = collect(id_loader)
        out_scores = collect(ood_loader)
        labels = torch.cat((torch.zeros(in_scores.numel()), torch.ones(out_scores.numel())))
        curves = binary_curves(torch.cat((in_scores, out_scores)), labels)
        metrics = {"auroc": curves["auroc"], "aupr": curves["aupr"], "fpr95": curves["fpr95"]}
        if return_details:
            return OODResult(metrics, in_scores, out_scores, curves, score)
        return metrics
