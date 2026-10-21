"""Optional Matplotlib visualizations. No Matplotlib import at package import time."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import torch


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Install plotting support with: pip install "stochact[plots]"') from exc
    return plt


def _save(fig, save_path):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return torch.as_tensor(value).detach().cpu().float().numpy()


def plot_training(history, *, metrics: Sequence[str] = ("loss", "accuracy", "uncertainty"),
                  save_path=None):
    """Plot available training/validation metrics from a TrainingHistory or records."""
    plt = _plt()
    records = history.records if hasattr(history, "records") else history
    if not records:
        raise ValueError("Training history is empty")
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("Choose at least one metric")
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 3.8), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        found = False
        for split in ("train", "val"):
            key = f"{split}_{metric}"
            points = [(record["epoch"], record[key]) for record in records if key in record]
            if points:
                found = True
                ax.plot([p[0] for p in points], [p[1] for p in points], label=split.capitalize())
        if not found:
            ax.text(0.5, 0.5, f"No {metric} data", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            ax.legend()
        ax.set(xlabel="Epoch", ylabel=metric.replace("_", " ").title())
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, save_path), axes[0]


def _show_image(array):
    import numpy as np
    arr = _numpy(array)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError("Image must be 2D grayscale or 3D CHW/HWC")
    if arr.ndim == 3:
        # Do not clip arbitrary model-normalized pixel values silently; normalize for display.
        arr = np.clip(arr, 0, 1) if arr.min() >= 0 and arr.max() <= 1 else (arr - arr.min()) / max(float(arr.max()-arr.min()), 1e-10)
    return arr


def plot_uncertainty(image, result, *, target=None, index: int = 0,
                     cmap: str = "magma", save_path=None):
    """For segmentation, display input, optional ground truth, prediction, error, MC variance.

    Inputs are [B,C,H,W] or a single image [C,H,W]/[H,W]; masks [B,H,W]/[H,W].
    The uncertainty color bar shows raw predictive variance, not error probability.
    """
    plt = _plt()
    if result.task != "segmentation" or result.uncertainty_map is None:
        raise ValueError("plot_uncertainty requires a segmentation PredictionResult")
    import numpy as np
    img = torch.as_tensor(image)
    if img.ndim == 4:
        img = img[index]
    pred = result.predictions[index]
    if target is not None:
        gt = torch.as_tensor(target)
        if gt.ndim == result.predictions.ndim + 1 and gt.shape[1] == 1:
            gt = gt[:, 0]
        if gt.ndim == result.predictions.ndim:
            gt = gt[index]
        if gt.shape != pred.shape:
            raise ValueError("target mask differs from predicted mask shape")
    else:
        gt = None
    if pred.ndim != 2 or result.uncertainty_map[index].ndim != 2:
        raise ValueError("plot_uncertainty currently supports 2D masks")
    panels = [("Input", _show_image(img), None)]
    if gt is not None:
        panels.append(("Ground truth", _numpy(gt), "gray"))
    panels.append(("Prediction", _numpy(pred), "gray"))
    if gt is not None:
        panels.append(("Prediction error", _numpy(pred != gt), "Reds"))
    panels.append(("MC variance", _numpy(result.uncertainty_map[index]), cmap))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1*len(panels), 3.6), squeeze=False)
    for ax, (name, data, colormap) in zip(axes[0], panels):
        im = ax.imshow(data, cmap=colormap, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
        if name == "MC variance":
            fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)
    fig.tight_layout()
    return _save(fig, save_path), axes[0]


def plot_reliability(evaluation, *, save_path=None):
    """Reliability diagram from Evaluator.evaluate(return_details=True)."""
    plt = _plt()
    if evaluation.task not in ("classification", "segmentation"):
        raise ValueError("Reliability diagrams require classification or segmentation predictions")
    stats = evaluation.reliability
    counts = _numpy(stats["count"])
    acc = _numpy(stats["accuracy"])
    conf = _numpy(stats["mean_confidence"])
    edges = _numpy(stats["bin_edges"])
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    valid = counts > 0
    ax.bar((edges[:-1] + edges[1:])/2, acc, width=(edges[1]-edges[0])*0.82,
           alpha=0.6, label="Empirical accuracy")
    ax.scatter(conf[valid], acc[valid], s=32, label="Observed bins")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy",
           title=f"Reliability diagram (ECE={evaluation.metrics['ece']:.4f})")
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path), ax


def plot_risk_coverage(evaluation, *, save_path=None):
    """Selective risk as lower-uncertainty examples are retained first.

    `per_example_error`: 0/1 for classification, mean pixel error for segmentation,
    MSE for regression. Higher uncertainty means reject earlier.
    """
    plt = _plt()
    unc = evaluation.uncertainty.flatten()
    error = evaluation.per_example_error.flatten()
    if unc.numel() == 0 or error.numel() != unc.numel():
        raise ValueError("Risk–coverage requires nonempty matching per-example arrays")
    order = torch.argsort(unc, stable=True)
    sorted_error = error[order].double()
    xs = torch.arange(1, len(error)+1, dtype=torch.float64) / len(error)
    ys = sorted_error.cumsum(0) / torch.arange(1, len(error)+1)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.plot(_numpy(xs), _numpy(ys), label="Selective risk")
    ax.axhline(float(error.mean()), linestyle="--", color="gray", label="Full-coverage risk")
    ax.set(xlabel="Coverage (fraction retained)", ylabel="Mean retained error",
           title="Risk–coverage curve", xlim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path), ax


def plot_ood(ood, *, save_path=None):
    """ROC and precision–recall plots from evaluate_ood(return_details=True)."""
    plt = _plt()
    curves = ood.curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(_numpy(curves["fpr"]), _numpy(curves["tpr"]),
                 label=f"AUROC={ood.metrics['auroc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="OOD ROC")
    axes[1].step(_numpy(curves["recall"]), _numpy(curves["precision"]), where="post",
                 label=f"AUPR={ood.metrics['aupr']:.3f}")
    baseline = len(ood.ood_scores) / (len(ood.ood_scores)+len(ood.id_scores))
    axes[1].axhline(baseline, linestyle="--", color="gray")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="OOD precision–recall")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    return _save(fig, save_path), axes


def plot_mc_convergence(result, *, save_path=None):
    """Track cumulative predictive variance vs sampled pass count.

    Requires `model.predict(..., return_samples=True)`; does not re-run inference.
    """
    plt = _plt()
    if result.samples is None:
        raise ValueError("Run model.predict(..., return_samples=True) first")
    if result.samples.shape[0] < 2:
        raise ValueError("At least two samples required for convergence plot")
    samples = result.samples.detach().float()
    # Plot a streaming mean of per-output population variance, averaged over the batch.
    running_mean = None
    m2 = None
    curve = []
    for t, sample in enumerate(samples, 1):
        if running_mean is None:
            running_mean = sample.clone()
            m2 = torch.zeros_like(sample)
        else:
            delta = sample - running_mean
            running_mean = running_mean + delta/t
            m2 = m2 + delta * (sample - running_mean)
        curve.append(float((m2/t).mean()))
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.plot(range(1, len(curve)+1), curve, marker="o", markersize=3)
    ax.set(xlabel="Stochastic passes", ylabel="Mean predictive variance",
           title="MC uncertainty convergence")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, save_path), ax
