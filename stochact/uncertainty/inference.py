"""Monte Carlo stochastic activation inference with streaming statistics."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

from ..core.activations import StochasticActivation
from .results import PredictionResult


@contextmanager
def _inference_mode(model: nn.Module, *, stochastic: bool) -> Iterator[None]:
    """Temporarily evaluate normally but control activation sampling explicitly.

    Restore even unusual per-submodule train flags and stochastic configurations,
    including after an exception. The model's parameters never change.
    """
    modes = [(module, module.training) for module in model.modules()]
    activations = [(module, module.stochastic_enabled) for module, _ in modes
                   if isinstance(module, StochasticActivation)]
    try:
        model.eval()  # freeze dropout / batch norm
        for module, _ in activations:
            module.set_stochastic(stochastic)
        with torch.inference_mode():
            yield
    finally:
        for module, previous in modes:
            module.training = previous
        for module, previous in activations:
            module.set_stochastic(previous)


def _extract_output(
    value: Any,
    output_key: str | None,
    output_index: int | None,
) -> torch.Tensor:
    if output_key is not None:
        if not isinstance(value, Mapping):
            raise TypeError("output_key requires a model that returns a mapping")
        if output_key not in value:
            raise KeyError(f"Model output does not include key '{output_key}'")
        value = value[output_key]
    elif output_index is not None:
        if not isinstance(value, (list, tuple)):
            raise TypeError("output_index requires a model that returns a list/tuple")
        value = value[output_index]
    elif isinstance(value, Mapping):
        if len(value) == 1:
            value = next(iter(value.values()))
        else:
            raise TypeError("Model returned a mapping with multiple keys; specify output_key")
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            value = value[0]
        else:
            raise TypeError("Model returned multiple outputs; specify output_index")
    if not isinstance(value, torch.Tensor):
        raise TypeError("Model output must be a Tensor; select a Tensor using output_key or output_index")
    if not value.is_floating_point():
        raise TypeError("Model prediction must be a floating-point tensor")
    if value.ndim < 1 or value.shape[0] == 0:
        raise ValueError("Model output must include a nonempty batch dimension")
    return value


def _validate_output_shape(value: torch.Tensor, *, task: str, mode: str, class_dim: int) -> str | None:
    if task == "regression":
        return None
    if task == "classification" and value.ndim not in (1, 2):
        raise ValueError("Classification output must have shape [B,C], [B,1], or [B]")
    if task == "segmentation" and value.ndim < 3:
        raise ValueError("Segmentation output must have shape [B,C,*spatial] or [B,*spatial] (binary)")
    if value.ndim == 1 or (task == "segmentation" and value.ndim == 3):
        detected = "binary"
    else:
        if not -value.ndim <= class_dim < value.ndim:
            raise ValueError("class_dim out of bounds")
        index = class_dim % value.ndim
        if index == 0:
            raise ValueError("class_dim cannot be the batch dimension")
        detected = "binary" if value.shape[index] == 1 else "multiclass"
    if mode != "auto" and mode != detected:
        raise ValueError(f"mode='{mode}' incompatible with output shape {tuple(value.shape)}; inferred '{detected}'")
    if task == "segmentation" and detected == "multiclass" and value.ndim < 4:
        raise ValueError("Multiclass segmentation requires explicit [B,C,*spatial] output")
    return detected


def _normalise(
    output: torch.Tensor, *,
    task: str, mode: str | None, class_dim: int, from_logits: bool,
) -> torch.Tensor:
    # Float32 accumulations under autocast/half precision are important for
    # mean, variance and logarithms; float64 inputs retain their precision.
    if output.dtype in (torch.float16, torch.bfloat16):
        output = output.float()
    if task == "regression":
        return output
    if from_logits:
        return output.sigmoid() if mode == "binary" else output.softmax(dim=class_dim)
    # Do not silently treat arbitrary logits as probabilities.
    if not torch.isfinite(output).all() or (output < 0).any() or (output > 1).any():
        raise ValueError("from_logits=False requires finite probabilities in [0,1]")
    if mode == "multiclass" and not torch.allclose(
        output.sum(dim=class_dim), torch.ones_like(output.sum(dim=class_dim)), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Multiclass probabilities must sum to one along class_dim")
    return output


def _entropy(p: torch.Tensor, mode: str, class_dim: int, task: str) -> torch.Tensor:
    eps = torch.finfo(p.dtype).eps
    if mode == "binary":
        clipped = p.clamp(eps, 1.0 - eps)
        ent = -(p * clipped.log() + (1.0 - p) * (1.0 - clipped).log())
        if (task == "classification" and p.ndim == 2 and p.shape[class_dim] == 1) or (
            task == "segmentation" and p.ndim >= 4 and p.shape[class_dim] == 1
        ):
            return ent.squeeze(class_dim)
        return ent
    return -(p * p.clamp_min(eps).log()).sum(dim=class_dim)


def predict_mc(
    model: nn.Module,
    x: torch.Tensor,
    *,
    task: str = "classification",
    mode: str = "auto",
    n_samples: int = 5,
    from_logits: bool = True,
    class_dim: int = 1,
    threshold: float = 0.5,
    return_samples: bool = False,
    stochastic: bool = True,
    output_key: str | None = None,
    output_index: int | None = None,
    **forward_kwargs: Any,
) -> PredictionResult:
    """Perform T stochastic passes and estimate predictive uncertainty.

    Args:
        model: StochasticModel or any nn.Module containing StochasticActivation.
        x: Input tensor, forwarded as first positional argument.
        task: 'classification', 'segmentation', or 'regression'.
        mode: 'auto', 'binary' or 'multiclass' (not used in regression).
        n_samples: Number of passes, default 5 as in the BMVC paper.
        from_logits: Apply sigmoid/softmax when True. Set False only if model
            already outputs normalized probabilities. Regression is unchanged.
        class_dim: Class axis, normally 1 for [B,C,...] outputs.
        threshold: Decision threshold for binary labels/masks.
        return_samples: Retain [T,B,...] outputs, consumes O(T) memory.
        stochastic: Temporarily switch activations ON (or OFF for a baseline).
        output_key, output_index: Select a tensor from structured model outputs.
        **forward_kwargs: Extra keyword arguments passed to model(x, ...).

    Return:
        PredictionResult with streaming mean, population variance, predictions,
        per-example uncertainty, entropy and MI (not for regression).

    Inference freezes dropout/BatchNorm, preserves model state, disables gradients,
    and samples *new* activation choices on every pass. Tensors remain on their
    original device. No GPU memory is allocated proportional to n_samples unless
    return_samples=True. This method does not change model weights.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if task not in ("classification", "segmentation", "regression"):
        raise ValueError("task must be 'classification', 'segmentation', or 'regression'")
    if mode not in ("auto", "binary", "multiclass"):
        raise ValueError("mode must be 'auto', 'binary', or 'multiclass'")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(class_dim, int) or isinstance(class_dim, bool) or class_dim == 0:
        raise ValueError("class_dim must be a nonzero integer")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    for value, name in ((from_logits, "from_logits"), (return_samples, "return_samples"), (stochastic, "stochastic")):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")
    if output_key is not None and output_index is not None:
        raise ValueError("Select either output_key or output_index, not both")
    if output_key is not None and not isinstance(output_key, str):
        raise TypeError("output_key must be a string")
    if output_index is not None and (not isinstance(output_index, int) or isinstance(output_index, bool)):
        raise TypeError("output_index must be an integer")
    if not any(isinstance(layer, StochasticActivation) for layer in model.modules()):
        raise ValueError("No StochasticActivation layers found; check the conversion report")

    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    sum_entropy: torch.Tensor | None = None
    samples: list[torch.Tensor] | None = [] if return_samples else None
    detected_mode: str | None = None
    output_shape: torch.Size | None = None
    with _inference_mode(model, stochastic=stochastic):
        for t in range(1, n_samples + 1):
            output = _extract_output(model(x, **forward_kwargs), output_key, output_index)
            current_mode = _validate_output_shape(output, task=task, mode=mode, class_dim=class_dim)
            if t == 1:
                detected_mode = current_mode
                output_shape = output.shape
            elif output.shape != output_shape or current_mode != detected_mode:
                raise RuntimeError("Model output shape or mode changed between MC passes")
            p = _normalise(output, task=task, mode=detected_mode, class_dim=class_dim,
                           from_logits=from_logits)
            if not torch.isfinite(p).all():
                raise ValueError("Model returned non-finite outputs during MC inference")
            if mean is None:
                mean = p.clone()
                m2 = torch.zeros_like(p)
            else:
                delta = p - mean
                mean = mean + delta / t
                m2 = m2 + delta * (p - mean)
            if task != "regression":
                ent = _entropy(p, detected_mode, class_dim, task)
                sum_entropy = ent if sum_entropy is None else sum_entropy + ent
            if samples is not None:
                samples.append(p)

    assert mean is not None and m2 is not None
    variance = (m2 / n_samples).clamp_min(0)
    uncertainty = variance.reshape(variance.shape[0], -1).mean(dim=1)
    predictions: torch.Tensor
    uncertainty_map: torch.Tensor | None = None
    predictive_entropy: torch.Tensor | None = None
    expected_entropy: torch.Tensor | None = None
    mutual_information: torch.Tensor | None = None

    if task == "regression":
        predictions = mean
    else:
        assert detected_mode is not None and sum_entropy is not None
        if detected_mode == "binary":
            if (task == "classification" and mean.ndim == 2 and mean.shape[class_dim] == 1) or (
                task == "segmentation" and mean.ndim >= 4 and mean.shape[class_dim] == 1
            ):
                predictions = (mean >= threshold).long().squeeze(class_dim)
            else:
                predictions = (mean >= threshold).long()
        else:
            predictions = mean.argmax(dim=class_dim)
        predictive_entropy = _entropy(mean, detected_mode, class_dim, task)
        expected_entropy = sum_entropy / n_samples
        mutual_information = (predictive_entropy - expected_entropy).clamp_min(0)
        if task == "segmentation":
            if detected_mode == "binary":
                uncertainty_map = variance.squeeze(class_dim) if variance.ndim >= 4 and variance.shape[class_dim] == 1 else variance
            else:
                uncertainty_map = variance.mean(dim=class_dim)

    return PredictionResult(
        task=task,
        mode=detected_mode,
        n_samples=n_samples,
        mean=mean,
        variance=variance,
        predictions=predictions,
        uncertainty=uncertainty,
        predictive_entropy=predictive_entropy,
        expected_entropy=expected_entropy,
        mutual_information=mutual_information,
        uncertainty_map=uncertainty_map,
        samples=None if samples is None else torch.stack(samples, dim=0),
    )
