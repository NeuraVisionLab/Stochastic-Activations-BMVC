"""Public universal nn.Module wrapper for stochastic activation conversion."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn

from .activations import StochasticActivation
from .conversion import ConversionReport, convert_activations
from .registry import ActivationSpec


def _patterns(value: str | Sequence[str] | None, arg: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) or not v for v in value):
        raise TypeError(f"{arg} must be a glob/regex string or a sequence of strings")
    return tuple(value)


class StochasticModel(nn.Module):
    """Copy and convert supported activations in any torch.nn.Module.

    The wrapper also provides Monte Carlo prediction; training and evaluation
    tools are available as independent optional utilities.

    Note: ``eval()`` does *not* disable stochastic activation switching.
    To get the original native-activation behavior use ``set_stochastic(False)``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        granularity: str = "node",
        activations: ActivationSpec | Sequence[ActivationSpec] | None = None,
        p: float | None = None,
        probs: Sequence[float] | None = None,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        inplace: bool = False,
        backend: str = "modules",
        channel_dim: int | None = None,
        batch_shared_channels: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module instance")
        if isinstance(model, StochasticModel):
            raise ValueError("model is already a StochasticModel")
        if backend != "modules":
            if backend == "fx":
                raise NotImplementedError(
                    "FX functional-activation conversion is planned for a later release; "
                    "currently use backend='modules'."
                )
            raise ValueError("backend must be 'modules'")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be bool")
        include_patterns = _patterns(include, "include")
        exclude_patterns = _patterns(exclude, "exclude")
        if inplace:
            working_model = model
        else:
            try:
                working_model = copy.deepcopy(model)
            except Exception as exc:
                raise RuntimeError(
                    "Could not deepcopy the model. Try inplace=True, or make "
                    "your model deepcopy-compatible."
                ) from exc
        self.model, self.conversion_report = convert_activations(
            working_model,
            granularity=granularity,
            activations=activations,
            p=p,
            probs=probs,
            include=include_patterns,
            exclude=exclude_patterns,
            channel_dim=channel_dim,
            batch_shared_channels=batch_shared_channels,
            inplace=inplace,
        )
        # nn.Module.__init__() defaults to training=True; conversion must not
        # silently change the externally visible mode of an already-eval model.
        # Avoid train(...), which would overwrite intentionally mixed child flags.
        self.training = self.model.training

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the converted model without changing output structure."""
        return self.model(*args, **kwargs)

    def set_stochastic(self, enabled: bool = True) -> "StochasticModel":
        """Switch all StochAct layers independently of model.train()/eval()."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        for module in self.model.modules():
            if isinstance(module, StochasticActivation):
                module.set_stochastic(enabled)
        return self

    @contextmanager
    def stochastic_mode(self, enabled: bool = True) -> Iterator["StochasticModel"]:
        """Temporarily enable or disable switching and restore per-layer state.

        This is independent of `.train()` and `.eval()`, and is safe to nest.
        Prior settings are restored even if the caller raises an exception.
        """
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        layers = list(self.stochastic_layers().values())
        original = [layer.stochastic_enabled for layer in layers]
        try:
            for layer in layers:
                layer.set_stochastic(enabled)
            yield self
        finally:
            for layer, was_enabled in zip(layers, original):
                layer.set_stochastic(was_enabled)

    def report(self, *, print_report: bool = True) -> ConversionReport:
        """Show and return the conversion report."""
        if print_report:
            print(self.conversion_report)
        return self.conversion_report

    def stochastic_layers(self) -> dict[str, StochasticActivation]:
        """Return named stochastic activation modules in the converted model."""
        return {
            name: layer for name, layer in self.model.named_modules()
            if isinstance(layer, StochasticActivation)
        }

    def predict(
        self,
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
    ) -> "PredictionResult":
        """Run Monte Carlo stochastic inference (see stochact.uncertainty.predict_mc).

        Classification is the default; set task='segmentation' or 'regression'
        for dense masks or continuous predictions. Result variance is computed
        across *probability* predictions, matching the BMVC methodology.
        """
        from ..uncertainty import predict_mc

        return predict_mc(
            self, x, task=task, mode=mode, n_samples=n_samples,
            from_logits=from_logits, class_dim=class_dim, threshold=threshold,
            return_samples=return_samples, stochastic=stochastic,
            output_key=output_key, output_index=output_index, **forward_kwargs,
        )
