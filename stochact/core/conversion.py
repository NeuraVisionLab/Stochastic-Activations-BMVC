"""Conversion of registered PyTorch activation modules into stochastic modules."""
from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import re
from typing import Any, Sequence

import torch
import torch.nn as nn

from .activations import StochasticActivation
from .registry import ActivationSpec, SUPPORTED_NATIVE_TYPES


@dataclass(frozen=True)
class ConvertedActivation:
    path: str
    original: str
    choices: tuple[str, ...]
    probabilities: tuple[float, ...]


@dataclass
class ConversionReport:
    model_name: str
    granularity: str
    backend: str
    inplace: bool
    original_parameters: int
    converted_parameters: int
    detected: int = 0
    replacements: list[ConvertedActivation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def replaced(self) -> int:
        return len(self.replacements)

    @property
    def parameters_added(self) -> int:
        return self.converted_parameters - self.original_parameters

    def __str__(self) -> str:
        lines = [
            "StochAct Conversion Report",
            "=" * 38,
            f"Model:                      {self.model_name}",
            f"Backend:                    {self.backend}",
            f"Granularity:                {self.granularity}",
            f"Conversion:                 {'in place' if self.inplace else 'deep copy'}",
            f"Supported activations found: {self.detected}",
            f"Activations replaced:       {self.replaced}",
            f"Activations skipped:        {len(self.skipped)}",
            f"Trainable parameters added: {self.parameters_added}",
        ]
        if self.replacements:
            lines += ["", "Converted layers:"]
            for item in self.replacements:
                choices = " / ".join(item.choices)
                probs = ", ".join(f"{v:.3f}" for v in item.probabilities)
                lines.append(f"  {item.path}: {item.original} -> [{choices}] (p=[{probs}])")
        if self.skipped:
            lines += ["", "Excluded layers:"] + [f"  {s}" for s in self.skipped]
        if self.warnings:
            lines += ["", "Notes:"] + [f"  - {s}" for s in self.warnings]
        lines.append("=" * 38)
        return "\n".join(lines)


def _matches(path: str, patterns: Sequence[str] | None) -> bool:
    if patterns is None:
        return False
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], path):
                return True
        elif fnmatch.fnmatchcase(path, pattern):
            return True
    return False


def _should_convert(path: str, include: Sequence[str] | None, exclude: Sequence[str] | None) -> bool:
    return (include is None or _matches(path, include)) and not _matches(path, exclude)


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def convert_activations(
    model: nn.Module,
    *,
    granularity: str,
    activations: ActivationSpec | Sequence[ActivationSpec] | None,
    p: float | None,
    probs: Sequence[float] | None,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    channel_dim: int | None,
    batch_shared_channels: bool,
    inplace: bool,
) -> tuple[nn.Module, ConversionReport]:
    """Replace eligible activation modules, maintaining shared-module identity."""
    original_parameters = _count_parameters(model)
    report = ConversionReport(
        model_name=type(model).__name__,
        granularity=granularity,
        backend="modules",
        inplace=inplace,
        original_parameters=original_parameters,
        converted_parameters=original_parameters,
    )
    # Same activation referenced by multiple attributes gets the same wrapper.
    # Nested/shared containers are visited once, as in conventional module walks.
    wrappers: dict[int, StochasticActivation] = {}
    visited_containers: set[int] = set()

    def reference_for(native: nn.Module, parent: nn.Module) -> torch.Tensor | None:
        """Find nearest floating tensor to align optional trainable candidates.

        Most built-in activations are parameter-free. When a user supplies a
        parameterized alternative (e.g., PReLU), its parameters must match the
        converted model even if that model was on CUDA/double before conversion.
        """
        for scope, recursive in ((native, True), (parent, False), (parent, True), (model, True)):
            tensors = list(scope.parameters(recurse=recursive)) + list(scope.buffers(recurse=recursive))
            for value in tensors:
                if value.is_floating_point():
                    return value
        return None

    def align_candidate_parameters(layer: StochasticActivation, ref: torch.Tensor | None) -> None:
        if ref is not None:
            # Only alternatives are moved. Native modules may be deliberately
            # distributed across devices and must never move implicitly.
            layer.alternatives.to(device=ref.device, dtype=ref.dtype)

    def visit(parent: nn.Module, prefix: str) -> None:
        if id(parent) in visited_containers:
            return
        visited_containers.add(id(parent))
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, StochasticActivation):
                continue
            if isinstance(child, SUPPORTED_NATIVE_TYPES):
                report.detected += 1
                if not _should_convert(path, include, exclude):
                    report.skipped.append(path)
                    continue
                if id(child) not in wrappers:
                    wrapped = StochasticActivation(
                        child, activations=activations, p=p, probs=probs,
                        granularity=granularity, channel_dim=channel_dim,
                        batch_shared_channels=batch_shared_channels,
                    )
                    align_candidate_parameters(wrapped, reference_for(child, parent))
                    wrapped.train(child.training)
                    wrappers[id(child)] = wrapped
                wrapped = wrappers[id(child)]
                parent._modules[name] = wrapped
                report.replacements.append(ConvertedActivation(
                    path=path,
                    original=type(child).__name__,
                    choices=wrapped.candidate_names,
                    probabilities=tuple(float(v) for v in wrapped.probabilities.tolist()),
                ))
            else:
                visit(child, path)

    if isinstance(model, SUPPORTED_NATIVE_TYPES):
        report.detected += 1
        if _should_convert("<root>", include, exclude):
            old = model
            model = StochasticActivation(
                old, activations=activations, p=p, probs=probs,
                granularity=granularity, channel_dim=channel_dim,
                batch_shared_channels=batch_shared_channels,
            )
            align_candidate_parameters(model, reference_for(old, old))
            model.train(old.training)
            report.replacements.append(ConvertedActivation(
                path="<root>", original=type(old).__name__,
                choices=model.candidate_names,
                probabilities=tuple(float(v) for v in model.probabilities.tolist()),
            ))
        else:
            report.skipped.append("<root>")
    else:
        visit(model, "")
    report.converted_parameters = _count_parameters(model)
    if report.replaced == 0:
        report.warnings.append("No registered supported activation modules were replaced.")
    report.warnings.append(
        "Functional activations (e.g. torch.nn.functional.relu) are not converted "
        "by the modules backend. FX-based conversion is planned for a later phase."
    )
    if report.parameters_added > 0:
        report.warnings.append(
            "One or more explicitly provided candidate activations have learnable "
            "parameters; the zero-extra-parameters property no longer holds."
        )
    return model, report
