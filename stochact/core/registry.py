"""Activation factories, public registration, and candidate resolution."""
from __future__ import annotations

import copy
from typing import Callable, TypeAlias

import torch.nn as nn

ActivationFactory: TypeAlias = Callable[[], nn.Module]
ActivationSpec: TypeAlias = str | nn.Module | ActivationFactory

_FACTORIES: dict[str, ActivationFactory] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "mish": nn.Mish,
    "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "hardswish": nn.Hardswish,
    "hardtanh": nn.Hardtanh,
    "prelu": nn.PReLU,
}

# Output activations (e.g. Sigmoid, Softmax) are intentionally excluded.
SUPPORTED_NATIVE_TYPES: tuple[type[nn.Module], ...] = (
    nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Tanh, nn.ELU,
    nn.SELU, nn.Mish, nn.LeakyReLU, nn.Softplus,
    nn.Hardswish, nn.Hardtanh, nn.PReLU,
)

_NATIVE_ALIASES = {"original", "native", "base"}


def register_activation(name: str, factory: ActivationFactory, *, overwrite: bool = False) -> None:
    """Register an activation factory returning an ``nn.Module``.

    Custom modules should be elementwise, shape-preserving, and free of
    running-state side effects. Parameterized modules are allowed, but add
    trainable parameters and are counted in conversion reports.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Activation name must be a nonempty string")
    key = name.strip().lower()
    if key in _NATIVE_ALIASES:
        raise ValueError(f"'{key}' is reserved for the model's native activation")
    if not callable(factory):
        raise TypeError("factory must be callable and return nn.Module")
    if key in _FACTORIES and not overwrite:
        raise ValueError(f"Activation '{key}' already exists; pass overwrite=True")
    # Validate at registration rather than surprising users during conversion.
    instance = factory()
    if not isinstance(instance, nn.Module):
        raise TypeError("Activation factory must return a torch.nn.Module")
    _FACTORIES[key] = factory


def available_activations() -> tuple[str, ...]:
    """Return registered activation names, sorted alphabetically."""
    return tuple(sorted(_FACTORIES))


def _disable_inplace(module: nn.Module) -> None:
    """Avoid mutation of tensors read by multiple activation candidates."""
    for child in module.modules():
        if hasattr(child, "inplace") and isinstance(getattr(child, "inplace"), bool):
            child.inplace = False


def _resolve_candidate(
    spec: ActivationSpec, native: nn.Module
) -> tuple[str, nn.Module | None]:
    """Return display name and module. None means use the registered native."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in _NATIVE_ALIASES:
            return type(native).__name__ + " (original)", None
        try:
            factory = _FACTORIES[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown activation '{spec}'. Choose from {available_activations()} "
                "or register a factory."
            ) from exc
        module = factory()
        label = key
    elif isinstance(spec, nn.Module):
        module = copy.deepcopy(spec)
        label = type(module).__name__
    elif callable(spec):
        module = spec()
        label = getattr(spec, "__name__", type(module).__name__)
    else:
        raise TypeError("Activation must be a string, nn.Module, or zero-argument factory")
    if not isinstance(module, nn.Module):
        raise TypeError(f"Activation '{label}' did not produce an nn.Module")
    _disable_inplace(module)
    return label, module
