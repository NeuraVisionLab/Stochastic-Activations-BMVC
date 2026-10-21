"""Stochastic activation module supporting node, channel and layer sampling."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .registry import ActivationSpec, _disable_inplace, _resolve_candidate


class StochasticActivation(nn.Module):
    """Switch among nonlinearities without masking the input signal.

    Args:
        native: Native activation from the backbone. Retained for deterministic mode.
        activations: None -> [native, SiLU]; single -> [native, specified];
            a list with 2+ entries -> exactly those candidates. Lists of length 1
            are treated like single entries. Use "original" in a list to explicitly
            include the native activation.
        p: For two candidates, probability of selecting the second candidate.
            Defaults to 0.5 when p and probs are both unspecified.
        probs: Probabilities for *all* candidates; mutually exclusive with p.
        granularity: "node", "channel", or "layer".
        channel_dim: Feature channel dimension for channel sampling. Default is
            last dimension for rank-3 tensors [B, tokens, features], dimension 1
            for rank-2 and rank >= 4 tensors.
        batch_shared_channels: Reuse one channel selection across a batch.
            Default False means each example gets its own channel selection.

    The module continues sampling during eval(); use set_stochastic(False) to
    select the native activation deterministically. Candidates should be
    elementwise/shape-preserving without stateful side effects.
    """

    def __init__(
        self,
        native: nn.Module,
        *,
        activations: ActivationSpec | Sequence[ActivationSpec] | None = None,
        p: float | None = None,
        probs: Sequence[float] | None = None,
        granularity: str = "node",
        channel_dim: int | None = None,
        batch_shared_channels: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(native, nn.Module):
            raise TypeError("native must be a torch.nn.Module")
        if granularity not in ("node", "channel", "layer"):
            raise ValueError("granularity must be 'node', 'channel', or 'layer'")
        if channel_dim is not None and (not isinstance(channel_dim, int) or isinstance(channel_dim, bool)):
            raise TypeError("channel_dim must be an integer or None")
        if not isinstance(batch_shared_channels, bool):
            raise TypeError("batch_shared_channels must be bool")
        self.native = native
        _disable_inplace(self.native)
        self.granularity = granularity
        self.channel_dim = channel_dim
        self.batch_shared_channels = batch_shared_channels
        self.stochastic_enabled = True

        if activations is None:
            specs: list[ActivationSpec] = ["original", "silu"]
        elif isinstance(activations, (list, tuple)):
            if not activations:
                raise ValueError("activations cannot be an empty sequence")
            specs = ["original", activations[0]] if len(activations) == 1 else list(activations)
        else:
            specs = ["original", activations]

        self.alternatives = nn.ModuleList()
        self._choice_refs: list[int | None] = []
        names: list[str] = []
        for spec in specs:
            name, module = _resolve_candidate(spec, self.native)
            if module is None:
                self._choice_refs.append(None)
            else:
                self._choice_refs.append(len(self.alternatives))
                self.alternatives.append(module)
            names.append(name)
        self.candidate_names = tuple(names)

        # For an explicit list, allow e.g. ["original", "tanh"] but reject
        # multiple references to the *same* native module.
        if self._choice_refs.count(None) > 1:
            raise ValueError("'original' may only appear once in an activation pool")

        n = len(specs)
        if p is not None and probs is not None:
            raise ValueError("Use either p or probs, not both")
        if p is not None and n != 2:
            raise ValueError("p applies only to two candidates; use probs for larger pools")
        if probs is not None:
            if len(probs) != n:
                raise ValueError(f"Expected {n} probabilities, received {len(probs)}")
            weights = torch.as_tensor(list(probs), dtype=torch.float64)
        elif p is not None:
            if not isinstance(p, (float, int)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
                raise ValueError("p must be between 0 and 1")
            weights = torch.tensor([1.0 - p, p], dtype=torch.float64)
        else:
            weights = torch.ones(n, dtype=torch.float64) / n
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("probs must be finite, nonnegative, and have positive sum")
        if probs is not None and abs(weights.sum().item() - 1.0) > 1e-6:
            raise ValueError("probs must sum to 1 (within 1e-6)")
        # A buffer travels automatically with .to(device), but does not add
        # trainable parameters. float64 avoids rounding all nonzero tiny weights.
        self.register_buffer("probabilities", (weights / weights.sum()).contiguous())

    def _apply(self, fn: Any) -> "StochasticActivation":
        """Move activation parameters with PyTorch without rounding sampling probabilities.

        PyTorch's ``half()``/``bfloat16()`` also converts floating buffers by
        default. Sampling weights remain float64 so very small probabilities
        survive dtype-only conversion and state_dict round-trips.
        """
        saved = self.probabilities.detach().clone()
        super()._apply(fn)
        self.probabilities = saved.to(device=self.probabilities.device, dtype=torch.float64)
        return self

    def set_stochastic(self, enabled: bool = True) -> "StochasticActivation":
        """Enable/disable switching, independently of train()/eval()."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.stochastic_enabled = enabled
        return self

    def _sample_shape(self, x: torch.Tensor) -> tuple[int, ...]:
        if self.granularity == "layer":
            return ()
        if self.granularity == "node":
            return tuple(x.shape)
        if x.ndim < 2:
            raise ValueError("Channel-level sampling requires a tensor of rank >= 2")
        feature_dim = self.channel_dim
        if feature_dim is None:
            feature_dim = -1 if x.ndim == 3 else 1
        if feature_dim < 0:
            feature_dim += x.ndim
        if feature_dim <= 0 or feature_dim >= x.ndim:
            raise ValueError(
                f"channel_dim must select a non-batch dimension in [1, {x.ndim - 1}]"
            )
        shape = [1] * x.ndim
        shape[0] = 1 if self.batch_shared_channels else x.shape[0]
        shape[feature_dim] = x.shape[feature_dim]
        return tuple(shape)

    def sample_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Sample candidate indices, with broadcastable shape for the granularity.

        This public method is useful for testing and inspecting granularity.
        Sampling does not change stochastic_enabled or the model's training mode.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a Tensor")
        shape = self._sample_shape(x)
        # For GPU models the buffer is on GPU too. Sampling on the input device
        # ensures the returned indices can index GPU activations without transfers.
        weights = self.probabilities.to(device=x.device, dtype=torch.float64)
        if len(self._choice_refs) == 2:
            # The binary case is the BMVC default. A direct Bernoulli draw is
            # much cheaper than multinomial sampling over millions of pixels.
            return (torch.rand(shape, device=x.device) < weights[1]).long()
        n = 1
        for size in shape:
            n *= size
        return torch.multinomial(weights, num_samples=n, replacement=True).reshape(shape)

    def _candidate(self, index: int) -> nn.Module:
        ref = self._choice_refs[index]
        return self.native if ref is None else self.alternatives[ref]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.stochastic_enabled:
            return self.native(x)
        # Avoid sampling and evaluating unused candidates for a degenerate pool.
        deterministic_choice = torch.nonzero(self.probabilities == 1.0).flatten()
        if deterministic_choice.numel() == 1:
            return self._candidate(int(deterministic_choice.item()))(x)
        choice = self.sample_indices(x)
        if self.granularity == "layer":
            return self._candidate(int(choice.item()))(x)
        out: torch.Tensor | None = None
        for i in range(len(self._choice_refs)):
            # Every branch receives the same *unmodified* x. This is why
            # inplace=True is disabled on activation candidates.
            candidate_out = self._candidate(i)(x)
            if candidate_out.shape != x.shape:
                raise RuntimeError(
                    f"Activation '{self.candidate_names[i]}' changed shape "
                    f"from {tuple(x.shape)} to {tuple(candidate_out.shape)}"
                )
            if out is None:
                out = candidate_out
            else:
                out = torch.where(choice == i, candidate_out, out)
        assert out is not None
        return out

    def extra_repr(self) -> str:
        probs = ", ".join(f"{v:.3g}" for v in self.probabilities.tolist())
        return (
            f"candidates={self.candidate_names}, probs=({probs}), "
            f"granularity={self.granularity}, stochastic={self.stochastic_enabled}"
        )
