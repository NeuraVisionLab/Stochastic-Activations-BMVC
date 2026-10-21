# Contributing to StochAct

StochAct is a general-purpose PyTorch library. Contributions should remain task- and model-agnostic unless a feature specifically concerns evaluation for a given task.

## Local development

1. Use Python >=3.10 and a compatible version of PyTorch.
2. `python -m pip install -e '.[dev,all]'`
3. `python -m pytest -q`
4. `python -m ruff check --select E4,E7,E9 stochact tests examples`
5. Run at least one example in `examples/` if your change affects public APIs.

## Design conventions

- Keep core activation switching independent of training and evaluation.
- Maintain deterministic-native evaluation and explicit stochastic-mode control.
- Prefer backward-compatible APIs, typed signatures, and clear error messages.
- For activation conversion, preserve the original user model unless `inplace=True`.
- Always test shape, device, dtype, gradient flow, batch dimensions, and failure recovery.
- New optional features should not add mandatory dependencies to the core package.
- Document cases that PyTorch FX, tracing, compilation, or custom model outputs cannot support.

## Pull requests

Please include a minimal runnable test, an explanation of intended behavior, and a documentation update for user-visible changes. Use synthetic data in tests. Avoid committing datasets, trained model weights, or private research assets.

## Security

Do not load untrusted `torch.save` checkpoints or arbitrary activation plugins. See [SECURITY.md](SECURITY.md).
