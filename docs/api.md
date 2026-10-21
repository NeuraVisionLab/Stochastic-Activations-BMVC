# API reference

| Public entry point | Responsibilities |
| --- | --- |
| `StochasticModel(model, **options)` | Deep-copy and convert registered activations; behaves as `nn.Module` |
| `StochasticActivation(native, **options)` | Manual stochastic module supporting node/channel/layer granularity |
| `register_activation(name, factory)` | Register a trusted, zero-argument `nn.Module` factory |
| `available_activations()` | List registered activation names |
| `StochasticModel.report(print_report=True)` | Inspect detected/replaced/skipped layers and warnings |
| `StochasticModel.set_stochastic(enabled)` | Enable/disable all stochastic switches |
| `StochasticModel.stochastic_mode(enabled)` | Context manager restoring previous per-layer switch settings |
| `StochasticModel.predict(x, **options)` | Monte Carlo predictive mean, variance, entropy and MI |
| `predict_mc(model, x, **options)` | Same inference for arbitrary models with StochAct layers |
| `Trainer(model, task=..., **options)` | Fit, validate, test, save/load checkpoints |
| `Evaluator(model, task=..., **options)` | Metrics for classification, segmentation, regression and OOD |
| `stochact.visualization` | Matplotlib figure utilities with optional dependency |

## Activation candidate rules

- `None` => `[native, SiLU]`
- Single name/module/factory => `[native, candidate]`
- List with two or more entries => exactly that pool (`'original'` names the native activation)
- For a two-choice pool, `p` selects the second option with probability `p` (defaults to 0.5).
- For any pool, `probs=[...]` sets all probabilities (nonnegative, sum approximately one).
- For three or more options with no `probs`, the selection is uniform.

`granularity` is `node`, `channel`, or `layer`. Channel-level sampling uses the tensor's feature axis; `channel_dim` can override inference from shape. `batch_shared_channels=True` chooses one mask across all examples. An activation candidate must be shape-preserving.

## Prediction options

`model.predict(x, task='classification', mode='auto', n_samples=5, from_logits=True, class_dim=1, threshold=0.5, return_samples=False, stochastic=True, output_key=None, output_index=None)`

- Classification: probabilities `[B,C]` and uncertainty `[B]`.
- Binary segmentation: probability/variance maps `[B,1,H,W]`, hard masks/uncertainty `[B,H,W]`.
- Multiclass segmentation: probabilities `[B,C,H,W]`, hard masks/uncertainty `[B,H,W]`.
- Regression: raw output mean and variance; no entropy or mutual information.
- `return_samples=True` adds `[T,B,...]` samples; normally only streaming statistics are retained.
- `output_key`/`output_index` select an output tensor from mappings/tuples.

## Training and evaluation

See [training guide](training.md) and [evaluation guide](evaluation.md). Task choice affects loss functions and metrics **without creating task-specific model wrappers**.
