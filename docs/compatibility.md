# Compatibility and limits

| Area | Intended / verified support | Important limits |
| --- | --- | --- |
| Python | >= 3.10 | Current manual verification: Python 3.13 only; CI matrix covers 3.10–3.13 when run on GitHub |
| PyTorch | >= 2.0 | Current manual verification: CPU PyTorch 2.10; other versions need CI validation |
| CPU | Verified with automated tests | Activation sampling adds compute vs. a single deterministic pass |
| CUDA | Device-aware implementation + conditional test | Not executed in the supplied CPU-only environment; do not claim CUDA validated |
| Registered `nn.Module` activations | ReLU, GELU, SiLU, Tanh, etc. | Custom modules should be elementwise and shape-preserving |
| Functional activations | Not automatically converted | `F.relu`, `torch.relu`, `torch.nn.functional.gelu` need manual changes or future FX backend |
| Classification | Binary, multiclass | The model must return a floating tensor or a selectable tensor in a tuple/mapping |
| Segmentation | Binary and multiclass, default BCHW | Specify `class_dim` where appropriate; trainer/evaluator currently expect conventional channel-first layouts |
| Regression | Raw tensor outputs | Classification entropy is not applicable |
| Vision Transformer | TorchVision-style registered GELU | A custom ViT using functional activations needs manual replacement |
| `torch.compile` / TorchScript / ONNX | No compatibility guarantee | Dynamic random routing can cause graph breaks; test manually for each configuration |
| Multi-GPU / distributed | No validated guarantee | No DDP/FSDP integration test in this release |

Converting a model creates a deep copy by default; models that cannot be deep-copied need `inplace=True`. Shared module references are preserved in conversion; conditional Python control flow and dynamically created activations might not be found by module walking.

## Precision and custom parameters

Sampling weights remain float64 even after `.half()`/`.bfloat16()` so rare nonzero candidate probabilities are not rounded to zero. Parameterized alternative activations are aligned to the closest available model floating tensor at conversion time; for unusual multi-device architectures, move or configure the converted wrapper explicitly and verify each component's placement.

## Validating your model

1. Inspect `model.report()`; ensure expected layers were replaced.
2. Verify output shape and native baseline under `with model.stochastic_mode(False)`.
3. Check gradients, dtype, device, and any custom layer behavior.
4. Compare `model.predict(..., n_samples=5)` with a manual stack of repeated probabilities on a small batch.
5. Check throughput and memory on the real intended hardware before deployment.
