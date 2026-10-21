# StochAct

**A reusable PyTorch library for stochastic activations, uncertainty estimation, training, evaluation, and visualization.**

StochAct is a general-purpose research library based on *Stochastic Nonlinearities Improve Uncertainty Estimation* (BMVC 2026). It converts supported registered activation layers in an ordinary `torch.nn.Module` into stochastic activation switches, then estimates uncertainty by repeating forward passes with the **same trained weights** and different activation choices.

StochAct is a general-purpose research toolkit. It does not require reproducing any specific BMVC dataset, configuration, or numerical result. **This source package is prepared for publication but is not yet published to PyPI.** Before public release, repository maintainers should confirm licensing and contributor credits.

## Installation

After publication to PyPI, install using `pip install stochact`. Until then, install from the checked-out source:


```bash
# In the extracted StochAct/ source directory
pip install -e .

# For tests / development:
pip install -e '.[dev]'
pytest -q

# For the TorchVision ResNet and ViT demos:
pip install -e '.[examples]'

# For visualization (Matplotlib optional):
pip install -e '.[plots]'

# Everything optional (TorchVision and plotting):
pip install -e '.[all]'
```

Python >= 3.10, PyTorch >= 2.0. No datasets, pretrained checkpoints or CUDA are required for the included examples. The core package only depends on PyTorch.

## 1. Convert any *supported* PyTorch model

```python
import torch
from torch import nn
from stochact import StochasticModel

base_model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5)
)

# Deep-copies base_model by default: the original stays untouched.
model = StochasticModel(
    base_model,
    granularity="node",           # node / channel / layer
    activations=None,             # native activation + SiLU
    p=0.5,                       # probability of selecting SiLU
)
model.report()

x = torch.randn(8, 20)
logits = model(x)                 # ordinary PyTorch forward call
```

The wrapper remains a normal `nn.Module`: `.to(device)`, `.parameters()`, optimizers, backward propagation, `.train()` and `.eval()` work. It introduces **no new trainable parameters** with the built-in parameter-free activation choices. `inplace=True` optionally modifies the provided original model instead.

### Activation pools

| Configuration | Stochastic choices |
| --- | --- |
| `activations=None` | Native + SiLU |
| `activations="tanh"` | Native + Tanh |
| `activations=["relu", "gelu"]` | Exactly ReLU / GELU |
| `activations=["original", "mish"]` | Native + Mish |
| `activations=["relu", "silu", "tanh"]` | Exactly three activations |
| `activations=nn.Softsign()` | Native + custom module |

A single activation or single-item list is paired with the backbone's native activation. Two or more explicit list entries define the **entire** stochastic activation pool. Native activation is always retained for deterministic evaluation even if omitted from the stochastic pool.

```python
model = StochasticModel(
    base_model,
    granularity="node",
    activations=["relu", "silu", "tanh"],
    probs=[0.2, 0.5, 0.3],
)
```

`p` applies only to a **two-candidate** pool, selecting the second candidate with probability `p`; `probs` sets each candidate's probability. Both cannot be supplied together. When neither is provided, probabilities are uniform. Multiple activations, unusual activation pairs, and custom modules are library extensions not validated as configurations in the BMVC paper.

Custom activation factories can be registered using `register_activation("name", MyActivation)`. Candidates should be elementwise, shape-preserving and free of running-state side effects.

### Granularity

- `node` *(default)* samples an activation choice for each individual tensor element.
- `channel` samples a choice for each feature channel and example, shared across spatial locations. For `[B,C,H,W]`, masks are `[B,C,1,1]`. For transformer `[B,tokens,features]`, masks default to `[B,1,features]`. Use `channel_dim` to override and `batch_shared_channels=True` to share across a batch.
- `layer` samples one choice for an entire activation layer and forward invocation.

### Conversion report and scope

```python
model = StochasticModel(base_model, include=["layer1.*"], exclude=["layer1.0.relu"])
report = model.report()                 # prints and returns ConversionReport
print(report.detected, report.replaced, report.skipped, report.parameters_added)
print(model.stochastic_layers().keys())
```

`include` and `exclude` accept glob paths, or regex prefixed `re:`. Supported native modules include ReLU, GELU, SiLU, Tanh, ELU, LeakyReLU, Mish, Softplus, PReLU and others. **Output Sigmoid/Softmax layers are not replaced by default.** Functional calls like `torch.nn.functional.relu` cannot be automatically converted with `backend="modules"`; an FX backend is planned separately. Models without registered supported activation layers will report zero replacements; `predict()` refuses to silently treat those as stochastic models.

### Stochastic vs deterministic forward

```python
model.eval()                     # freezes normal dropout and batchnorm
model.set_stochastic(True)       # samples activations even in eval mode
stochastic_logits = model(x)
model.set_stochastic(False)      # uses original native activations
native_logits = model(x)
model.set_stochastic(True)

# Temporary override, restoring even mixed per-layer settings on exit:
with model.stochastic_mode(False):
    deterministic_result = model(x)
```

Model `.train()` and `.eval()` do **not** control stochastic activation switching. The two settings are independent.

## 2. Monte Carlo inference and uncertainty

```python
result = model.predict(x, task="classification", n_samples=5)

result.mean                  # [B,C] predictive mean of class probabilities
result.probabilities         # alias for result.mean
result.predictions           # [B] predicted class IDs
result.classes               # alias for result.predictions
result.variance              # [B,C] population variance across probabilities
result.uncertainty           # [B] class-mean variance, one score per sample
result.predictive_entropy    # [B]
result.expected_entropy      # [B]
result.mutual_information    # [B]
result.samples               # None by default; no T-sized prediction stack
```

If needed, `return_samples=True` includes a `[T,B,C,...]` tensor of probabilities; this uses memory proportional to the number of passes. With its default `False`, predictive **mean and population variance** are calculated online using a numerically stable Welford update: O(output size) memory, independent of `n_samples`.

Inference uses `torch.inference_mode()`, calls `.eval()` temporarily to freeze ordinary dropout/BatchNorm, **enables stochastic activations**, and restores the previous training and stochastic switches even if an exception occurs. The returned tensors stay on the model's output device. The code assumes input tensors are already on the correct device.

### Classification

For logits `[B,C]`, `mode="auto"` uses multiclass softmax if `C>1`, binary sigmoid if `C=1`. For binary classification, `result.classes` is `[B]` and a configurable `threshold` defaults to 0.5. If the model already outputs normalized probabilities, pass `from_logits=False`.

### Binary segmentation

```python
result = model.predict(images, task="segmentation", n_samples=5)
result.mean                 # [B,1,H,W] positive-class probabilities
result.variance             # [B,1,H,W] probability variance
result.mask                 # [B,H,W] binary mask
result.uncertainty_map      # [B,H,W] pixel uncertainty
result.uncertainty          # [B] spatial-mean variance
result.predictive_entropy  # [B,H,W] Bernoulli entropy
result.mutual_information   # [B,H,W]
```

Supports `[B,H,W]` single-channel-free binary outputs too.

### Multiclass segmentation

For logits `[B,C,H,W]` with `C>1`, probabilities and variance are `[B,C,H,W]`, while masks and entropy/MI maps are `[B,H,W]`. `uncertainty_map` is the **mean over class-wise predictive variances**; `uncertainty` is its spatial mean. The class axis defaults to `class_dim=1`, but other arrangements such as NHWC can be specified.

### Regression

```python
result = model.predict(x, task="regression", n_samples=5)
result.mean                # Mean of raw continuous outputs
result.variance            # Population variance of continuous outputs
result.predictions         # Alias for mean
result.uncertainty         # Mean variance across output dimensions, [B]
```

Categorical entropy and mutual information are `None` for regression. Neither classification nor segmentation uncertainty scores should be read as calibrated probabilities of error. These sample-based quantities also do not provide a guaranteed separation of epistemic and aleatoric uncertainty.

### Advanced options

```python
result = model.predict(
    x,
    n_samples=20,
    return_samples=False,
    stochastic=True,         # False gives deterministic native baseline
    output_key="out",        # for torchvision-style dict outputs
    # output_index=0,        # alternatively for tuple/list outputs
    from_logits=True,
    class_dim=1,
)
```

Additional keyword arguments are forwarded to the underlying model. `predict_mc(model, x, ...)` is also available as a standalone function for any model that contains `StochasticActivation` layers.

**Probability definitions:** each pass converts raw logits to probabilities using sigmoid or softmax, then computes the mean and population variance across **probabilities** (not logits). Predictive entropy is entropy of the mean probability, expected entropy is the average per-pass entropy, and mutual information is their nonnegative difference (subject to numerical rounding). For deterministic mode and one pass, predictive variance should be zero up to floating-point error.


## 3. Standard Trainer

`Trainer` is **optional** and accepts either `StochasticModel` or a normal `nn.Module`. It does not introduce separate `Classifier` or `Segmenter` wrappers. Set `task` to `"classification"`, `"segmentation"` or `"regression"`. It provides PyTorch training/validation loops, losses, basic task metrics, optimizers, schedulers, AMP, gradient clipping, accumulation, callbacks, checkpoints, early stopping, and opt-in MC uncertainty diagnostics. **It is not a high-level experiment manager or replacement for custom training loops.**

```python
from stochact import StochasticModel, Trainer

model = StochasticModel(your_backbone, granularity="node", activations="silu")
trainer = Trainer(
    model, task="classification",
    optimizer="adamw", lr=1e-3, weight_decay=1e-4,
    scheduler="cosine", mixed_precision=True,
    grad_clip_norm=1.0, accumulation_steps=1,
)

history = trainer.fit(
    train_loader, val_loader, epochs=50,
    uncertainty_every=5, uncertainty_samples=5, uncertainty_batches=2,
    checkpoint_dir="./checkpoints", early_stopping_patience=8,
)
print(history[-1])             # dict of train/validation epoch metrics
print(history["val_loss"])  # list of available validation losses
print(trainer.test(test_loader, n_samples=5))  # predictive-mean MC testing
```

**Default tasks and losses**

| Task / output | Target shape | Default loss | Lightweight metrics |
| --- | --- | --- | --- |
| Classification `[B,C]`, C>1 | `[B]` integer class IDs | Cross-entropy | Accuracy |
| Binary classification `[B]` or `[B,1]` | `[B]` or `[B,1]` | BCE-with-logits | Accuracy |
| Multiclass segmentation `[B,C,H,W]`, C>1 | `[B,H,W]` integer mask | Cross-entropy | Pixel accuracy, micro foreground Dice |
| Binary segmentation `[B,1,H,W]` / `[B,H,W]` | `[B,H,W]` or matching output shape | BCE-with-logits | Pixel accuracy, foreground Dice |
| Regression (any continuous output) | Same shape as output | MSE | MAE, MSE |

For already-normalized probabilities, set `from_logits=False`; the built-in default losses then use probability-safe BCE/NLL formulations. For specialized losses (Dice/CE combinations, weighted losses, etc.) pass `loss_fn=lambda outputs, targets: ...`. For multiple model output tensors select `output_key="out"` or `output_index=0`. Use `ignore_index=255` to exclude unlabeled pixels. The lightweight segmentation Dice metric is class-sensitive micro Dice over **foreground classes** (`0` treated as background), and is **not** the full calibration/structural evaluation suite in the `Evaluator`.

**Uncertainty diagnostics:** `uncertainty_every=N` computes mean per-example predictive variance (not a calibration score) on up to `uncertainty_batches` of both training and validation splits at epoch 1 and then every N epochs. It runs T additional stochastic passes per selected batch with gradients off and BN/dropout frozen, without running T passes for every training update. Requires a converted model and simple tensor inputs for diagnostics. Training itself uses **one stochastic pass per batch**. `uncertainty_every=None` (default) disables the overhead. Training and validation batches can be `(inputs, targets)` or mappings with keys such as `inputs`/`targets`, `x`/`y`, `image`/`mask`. Tuple/dict multi-input networks work for ordinary fit/evaluate, but MC monitoring and MC test in the Trainer currently require a single input tensor.

**Training utilities:** built-in optimizers are `"adam"`, `"adamw"`, and `"sgd"` (or pass your own `torch.optim.Optimizer`). Schedulers are `None` (default), `"cosine"`, `"step"`, `"plateau"`, or a PyTorch scheduler instance. A cosine schedule is created when `fit()` knows the total requested epochs; for resumed runs, its original scheduling horizon is restored from the checkpoint. Mixed precision uses CUDA float16 + gradient scaling on GPU and CPU bfloat16 autocast on CPU; CUDA behavior has not been tested on a GPU in this release. `seed=` seeds PyTorch/Python when creating the Trainer; for full experiment reproducibility also seed prior to creating the model, and manage DataLoader worker seeds.

**Checkpointing and early stopping:**

```python
trainer.fit(train_loader, val_loader, epochs=100,
            checkpoint_dir="runs/demo",
            early_stopping_patience=10, min_delta=0.0001,
            restore_best_weights=True)
# Saves runs/demo/last.pt and runs/demo/best.pt

# Reinitialize the *same* backbone, optimizer and scheduler configuration first:
resumed = Trainer(model, task="classification", scheduler="cosine")
resumed.load_checkpoint("runs/demo/last.pt", restore_rng=True)
resumed.fit(train_loader, val_loader, epochs=5)
```

Best epoch is determined from validation loss when a validation loader is supplied, otherwise training loss. Early stopping is optional and uses loss improvements. Checkpoints contain model, optimizer, scheduler, AMP scaler, Python/PyTorch RNG states, current epoch and `TrainingHistory`. `restore_best_weights=True` restores the best weights seen **in the current fit call** after saving the final checkpoint; for a multi-session resume, `best.pt` is separately preserved. Load checkpoint files **only from trusted sources** because they use pickle. DataLoader shuffle order/worker state is not automatically checkpointed. Custom epoch callbacks (`callbacks=[lambda trainer, row: ...]`) receive a dictionary of numeric logs.

`trainer.validate(loader)` and `trainer.test(loader, n_samples=1)` perform one stochastic forward pass per batch. With `n_samples>1`, test metrics and default loss use the mean MC probabilities (or regression means). A custom `loss_fn` in this case receives mean probabilities/outputs, not logits; design it accordingly. For ECE, Brier, OOD, structural metrics and publication figures, use the optional `Evaluator` and visualization tools; the lightweight `Trainer.test()` API remains unchanged.

### Training examples (no datasets or weights downloaded)

```bash
python examples/training/train_resnet.py  # TorchVision ResNet18, classification
python examples/training/train_unet.py    # Pure PyTorch U-Net, segmentation
python examples/training/train_vit.py     # TorchVision Vision Transformer, classification
```

All three examples use small synthetic inputs/labels to verify API behavior, **not** to demonstrate scientific performance. Install the optional TorchVision dependency for ResNet and ViT: `pip install -e '.[examples]'`.

## Examples (all run without downloads)

```bash
python examples/mlp_quickstart.py
python examples/cnn_quickstart.py
python examples/unet_quickstart.py
python examples/uncertainty_quickstart.py

# TorchVision dependency required:
python examples/resnet18_quickstart.py
python examples/vision_transformer_quickstart.py
```

The **Vision Transformer** example uses a small, real TorchVision `VisionTransformer` with two encoder blocks and registered GELU activations. It converts native GELU to GELU/SiLU at channel-level granularity and reports MC uncertainty. TorchVision initializes its ViT classification head to zero; our synthetic demo initializes it to small random weights **for nontrivial demo predictions**. This is not a trained model, so its uncertainty values are demonstrations of the API, **not measurements of meaningful predictive reliability**.

Other examples also use synthetic data rather than BMVC benchmark datasets. The original network remains unmodified by default.

## 4. General-purpose Evaluator

The `Evaluator` is independent of `Trainer` and works with a converted `StochasticModel`. Classification, segmentation and regression use one interface; there are **no task-specific model wrapper classes**. Its default `evaluate(loader)` returns a plain dictionary of metrics. Pass `return_details=True` when you need additional arrays for plotting.

```python
from stochact import Evaluator

evaluator = Evaluator(model, task="classification", n_samples=5, n_bins=15)
metrics = evaluator.evaluate(test_loader)
print(metrics)  # accuracy, precision_macro, recall_macro, f1_macro, ...

# Additional small arrays for figures:
details = evaluator.evaluate(test_loader, return_details=True)
print(details.metrics, details.confusion_matrix)
```

For segmentation: `Evaluator(model, task="segmentation", mode="binary", n_samples=5, ignore_index=255, ccq_tolerance=0)`; for multiclass segmentation set `mode="multiclass"` or leave `"auto"`. For regression use `task="regression"`. Inputs must currently be a single tensor; batches are `(x, y)` or dictionaries with recognized input/target keys, matching the MC inference interface.

**Available metrics**

| Task | Metric names | Notes |
| --- | --- | --- |
| Classification | `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `f1_weighted`, `ece`, `brier`, `nll` | Macro averages include all model classes; the Brier score is a class-wise squared-error sum |
| Segmentation | `pixel_accuracy`, `foreground_dice`, `foreground_iou`, `foreground_precision`, `foreground_recall`, `dice_macro_fg`, `iou_macro_fg`, `ece`, `brier`, `nll`, `uncertainty_error_corr` | Class 0 is background; metrics are reduced over valid pixels, not averaged by minibatch |
| Binary 2D segmentation | `correctness`, `completeness`, `quality` | With `ccq_tolerance=0`, equivalent to foreground precision, recall and IoU; increasing pixel tolerance allows approximate spatial matching. This is **not** a benchmark-specific centerline CCQ protocol |
| Regression | `mae`, `mse`, `rmse` | Raw continuous predictions |

ECE uses equal-width confidence bins (`n_bins`, default 15) and top-label accuracy; binary classification confidence follows the predicted label. Brier is `(p-y)^2` for binary tasks and `sum_c(p_c-1[y=c])^2` for multiclass tasks. NLL uses probability aggregation over stochastic samples before taking logarithms. `uncertainty_error_corr` computes Pearson correlation of pixel variance and binary prediction error. It returns NaN if either variable is constant, rather than falsely reporting zero. `ignore_index` excludes pixels from loss/calibration/overlap calculations.

The standalone utilities `calibration_metrics(probabilities, targets, mode=...)`, `binary_ccq(pred, truth, tolerance=...)` and `binary_curves(scores, labels)` are also available. They can be used without StochAct's model wrapper.

### Out-of-distribution detection

```python
# id_loader, ood_loader may provide images only, or (images, labels).
metrics = evaluator.evaluate_ood(id_loader, ood_loader, score="variance")
print(metrics)  # AUROC, AUPR, FPR95

# For ROC and PR figure data:
ood = evaluator.evaluate_ood(id_loader, ood_loader,
                             score="mutual_information", return_details=True)
```

Use `score="variance"`, `"predictive_entropy"`, `"mutual_information"`, or `"max_probability"` (negative maximum predicted probability). One higher-is-more-OOD score is computed per image: for segmentation, spatial scores are averaged. OOD is treated as positive; AUROC uses trapezoidal ROC integration, AUPR uses stepwise average precision, and FPR95 selects the minimum false-positive rate achieving at least 95% OOD true-positive rate. **AUPR depends on the in-/out-of-distribution example ratio**, so comparisons require matched class prevalence. OOD datasets are optional; this is a generic detection utility, not proof of calibrated OOD confidence.

### Plotting and figures

Install the optional Matplotlib extra (`pip install -e '.[plots]'`). Every plot returns `(fig, ax_or_axes)` so it can be further customized or exported to PNG/PDF/SVG via `save_path`.

```python
from stochact.visualization import (
    plot_training, plot_uncertainty, plot_reliability,
    plot_risk_coverage, plot_ood, plot_mc_convergence,
)

plot_training(history, metrics=["loss", "accuracy", "uncertainty"],
              save_path="figures/training.pdf")
plot_reliability(details, save_path="figures/calibration.svg")
plot_risk_coverage(details, save_path="figures/risk.png")
plot_ood(ood, save_path="figures/ood.png")

# 2D binary or multiclass segmentation:
seg_result = model.predict(images, task="segmentation", n_samples=5)
plot_uncertainty(images, seg_result, target=masks,
                 save_path="figures/segmentation.png")

# Convergence requires retaining T stochastic predictions:
mc = model.predict(images, n_samples=20, return_samples=True)
plot_mc_convergence(mc, save_path="figures/convergence.png")
```

`plot_uncertainty` displays actual predictive **variance**, not the probability of a pixel being erroneous. MC convergence plotting retains all samples only when `return_samples=True`; normal prediction and evaluation keep streaming statistics. The extra plotting dependency is loaded lazily so core users do not need Matplotlib.

**Note:** Trainer loss/accuracy history is recorded during training; detailed ECE and Brier histories are calculated by running `Evaluator` at your chosen validation checkpoints rather than increasing normal training costs automatically.

**General examples:** `examples/evaluation/classification_evaluation.py` and `examples/evaluation/segmentation_evaluation.py` use small synthetic tensors to demonstrate the API. They are functional smoke tests, not claims of model quality.

## Included components

StochAct provides model conversion, stochastic activation sampling, uncertainty inference, supervised training, evaluation metrics, visualization utilities, examples, and automated tests. These features are available through a shared API without task-specific model wrappers.

## Tests

```bash
python -m pytest -q
```

The suite covers conversion and sampling, gradients, state restoration, MC inference, segmentation, regression, Vision Transformers, training, checkpoint resume, calibration, OOD metrics, CCQ, plotting, and precision/compatibility edge cases. CPU tests run automatically; a CUDA test is skipped unless hardware is available. Multi-platform CI is configured but has not yet been executed remotely.

## Citation

Based on: Mohammad Sarhangzadeh, Amirreza M. Shebly, and Doruk Oner, *Stochastic Nonlinearities Improve Uncertainty Estimation*, BMVC 2026. The multi-activation and model-agnostic usability extensions are features of this software, and should not be presented as experiments validated by the paper.

## Documentation and contributing

- [Quickstart](docs/quickstart.md) · [API reference](docs/api.md) · [Training](docs/training.md) · [Evaluation](docs/evaluation.md)
- [Compatibility and limitations](docs/compatibility.md) · [Maintainer release guide](docs/release.md)
- [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

The project includes GitHub Actions definitions for Linux/Windows/macOS and a package-build check; CI has **not** yet run on an actual GitHub repository. The provided environment was CPU-only, so CUDA-specific validation is conditional and skipped here. The MIT license is included but project maintainers should confirm publication rights before a public release.
