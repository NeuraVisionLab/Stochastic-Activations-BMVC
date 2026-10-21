# StochAct — included capabilities

StochAct is a general-purpose PyTorch framework for stochastic activation switching and uncertainty estimation.

## Core and compatibility

- Automatic conversion of supported registered activation modules, with a conversion report.
- Node-, channel-, and layer-level stochastic activation sampling; configurable probabilities and custom activations.
- Copy-by-default model conversion and explicit stochastic/deterministic controls.
- Correct activation-probability handling for `half()` and `bfloat16()` conversions.
- Device/dtype alignment for parameterized activation alternatives, and preservation of training state.
- Nesting-safe `StochasticModel.stochastic_mode(enabled)` context manager.

## Inference and workflows

- Memory-efficient Monte Carlo predictions, variance, entropy, and mutual information.
- Shared training framework: optimizer and scheduler options, validation, checkpoints, early stopping, and monitoring.
- Evaluator for classification, segmentation, regression, calibration, and out-of-distribution detection.
- Optional Matplotlib visualizations and runnable examples using ResNet, U-Net, and a Vision Transformer.

## Engineering

- Automated tests, contribution and security guidance, public API documentation, and GitHub Actions definitions.
- Packaging for source installation, standard Python wheel, and source distribution.
- External CI and CUDA validation remain pending until exercised on the actual supported environments.
