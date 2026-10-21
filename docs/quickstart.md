# Quickstart

Install from the extracted source directory:

```bash
python -m pip install -e '.[all]'
```

Convert a built-in activation layer in any supported `nn.Module`:

```python
import torch
from torch import nn
from stochact import StochasticModel

backbone = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 3))
model = StochasticModel(backbone, activations='tanh', granularity='node', p=0.5)
model.report()

x = torch.randn(8, 12)
logits = model(x)  # standard forward; gradients work normally

result = model.predict(x, task='classification', n_samples=5)
print(result.mean.shape, result.variance.shape, result.mutual_information.shape)

# Original backbone behavior, without sampling:
with model.stochastic_mode(False):
    native_logits = model(x)
```

Notes:

- The wrapper *deep-copies* the backbone by default. Pass `inplace=True` to modify it instead.
- Registered activation modules (ReLU, GELU, etc.) are replaced; functional calls (`F.relu`) aren't rewritten.
- Switching stays enabled under `.eval()`. `predict()` explicitly freezes regular training-mode effects.
- Predictions use averaged *probabilities*, not averaged logits.

For supervised learning use `Trainer(model, task='classification', optimizer='adamw')`. For test-set metrics use `Evaluator(model, task='classification', n_samples=5)`. For a trained U-Net set `task='segmentation'`, `mode='binary'` or `'multiclass'` as appropriate.

See [API reference](api.md), [training guide](training.md), and [compatibility matrix](compatibility.md).
