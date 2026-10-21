# Supervised training

```python
from stochact import Trainer, StochasticModel

model = StochasticModel(backbone, granularity='node', activations='silu')
trainer = Trainer(model, task='classification', optimizer='adamw', lr=1e-3,
                  scheduler='cosine', mixed_precision=True,
                  accumulation_steps=2, grad_clip_norm=1.0)
history = trainer.fit(train_loader, val_loader, epochs=20,
                      uncertainty_every=5, uncertainty_samples=5,
                      checkpoint_dir='checkpoints', early_stopping_patience=5)
metrics = trainer.test(test_loader, n_samples=5)
```

- Batch format: `(inputs, targets)` or mapping with supported input and target keys.
- Tasks: classification, segmentation, regression. Standard loss is CE/BCE-with-logits or MSE; you can provide a scalar-returning custom loss.
- Optimizers: Adam, AdamW, SGD, or an optimizer object. Schedulers: cosine, step, plateau or an initialized scheduler.
- MC monitoring runs on a limited subset and is *not* an extra uncertainty training loss.
- Mixed precision uses CPU bfloat16 or CUDA float16 with gradient scaling as applicable.
- Gradient accumulation, clipping, checkpointing, early stopping and epoch callbacks are supported.
- `Trainer.test(..., n_samples>1)` evaluates using mean predicted probabilities. A custom loss in MC testing receives mean predictions instead of raw logits.

## Checkpoints and security

`trainer.save_checkpoint('last.pt')` stores training state. `trainer.load_checkpoint('last.pt', load_optimizer=True, restore_rng=True)` restores a **trusted** checkpoint. **Never use pickle-based checkpoints from unknown sources**; `load_checkpoint` currently uses `weights_only=False`. To load only parameter tensors safely, see the [security guide](../SECURITY.md).

`Trainer` is optional. Existing custom loops can train `StochasticModel` directly with standard autograd and optimizers.
