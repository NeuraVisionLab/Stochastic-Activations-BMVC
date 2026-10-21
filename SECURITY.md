# Security guidance

StochAct operates inside Python/PyTorch code supplied by the user. It is a research library, not a sandbox or a security boundary.

- **Only load trusted checkpoints.** `Trainer.load_checkpoint()` uses `torch.load(weights_only=False)` to restore optimizer, scheduler, Python RNG, and training history; malicious pickle payloads can run code. For weights-only use, prefer `model.load_state_dict(torch.load(file, weights_only=True))` from a trusted, expected source.
- **Only register trusted activation factories.** Custom activation callables are executable Python code.
- Avoid placing secrets in notebooks, training logs, uploaded datasets or issue reports.
- Do not expect StochAct to validate or sanitize arbitrary model input, dataset content, or output tensors beyond the documented checks.

Please report suspected vulnerabilities privately to repository maintainers when a contact channel has been established; do not publish exploits against another person's infrastructure.
