"""Train a real TorchVision ResNet18 on synthetic labels (no downloads).

Run from the project root: python examples/training/train_resnet.py
Requires torchvision: pip install -e '.[examples]'
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from stochact import StochasticModel, Trainer


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    base = resnet18(weights=None, num_classes=3)
    model = StochasticModel(base, granularity="node", activations="silu", p=0.5)
    model.report()
    images = torch.randn(8, 3, 64, 64)
    labels = torch.randint(0, 3, (8,))
    train_data = DataLoader(TensorDataset(images[:6], labels[:6]), batch_size=2)
    val_data = DataLoader(TensorDataset(images[6:], labels[6:]), batch_size=2)
    trainer = Trainer(model, task="classification", optimizer="adamw", lr=1e-4,
                      scheduler="cosine", mixed_precision=False)
    history = trainer.fit(train_data, val_data, epochs=2,
                          uncertainty_every=1, uncertainty_samples=3)
    print("Training records:", history.records)
    print("MC evaluation:", trainer.test(val_data, n_samples=3))
    print("Prediction uncertainty:", model.predict(images[6:], n_samples=3).uncertainty)
    print("Note: labels are random; results validate usage, not classification quality.")


if __name__ == "__main__":
    main()
