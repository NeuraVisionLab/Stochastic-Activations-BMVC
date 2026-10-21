"""Train a real tiny TorchVision VisionTransformer (random synthetic data).

Run: python examples/training/train_vit.py
Requires torchvision. No downloads, no pretrained weights.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

from stochact import StochasticModel, Trainer


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    vit = VisionTransformer(
        image_size=32, patch_size=8, num_layers=1,
        num_heads=4, hidden_dim=64, mlp_dim=128, num_classes=3,
        dropout=0.0, attention_dropout=0.0,
    )
    # TorchVision initially zeroes the ViT prediction head. Small random weights
    # make stochastic activation differences visible from the first forward pass.
    nn.init.normal_(vit.heads.head.weight, std=0.02)
    model = StochasticModel(vit, granularity="channel", activations="silu", p=0.5)
    model.report()
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 3, (8,))
    train_data = DataLoader(TensorDataset(images[:6], labels[:6]), batch_size=2)
    val_data = DataLoader(TensorDataset(images[6:], labels[6:]), batch_size=2)
    trainer = Trainer(model, task="classification", optimizer="adamw", lr=5e-4,
                      scheduler="cosine", grad_clip_norm=1.0)
    history = trainer.fit(train_data, val_data, epochs=2, uncertainty_every=1,
                          uncertainty_samples=3)
    print("Last epoch:", history[-1])
    print("MC ViT metrics:", trainer.test(val_data, n_samples=3))
    print("MC variance shape:", model.predict(images[6:], n_samples=3).variance.shape)
    print("Synthetic labels are for API verification only, not accuracy evaluation.")


if __name__ == "__main__":
    main()
