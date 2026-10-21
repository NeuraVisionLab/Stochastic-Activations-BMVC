"""Binary segmentation training with synthetic U-Net data (no downloads)."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from stochact import StochasticModel, Trainer


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        skip = self.encoder(x)
        encoded = self.bottleneck(F.avg_pool2d(skip, 2))
        up = F.interpolate(encoded, size=x.shape[-2:], mode="nearest")
        return self.head(self.decoder(torch.cat([skip, up], 1)))


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = StochasticModel(TinyUNet(), granularity="channel", activations="tanh")
    images = torch.randn(8, 3, 32, 32)
    masks = (images[:, 0] > 0).float()
    train_data = DataLoader(TensorDataset(images[:6], masks[:6]), batch_size=2)
    val_data = DataLoader(TensorDataset(images[6:], masks[6:]), batch_size=2)
    trainer = Trainer(model, task="segmentation", optimizer="adamw", lr=3e-3)
    history = trainer.fit(train_data, val_data, epochs=2, uncertainty_every=1,
                          uncertainty_samples=3)
    print("Last epoch:", history[-1])
    print("MC segmentation metrics:", trainer.test(val_data, n_samples=3))
    pred = model.predict(images[6:], task="segmentation", n_samples=3)
    print("Masks:", pred.mask.shape, "Uncertainty maps:", pred.uncertainty_map.shape)


if __name__ == "__main__":
    main()
