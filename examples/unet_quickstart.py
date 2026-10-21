"""Small, fully self-contained U-Net example using native PyTorch modules."""
import torch
from torch import nn
from torch.nn import functional as F

from stochact import StochasticModel


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = conv_block(3, 16)
        self.middle = conv_block(16, 32)
        self.decoder = conv_block(48, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.encoder(x)
        down = F.avg_pool2d(skip, 2)
        middle = self.middle(down)
        up = F.interpolate(middle, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder(torch.cat([skip, up], dim=1)))


base = TinyUNet()
model = StochasticModel(base, granularity="channel", activations="tanh", p=0.5)
model.report()
model.eval()
result = model.predict(torch.randn(2, 3, 48, 48), task="segmentation", n_samples=5)
print("Binary segmentation mask shape:", tuple(result.mask.shape))
print("Uncertainty map shape:", tuple(result.uncertainty_map.shape))
