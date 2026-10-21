"""ResNet18 (optional torchvision dependency); no pretrained weights downloaded."""
import torch
from torchvision.models import resnet18

from stochact import StochasticModel

base = resnet18(weights=None, num_classes=10)
model = StochasticModel(
    base,
    granularity="node",
    activations="silu",
    p=0.5,
)
model.report()
model.eval()
x = torch.randn(2, 3, 64, 64)
result = model.predict(x, n_samples=5)
print("ResNet probability shape:", tuple(result.mean.shape))
print("ResNet variance shape:", tuple(result.variance.shape))
print("Native model remains unmodified:", type(base.relu).__name__)
