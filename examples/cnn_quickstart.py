"""Quickstart for convolutional segmentation-style outputs."""
import torch
from torch import nn
from stochact import StochasticModel

network = nn.Sequential(
    nn.Conv2d(3, 12, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(12, 1, 1),
)
model = StochasticModel(network, granularity="channel", activations="tanh", p=0.5)
model.report()
images = torch.randn(2, 3, 32, 32)
print("Output logits shape:", model(images).shape)
