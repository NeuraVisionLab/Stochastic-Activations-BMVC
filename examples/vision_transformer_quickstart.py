"""Real Vision Transformer example using TorchVision, without downloads.

Run: python examples/vision_transformer_quickstart.py
Requires: pip install -e '.[examples]'

Small ViT configuration keeps CPU quickstart practical. TorchVision's ViT MLP
contains registered nn.GELU modules, converted using the standard modules backend.
"""
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from stochact import StochasticModel


def main() -> None:
    torch.manual_seed(2026)
    vit = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=4,
        hidden_dim=96,
        mlp_dim=192,
        num_classes=10,
        dropout=0.0,
        attention_dropout=0.0,
    )
    # TorchVision initializes its classification head to exactly zero. For an
    # untrained demonstration, initialize the head to nonzero random weights;
    # otherwise *every* MC pass is identically zero regardless of activations.
    nn.init.normal_(vit.heads.head.weight, std=0.02)
    nn.init.zeros_(vit.heads.head.bias)
    model = StochasticModel(
        vit,
        granularity="channel",  # [B, tokens, hidden] -> sample per feature
        activations="silu",     # native GELU + SiLU
        p=0.5,
    )
    report = model.report()
    assert report.replaced > 0, "ViT did not expose convertible GELU modules"
    images = torch.randn(2, 3, 32, 32)
    model.eval()
    result = model.predict(images, task="classification", n_samples=5)
    print("ViT model:", type(vit).__name__)
    print("Converted activations:", report.replaced)
    print("Mean probabilities:", tuple(result.mean.shape))
    print("Predicted classes:", result.classes.tolist())
    print("Variance scores:", result.uncertainty.tolist())
    print("Mutual information:", result.mutual_information.tolist())
    print("Original model GELU untouched:", 'GELU' in repr(vit))


if __name__ == "__main__":
    main()
