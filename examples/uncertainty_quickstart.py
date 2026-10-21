"""MC uncertainty inference demo for all three task types."""
import torch
from torch import nn
from stochact import StochasticModel


def main() -> None:
    torch.manual_seed(2026)
    # Multiclass classification: [B, C]
    classifier = StochasticModel(nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4)))
    classes = classifier.predict(torch.randn(6, 12), n_samples=5)
    print("Classification probabilities:", classes.mean.shape)
    print("Class labels:", classes.classes.tolist())
    print("Uncertainty scores:", classes.uncertainty.tolist())
    print("Mutual information:", classes.mutual_information.tolist())

    # Binary segmentation: [B, 1, H, W]
    segmenter = StochasticModel(nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(),
                                               nn.Conv2d(12, 1, 1)), activations="tanh")
    masks = segmenter.predict(torch.randn(2, 3, 32, 32), task="segmentation", n_samples=5)
    print("Segmentation masks:", masks.mask.shape)
    print("Pixel uncertainty maps:", masks.uncertainty_map.shape)

    # Regression: raw model outputs; no entropy or MI.
    regressor = StochasticModel(nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1)))
    values = regressor.predict(torch.randn(3, 5), task="regression", n_samples=5)
    print("Regression means:", values.mean.shape)
    print("Regression variances:", values.variance.shape)


if __name__ == "__main__":
    main()
