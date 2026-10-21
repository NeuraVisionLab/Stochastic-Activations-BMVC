"""Binary segmentation evaluation/uncertainty visualization (synthetic data)."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from stochact import StochasticModel, Evaluator
from stochact.visualization import plot_uncertainty


def main():
    torch.manual_seed(8)
    raw = nn.Sequential(nn.Conv2d(1, 6, kernel_size=3, padding=1), nn.ReLU(),
                        nn.Conv2d(6, 1, kernel_size=1))
    model = StochasticModel(raw, activations="tanh", granularity="node")
    images = torch.rand(6, 1, 32, 32)
    masks = (images[:, 0] > 0.5).long()
    val = DataLoader(TensorDataset(images, masks), batch_size=2)
    report = Evaluator(model, task="segmentation", n_samples=5, ccq_tolerance=1).evaluate(val)
    print("Segmentation metrics:", report)
    predictions = model.predict(images[:2], task="segmentation", n_samples=5)
    out = Path("example_figures")
    plot_uncertainty(images[:2], predictions, target=masks[:2],
                     save_path=out / "segmentation_uncertainty.png")
    print("Saved synthetic demonstration figure in", out.resolve())


if __name__ == "__main__":
    main()
