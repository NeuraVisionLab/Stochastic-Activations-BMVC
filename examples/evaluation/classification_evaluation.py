"""Classification uncertainty, calibration and OOD plots (synthetic data)."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from stochact import StochasticModel, Evaluator
from stochact.visualization import plot_reliability, plot_risk_coverage, plot_ood, plot_mc_convergence


def main():
    torch.manual_seed(4)
    raw = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    model = StochasticModel(raw, activations="silu", granularity="node")
    model.report()
    x = torch.randn(20, 16)
    y = torch.randint(0, 4, (20,))
    val = DataLoader(TensorDataset(x, y), batch_size=5)
    evaluator = Evaluator(model, task="classification", n_samples=5)
    report = evaluator.evaluate(val, return_details=True)
    print("Classification metrics:", report.metrics)
    out = Path("example_figures")
    plot_reliability(report, save_path=out / "reliability.png")
    plot_risk_coverage(report, save_path=out / "risk_coverage.png")
    ood = evaluator.evaluate_ood([torch.randn(8, 16)], [torch.randn(8, 16)+3],
                                  score="variance", return_details=True)
    print("OOD metrics:", ood.metrics)
    plot_ood(ood, save_path=out / "ood.png")
    result = model.predict(x[:4], n_samples=10, return_samples=True)
    plot_mc_convergence(result, save_path=out / "mc_convergence.png")
    print("Saved synthetic demonstration figures in", out.resolve())


if __name__ == "__main__":
    main()
