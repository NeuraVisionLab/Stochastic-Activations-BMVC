"""Visualization functions tested with real evaluation/inference results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch
from torch import nn
from stochact import StochasticModel, Evaluator
from stochact.visualization import (
    plot_training, plot_uncertainty, plot_reliability, plot_risk_coverage,
    plot_ood, plot_mc_convergence,
)


def model_cls():
    return StochasticModel(nn.Sequential(nn.Linear(3, 9), nn.ReLU(), nn.Linear(9, 2)))


def model_seg():
    return StochasticModel(nn.Sequential(nn.Conv2d(1, 5, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(5, 1, 1)))


def test_plot_training(tmp_path):
    h = [{"epoch": 1, "train_loss": 2., "val_loss": 2.2, "train_accuracy": .3},
         {"epoch": 2, "train_loss": 1., "val_loss": 1.4, "train_accuracy": .8}]
    out = tmp_path / "training.pdf"
    fig, axes = plot_training(h, save_path=out)
    assert len(axes) == 3
    assert out.is_file()
    plt.close(fig)


def test_plot_training_empty():
    with pytest.raises(ValueError):
        plot_training([])


def test_plot_reliability(tmp_path):
    model = model_cls()
    x = torch.randn(5, 3)
    y = torch.randint(0, 2, (5,))
    r = Evaluator(model).evaluate([(x, y)], return_details=True)
    fig, ax = plot_reliability(r, save_path=tmp_path / "reliability.svg")
    assert ax.get_xlabel() == "Confidence"
    assert (tmp_path / "reliability.svg").exists()
    plt.close(fig)


def test_plot_risk_coverage(tmp_path):
    model = model_cls()
    x = torch.randn(8,3)
    y = torch.randint(0,2,(8,))
    r = Evaluator(model).evaluate([(x,y)], return_details=True)
    fig, ax = plot_risk_coverage(r, save_path=tmp_path / "risk.png")
    assert "Coverage" in ax.get_xlabel()
    assert (tmp_path / "risk.png").exists()
    plt.close(fig)


def test_plot_ood(tmp_path):
    model = model_cls()
    e = Evaluator(model)
    r = e.evaluate_ood([torch.randn(3,3)], [torch.randn(4,3)], return_details=True)
    fig, axes = plot_ood(r, save_path=tmp_path / "ood.png")
    assert len(axes) == 2
    assert (tmp_path / "ood.png").exists()
    plt.close(fig)


def test_plot_uncertainty(tmp_path):
    model = model_seg()
    x = torch.rand(2, 1, 16, 16)
    result = model.predict(x, task="segmentation", n_samples=4)
    gt = torch.randint(0, 2, (2, 16, 16))
    fig, axes = plot_uncertainty(x, result, target=gt, save_path=tmp_path / "map.png")
    assert len(axes) == 5
    assert (tmp_path / "map.png").exists()
    plt.close(fig)


def test_plot_uncertainty_without_gt():
    model = model_seg()
    x = torch.rand(1, 1, 8, 8)
    result = model.predict(x, task="segmentation", n_samples=3)
    fig, axes = plot_uncertainty(x, result)
    assert len(axes) == 3
    plt.close(fig)


def test_plot_uncertainty_reject_classification():
    r = model_cls().predict(torch.rand(2,3), n_samples=3)
    with pytest.raises(ValueError):
        plot_uncertainty(torch.rand(2,3), r)


def test_mc_convergence(tmp_path):
    model = model_cls()
    r = model.predict(torch.rand(3, 3), n_samples=6, return_samples=True)
    fig, ax = plot_mc_convergence(r, save_path=tmp_path / "mc.svg")
    assert len(ax.lines[0].get_xdata()) == 6
    assert (tmp_path / "mc.svg").exists()
    plt.close(fig)


def test_mc_convergence_needs_samples():
    model = model_cls()
    r = model.predict(torch.rand(3, 3), n_samples=3)
    with pytest.raises(ValueError):
        plot_mc_convergence(r)


def test_reliability_rejects_regression():
    x = torch.randn(5,3)
    y = torch.zeros(5,2)
    r = Evaluator(model_cls(), task="regression").evaluate([(x,y)], return_details=True)
    with pytest.raises(ValueError):
        plot_reliability(r)
