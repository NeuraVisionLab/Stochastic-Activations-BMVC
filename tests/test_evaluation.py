"""Evaluation mathematics, integration, and API contracts."""
import math
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stochact import StochasticModel, Evaluator, calibration_metrics, binary_ccq, binary_curves
from stochact.evaluation import EvaluationResult, OODResult


def classify_model(out=3):
    torch.manual_seed(10)
    return StochasticModel(nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, out)),
                           activations="tanh", granularity="node")


def segment_model(out=1):
    torch.manual_seed(11)
    return StochasticModel(nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(8, out, 1)), activations="silu")


def loader_cls(n=13):
    torch.manual_seed(13)
    x = torch.randn(n, 4)
    y = torch.arange(n) % 3
    return DataLoader(TensorDataset(x, y), batch_size=5)


def loader_seg(out=1, n=7):
    torch.manual_seed(14)
    x = torch.randn(n, 1, 8, 8)
    y = torch.randint(0, 2 if out == 1 else out, (n, 8, 8))
    return DataLoader(TensorDataset(x, y), batch_size=3)


def test_calibration_perfect_binary():
    p = torch.tensor([0., 1., 0., 1.])
    y = torch.tensor([0, 1, 0, 1])
    r = calibration_metrics(p, y, mode="binary")
    assert r["ece"] == pytest.approx(0)
    assert r["brier"] == pytest.approx(0)
    assert r["nll"] < 1e-5


def test_calibration_multiclass_manual():
    p = torch.tensor([[.8, .2], [.25, .75]])
    y = torch.tensor([0, 1])
    r = calibration_metrics(p, y, mode="multiclass", n_bins=10)
    assert r["brier"] == pytest.approx((2*.2**2+2*.25**2)/2)
    assert r["nll"] == pytest.approx((-math.log(.8)-math.log(.75))/2)
    assert r["ece"] == pytest.approx((.2+.25)/2)


def test_calibration_ignore_index():
    a = calibration_metrics(torch.tensor([.9, .3]), torch.tensor([1, -1]),
                            mode="binary", ignore_index=-1)
    b = calibration_metrics(torch.tensor([.9]), torch.tensor([1]), mode="binary")
    assert a == pytest.approx(b)


def test_calibration_segmentation_ignores_pixels():
    probs = torch.tensor([[[[0.9, 0.2]]]])
    y = torch.tensor([[[1, 255]]])
    result = calibration_metrics(probs, y, mode="binary", task="segmentation", ignore_index=255)
    assert result["brier"] == pytest.approx(.01)


@pytest.mark.parametrize("bad", [0, -2, True])
def test_calibration_bad_bins(bad):
    with pytest.raises(ValueError):
        calibration_metrics(torch.tensor([.2]), torch.tensor([0]), mode="binary", n_bins=bad)


def test_calibration_bad_probs():
    with pytest.raises(ValueError):
        calibration_metrics(torch.tensor([[.6, .6]]), torch.tensor([0]), mode="multiclass")


def test_calibration_bad_targets():
    with pytest.raises(ValueError):
        calibration_metrics(torch.tensor([.5]), torch.tensor([5]), mode="binary")


def test_binary_ccq_exact():
    pred = torch.tensor([[[1, 1], [0, 0]]])
    gt = torch.tensor([[[1, 0], [1, 0]]])
    x = binary_ccq(pred, gt)
    assert x == pytest.approx({"correctness": .5, "completeness": .5, "quality": 1/3})


def test_binary_ccq_tolerance():
    a = torch.zeros(1, 7, 7)
    b = a.clone()
    a[0, 3, 3] = 1
    b[0, 3, 4] = 1
    assert binary_ccq(a, b, tolerance=0)["quality"] == 0
    assert binary_ccq(a, b, tolerance=1)["quality"] == pytest.approx(1)


def test_binary_ccq_empty():
    zeros = torch.zeros(1, 8, 8)
    assert binary_ccq(zeros, zeros) == pytest.approx({"correctness":1, "completeness":1, "quality":1})


def test_binary_ccq_bad():
    with pytest.raises(ValueError):
        binary_ccq(torch.zeros(1,8,8), torch.zeros(1,8,8), tolerance=-1)


def test_ood_curves_perfect():
    s = torch.tensor([.1, .2, .8, .9])
    y = torch.tensor([0,0,1,1])
    x = binary_curves(s, y)
    for m in ("auroc", "aupr"):
        assert x[m] == pytest.approx(1)
    assert x["fpr95"] == pytest.approx(0)


def test_ood_curves_reversed():
    x = binary_curves(torch.tensor([.8, .9, .1, .2]), torch.tensor([0,0,1,1]))
    assert x["auroc"] == pytest.approx(0)
    assert x["fpr95"] == pytest.approx(1)


def test_ood_curves_all_ties():
    x = binary_curves(torch.ones(6), torch.tensor([0,0,0,1,1,1]))
    assert x["auroc"] == pytest.approx(.5)
    assert x["aupr"] == pytest.approx(.5)
    assert x["fpr95"] == pytest.approx(1)


def test_ood_curves_invalid():
    with pytest.raises(ValueError):
        binary_curves([1, 2], [1, 1])


def test_evaluator_classification():
    ev = Evaluator(classify_model(), task="classification", n_samples=3)
    r = ev.evaluate(loader_cls(), return_details=True)
    assert isinstance(r, EvaluationResult)
    assert r.confusion_matrix.shape == (3,3)
    assert r.uncertainty.shape == (13,)
    assert r.per_example_error.shape == (13,)
    assert r.confusion_matrix.sum().item() == 13
    assert set(("accuracy", "f1_macro", "f1_weighted", "ece", "brier", "nll")) <= r.metrics.keys()
    assert 0 <= r["accuracy"] <= 1
    assert r.reliability["count"].sum() == 13


def test_evaluator_dict_default():
    r = Evaluator(classify_model(), n_samples=2).evaluate(loader_cls())
    assert isinstance(r, dict)
    assert "accuracy" in r


def test_evaluator_binary_classification():
    model = classify_model(out=1)
    x = torch.randn(6,4)
    y = torch.randint(0,2,(6,))
    r = Evaluator(model, mode="binary").evaluate(DataLoader(TensorDataset(x,y),batch_size=3))
    assert "accuracy" in r and "ece" in r


def test_evaluator_binary_segmentation():
    r = Evaluator(segment_model(), task="segmentation", n_samples=3).evaluate(
        loader_seg(), return_details=True)
    for metric in ("pixel_accuracy", "foreground_dice", "foreground_iou", "dice_macro_fg", "iou_macro_fg", "uncertainty_error_corr"):
        assert metric in r.metrics
    assert r.uncertainty.shape == (7,)
    assert r.confusion_matrix.shape == (2, 2)
    assert r.reliability["count"].sum() == 7*8*8


def test_evaluator_multiclass_segmentation():
    r = Evaluator(segment_model(out=3), task="segmentation").evaluate(loader_seg(out=3))
    assert 0 <= r["foreground_dice"] <= 1
    assert 0 <= r["iou_macro_fg"] <= 1


def test_evaluator_ignore_index():
    model = segment_model()
    x = torch.randn(3, 1, 8, 8)
    y = torch.ones(3, 8, 8, dtype=torch.long)
    y[0, :4, :] = 255
    ev = Evaluator(model, task="segmentation", ignore_index=255)
    r = ev.evaluate([(x,y)], return_details=True)
    assert r.reliability["count"].sum() == y.numel()-32


def test_evaluator_regression():
    model = classify_model(out=2)
    data = [(torch.randn(7,4), torch.zeros(7,2))]
    r = Evaluator(model, task="regression").evaluate(data, return_details=True)
    assert set(r.metrics) == {"mae", "mse", "rmse"}
    assert r["rmse"] == pytest.approx(math.sqrt(r["mse"]))
    assert r.uncertainty.shape == (7,)


def test_evaluator_regression_wrong_shape():
    with pytest.raises(ValueError):
        Evaluator(classify_model(2), task="regression").evaluate([(torch.randn(3,4),torch.ones(3))])


@pytest.mark.parametrize("score", ["variance", "predictive_entropy", "mutual_information", "max_probability"])
def test_evaluator_ood_scores(score):
    model = classify_model()
    ev = Evaluator(model, n_samples=3)
    a = DataLoader(TensorDataset(torch.randn(9,4)),batch_size=4)
    b = DataLoader(TensorDataset(torch.randn(11,4)+2),batch_size=4)
    r = ev.evaluate_ood(a,b,score=score,return_details=True)
    assert isinstance(r, OODResult)
    assert r.id_scores.shape == (9,)
    assert r.ood_scores.shape == (11,)
    for key in ("auroc", "aupr", "fpr95"):
        assert 0 <= r[key] <= 1


def test_evaluator_ood_labeled_batches():
    e = Evaluator(classify_model())
    a = loader_cls(n=6)
    b = loader_cls(n=7)
    assert "auroc" in e.evaluate_ood(a, b)


def test_evaluator_ood_regression_restriction():
    e = Evaluator(classify_model(2), task="regression")
    with pytest.raises(ValueError):
        e.evaluate_ood([], [], score="mutual_information")


def test_evaluator_empty():
    with pytest.raises(ValueError):
        Evaluator(classify_model()).evaluate([])


def test_evaluator_requires_tensor_inputs():
    with pytest.raises(TypeError):
        Evaluator(classify_model()).evaluate([({"x": torch.rand(1,4)}, torch.tensor([0]))])


def test_evaluator_invalid_n_samples():
    with pytest.raises(ValueError):
        Evaluator(classify_model(), n_samples=0)


def test_evaluator_maintains_training_flag():
    model = classify_model()
    model.train()
    Evaluator(model).evaluate(loader_cls(6))
    assert model.training


def test_segmentation_ccq_in_evaluator():
    x = torch.rand(2, 1, 9, 9)
    y = torch.zeros(2, 9, 9, dtype=torch.long)
    y[:, 4, 4] = 1
    ev = Evaluator(segment_model(), task="segmentation", ccq_tolerance=1)
    results = ev.evaluate([(x,y)])
    assert set(("correctness", "completeness", "quality")) <= results.keys()
    assert 0 <= results["quality"] <= 1


def test_segmentation_ccq_bad_tolerance():
    with pytest.raises(ValueError):
        Evaluator(segment_model(), task="segmentation", ccq_tolerance=-1)


def test_evaluator_segmentation_ood_entropy():
    e = Evaluator(segment_model(), task="segmentation", n_samples=3)
    r = e.evaluate_ood([torch.randn(3,1,8,8)], [torch.randn(4,1,8,8)+2],
                       score="predictive_entropy", return_details=True)
    assert r.id_scores.shape == (3,)
    assert r.ood_scores.shape == (4,)


def test_evaluator_regression_ood_variance():
    e = Evaluator(classify_model(out=2), task="regression", n_samples=3)
    r = e.evaluate_ood([torch.randn(3,4)], [torch.randn(4,4)+2], score="variance")
    assert 0 <= r["auroc"] <= 1


def test_ood_score_validation():
    with pytest.raises(ValueError):
        Evaluator(classify_model()).evaluate_ood([], [], score="bogus")
