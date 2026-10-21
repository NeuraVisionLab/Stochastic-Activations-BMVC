"""Training integration and numerical regression tests (CPU-compatible)."""
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stochact import StochasticModel, Trainer, TrainingHistory


def classifier(binary=False):
    torch.manual_seed(5)
    return StochasticModel(nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1 if binary else 3)),
                           activations="silu", p=0.5)


def loader(n=16, batch_size=4, *, binary=False):
    torch.manual_seed(9)
    x = torch.randn(n, 4)
    y = torch.randint(0, 2 if binary else 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def conv_model(channels=1):
    return StochasticModel(nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(8, channels, 1)), activations="tanh")


def seg_loader(channels=1):
    x = torch.randn(6, 3, 8, 8)
    y = torch.randint(0, max(2, channels), (6, 8, 8))
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_fit_multiclass_with_diagnostics_history_and_optimizer_updates(tmp_path):
    model = classifier()
    starting = model.model[0].weight.detach().clone()
    train = Trainer(model, optimizer="adamw", scheduler="cosine", lr=0.005)
    history = train.fit(loader(), loader(8), epochs=3, uncertainty_every=1, uncertainty_samples=3,
                        checkpoint_dir=tmp_path, verbose=False)
    assert len(history) == 3
    assert history.best_epoch in [1, 2, 3]
    assert 0 <= history.best_value < 5
    assert history["train_loss"][0] > 0
    assert 0 <= history["val_accuracy"][0] <= 1
    assert all(x >= 0 for x in history["train_uncertainty"])
    assert all(x >= 0 for x in history["val_uncertainty"])
    assert train.current_epoch == 3
    assert not torch.equal(starting, model.model[0].weight)
    assert (tmp_path / "last.pt").is_file() and (tmp_path / "best.pt").is_file()
    assert isinstance(history.as_dict(), dict)


def test_checkpoint_load_model_optimizer_scheduler_history_and_rng(tmp_path):
    t = Trainer(classifier(), optimizer="sgd", lr=0.02, scheduler="step", seed=11)
    t.fit(loader(), epochs=2, checkpoint_dir=tmp_path, verbose=False, restore_best_weights=False)
    weights = {key: v.clone() for key, v in t.model.state_dict().items()}
    t2 = Trainer(classifier(), optimizer="sgd", lr=0.02, scheduler="step")
    t2.load_checkpoint(tmp_path / "last.pt", restore_rng=True)
    assert t2.current_epoch == 2 and len(t2.history) == 2
    assert t2.optimizer.param_groups[0]["lr"] == t.optimizer.param_groups[0]["lr"]
    for name, value in weights.items():
        assert torch.equal(value, t2.model.state_dict()[name]), name
    h = t2.fit(loader(), epochs=1, verbose=False, restore_best_weights=False)
    assert len(h) == 3 and h[-1]["epoch"] == 3


def test_gradient_accumulation_tail_matches_combined_batch_for_deterministic_network():
    torch.manual_seed(24)
    base = nn.Linear(4, 3)
    a = nn.Linear(4, 3)
    b = nn.Linear(4, 3)
    a.load_state_dict(base.state_dict())
    b.load_state_dict(base.state_dict())
    inputs = torch.randn(6, 4)
    targets = torch.randint(0, 3, (6,))
    small = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    big = DataLoader(TensorDataset(inputs, targets), batch_size=6)
    tra = Trainer(a, optimizer="sgd", lr=0.1, accumulation_steps=4)
    trb = Trainer(b, optimizer="sgd", lr=0.1)
    tra.fit(small, epochs=1, verbose=False)
    trb.fit(big, epochs=1, verbose=False)
    assert torch.allclose(a.weight, b.weight, atol=1e-6)


def test_binary_classification_and_binary_segmentation():
    binary = Trainer(classifier(binary=True), task="classification")
    binary.fit(loader(binary=True), epochs=1, verbose=False)
    assert 0 <= binary.test(loader(binary=True))["accuracy"] <= 1
    segmenter = Trainer(conv_model(), task="segmentation")
    hist = segmenter.fit(seg_loader(), epochs=1, verbose=False)
    assert "train_foreground_dice" in hist[-1]
    scores = segmenter.test(seg_loader(), n_samples=3)
    assert 0 <= scores["foreground_dice"] <= 1
    assert 0 <= scores["pixel_accuracy"] <= 1


def test_multiclass_segmentation_loss_ignore_index_and_mc_testing():
    model = conv_model(channels=3)
    inp = torch.randn(4, 3, 6, 6)
    target = torch.randint(0, 3, (4, 6, 6))
    target[0, 0, 0] = 255
    batch = [(inp, target)]
    t = Trainer(model, task="segmentation", ignore_index=255, optimizer="adam")
    t.fit(batch, epochs=1, verbose=False)
    assert "foreground_dice" in t.test(batch, n_samples=2)


def test_regression_and_dict_batch_and_probability_losses():
    net = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 2))
    t = Trainer(StochasticModel(net), task="regression")
    x, y = torch.randn(5, 4), torch.randn(5, 2)
    t.fit([{"inputs": x, "targets": y}], epochs=1, verbose=False)
    assert t.test([{"inputs": x, "targets": y}])["mse"] >= 0
    class Probabilities(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        def forward(self, x):
            return self.net(x).softmax(dim=1)
    prob = Trainer(StochasticModel(Probabilities()), from_logits=False)
    prob.fit(loader(8), epochs=1, verbose=False)
    assert prob.test(loader(8), n_samples=3)["loss"] > 0


def test_early_stopping_with_no_learning(tmp_path):
    t = Trainer(classifier(), optimizer="sgd", lr=0.01)
    # Construct constant custom loss with an autograd path but zero gradients.
    t.loss_fn = lambda output, target: output.sum() * 0 + 1.0
    h = t.fit(loader(8), loader(8), epochs=10, early_stopping_patience=2,
              checkpoint_dir=tmp_path, verbose=False)
    assert h.stopped_early is True
    assert len(h) == 3 and h.best_epoch == 1
    assert (tmp_path / "best.pt").is_file()


def test_stochastic_activation_kept_on_and_batchnorm_validation_frozen():
    model = StochasticModel(nn.Sequential(nn.Linear(4, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10, 3)))
    t = Trainer(model)
    t.fit(loader(8, batch_size=4), epochs=1, verbose=False)
    before = model.model[1].running_mean.detach().clone()
    model.set_stochastic(False)
    t.validate(loader(8, batch_size=4))
    assert torch.equal(before, model.model[1].running_mean)
    assert all(x.stochastic_enabled for x in model.stochastic_layers().values())


def test_amp_cpu_bfloat16_and_gradient_clipping():
    t = Trainer(classifier(), mixed_precision=True, grad_clip_norm=1.0)
    h = t.fit(loader(8), epochs=1, verbose=False)
    assert h[-1]["train_loss"] >= 0


def test_custom_optimizer_loss_and_mapping_output():
    class MappingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 3))
        def forward(self, x):
            y = self.layers(x)
            return {"out": y, "aux": y + 1}
    model = StochasticModel(MappingNet())
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = lambda output, target: nn.functional.cross_entropy(output, target)
    t = Trainer(model, optimizer=optim, output_key="out", loss_fn=loss)
    t.fit(loader(8), epochs=1, verbose=False)
    assert t.test(loader(8), n_samples=2)["accuracy"] >= 0


def test_restore_best_weights_matches_best_checkpoint(tmp_path):
    model = classifier()
    t = Trainer(model)
    t.fit(loader(8), epochs=3, checkpoint_dir=tmp_path, verbose=False, restore_best_weights=True)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    for key, value in model.state_dict().items():
        assert torch.equal(value, best["model"][key]), key


def test_training_history_and_validation():
    hist = TrainingHistory([{"epoch": 1., "train_loss": .3}], 1, .3)
    assert TrainingHistory.from_dict(hist.as_dict())["train_loss"] == [.3]
    t = Trainer(classifier())
    assert t.validate(loader(8))["loss"] > 0


@pytest.mark.parametrize('kwargs,err', [
    ({'task': 'detection'}, 'task'),
    ({'optimizer': 'invented'}, 'optimizer'),
    ({'accumulation_steps': 0}, 'accumulation_steps'),
    ({'lr': 0}, 'lr'),
    ({'threshold': -1}, 'threshold'),
    ({'grad_clip_norm': -1}, 'grad_clip_norm'),
    ({'output_key': 'a', 'output_index': 0}, 'output_key'),
])
def test_invalid_trainer_constructor(kwargs, err):
    with pytest.raises((ValueError, TypeError), match=err):
        Trainer(classifier(), **kwargs)


def test_invalid_fit_and_monitoring():
    t = Trainer(classifier())
    with pytest.raises(ValueError, match='epochs'):
        t.fit(loader(), epochs=0)
    with pytest.raises(ValueError, match='uncertainty_every'):
        t.fit(loader(), epochs=1, uncertainty_every=0)
    with pytest.raises(ValueError, match='scheduler'):
        Trainer(classifier(), scheduler="mystery").fit(loader(), epochs=1)
    with pytest.raises(ValueError, match='Uncertainty monitoring'):
        Trainer(nn.Linear(4, 3)).fit(loader(), epochs=1, uncertainty_every=1, verbose=False)


def test_empty_loader_and_input_validation():
    t = Trainer(classifier())
    with pytest.raises(ValueError, match='no batches'):
        t.fit([], epochs=1, verbose=False)
    with pytest.raises(ValueError, match='Each batch'):
        t.fit([torch.randn(2, 4)], epochs=1, verbose=False)


def test_custom_multitensor_input_and_mc_monitor_limitation():
    class TwoInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()
            self.fc = nn.Linear(4, 3)
        def forward(self, a, b):
            return self.fc(self.act(a + b))
    model = StochasticModel(TwoInput())
    t = Trainer(model)
    x, y, labels = torch.randn(4, 4), torch.randn(4, 4), torch.randint(0, 3, (4,))
    batch = [((x, y), labels)]
    t.fit(batch, epochs=1, verbose=False)
    with pytest.raises(TypeError, match='tensor inputs'):
        t.fit(batch, epochs=1, uncertainty_every=1, verbose=False)


def test_ignore_index_binary_and_class_sensitive_foreground_dice():
    from stochact.training.trainer import _TaskStats
    # Foreground 1 -> predicted foreground 2 is *incorrect*, not a true positive.
    stats = _TaskStats('segmentation', from_logits=True, threshold=.5, ignore_index=None)
    logits = torch.tensor([[[[0.]], [[5.]], [[0.]]], [[[0.]], [[0.]], [[5.]]]])
    targets = torch.tensor([[[2]], [[1]]])
    stats.update(logits, targets)
    assert stats.compute()['foreground_dice'] == 0.0
    t = Trainer(conv_model(), task='segmentation', ignore_index=255)
    x = torch.randn(2, 3, 3, 3)
    y = torch.randint(0, 2, (2, 3, 3))
    y[0, 1, 0] = 255
    t.fit([(x, y)], epochs=1, verbose=False)
    assert t.test([(x, y)])['pixel_accuracy'] >= 0


def test_epoch_callback_receives_metrics_and_trainer():
    t = Trainer(classifier())
    received = []
    t.fit(loader(8), epochs=2, verbose=False,
          callbacks=[lambda trainer, row: received.append((trainer is t, row["epoch"]))])
    assert received == [(True, 1.0), (True, 2.0)]
    with pytest.raises(TypeError, match='callbacks'):
        t.fit(loader(8), epochs=1, callbacks=[None], verbose=False)
