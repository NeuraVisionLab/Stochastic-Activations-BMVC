import copy

import pytest
import torch
from torch import nn

from stochact import StochasticActivation, StochasticModel


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 4),
            nn.GELU(),
            nn.Linear(4, 2),
        )
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        return self.output_activation(self.net(x))


def test_copy_preserves_original_and_parameters():
    base = TinyNet()
    weights_before = copy.deepcopy(base.state_dict())
    wrapped = StochasticModel(base)
    assert isinstance(base.net[1], nn.ReLU)
    assert isinstance(base.net[3], nn.GELU)
    assert isinstance(wrapped.model.net[1], StochasticActivation)
    assert isinstance(wrapped.model.net[3], StochasticActivation)
    assert isinstance(wrapped.model.output_activation, nn.Sigmoid)
    assert wrapped.conversion_report.replaced == 2
    assert wrapped.conversion_report.detected == 2
    assert wrapped.conversion_report.parameters_added == 0
    for key, weight in base.state_dict().items():
        assert torch.equal(weight, weights_before[key])
    wrapped.set_stochastic(False).eval()
    base.eval()
    with torch.no_grad():
        x = torch.randn(3, 6)
        assert torch.allclose(wrapped(x), base(x))


def test_inplace_option_mutates_original():
    base = TinyNet()
    converted = StochasticModel(base, inplace=True)
    assert converted.model is base
    assert isinstance(base.net[1], StochasticActivation)


def test_matching_include_exclude_patterns():
    base = TinyNet()
    model = StochasticModel(base, include="net.*", exclude=["net.3"])
    assert model.conversion_report.replaced == 1
    assert model.conversion_report.skipped == ["net.3"]
    model2 = StochasticModel(base, include="re:net\\.3$")
    assert model2.conversion_report.replaced == 1
    assert model2.conversion_report.replacements[0].path == "net.3"


def test_all_layer_variation_without_mutating_weights():
    torch.manual_seed(9)
    base = TinyNet()
    model = StochasticModel(base, activations="tanh", p=1.0)
    weights_before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(8, 6)
    changed = model(x)
    model.set_stochastic(False)
    baseline = model(x)
    assert not torch.allclose(changed, baseline)
    for before, now in zip(weights_before, model.parameters()):
        assert torch.equal(before, now.detach())


def test_wrapper_training_gradients_and_optimizer():
    torch.manual_seed(4)
    model = StochasticModel(TinyNet())
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(8, 6)
    y = torch.rand(8, 2)
    out = model(x)
    loss = (out - y).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optim.step()


def test_eval_does_not_disable_stochastic_sampling():
    torch.manual_seed(77)
    model = StochasticModel(nn.Sequential(nn.Linear(64, 64), nn.ReLU()), activations="tanh")
    model.eval()
    assert all(layer.stochastic_enabled for layer in model.stochastic_layers().values())
    inputs = torch.randn(16, 64)
    # With hundreds of node-level choices output differs with overwhelming probability.
    first = model(inputs)
    second = model(inputs)
    assert not torch.equal(first, second)
    model.set_stochastic(False)
    assert torch.equal(model(inputs), model(inputs))


def test_report_and_no_functional_replacement():
    class Functional(nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x)
    model = StochasticModel(Functional())
    report = model.report(print_report=False)
    assert report.replaced == 0
    assert any("Functional activations" in line for line in report.warnings)
    assert "StochAct Conversion Report" in str(report)
    with pytest.raises(NotImplementedError, match="FX"):
        StochasticModel(Functional(), backend="fx")


def test_shared_activation_references():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            r = nn.ReLU()
            self.first = r
            self.second = r
        def forward(self, x):
            return self.second(self.first(x))
    model = StochasticModel(Shared())
    assert model.model.first is model.model.second
    assert model.conversion_report.replaced == 2


def test_root_activation_conversion():
    model = StochasticModel(nn.ReLU(), activations="tanh", p=1.0)
    assert model.conversion_report.replaced == 1
    assert isinstance(model.model, StochasticActivation)
    x = torch.tensor([-1., 2.])
    assert torch.allclose(model(x), x.tanh())


def test_save_load_state_dict_with_probabilities(tmp_path):
    model = StochasticModel(TinyNet(), activations=["relu", "silu", "tanh"], probs=[0.2,0.3,0.5])
    torch.save(model.state_dict(), tmp_path / "state.pt")
    new_model = StochasticModel(TinyNet(), activations=["relu", "silu", "tanh"], probs=[0.2,0.3,0.5])
    new_model.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    assert all(torch.equal(model.state_dict()[k], new_model.state_dict()[k]) for k in model.state_dict())


def test_conv_input_and_channel_granularity():
    base = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    model = StochasticModel(base, granularity="channel", activations="tanh")
    assert model(torch.randn(2,3,16,16)).shape == (2,1,16,16)


def test_invalid_inputs():
    with pytest.raises(TypeError):
        StochasticModel("not a module")
    with pytest.raises(ValueError, match="already"):
        StochasticModel(StochasticModel(TinyNet()))
    with pytest.raises(ValueError, match="backend"):
        StochasticModel(TinyNet(), backend="something_else")
