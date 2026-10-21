import pytest
import torch
from torch import nn

from stochact import StochasticActivation, available_activations, register_activation


def test_native_vs_alternative_and_eval_independence():
    layer = StochasticActivation(nn.ReLU(), activations="tanh", p=1.0)
    x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    layer.eval()
    assert torch.allclose(layer(x), torch.tanh(x))
    layer.set_stochastic(False)
    assert torch.equal(layer(x), torch.relu(x))
    layer.train()
    assert torch.equal(layer(x), torch.relu(x))
    layer.set_stochastic(True)
    assert torch.allclose(layer(x), torch.tanh(x))


def test_zero_probability_deterministic_native():
    layer = StochasticActivation(nn.LeakyReLU(negative_slope=0.3), activations="silu", p=0)
    x = torch.tensor([-2.0, 1.0], requires_grad=True)
    assert torch.allclose(layer(x), nn.functional.leaky_relu(x, negative_slope=0.3))


def test_multiple_candidates_with_custom_probabilities():
    layer = StochasticActivation(nn.ReLU(), activations=["relu", "tanh", "silu"], probs=[0, 1, 0])
    x = torch.randn(4, 5)
    assert torch.allclose(layer(x), x.tanh())
    assert layer.candidate_names == ("relu", "tanh", "silu")
    assert layer.probabilities.tolist() == [0.0, 1.0, 0.0]


def test_explicit_two_activation_list_omits_native():
    layer = StochasticActivation(nn.Tanh(), activations=["relu", "silu"], p=1.0)
    x = torch.tensor([-1.0, 0.3])
    assert torch.allclose(layer(x), nn.functional.silu(x))
    layer.set_stochastic(False)
    assert torch.allclose(layer(x), x.tanh())


def test_node_channel_layer_shapes_and_transformer_default():
    x = torch.rand(2, 3, 7, 5)
    assert StochasticActivation(nn.ReLU(), granularity="node").sample_indices(x).shape == x.shape
    assert StochasticActivation(nn.ReLU(), granularity="channel").sample_indices(x).shape == (2, 3, 1, 1)
    assert StochasticActivation(nn.ReLU(), granularity="channel", batch_shared_channels=True).sample_indices(x).shape == (1, 3, 1, 1)
    assert StochasticActivation(nn.ReLU(), granularity="layer").sample_indices(x).shape == ()
    tokens = torch.rand(4, 32, 128)
    assert StochasticActivation(nn.GELU(), granularity="channel").sample_indices(tokens).shape == (4, 1, 128)
    assert StochasticActivation(nn.GELU(), granularity="channel", channel_dim=1).sample_indices(tokens).shape == (4, 32, 1)


def test_granularity_sampled_indices_match_distribution():
    torch.manual_seed(19)
    x = torch.randn(64, 64)
    layer = StochasticActivation(nn.ReLU(), activations="tanh", p=0.25)
    idx = layer.sample_indices(x)
    fraction = idx.float().mean().item()
    assert 0.21 < fraction < 0.29


def test_stochastic_gradients_flow_through_selected_activations():
    torch.manual_seed(1)
    layer = StochasticActivation(nn.ReLU(inplace=True), activations="silu", p=0.5)
    x = torch.randn(5, 7, requires_grad=True)
    y = layer(x)
    y.square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.native.inplace is False
    assert torch.any(x.grad != 0)


def test_input_not_mutated_by_inplace_native():
    layer = StochasticActivation(nn.ReLU(inplace=True), activations="tanh")
    x = torch.tensor([[-1.0, 2.0]])
    copy = x.clone()
    layer(x)
    assert torch.equal(x, copy)


def test_single_list_treated_as_native_plus_one():
    layer = StochasticActivation(nn.ReLU(), activations=["tanh"], p=1)
    x = torch.tensor([-2.])
    assert torch.allclose(layer(x), x.tanh())


def test_user_supplied_activation_module():
    layer = StochasticActivation(nn.ReLU(), activations=nn.Softsign(), p=1)
    x = torch.randn(4, 4)
    assert torch.allclose(layer(x), nn.functional.softsign(x))


def test_custom_activation_registry():
    name = "unit_test_square"
    class Square(nn.Module):
        def forward(self, x):
            return x.square()
    register_activation(name, Square, overwrite=True)
    assert name in available_activations()
    layer = StochasticActivation(nn.ReLU(), activations=name, p=1)
    x = torch.tensor([-2., 3.])
    assert torch.equal(layer(x), x.square())


@pytest.mark.parametrize("kwargs,error", [
    ({"granularity": "batch"}, "granularity"),
    ({"activations": []}, "empty"),
    ({"activations": ["relu", "silu", "tanh"], "p": 0.5}, "only to two"),
    ({"p": 1.2}, "between"),
    ({"p": -0.2}, "between"),
    ({"p": 0.5, "probs": [0.5, 0.5]}, "either"),
    ({"probs": [0.5, 0.8]}, "sum to 1"),
    ({"probs": [-0.1, 1.1]}, "nonnegative"),
    ({"probs": [0, 0]}, "positive sum"),
    ({"activations": ["original", "original"]}, "only appear once"),
    ({"activations": "bad_name"}, "Unknown activation"),
])
def test_invalid_config(kwargs, error):
    with pytest.raises((ValueError, TypeError), match=error):
        StochasticActivation(nn.ReLU(), **kwargs)


def test_invalid_channel_dimension_fails_clearly():
    layer = StochasticActivation(nn.ReLU(), granularity="channel", channel_dim=0)
    with pytest.raises(ValueError, match="non-batch"):
        layer(torch.randn(2, 3, 4))


def test_malformed_custom_activation_output_fails_clearly():
    class Shrink(nn.Module):
        def forward(self, x):
            return x[..., :-1]
    layer = StochasticActivation(nn.ReLU(), activations=Shrink(), p=0.5)
    with pytest.raises(RuntimeError, match="changed shape"):
        layer(torch.randn(3, 4))


def test_sampling_reproducibility_with_torch_manual_seed():
    x = torch.randn(6, 12)
    layer = StochasticActivation(nn.ReLU(), activations="silu", p=0.5)
    torch.manual_seed(2026)
    a = layer(x)
    torch.manual_seed(2026)
    b = layer(x)
    assert torch.equal(a, b)
