"""Public contracts, dtype/device safety, integration and failure cases."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from stochact import StochasticActivation, StochasticModel, predict_mc


def tiny_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(inplace=True), nn.Linear(8, 3))


def test_copy_is_default_and_preserves_original():
    base = tiny_model()
    original = base[1]
    converted = StochasticModel(base)
    assert converted.model is not base
    assert base[1] is original and original.inplace is True
    assert converted.model[1].native.inplace is False
    assert converted.conversion_report.parameters_added == 0


def test_inplace_conversion_changes_original_tree():
    base = tiny_model()
    converted = StochasticModel(base, inplace=True)
    assert base is converted.model
    assert isinstance(base[1], StochasticActivation)


@pytest.mark.parametrize("granularity,shape", [
    ("node", (3, 5, 7, 11)),
    ("channel", (3, 5, 1, 1)),
    ("layer", ()),
])
def test_mask_shapes_for_conv(granularity, shape):
    layer = StochasticActivation(nn.ReLU(), granularity=granularity)
    assert layer.sample_indices(torch.randn(3, 5, 7, 11)).shape == shape


def test_transformer_channel_shape_and_explicit_axis():
    x = torch.randn(2, 10, 16)
    assert StochasticActivation(nn.GELU(), granularity="channel").sample_indices(x).shape == (2, 1, 16)
    assert StochasticActivation(nn.GELU(), granularity="channel", channel_dim=1).sample_indices(x).shape == (2, 10, 1)
    assert StochasticActivation(nn.GELU(), granularity="channel", batch_shared_channels=True).sample_indices(x).shape == (1, 1, 16)


def test_probability_sampling_tracks_specified_distribution():
    layer = StochasticActivation(nn.ReLU(), activations=["relu", "tanh", "silu"], probs=[0.1, 0.2, 0.7])
    torch.manual_seed(123)
    indices = layer.sample_indices(torch.empty(100_000))
    histogram = torch.bincount(indices, minlength=3).float() / indices.numel()
    torch.testing.assert_close(histogram, torch.tensor([0.1, 0.2, 0.7]), atol=0.006, rtol=0)


def test_half_and_bfloat16_do_not_round_sampling_weights():
    layer = StochasticActivation(nn.ReLU(), activations="tanh", p=0.00001)
    assert layer.half().probabilities.dtype == torch.float64
    assert layer.probabilities[1].item() == pytest.approx(0.00001)
    assert layer.bfloat16().probabilities[1].item() == pytest.approx(0.00001)


def test_parameterized_alternative_matches_double_backbone():
    base = tiny_model().double().eval()
    converted = StochasticModel(base, activations=nn.PReLU(), p=1.0)
    layer = converted.model[1]
    assert layer.native.training is False and layer.training is False
    assert layer.alternatives[0].weight.dtype == torch.float64
    output = converted(torch.randn(3, 4, dtype=torch.float64))
    assert output.dtype == torch.float64
    output.sum().backward()
    assert layer.alternatives[0].weight.grad is not None
    assert converted.conversion_report.parameters_added == 1


def test_parameterized_native_keeps_dtype_and_stochastic_alternative_aligns():
    base = nn.Sequential(nn.Linear(4, 4), nn.PReLU(), nn.Linear(4, 2)).double()
    converted = StochasticModel(base, activations=nn.PReLU(), p=0.5)
    layer = converted.model[1]
    assert layer.native.weight.dtype == torch.float64
    assert layer.alternatives[0].weight.dtype == torch.float64
    converted(torch.randn(2, 4, dtype=torch.float64)).sum().backward()
    assert layer.native.weight.grad is not None or layer.alternatives[0].weight.grad is not None


def test_stochastic_mode_context_restores_mixed_state_even_on_error():
    converted = StochasticModel(nn.Sequential(nn.ReLU(), nn.Tanh()), activations="silu")
    layers = list(converted.stochastic_layers().values())
    layers[0].set_stochastic(False)
    before = [layer.stochastic_enabled for layer in layers]
    with pytest.raises(RuntimeError, match="boom"):
        with converted.stochastic_mode(True):
            assert all(layer.stochastic_enabled for layer in layers)
            with converted.stochastic_mode(False):
                assert not any(layer.stochastic_enabled for layer in layers)
            raise RuntimeError("boom")
    assert [layer.stochastic_enabled for layer in layers] == before


def test_stochastic_mode_bad_flag():
    with pytest.raises(TypeError, match="bool"):
        with StochasticModel(tiny_model()).stochastic_mode("yes"):
            pass


def test_native_deterministic_matches_base_forward():
    torch.manual_seed(1)
    base = tiny_model().eval()
    x = torch.randn(5, 4)
    result = StochasticModel(base)
    with result.stochastic_mode(False):
        torch.testing.assert_close(result(x), base(x), rtol=0, atol=0)


def test_probability_buffer_saves_and_loads_after_dtype_conversion(tmp_path):
    a = StochasticModel(tiny_model(), activations="tanh", p=0.00001)
    a.half()
    target = tmp_path / "safe_weights.pt"
    torch.save(a.state_dict(), target)
    b = StochasticModel(tiny_model(), activations="tanh")
    b.load_state_dict(torch.load(target, weights_only=True))
    assert b.model[1].probabilities.dtype == torch.float64
    assert b.model[1].probabilities[1].item() == pytest.approx(0.00001)


def test_nonuniform_training_flags_stay_after_mc():
    model = StochasticModel(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))).train()
    model.model[0].eval()
    before = [(m, m.training) for m in model.modules()]
    predict_mc(model, torch.randn(3, 4), n_samples=3)
    assert all(m.training == old for m, old in before)


def test_mc_classification_has_valid_normalized_probabilities():
    model = StochasticModel(tiny_model(), activations="tanh")
    result = model.predict(torch.randn(6, 4), n_samples=7)
    assert torch.allclose(result.mean.sum(dim=1), torch.ones(6), atol=1e-6)
    assert result.variance.min() >= 0
    assert torch.isfinite(result.expected_entropy).all()
    assert torch.isfinite(result.mutual_information).all()


def test_welford_variance_matches_manual_reference():
    model = StochasticModel(tiny_model(), activations="tanh")
    x = torch.randn(8, 4)
    torch.manual_seed(12)
    samples = torch.stack([model.eval()(x).softmax(dim=1) for _ in range(11)])
    torch.manual_seed(12)
    result = model.predict(x, n_samples=11)
    torch.testing.assert_close(result.mean, samples.mean(0), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.variance, samples.var(dim=0, unbiased=False), atol=1e-6, rtol=0)


def test_unconverted_functional_activation_reports_zero_conversions():
    class OnlyFunctional(nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x)
    model = StochasticModel(OnlyFunctional())
    assert model.conversion_report.replaced == 0
    with pytest.raises(ValueError, match="No StochasticActivation"):
        model.predict(torch.randn(2, 3))


def test_cuda_conversion_and_training_smoke_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in test environment")
    torch.manual_seed(7)
    model = StochasticModel(tiny_model().cuda(), activations=nn.PReLU(), granularity="node")
    x = torch.randn(6, 4, device="cuda")
    result = model.predict(x, n_samples=4)
    assert result.mean.is_cuda
    logits = model(x)
    logits.square().mean().backward()
    assert model.model[1].alternatives[0].weight.is_cuda
    assert model.model[1].alternatives[0].weight.grad is not None


def test_vit_registered_gelu_conversion():
    pytest.importorskip("torchvision")
    from torchvision.models.vision_transformer import VisionTransformer
    vit = VisionTransformer(image_size=32, patch_size=16, num_layers=1,
                            num_heads=2, hidden_dim=64, mlp_dim=128, num_classes=3)
    converted = StochasticModel(vit, activations="silu", granularity="channel")
    assert converted.conversion_report.replaced >= 1
    assert any("GELU" in x.original for x in converted.conversion_report.replacements)
    output = converted(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 3)


def test_report_counts_shared_activation_references():
    shared = nn.ReLU()
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = shared
            self.second = shared
        def forward(self, x):
            return self.second(self.first(x))
    converted = StochasticModel(Shared())
    assert converted.model.first is converted.model.second
    assert converted.conversion_report.replaced == 2


def test_conversion_report_is_human_readable():
    model = StochasticModel(tiny_model(), activations=["relu", "tanh", "silu"], probs=[0.1,0.3,0.6])
    rendered = str(model.report(print_report=False))
    assert "StochAct Conversion Report" in rendered
    assert "node" in rendered
    assert "Trainable parameters added" in rendered


def test_wrapping_eval_backbone_preserves_root_and_child_modes():
    base = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.ReLU()).eval()
    converted = StochasticModel(base)
    assert converted.training is False
    assert converted.model.training is False
    assert converted.model[1].training is False
    assert converted.model[2].training is False
    assert converted.model[2].native.training is False
    assert base.training is False  # untouched copy source
