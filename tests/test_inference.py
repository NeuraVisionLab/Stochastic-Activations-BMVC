import pytest
import torch
from torch import nn

from stochact import PredictionResult, StochasticModel, predict_mc


def classifier():
    return StochasticModel(nn.Sequential(nn.Linear(6, 24), nn.ReLU(), nn.Linear(24, 3)),
                           activations="tanh", p=0.5)


def binary_seg():
    return StochasticModel(nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 1, 1)),
                           activations="tanh", p=0.5)


def multiclass_seg():
    return StochasticModel(nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 4, 1)),
                           activations="tanh", p=0.5)


def test_streaming_mc_matches_manual_reference_and_mean_probabilities():
    torch.manual_seed(1)
    model = classifier().eval()
    x = torch.randn(4, 6)
    torch.manual_seed(123)
    manual = torch.stack([model(x).softmax(dim=1) for _ in range(7)])
    torch.manual_seed(123)
    result = model.predict(x, n_samples=7, return_samples=True)
    assert isinstance(result, PredictionResult)
    assert result.task == "classification" and result.mode == "multiclass"
    assert torch.allclose(result.samples, manual)
    assert torch.allclose(result.mean, manual.mean(dim=0), atol=1e-7)
    assert torch.allclose(result.variance, manual.var(dim=0, unbiased=False), atol=1e-7)
    assert result.uncertainty.shape == (4,)
    assert result.predictions.shape == (4,)
    assert torch.equal(result.classes, manual.mean(0).argmax(1))
    assert torch.allclose(result.probabilities.sum(1), torch.ones(4))
    assert result.predictive_entropy.shape == (4,)
    assert result.expected_entropy.shape == (4,)
    assert result.mutual_information.shape == (4,)
    assert result.mutual_information.min() >= 0


def test_streaming_without_storing_sample_stack():
    model = classifier()
    result = model.predict(torch.randn(3, 6), n_samples=3)
    assert result.samples is None
    assert result.variance.shape == (3, 3)
    assert result.uncertainty_map is None


def test_binary_segmentation_shapes_and_maps():
    model = binary_seg()
    result = model.predict(torch.randn(2, 3, 6, 7), task="segmentation", n_samples=4)
    assert result.mean.shape == (2, 1, 6, 7)
    assert result.mask.shape == (2, 6, 7)
    assert result.uncertainty_map.shape == (2, 6, 7)
    assert torch.allclose(result.uncertainty, result.uncertainty_map.flatten(1).mean(1))
    assert result.predictive_entropy.shape == (2, 6, 7)
    assert result.expected_entropy.shape == (2, 6, 7)
    assert result.mutual_information.shape == (2, 6, 7)
    assert torch.equal(result.predictions, (result.mean[:, 0] >= .5).long())


def test_multiclass_segmentation_shapes():
    model = multiclass_seg()
    result = model.predict(torch.randn(2, 3, 7, 9), task="segmentation", n_samples=4)
    assert result.mean.shape == (2, 4, 7, 9)
    assert result.predictions.shape == (2, 7, 9)
    assert result.uncertainty_map.shape == (2, 7, 9)
    assert torch.allclose(result.uncertainty_map, result.variance.mean(dim=1))
    assert result.predictive_entropy.shape == (2, 7, 9)


def test_classification_binary_logits():
    model = StochasticModel(nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 1)), activations="tanh")
    result = model.predict(torch.randn(5, 6), n_samples=3, mode="binary", threshold=.7)
    assert result.mean.shape == (5, 1)
    assert result.classes.shape == (5,)
    assert torch.equal(result.classes, (result.mean[:, 0] >= .7).long())
    assert result.predictive_entropy.shape == (5,)


def test_regression_raw_outputs_and_no_entropy():
    model = classifier()
    x = torch.randn(3, 6)
    torch.manual_seed(8)
    manual = torch.stack([model(x) for _ in range(4)])
    torch.manual_seed(8)
    result = model.predict(x, task="regression", n_samples=4)
    assert torch.allclose(result.mean, manual.mean(0), atol=1e-6)
    assert torch.allclose(result.variance, manual.var(0, unbiased=False), atol=1e-6)
    assert result.predictive_entropy is None
    assert result.expected_entropy is None
    assert result.mutual_information is None
    assert torch.equal(result.predictions, result.mean)
    with pytest.raises(AttributeError):
        _ = result.probabilities


def test_model_modes_and_stochastic_states_restored_after_predict():
    model = classifier().train()
    model.model[0].eval()  # deliberately nonuniform submodule flags
    model.stochastic_layers()["1"].set_stochastic(False)
    before = [(id(module), module.training) for module in model.modules()]
    before_sampling = [s.stochastic_enabled for s in model.stochastic_layers().values()]
    x = torch.randn(3, 6, requires_grad=True)
    result = model.predict(x, n_samples=3)
    assert not result.mean.requires_grad
    assert x.grad is None
    assert before == [(id(module), module.training) for module in model.modules()]
    assert before_sampling == [s.stochastic_enabled for s in model.stochastic_layers().values()]
    # Prediction intentionally switches activations on while evaluating.
    assert result.variance.sum() > 0


def test_batchnorm_freezes_stats_and_dropout_deactivated():
    torch.manual_seed(5)
    base = nn.Sequential(nn.Linear(6, 16), nn.BatchNorm1d(16), nn.ReLU(),
                         nn.Dropout(.8), nn.Linear(16, 3))
    model = StochasticModel(base).train()
    before_mean = model.model[1].running_mean.clone()
    x = torch.randn(6, 6)
    model.predict(x, n_samples=4)
    assert torch.equal(model.model[1].running_mean, before_mean)
    assert model.training and model.model[1].training and model.model[3].training


def test_seed_reproducibility():
    model = classifier()
    x = torch.randn(2, 6)
    torch.manual_seed(45)
    first = model.predict(x, n_samples=4)
    torch.manual_seed(45)
    second = model.predict(x, n_samples=4)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.variance, second.variance)


def test_deterministic_baseline_and_one_pass():
    model = classifier()
    x = torch.randn(3, 6)
    result = model.predict(x, n_samples=4, stochastic=False)
    assert torch.count_nonzero(result.variance) == 0
    assert torch.all(result.mutual_information < 1e-6)
    single = model.predict(x, n_samples=1)
    assert torch.count_nonzero(single.variance) == 0
    assert torch.all(single.mutual_information < 1e-6)


def test_probability_input_option():
    class Probability(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
        def forward(self, x):
            return self.features(x).softmax(dim=1)
    model = StochasticModel(Probability())
    result = model.predict(torch.randn(2, 6), from_logits=False)
    assert torch.allclose(result.mean.sum(1), torch.ones(2), atol=1e-6)


def test_return_tensor_from_mapping_or_tuple():
    class MapOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
        def forward(self, x):
            y = self.layers(x)
            return {'out': y, 'aux': y * 2}
    class TupleOutput(MapOutput):
        def forward(self, x):
            y = self.layers(x)
            return (y, y * 2)
    x = torch.randn(2, 6)
    assert StochasticModel(MapOutput()).predict(x, output_key='out').mean.shape == (2, 3)
    assert StochasticModel(TupleOutput()).predict(x, output_index=0).mean.shape == (2, 3)
    with pytest.raises(TypeError, match='output_key'):
        StochasticModel(MapOutput()).predict(x)
    with pytest.raises(TypeError, match='output_index'):
        StochasticModel(TupleOutput()).predict(x)


def test_restoration_on_forward_exception():
    class Fails(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = nn.ReLU()
        def forward(self, x):
            self.activation(x)
            raise RuntimeError('injected failure')
    model = StochasticModel(Fails()).train()
    layer = next(iter(model.stochastic_layers().values()))
    layer.set_stochastic(False)
    with pytest.raises(RuntimeError, match='injected failure'):
        model.predict(torch.randn(3, 6))
    assert model.training and layer.stochastic_enabled is False


@pytest.mark.parametrize('kwargs, exception, pattern', [
    ({'n_samples': 0}, ValueError, 'positive'),
    ({'n_samples': True}, ValueError, 'positive'),
    ({'task': 'detection'}, ValueError, 'task'),
    ({'mode': 'weird'}, ValueError, 'mode'),
    ({'mode': 'binary'}, ValueError, 'incompatible'),
    ({'class_dim': 0}, ValueError, 'class_dim'),
    ({'threshold': -0.1}, ValueError, 'threshold'),
    ({'output_key': 'out', 'output_index': 0}, ValueError, 'either'),
    ({'return_samples': 1}, TypeError, 'return_samples'),
])
def test_invalid_inference_options(kwargs, exception, pattern):
    model = classifier()
    with pytest.raises(exception, match=pattern):
        model.predict(torch.randn(2, 6), **kwargs)


def test_no_stochastic_layers_error_and_standalone_api():
    net = nn.Sequential(nn.Linear(5, 2))
    with pytest.raises(ValueError, match='No StochasticActivation'):
        predict_mc(net, torch.randn(2, 5))
    model = classifier()
    result = predict_mc(model, torch.randn(2, 6), n_samples=2)
    assert result.mean.shape == (2, 3)


def test_segmentation_without_channel_axis_and_height_one():
    class Unchannelled(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Conv2d(3, 5, 1), nn.ReLU(), nn.Conv2d(5, 1, 1))
        def forward(self, x):
            return self.layers(x)[:, 0]  # [B,H,W], H=1 in test
    result = StochasticModel(Unchannelled()).predict(
        torch.randn(2, 3, 1, 6), task="segmentation", mode="binary", n_samples=2)
    assert result.mean.shape == (2, 1, 6)
    assert result.mask.shape == (2, 1, 6)
    assert result.uncertainty_map.shape == (2, 1, 6)
    assert result.predictive_entropy.shape == (2, 1, 6)


def test_channels_last_segmentation_and_other_forward_kwargs():
    class NHWC(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 5, 1), nn.ReLU(), nn.Conv2d(5, 3, 1))
        def forward(self, x, factor=1.0):
            return (self.features(x) * factor).permute(0, 2, 3, 1)
    model = StochasticModel(NHWC())
    result = model.predict(torch.randn(2, 3, 4, 5), task="segmentation", class_dim=-1, factor=1.2)
    assert result.mean.shape == (2, 4, 5, 3)
    assert result.predictions.shape == (2, 4, 5)
    assert result.uncertainty_map.shape == (2, 4, 5)


def test_no_altered_parameters_or_batch_stats_after_mc():
    model = binary_seg().train()
    state = {k: v.clone() for k,v in model.state_dict().items()}
    model.predict(torch.randn(2, 3, 4, 4), task="segmentation", n_samples=4)
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in state.items())


def test_torchvision_small_vision_transformer_converted():
    tv = pytest.importorskip("torchvision")
    from torchvision.models.vision_transformer import VisionTransformer
    torch.manual_seed(2026)
    backbone = VisionTransformer(image_size=32, patch_size=8, num_layers=1,
                                 num_heads=4, hidden_dim=64, mlp_dim=128,
                                 num_classes=5, dropout=0, attention_dropout=0)
    nn.init.normal_(backbone.heads.head.weight, std=0.02)
    wrapped = StochasticModel(backbone, granularity="channel", activations="silu")
    assert wrapped.conversion_report.replaced == 1
    assert wrapped.conversion_report.parameters_added == 0
    assert type(backbone.encoder.layers.encoder_layer_0.mlp[1]).__name__ == 'GELU'
    result = wrapped.predict(torch.randn(2,3,32,32), n_samples=4)
    assert result.mean.shape == (2,5)
    assert result.variance.sum() > 0
