"""StochAct: stochastic nonlinearities for PyTorch."""
from .core import (
    ConversionReport,
    ConvertedActivation,
    StochasticActivation,
    StochasticModel,
    available_activations,
    register_activation,
)

from .uncertainty import PredictionResult, predict_mc
from .training import Trainer, TrainingHistory
from .evaluation import Evaluator, EvaluationResult, OODResult, calibration_metrics, binary_ccq, binary_curves

__version__ = "1.0.0"
__all__ = [
    "StochasticModel",
    "Trainer",
    "Evaluator",
    "EvaluationResult",
    "OODResult",
    "calibration_metrics",
    "binary_ccq",
    "binary_curves",
    "TrainingHistory",
    "PredictionResult",
    "predict_mc",
    "StochasticActivation",
    "ConversionReport",
    "ConvertedActivation",
    "register_activation",
    "available_activations",
]
