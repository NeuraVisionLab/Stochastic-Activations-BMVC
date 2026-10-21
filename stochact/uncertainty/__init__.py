"""Memory-efficient Monte Carlo uncertainty inference."""
from .inference import predict_mc
from .results import PredictionResult

__all__ = ["predict_mc", "PredictionResult"]
