"""Anomaly detection module.

Provides unified interface for various anomaly detection models:
- EfficientAD (ONNX)
- PatchCore (planned)
- UniAD (planned)
"""
from .base import (
    AnomalyResult,
    BaseAnomalyModel,
    PerClassAnomalyModel,
    UnifiedAnomalyModel,
)
from .dummy_edge import DummyEdgeAnomaly
from .efficientad_onnx import EfficientADOnnx, EfficientADModelManager

__all__ = [
    # Base classes
    "AnomalyResult",
    "BaseAnomalyModel",
    "PerClassAnomalyModel",
    "UnifiedAnomalyModel",
    # Implementations
    "DummyEdgeAnomaly",
    "EfficientADOnnx",
    "EfficientADModelManager",
]
