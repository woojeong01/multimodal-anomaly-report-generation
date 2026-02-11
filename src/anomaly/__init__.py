"""Anomaly detection module.

Provides unified interface for anomaly detection models:
- PatchCore (ONNX)
"""
from .base import (
    AnomalyResult,
    BaseAnomalyModel,
    PerClassAnomalyModel,
    UnifiedAnomalyModel,
)
from .patchcore_onnx import PatchCoreOnnx, PatchCoreModelManager

__all__ = [
    # Base classes
    "AnomalyResult",
    "BaseAnomalyModel",
    "PerClassAnomalyModel",
    "UnifiedAnomalyModel",
    # PatchCore
    "PatchCoreOnnx",
    "PatchCoreModelManager",
]
