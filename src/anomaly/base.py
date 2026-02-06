"""Base class for anomaly detection models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class AnomalyResult:
    """Anomaly detection result.

    Attributes:
        anomaly_score: Image-level anomaly score (0-1, higher = more anomalous)
        anomaly_map: Pixel-level anomaly heatmap (H, W), values 0-1
        is_anomaly: Binary prediction (True if anomalous)
        threshold: Threshold used for binary prediction
        metadata: Additional model-specific information
    """
    anomaly_score: float
    anomaly_map: Optional[np.ndarray] = None
    is_anomaly: bool = False
    threshold: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "anomaly_score": float(self.anomaly_score),
            "is_anomaly": self.is_anomaly,
            "threshold": float(self.threshold),
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BaseAnomalyModel(ABC):
    """Abstract base class for anomaly detection models.

    Supports both per-class models (EfficientAD, PatchCore) and
    unified models (UniAD).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize anomaly detection model.

        Args:
            model_path: Path to model file (ONNX, checkpoint, etc.)
            threshold: Anomaly threshold for binary prediction
            device: Device to run inference ("cpu", "cuda", "mps")
        """
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.device = device
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load model from model_path. Must be implemented by subclass."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> AnomalyResult:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            AnomalyResult with anomaly score and map
        """
        pass

    def predict_batch(self, images: List[np.ndarray]) -> List[AnomalyResult]:
        """Run inference on multiple images.

        Default implementation processes images sequentially.
        Override for batch processing support.

        Args:
            images: List of input images

        Returns:
            List of AnomalyResult
        """
        return [self.predict(img) for img in images]

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path}, device={self.device})"


class PerClassAnomalyModel(BaseAnomalyModel):
    """Base class for per-class anomaly models (EfficientAD, PatchCore).

    These models have separate model files for each class.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        class_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize per-class anomaly model.

        Args:
            models_dir: Directory containing class-specific models
            class_name: Name of the class (e.g., "cigarette_box")
        """
        super().__init__(**kwargs)
        self.models_dir = Path(models_dir) if models_dir else None
        self.class_name = class_name

    @classmethod
    def list_available_classes(cls, models_dir: Union[str, Path]) -> List[str]:
        """List available classes in the models directory.

        Args:
            models_dir: Directory containing class-specific models

        Returns:
            List of class names
        """
        models_dir = Path(models_dir)
        if not models_dir.exists():
            return []
        return [d.name for d in models_dir.iterdir() if d.is_dir()]


class UnifiedAnomalyModel(BaseAnomalyModel):
    """Base class for unified anomaly models (UniAD).

    These models have a single model file for all classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._supported_classes: List[str] = []

    @property
    def supported_classes(self) -> List[str]:
        """List of classes supported by this model."""
        return self._supported_classes

    @abstractmethod
    def predict_with_class(
        self, image: np.ndarray, class_name: str
    ) -> AnomalyResult:
        """Run inference with class information.

        Args:
            image: Input image
            class_name: Class name for conditioning

        Returns:
            AnomalyResult
        """
        pass
