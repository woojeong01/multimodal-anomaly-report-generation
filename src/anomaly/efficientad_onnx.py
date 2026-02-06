"""EfficientAD ONNX inference adapter.

Supports inference using ONNX-exported EfficientAD models.
Works with models trained using anomalib.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import AnomalyResult, PerClassAnomalyModel

logger = logging.getLogger(__name__)


class EfficientADOnnx(PerClassAnomalyModel):
    """EfficientAD model using ONNX runtime for inference.

    Attributes:
        input_size: Model input size (height, width)
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
    """

    # ImageNet normalization (used by EfficientAD)
    NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        class_name: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize EfficientAD ONNX model.

        Args:
            model_path: Direct path to ONNX file
            models_dir: Directory containing per-class ONNX models
            class_name: Class name (used with models_dir)
            input_size: Model input size (height, width)
            threshold: Anomaly threshold
            device: Inference device ("cpu", "cuda")
        """
        super().__init__(
            model_path=model_path,
            models_dir=models_dir,
            class_name=class_name,
            threshold=threshold,
            device=device,
            **kwargs,
        )
        self.input_size = input_size
        self._session = None
        self._input_name = None
        self._output_names = None

        # Resolve model path from models_dir if needed
        if self.model_path is None and self.models_dir and self.class_name:
            self.model_path = self._find_model_path()

    def _find_model_path(self) -> Optional[Path]:
        """Find ONNX model path for the given class."""
        if not self.models_dir or not self.class_name:
            return None

        models_dir = Path(self.models_dir)
        # Try common patterns
        patterns = [
            models_dir / f"{self.class_name}.onnx",
            models_dir / self.class_name / "model.onnx",
            models_dir / self.class_name / "efficientad.onnx",
        ]

        for pattern in patterns:
            if pattern.exists():
                return pattern

        return None

    def load_model(self) -> None:
        """Load ONNX model using onnxruntime."""
        if self._session is not None:
            return

        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")

        # Configure providers based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Get input/output info
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        logger.info(f"Loaded EfficientAD ONNX model: {self.model_path}")
        logger.debug(f"Input: {self._input_name}, Outputs: {self._output_names}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Preprocessed tensor (1, C, H, W) in RGB format, normalized
        """
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image_resized = cv2.resize(
            image_rgb, (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to float and normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image_norm = (image_float - self.NORMALIZE_MEAN) / self.NORMALIZE_STD

        # HWC to CHW
        image_chw = np.transpose(image_norm, (2, 0, 1))

        # Add batch dimension
        return np.expand_dims(image_chw, axis=0).astype(np.float32)

    def postprocess(
        self, outputs: List[np.ndarray], original_size: Tuple[int, int]
    ) -> Tuple[float, np.ndarray]:
        """Postprocess model outputs.

        Args:
            outputs: Model outputs (anomaly_map and/or anomaly_score)
            original_size: Original image size (height, width)

        Returns:
            Tuple of (anomaly_score, anomaly_map resized to original)
        """
        # EfficientAD outputs anomaly map
        # The exact output format depends on how the model was exported
        if len(outputs) >= 2:
            # If both map and score are outputs
            anomaly_map = outputs[0]
            anomaly_score = outputs[1]
        else:
            # If only map is output, compute score from map
            anomaly_map = outputs[0]
            anomaly_score = None

        # Remove batch dimension if present
        if anomaly_map.ndim == 4:
            anomaly_map = anomaly_map[0]
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map[0]  # Remove channel dim

        # Normalize anomaly map to [0, 1]
        map_min = anomaly_map.min()
        map_max = anomaly_map.max()
        if map_max > map_min:
            anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
        else:
            anomaly_map = np.zeros_like(anomaly_map)

        # Resize to original size
        anomaly_map = cv2.resize(
            anomaly_map.astype(np.float32),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Compute score from map if not provided
        if anomaly_score is None:
            anomaly_score = float(anomaly_map.max())
        elif isinstance(anomaly_score, np.ndarray):
            anomaly_score = float(anomaly_score.flatten()[0])

        return anomaly_score, anomaly_map

    def predict(self, image: np.ndarray) -> AnomalyResult:
        """Run inference on a single image.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            AnomalyResult with anomaly score and map
        """
        if self._session is None:
            self.load_model()

        original_size = image.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # Postprocess
        anomaly_score, anomaly_map = self.postprocess(outputs, original_size)

        # Determine if anomaly
        is_anomaly = anomaly_score > self.threshold

        return AnomalyResult(
            anomaly_score=anomaly_score,
            anomaly_map=anomaly_map,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            metadata={
                "class_name": self.class_name,
                "model_type": "efficientad_onnx",
            }
        )

    @classmethod
    def list_available_classes(cls, models_dir: Union[str, Path]) -> List[str]:
        """List available classes in the ONNX models directory.

        Args:
            models_dir: Directory containing class-specific ONNX models

        Returns:
            List of class names
        """
        models_dir = Path(models_dir)
        if not models_dir.exists():
            return []

        classes = []
        # Check for direct .onnx files
        for f in models_dir.glob("*.onnx"):
            classes.append(f.stem)

        # Check for subdirectories with model.onnx
        for d in models_dir.iterdir():
            if d.is_dir():
                if (d / "model.onnx").exists() or (d / "efficientad.onnx").exists():
                    classes.append(d.name)

        return sorted(set(classes))


class EfficientADModelManager:
    """Manager for loading multiple EfficientAD models.

    Handles per-class model loading and provides a unified inference interface.
    """

    def __init__(
        self,
        models_dir: Union[str, Path],
        input_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize model manager.

        Args:
            models_dir: Directory containing per-class ONNX models
            input_size: Model input size
            threshold: Default anomaly threshold
            device: Inference device
        """
        self.models_dir = Path(models_dir)
        self.input_size = input_size
        self.threshold = threshold
        self.device = device
        self._models: Dict[str, EfficientADOnnx] = {}

    def get_model(self, class_name: str) -> EfficientADOnnx:
        """Get or load model for a specific class.

        Args:
            class_name: Class name

        Returns:
            Loaded EfficientADOnnx model
        """
        if class_name not in self._models:
            model = EfficientADOnnx(
                models_dir=self.models_dir,
                class_name=class_name,
                input_size=self.input_size,
                threshold=self.threshold,
                device=self.device,
            )
            model.load_model()
            self._models[class_name] = model

        return self._models[class_name]

    def predict(self, image: np.ndarray, class_name: str) -> AnomalyResult:
        """Run inference for a specific class.

        Args:
            image: Input image
            class_name: Class name

        Returns:
            AnomalyResult
        """
        model = self.get_model(class_name)
        return model.predict(image)

    def list_available_classes(self) -> List[str]:
        """List available classes."""
        return EfficientADOnnx.list_available_classes(self.models_dir)

    def preload_all(self) -> None:
        """Preload all available models."""
        for class_name in self.list_available_classes():
            self.get_model(class_name)
