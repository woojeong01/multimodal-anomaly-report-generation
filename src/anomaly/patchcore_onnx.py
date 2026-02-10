"""PatchCore ONNX inference module."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import AnomalyResult, PerClassAnomalyModel


class PatchCoreOnnx(PerClassAnomalyModel):
    """PatchCore ONNX inference class.

    Loads ONNX model and performs inference using onnxruntime.

    Usage:
        model = PatchCoreOnnx(
            model_path="models/onnx/GoodsAD/cigarette_box/model.onnx",
            threshold=0.5,
        )
        model.load_model()
        result = model.predict(image_bgr)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
        device: str = "cpu",
        input_size: Tuple[int, int] = (224, 224),
        **kwargs,
    ):
        """Initialize PatchCore ONNX model.

        Args:
            model_path: Path to ONNX model file
            threshold: Anomaly threshold for binary prediction
            device: Device to run inference ("cpu" or "cuda")
            input_size: Model input size (height, width)
        """
        super().__init__(model_path=model_path, threshold=threshold, device=device, **kwargs)
        self.input_size = input_size
        self._session = None
        self._input_name = None
        self._output_names = None

    def load_model(self) -> None:
        """Load ONNX model using onnxruntime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Set providers based on device
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

        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._model = True  # Mark as loaded

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Resize
        img = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def predict(self, image: np.ndarray) -> AnomalyResult:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            AnomalyResult with anomaly score and map
        """
        if not self.is_loaded():
            self.load_model()

        original_size = image.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(image)

        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # Parse outputs (depends on export format)
        if len(outputs) >= 2:
            anomaly_map = outputs[0]
            pred_score = outputs[1]
        else:
            anomaly_map = outputs[0]
            pred_score = anomaly_map.max()

        # Process anomaly map
        if isinstance(anomaly_map, np.ndarray):
            if anomaly_map.ndim == 4:
                anomaly_map = anomaly_map[0, 0]  # Remove batch and channel dims
            elif anomaly_map.ndim == 3:
                anomaly_map = anomaly_map[0]  # Remove batch dim

            # Resize to original size
            anomaly_map = cv2.resize(
                anomaly_map, (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalize to [0, 1]
            map_min, map_max = anomaly_map.min(), anomaly_map.max()
            if map_max > map_min:
                anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
            else:
                anomaly_map = np.zeros_like(anomaly_map)

        # Process score
        if isinstance(pred_score, np.ndarray):
            pred_score = float(pred_score.flatten()[0])
        else:
            pred_score = float(pred_score)

        # Binary prediction
        is_anomaly = pred_score > self.threshold

        return AnomalyResult(
            anomaly_score=pred_score,
            anomaly_map=anomaly_map,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
        )

    def predict_batch(self, images: List[np.ndarray]) -> List[AnomalyResult]:
        """Run inference on multiple images.

        Args:
            images: List of input images

        Returns:
            List of AnomalyResult
        """
        # For now, process sequentially
        # TODO: Implement batched inference for better performance
        return [self.predict(img) for img in images]


class PatchCoreModelManager:
    """Manager for multiple PatchCore ONNX models.

    Handles loading models for different datasets/categories.

    Usage:
        manager = PatchCoreModelManager(models_dir="models/onnx")
        result = manager.predict("GoodsAD", "cigarette_box", image_bgr)
    """

    def __init__(
        self,
        models_dir: Union[str, Path],
        threshold: float = 0.5,
        device: str = "cpu",
        input_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize model manager.

        Args:
            models_dir: Directory containing ONNX models
            threshold: Default anomaly threshold
            device: Device for inference
            input_size: Model input size
        """
        self.models_dir = Path(models_dir)
        self.threshold = threshold
        self.device = device
        self.input_size = input_size
        self._models: Dict[str, PatchCoreOnnx] = {}

    def get_model_path(self, dataset: str, category: str) -> Path:
        """Get model path for dataset/category."""
        return self.models_dir / dataset / category / "model.onnx"

    def get_model(self, dataset: str, category: str) -> PatchCoreOnnx:
        """Get or load model for dataset/category."""
        key = f"{dataset}/{category}"

        if key not in self._models:
            model_path = self.get_model_path(dataset, category)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            model = PatchCoreOnnx(
                model_path=model_path,
                threshold=self.threshold,
                device=self.device,
                input_size=self.input_size,
            )
            model.load_model()
            self._models[key] = model

        return self._models[key]

    def predict(self, dataset: str, category: str, image: np.ndarray) -> AnomalyResult:
        """Run inference for specific dataset/category.

        Args:
            dataset: Dataset name (e.g., "GoodsAD")
            category: Category name (e.g., "cigarette_box")
            image: Input image in BGR format

        Returns:
            AnomalyResult
        """
        model = self.get_model(dataset, category)
        return model.predict(image)

    def list_available_models(self) -> List[Tuple[str, str]]:
        """List available dataset/category pairs."""
        available = []

        if not self.models_dir.exists():
            return available

        for dataset_dir in self.models_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for category_dir in dataset_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                if (category_dir / "model.onnx").exists():
                    available.append((dataset_dir.name, category_dir.name))

        return sorted(available)

    def clear_cache(self) -> None:
        """Clear loaded models from cache."""
        self._models.clear()
