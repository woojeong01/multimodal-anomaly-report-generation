"""PatchCore ONNX inference module.

Uses backbone ONNX + memory bank numpy for optimized inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import AnomalyResult, PerClassAnomalyModel


class PatchCoreOnnx(PerClassAnomalyModel):
    """PatchCore ONNX inference class.

    Uses backbone ONNX for feature extraction + numpy memory bank for anomaly scoring.
    This approach avoids anomalib ONNX export issues while enabling fast inference.

    Expected files in model_dir:
        - backbone.onnx: Feature extractor
        - memory_bank.npy: Coreset embeddings
        - config.json: Model configuration

    Usage:
        model = PatchCoreOnnx(
            model_path="models/onnx/GoodsAD/cigarette_box",
            device="cuda",
        )
        model.load_model()
        result = model.predict(image_bgr)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize PatchCore ONNX model.

        Args:
            model_path: Path to model directory (containing backbone.onnx, memory_bank.npy)
            threshold: Anomaly threshold for binary prediction
            device: Device to run inference ("cpu" or "cuda")
        """
        super().__init__(model_path=model_path, threshold=threshold, device=device, **kwargs)
        self._session = None
        self._memory_bank = None
        self._config = None
        self._input_name = None
        self._output_name = None
        self._num_neighbors = 1

    def load_model(self) -> None:
        """Load backbone ONNX and memory bank."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        model_dir = Path(self.model_path)

        # Check for new format (backbone.onnx + memory_bank.npy)
        backbone_path = model_dir / "backbone.onnx"
        memory_bank_path = model_dir / "memory_bank.npy"
        config_path = model_dir / "config.json"

        # Fallback to old format (model.onnx)
        if not backbone_path.exists():
            old_model_path = model_dir / "model.onnx"
            if old_model_path.exists():
                backbone_path = old_model_path
                memory_bank_path = None
            else:
                raise FileNotFoundError(f"Model not found in {model_dir}")

        # Load config
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {
                "input_size": [224, 224],
                "feature_dim": 1536,
                "num_neighbors": 1,
            }
        self._num_neighbors = int(self._config.get("num_neighbors", 1))

        # Set providers
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(backbone_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Print actual provider being used
        actual_providers = self._session.get_providers()
        print(f"  ONNX Providers: {actual_providers}")

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Load memory bank
        if memory_bank_path and memory_bank_path.exists():
            self._memory_bank = np.load(memory_bank_path)
        else:
            self._memory_bank = None

        self._model = True

    @property
    def input_size(self) -> Tuple[int, int]:
        """Get model input size."""
        return tuple(self._config.get("input_size", [224, 224]))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        h, w = self.input_size

        # Resize
        img = cv2.resize(image, (w, h))

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
        return np.expand_dims(img, axis=0).astype(np.float32)

    def predict(self, image: np.ndarray) -> AnomalyResult:
        """Run inference on a single image."""
        if not self.is_loaded():
            self.load_model()

        original_size = image.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(image)

        # Extract features with backbone
        features = self._session.run([self._output_name], {self._input_name: input_tensor})[0]

        # If we have memory bank, compute anomaly score
        if self._memory_bank is not None:
            anomaly_map, anomaly_score = self._compute_anomaly(features, original_size)
        else:
            # Old format: output is already anomaly map
            anomaly_map = features[0, 0] if features.ndim == 4 else features[0]
            anomaly_map = cv2.resize(anomaly_map, (original_size[1], original_size[0]))
            anomaly_score = float(anomaly_map.max())

        # Normalize anomaly map
        map_min, map_max = anomaly_map.min(), anomaly_map.max()
        if map_max > map_min:
            anomaly_map = (anomaly_map - map_min) / (map_max - map_min)

        is_anomaly = anomaly_score > self.threshold

        return AnomalyResult(
            anomaly_score=anomaly_score,
            anomaly_map=anomaly_map,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
        )

    def _compute_anomaly(
        self, features: np.ndarray, original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float]:
        """Compute anomaly map and score from features and memory bank.

        Args:
            features: Feature tensor from backbone [B, C, H, W]
            original_size: Original image size (H, W)

        Returns:
            anomaly_map: Anomaly heatmap resized to original size
            anomaly_score: Maximum anomaly score
        """
        # features: [1, C, H, W] -> [H*W, C]
        b, c, h, w = features.shape
        features_flat = features[0].transpose(1, 2, 0).reshape(-1, c)  # [H*W, C]

        # PatchCore nearest-neighbor search against the memory bank.
        distances = self._euclidean_dist(features_flat, self._memory_bank)
        min_locations = np.argmin(distances, axis=1)  # [H*W]
        min_distances = distances[np.arange(distances.shape[0]), min_locations]  # [H*W]
        anomaly_map = min_distances.reshape(h, w)

        # Upsample to original size
        anomaly_map = cv2.resize(anomaly_map, (original_size[1], original_size[0]))

        # Match anomalib PatchCore image-level scoring.
        anomaly_score = float(self._compute_image_score(min_distances, min_locations, features_flat))

        return anomaly_map, anomaly_score

    def predict_batch(self, images: List[np.ndarray]) -> List[AnomalyResult]:
        """Run inference on multiple images."""
        return [self.predict(img) for img in images]

    @staticmethod
    def _euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute pairwise L2 distances between rows of a and b."""
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)          # [M, 1]
        b_sq = np.sum(b ** 2, axis=1, keepdims=True).T        # [1, N]
        cross = a @ b.T                                        # [M, N]
        return np.sqrt(np.maximum(a_sq + b_sq - 2.0 * cross, 0.0))

    def _nearest_neighbors(self, embedding: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors in the memory bank."""
        distances = self._euclidean_dist(embedding, self._memory_bank)  # [M, N]
        if n_neighbors == 1:
            locations = np.argmin(distances, axis=1, keepdims=True)  # [M, 1]
            patch_scores = distances[np.arange(distances.shape[0]), locations[:, 0]][:, None]
            return patch_scores, locations

        idx = np.argpartition(distances, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        row = np.arange(distances.shape[0])[:, None]
        vals = distances[row, idx]
        order = np.argsort(vals, axis=1)
        sorted_idx = idx[row, order]
        sorted_vals = vals[row, order]
        return sorted_vals, sorted_idx

    def _compute_image_score(
        self,
        patch_scores: np.ndarray,
        locations: np.ndarray,
        embedding: np.ndarray,
    ) -> float:
        """Compute image-level score using anomalib PatchCore weighting."""
        if patch_scores.size == 0:
            return 0.0
        if self._num_neighbors <= 1:
            return float(np.max(patch_scores))

        max_patch_idx = int(np.argmax(patch_scores))
        max_patch_feature = embedding[max_patch_idx]          # [C]
        score = float(patch_scores[max_patch_idx])            # s*
        nn_index = int(locations[max_patch_idx])              # m*

        memory_bank_size = int(self._memory_bank.shape[0])
        k = min(max(1, self._num_neighbors), memory_bank_size)

        nn_sample = self._memory_bank[nn_index][None, :]      # [1, C]
        _, support_idx = self._nearest_neighbors(nn_sample, n_neighbors=k)  # [1, K]
        support = self._memory_bank[support_idx[0]]           # [K, C]

        d = self._euclidean_dist(max_patch_feature[None, :], support)  # [1, K]
        d = d - np.max(d, axis=1, keepdims=True)
        softmax = np.exp(d)
        softmax = softmax / np.sum(softmax, axis=1, keepdims=True)
        weight = 1.0 - float(softmax[0, 0])
        return weight * score


class PatchCoreModelManager:
    """Manager for multiple PatchCore ONNX models."""

    def __init__(
        self,
        models_dir: Union[str, Path],
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize model manager.

        Args:
            models_dir: Directory containing ONNX models
            threshold: Default anomaly threshold
            device: Device for inference
        """
        self.models_dir = Path(models_dir)
        self.threshold = threshold
        self.device = device
        self._models: Dict[str, PatchCoreOnnx] = {}

    def get_model(self, dataset: str, category: str) -> PatchCoreOnnx:
        """Get or load model for dataset/category."""
        key = f"{dataset}/{category}"

        if key not in self._models:
            model_path = self.models_dir / dataset / category

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            model = PatchCoreOnnx(
                model_path=model_path,
                threshold=self.threshold,
                device=self.device,
            )
            model.load_model()
            self._models[key] = model

        return self._models[key]

    def predict(self, dataset: str, category: str, image: np.ndarray) -> AnomalyResult:
        """Run inference for specific dataset/category."""
        model = self.get_model(dataset, category)
        return model.predict(image)

    def get_model_path(self, dataset: str, category: str) -> Path:
        """Get model directory path for dataset/category."""
        return self.models_dir / dataset / category

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
                # Check for new format or old format
                if (category_dir / "backbone.onnx").exists() or (category_dir / "model.onnx").exists():
                    available.append((dataset_dir.name, category_dir.name))

        return sorted(available)

    def clear_cache(self) -> None:
        """Clear loaded models from cache."""
        self._models.clear()
