"""PatchCore adapter using anomalib"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from anomalib.models import Patchcore
from ..common.types import AnomalyResult
from ..utils.device import get_device
from ..utils.checkpoint import load_checkpoint
from ..utils.log import setup_logger
from ..datasets.preprocess import get_transforms, prepare_input

logger = setup_logger(name="PatchCore", log_prefix="patchcore")

# 기본값
DEFAULTS = {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "image_size": (1024, 1024),
}


class PatchCoreAdapter:
    """PatchCore 모델 어댑터"""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        backbone: str | None = None,
        layers: list[str] | None = None,
        image_size: tuple[int, int] | list[int] | None = None,
    ):
        # null이면 기본값 사용
        self.backbone = backbone or DEFAULTS["backbone"]
        self.layers = layers or DEFAULTS["layers"]
        self.image_size = tuple(image_size) if image_size else DEFAULTS["image_size"]

        self.device = get_device(verbose=False)
        self.model = Patchcore(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=True,
        )
        self.model.to(self.device)
        self.transform = get_transforms(self.image_size)

        self.trained = False
        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(self.model, checkpoint_path, self.device)
            self.trained = True
            logger.info(f"Checkpoint loaded: {checkpoint_path}")

        logger.info(
            f"Initialized - backbone: {self.backbone}, "
            f"layers: {self.layers}, image_size: {self.image_size}, device: {self.device}"
        )

    def infer(
        self,
        image_bgr: np.ndarray,
        *,
        templates_bgr: list[np.ndarray] | None = None,
    ) -> AnomalyResult:
        """이미지에서 이상 탐지 수행"""
        del templates_bgr

        if not self.trained:
            raise RuntimeError("Model not trained. Provide checkpoint_path or train first.")

        input_tensor = prepare_input(image_bgr, self.transform).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        # postprocess
        if hasattr(output, "pred_score"):
            score = float(output.pred_score.item())
        elif hasattr(output, "pred_scores"):
            score = float(output.pred_scores.item())
        else:
            score = 0.5

        if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            heatmap = (anomaly_map - anomaly_map.min()) / (
                anomaly_map.max() - anomaly_map.min() + 1e-8
            )
            heatmap = heatmap.astype(np.float32)
        else:
            heatmap = np.zeros(self.image_size, dtype=np.float32)

        return AnomalyResult(score=score, heatmap=heatmap)

    @property
    def is_trained(self) -> bool:
        return self.trained
