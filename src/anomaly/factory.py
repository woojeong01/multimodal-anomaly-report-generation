"""Anomaly Model Factory"""
from __future__ import annotations

from .base import AnomalyModel
from .dummy_edge import DummyEdgeAnomaly
from .patchcore_adapter import PatchCoreAdapter
from ..utils.loaders import load_config

MODEL_REGISTRY: dict[str, type] = {
    "dummy": DummyEdgeAnomaly,
    "patchcore": PatchCoreAdapter,
}


def load_model(config_path: str) -> AnomalyModel:
    """Config에서 anomaly 모델 로드"""
    config = load_config(config_path)
    model_name = config.get("anomaly", {}).get("model", "patchcore")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    params = config.get("anomaly", {}).get(model_name, {})

    # null 값 필터링 (adapter에서 기본값 사용하도록)
    params = {k: v for k, v in params.items() if v is not None}

    # image_size: list → tuple
    if "image_size" in params:
        params["image_size"] = tuple(params["image_size"])

    return MODEL_REGISTRY[model_name](**params)


def list_models() -> list[str]:
    """사용 가능한 모델 목록"""
    return list(MODEL_REGISTRY.keys())
