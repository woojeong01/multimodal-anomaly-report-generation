"""AD prediction JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def normalize_image_key(image_key: str) -> str:
    """Normalize keys for robust image-path matching across platforms."""
    return image_key.lstrip("./").replace("\\", "/")


def _index_prediction_list(predictions: list[Any]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict):
            continue
        image_key = item.get("image_path") or item.get("image") or item.get("img_path") or item.get("path")
        if not isinstance(image_key, str) or not image_key:
            continue
        indexed[normalize_image_key(image_key)] = item
    return indexed


def parse_ad_predictions_payload(payload: Any) -> Dict[str, Dict[str, Any]]:
    """Parse legacy/report-style AD outputs into a normalized dict index."""
    if isinstance(payload, list):
        return _index_prediction_list(payload)

    if isinstance(payload, dict):
        predictions = payload.get("predictions")
        if isinstance(predictions, list):
            return _index_prediction_list(predictions)

        # Already-indexed mapping fallback.
        if all(isinstance(k, str) and isinstance(v, dict) for k, v in payload.items()):
            return {normalize_image_key(k): v for k, v in payload.items()}

    return {}


def load_ad_predictions_file(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return parse_ad_predictions_payload(payload)

