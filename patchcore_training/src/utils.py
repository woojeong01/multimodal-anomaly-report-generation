"""Utility functions for PatchCore training."""

import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device.

    Args:
        device_str: Device string ("cuda", "cpu", "mps", "auto")

    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("MPS not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def setup_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_anomaly_map(
    anomaly_map: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize anomaly map to [0, 1] range.

    Args:
        anomaly_map: Input anomaly map
        method: Normalization method ("minmax", "sigmoid")

    Returns:
        Normalized anomaly map
    """
    if method == "minmax":
        min_val = anomaly_map.min()
        max_val = anomaly_map.max()
        if max_val > min_val:
            return (anomaly_map - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(anomaly_map)
    elif method == "sigmoid":
        return 1 / (1 + np.exp(-anomaly_map))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_defect_location(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute defect location information from anomaly map.

    Args:
        anomaly_map: Normalized anomaly map (H, W) with values 0-1
        threshold: Threshold for defect detection

    Returns:
        Dictionary with location information
    """
    h, w = anomaly_map.shape

    defect_mask = anomaly_map > threshold

    if not defect_mask.any():
        return {
            "has_defect": False,
            "region": "none",
            "bbox": None,
            "center": None,
            "area_ratio": 0.0,
        }

    # Find bounding box
    coords = np.where(defect_mask)
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())

    # Compute center
    center_y = (y_min + y_max) / 2
    center_x = (x_min + x_max) / 2

    # Determine region (3x3 grid)
    region_y = "top" if center_y < h / 3 else ("bottom" if center_y > 2 * h / 3 else "center")
    region_x = "left" if center_x < w / 3 else ("right" if center_x > 2 * w / 3 else "center")

    if region_y == "center" and region_x == "center":
        region = "center"
    else:
        region = f"{region_y}-{region_x}"

    # Compute area ratio
    area_ratio = float(defect_mask.sum()) / (h * w)

    return {
        "has_defect": True,
        "region": region,
        "bbox": [x_min, y_min, x_max, y_max],
        "center": [float(center_x), float(center_y)],
        "area_ratio": round(area_ratio, 4),
    }
