"""Defect structure analysis from anomaly heatmaps.

Provides functions to extract structured defect information from anomaly maps
for LLM context.
"""
from __future__ import annotations

import cv2
import numpy as np


def nine_grid_location(cx: float, cy: float) -> str:
    """Get location description in 3x3 grid.

    Args:
        cx: Normalized x coordinate (0-1)
        cy: Normalized y coordinate (0-1)

    Returns:
        Location string like "top-left", "center", "bottom-right"
    """
    xs = "left" if cx < 1/3 else ("center" if cx < 2/3 else "right")
    ys = "top" if cy < 1/3 else ("middle" if cy < 2/3 else "bottom")

    if ys == "middle" and xs == "center":
        return "center"
    return f"{ys}-{xs}"


def structure_from_heatmap(heatmap: np.ndarray, *, thr: float = 0.35) -> dict:
    """Extract structured defect information from anomaly heatmap.

    Args:
        heatmap: Anomaly map (H, W) with values typically 0-1
        thr: Threshold for defect detection

    Returns:
        Dictionary with defect structure:
        - has_defect: bool
        - location: str (e.g., "top-left", "center")
        - area_ratio: float (0-1)
        - shape: str ("blob" or "elongated")
        - severity: str ("low", "medium", "high")
        - bbox: [x1, y1, x2, y2] or None
        - center: [cx, cy] normalized or None
        - confidence: float (max score in defect region) or None
    """
    hm = np.asarray(heatmap)
    if hm.ndim != 2:
        raise ValueError("heatmap must be HxW")

    H, W = hm.shape
    binm = (hm >= thr).astype(np.uint8)
    area_ratio = float(binm.mean())
    has_defect = area_ratio > 0.001

    if not has_defect:
        return {
            "has_defect": False,
            "location": "none",
            "area_ratio": area_ratio,
            "shape": "none",
            "severity": "low",
            "bbox": None,
            "center": None,
            "confidence": None,
            "thr": thr,
        }

    # Connected components analysis
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)

    if num <= 1:
        # No components found (shouldn't happen if has_defect is True)
        cx, cy = 0.5, 0.5
        shape = "blob"
        bbox = None
        confidence = float(hm.max())
    else:
        # Find largest component (excluding background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1

        # Centroid (normalized)
        cx_px, cy_px = centroids[idx]
        cx, cy = float(cx_px / max(W, 1)), float(cy_px / max(H, 1))

        # Bounding box
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        bbox = [int(x), int(y), int(x + w), int(y + h)]

        # Shape determination
        aspect_ratio = max(w, h) / max(1, min(w, h))
        shape = "elongated" if aspect_ratio >= 2.5 else "blob"

        # Confidence (max anomaly score in the defect region)
        component_mask = (labels == idx)
        confidence = float(hm[component_mask].max())

    location = nine_grid_location(cx, cy)

    # Severity based on area ratio
    if area_ratio < 0.01:
        severity = "low"
    elif area_ratio < 0.05:
        severity = "medium"
    else:
        severity = "high"

    return {
        "has_defect": True,
        "location": location,
        "area_ratio": round(area_ratio, 4),
        "shape": shape,
        "severity": severity,
        "bbox": bbox,
        "center": [round(cx, 3), round(cy, 3)] if cx is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "thr": thr,
    }


def get_defect_description(structure: dict) -> str:
    """Generate human-readable description from defect structure.

    Args:
        structure: Output from structure_from_heatmap()

    Returns:
        Human-readable description string
    """
    if not structure.get("has_defect", False):
        return "No defect detected."

    parts = []

    # Location
    location = structure.get("location", "unknown")
    parts.append(f"Defect detected at {location} region")

    # Shape and severity
    shape = structure.get("shape", "unknown")
    severity = structure.get("severity", "unknown")
    parts.append(f"({shape} shape, {severity} severity)")

    # Area
    area_ratio = structure.get("area_ratio", 0)
    if area_ratio > 0:
        parts.append(f"covering {area_ratio*100:.1f}% of image area")

    # Confidence
    confidence = structure.get("confidence")
    if confidence is not None:
        parts.append(f"with {confidence:.0%} confidence")

    return " ".join(parts) + "."
