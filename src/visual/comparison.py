"""Visualization module for comparing GT masks with model predictions.

Provides functions to create side-by-side comparison images for anomaly detection
evaluation and debugging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def create_comparison_image(
    original: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_heatmap: np.ndarray,
    anomaly_score: float,
    threshold: float,
    is_anomaly_gt: bool,
    defect_type: str = "",
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Layout: Original | GT Mask | Prediction Heatmap | Prediction Highlight

    Args:
        original: Original image (H, W, 3) BGR
        gt_mask: Ground truth mask (H, W) or None
        pred_heatmap: Predicted anomaly map (H, W), values 0-1
        anomaly_score: Anomaly score (0-1)
        threshold: Threshold used for binary prediction
        is_anomaly_gt: Ground truth label (True if anomaly)
        defect_type: Defect type string for display

    Returns:
        Combined comparison image (H, W*4 + title_height, 3)
    """
    h, w = original.shape[:2]

    # 1. Original image
    orig_display = original.copy()

    # 2. Ground truth mask visualization
    if gt_mask is not None and gt_mask.max() > 0:
        gt_colored = np.zeros_like(original)
        gt_colored[gt_mask > 0] = [0, 0, 255]  # Red for defect
        gt_display = cv2.addWeighted(original, 0.6, gt_colored, 0.4, 0)

        mask_uint8 = (gt_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_display, contours, -1, (0, 255, 0), 2)
    elif is_anomaly_gt:
        gt_display = original.copy()
        overlay = np.zeros_like(original)
        overlay[:] = (0, 100, 100)
        gt_display = cv2.addWeighted(gt_display, 0.8, overlay, 0.2, 0)
        cv2.putText(gt_display, "No Mask", (10, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        gt_display = np.zeros_like(original)
        cv2.putText(gt_display, "Normal", (w//2 - 40, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 3. Prediction heatmap
    heatmap_norm = np.clip(pred_heatmap, 0, 1)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_display = cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)

    # 4. Prediction highlight
    is_anomaly_pred = anomaly_score > threshold
    highlight_display = original.copy()

    if is_anomaly_pred:
        mask_pred = (pred_heatmap > threshold).astype(np.uint8)

        if mask_pred.sum() > 0:
            contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(highlight_display, contours, -1, (0, 0, 255), 2)

            red_overlay = original.copy()
            red_overlay[mask_pred == 1] = [0, 0, 255]
            highlight_display = cv2.addWeighted(highlight_display, 0.7, red_overlay, 0.3, 0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Original - GT label
    gt_label = "ANOMALY" if is_anomaly_gt else "NORMAL"
    gt_color = (0, 0, 255) if is_anomaly_gt else (0, 255, 0)
    cv2.putText(orig_display, f"GT: {gt_label}", (5, 20), font, font_scale, gt_color, thickness)
    if is_anomaly_gt and defect_type:
        cv2.putText(orig_display, f"Type: {defect_type}", (5, 40), font, font_scale, (255, 255, 255), thickness)

    # GT Mask label
    cv2.putText(gt_display, "Ground Truth", (5, 20), font, font_scale, (255, 255, 255), thickness)

    # Heatmap label
    cv2.putText(heatmap_display, "Pred Heatmap", (5, 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(heatmap_display, f"Score: {anomaly_score:.3f}", (5, 40), font, font_scale, (255, 255, 255), thickness)

    # Highlight label
    pred_label = "ANOMALY" if is_anomaly_pred else "NORMAL"
    pred_color = (0, 0, 255) if is_anomaly_pred else (0, 255, 0)
    cv2.putText(highlight_display, f"Pred: {pred_label}", (5, 20), font, font_scale, pred_color, thickness)

    correct = is_anomaly_gt == is_anomaly_pred
    result_text = "CORRECT" if correct else "WRONG"
    result_color = (0, 255, 0) if correct else (0, 0, 255)
    cv2.putText(highlight_display, result_text, (5, 40), font, font_scale, result_color, thickness)

    # Combine horizontally
    combined = np.hstack([orig_display, gt_display, heatmap_display, highlight_display])

    # Add title bar
    title_height = 30
    title_bar = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)

    col_width = w
    titles = ["Original", "Ground Truth", "Prediction", "Result"]
    for i, title in enumerate(titles):
        x = i * col_width + col_width // 2 - len(title) * 4
        cv2.putText(title_bar, title, (x, 20), font, 0.5, (255, 255, 255), 1)

    combined = np.vstack([title_bar, combined])

    return combined


def create_grid_visualization(
    images: list[np.ndarray],
    max_cols: int = 2,
    spacing: int = 5,
) -> np.ndarray:
    """Create a grid visualization from multiple images.

    Args:
        images: List of images to combine
        max_cols: Maximum number of columns
        spacing: Spacing between images in pixels

    Returns:
        Combined grid image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Resize all to same width
    max_width = max(img.shape[1] for img in images)
    resized = []
    for img in images:
        if img.shape[1] != max_width:
            scale = max_width / img.shape[1]
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (max_width, new_h))
        resized.append(img)

    # Create rows
    rows = []
    for i in range(0, len(resized), max_cols):
        row_images = resized[i:i + max_cols]

        # Pad row if needed
        if len(row_images) < max_cols:
            pad_h = row_images[0].shape[0]
            pad_img = np.zeros((pad_h, max_width, 3), dtype=np.uint8)
            while len(row_images) < max_cols:
                row_images.append(pad_img)

        # Add horizontal spacing
        spaced_row = []
        for j, img in enumerate(row_images):
            spaced_row.append(img)
            if j < len(row_images) - 1:
                spaced_row.append(np.ones((img.shape[0], spacing, 3), dtype=np.uint8) * 50)

        row = np.hstack(spaced_row)
        rows.append(row)

    # Add vertical spacing
    spaced_rows = []
    for i, row in enumerate(rows):
        spaced_rows.append(row)
        if i < len(rows) - 1:
            spaced_rows.append(np.ones((spacing, row.shape[1], 3), dtype=np.uint8) * 50)

    return np.vstack(spaced_rows)


def save_comparison(
    output_path: str | Path,
    original: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_heatmap: np.ndarray,
    anomaly_score: float,
    threshold: float,
    is_anomaly_gt: bool,
    defect_type: str = "",
) -> None:
    """Save comparison image to file.

    Args:
        output_path: Path to save the image
        original: Original image (H, W, 3) BGR
        gt_mask: Ground truth mask (H, W) or None
        pred_heatmap: Predicted anomaly map (H, W)
        anomaly_score: Anomaly score
        threshold: Threshold used
        is_anomaly_gt: Ground truth label
        defect_type: Defect type string
    """
    comparison = create_comparison_image(
        original=original,
        gt_mask=gt_mask,
        pred_heatmap=pred_heatmap,
        anomaly_score=anomaly_score,
        threshold=threshold,
        is_anomaly_gt=is_anomaly_gt,
        defect_type=defect_type,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)
