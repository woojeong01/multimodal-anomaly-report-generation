#!/usr/bin/env python
"""Visualize model predictions vs ground truth masks.

Creates side-by-side comparison images for each category:
- Normal samples: 5 random images
- Anomaly samples: 5 random images per defect type

Usage:
    python patchcore_training/scripts/visualize_comparison.py \
        --config patchcore_training/config/config.yaml \
        --thresholds patchcore_training/config/thresholds.yaml \
        --output-dir output/visualizations_comparison \
        --samples-per-type 5
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add patchcore_training to path
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PATCHCORE_ROOT))

from src.model import PatchCore
from src.dataset import get_dataloader
from src.utils import load_config, setup_seed, get_device


def load_thresholds(thresholds_path: str) -> dict:
    """Load per-category thresholds from YAML file."""
    import yaml

    with open(thresholds_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return {
        "global": config.get("global", 3.0),
        "categories": config.get("categories", {}),
    }


def load_models(config: dict, device: torch.device) -> dict:
    """Load all trained PatchCore models."""
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    datasets_config = config["data"].get("datasets", {})

    models = {}
    for dataset_name, categories in datasets_config.items():
        for category in categories:
            pt_path = checkpoint_dir / dataset_name / category / "model.pt"
            if pt_path.exists():
                model = PatchCore.load(str(pt_path), device)
                model.eval()
                key = f"{dataset_name}/{category}"
                models[key] = model
                print(f"Loaded: {key}")

    return models


def create_comparison_image(
    original: np.ndarray,
    gt_mask: np.ndarray,
    pred_heatmap: np.ndarray,
    anomaly_score: float,
    threshold: float,
    is_anomaly_gt: bool,
    defect_type: str,
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Layout: Original | GT Mask | Prediction Heatmap | Prediction Highlight

    Args:
        original: Original image (H, W, 3) BGR
        gt_mask: Ground truth mask (H, W) or None
        pred_heatmap: Predicted anomaly map (H, W)
        anomaly_score: Anomaly score
        threshold: Threshold used
        is_anomaly_gt: Ground truth label
        defect_type: Defect type string

    Returns:
        Combined comparison image
    """
    h, w = original.shape[:2]

    # 1. Original image (add label)
    orig_display = original.copy()

    # 2. Ground truth mask visualization
    if gt_mask is not None and gt_mask.max() > 0:
        # Create colored mask overlay
        gt_colored = np.zeros_like(original)
        gt_colored[gt_mask > 0] = [0, 0, 255]  # Red for defect
        gt_display = cv2.addWeighted(original, 0.6, gt_colored, 0.4, 0)

        # Draw contours
        mask_uint8 = (gt_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_display, contours, -1, (0, 255, 0), 2)
    elif is_anomaly_gt:
        # Anomaly but no mask available - show original with warning
        gt_display = original.copy()
        # Add semi-transparent overlay to indicate no mask
        overlay = np.zeros_like(original)
        overlay[:] = (0, 100, 100)  # Dark yellow tint
        gt_display = cv2.addWeighted(gt_display, 0.8, overlay, 0.2, 0)
        cv2.putText(gt_display, "No Mask Available", (10, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # Normal sample - show black/empty mask (no defect region)
        gt_display = np.zeros_like(original)
        # Add text to indicate it's normal (no defect)
        cv2.putText(gt_display, "Normal", (w//2 - 40, h//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(gt_display, "(No Defect)", (w//2 - 55, h//2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3. Prediction heatmap
    # Normalize heatmap for visualization
    heatmap_norm = np.clip(pred_heatmap / (threshold * 2), 0, 1)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_display = cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)

    # 4. Prediction highlight
    is_anomaly_pred = anomaly_score > threshold
    highlight_display = original.copy()

    if is_anomaly_pred:
        # Threshold the heatmap to find anomaly regions
        map_threshold = threshold / (threshold * 2)  # normalized threshold
        mask_pred = (heatmap_norm > map_threshold).astype(np.uint8)

        if mask_pred.sum() > 0:
            # Draw contours
            contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(highlight_display, contours, -1, (0, 0, 255), 2)

            # Fill with semi-transparent red
            red_overlay = original.copy()
            red_overlay[mask_pred == 1] = [0, 0, 255]
            highlight_display = cv2.addWeighted(highlight_display, 0.7, red_overlay, 0.3, 0)

    # Add labels to each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Original - show GT label and defect type
    gt_label = "ANOMALY" if is_anomaly_gt else "NORMAL"
    gt_color = (0, 0, 255) if is_anomaly_gt else (0, 255, 0)
    cv2.putText(orig_display, f"GT: {gt_label}", (10, 25), font, font_scale, gt_color, thickness)
    if is_anomaly_gt and defect_type != "good":
        cv2.putText(orig_display, f"Type: {defect_type}", (10, 50), font, font_scale, (255, 255, 255), thickness)

    # GT Mask label
    cv2.putText(gt_display, "Ground Truth", (10, 25), font, font_scale, (255, 255, 255), thickness)

    # Heatmap label
    cv2.putText(heatmap_display, "Pred Heatmap", (10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(heatmap_display, f"Score: {anomaly_score:.3f}", (10, 50), font, font_scale, (255, 255, 255), thickness)

    # Highlight label - show prediction result
    pred_label = "ANOMALY" if is_anomaly_pred else "NORMAL"
    pred_color = (0, 0, 255) if is_anomaly_pred else (0, 255, 0)
    cv2.putText(highlight_display, f"Pred: {pred_label}", (10, 25), font, font_scale, pred_color, thickness)

    # Check if prediction is correct
    correct = is_anomaly_gt == is_anomaly_pred
    result_text = "CORRECT" if correct else "WRONG"
    result_color = (0, 255, 0) if correct else (0, 0, 255)
    cv2.putText(highlight_display, result_text, (10, 50), font, font_scale, result_color, thickness)

    # Combine horizontally
    combined = np.hstack([orig_display, gt_display, heatmap_display, highlight_display])

    # Add title bar
    title_height = 40
    title_bar = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)

    # Column titles
    col_width = w
    titles = ["Original", "Ground Truth Mask", "Prediction Heatmap", "Prediction Highlight"]
    for i, title in enumerate(titles):
        x = i * col_width + col_width // 2 - len(title) * 5
        cv2.putText(title_bar, title, (x, 28), font, 0.7, (255, 255, 255), 2)

    combined = np.vstack([title_bar, combined])

    return combined


def process_category(
    model: PatchCore,
    config: dict,
    dataset_name: str,
    category: str,
    threshold: float,
    output_dir: Path,
    samples_per_type: int,
    device: torch.device,
):
    """Process a single category and save comparison images."""

    print(f"\nProcessing: {dataset_name}/{category} (threshold={threshold:.2f})")

    # Create dataloader
    data_root = Path(config["data"]["root"])
    image_size = config["data"].get("image_size", 224)

    dataloader = get_dataloader(
        root=data_root,
        dataset_name=dataset_name,
        category=category,
        split="test",
        image_size=image_size,
        batch_size=1,  # Process one at a time for visualization
        num_workers=0,
        include_mask=True,
    )

    if len(dataloader.dataset) == 0:
        print(f"  No test samples found")
        return

    # Collect samples by defect type
    samples_by_type = defaultdict(list)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Collecting samples", leave=False):
            image = batch["image"].to(device)
            label = batch["label"].item()
            mask = batch["mask"].numpy().squeeze()  # (H, W)
            image_path = batch["image_path"][0]
            defect_type = batch["defect_type"][0]

            # Get prediction
            scores, maps = model.predict(image)
            anomaly_score = scores[0].cpu().item()
            anomaly_map = maps[0].cpu().numpy()  # (H, W)

            # Load original image (full resolution)
            full_path = data_root / image_path
            original = cv2.imread(str(full_path))
            if original is None:
                continue

            # Resize mask and map to original size
            h, w = original.shape[:2]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            if anomaly_map.shape != (h, w):
                anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

            sample_info = {
                "original": original,
                "mask": mask,
                "anomaly_map": anomaly_map,
                "anomaly_score": anomaly_score,
                "label": label,
                "defect_type": defect_type,
                "image_path": image_path,
            }

            samples_by_type[defect_type].append(sample_info)

    # Create output directory for this category
    cat_output_dir = output_dir / dataset_name / category
    cat_output_dir.mkdir(parents=True, exist_ok=True)

    # Sample and save for each defect type
    for defect_type, samples in samples_by_type.items():
        # Random sample
        n_samples = min(samples_per_type, len(samples))
        selected = random.sample(samples, n_samples)

        print(f"  {defect_type}: {len(samples)} samples, selected {n_samples}")

        # Create comparison images
        comparison_images = []
        for i, sample in enumerate(selected):
            comp_img = create_comparison_image(
                original=sample["original"],
                gt_mask=sample["mask"],
                pred_heatmap=sample["anomaly_map"],
                anomaly_score=sample["anomaly_score"],
                threshold=threshold,
                is_anomaly_gt=sample["label"] == 1,
                defect_type=sample["defect_type"],
            )
            comparison_images.append(comp_img)

            # Also save individual comparison
            individual_path = cat_output_dir / f"{defect_type}_{i+1}.png"
            cv2.imwrite(str(individual_path), comp_img)

        # Create combined grid (vertical stack)
        if comparison_images:
            # Add spacing between rows
            spacing = 10
            spaced_images = []
            for img in comparison_images:
                spaced_images.append(img)
                spaced_images.append(np.ones((spacing, img.shape[1], 3), dtype=np.uint8) * 50)

            if spaced_images:
                spaced_images = spaced_images[:-1]  # Remove last spacer
                combined_grid = np.vstack(spaced_images)

                grid_path = cat_output_dir / f"{defect_type}_grid.png"
                cv2.imwrite(str(grid_path), combined_grid)

    print(f"  Saved to: {cat_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model predictions vs ground truth")

    parser.add_argument(
        "--config",
        type=str,
        default=str(PATCHCORE_ROOT / "config" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=str(PATCHCORE_ROOT / "config" / "thresholds.yaml"),
        help="Path to per-category thresholds YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualizations_comparison",
        help="Output directory for comparison images",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=5,
        help="Number of samples per defect type",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Process only specific category (format: Dataset/category)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)
    random.seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Load thresholds
    print(f"Loading thresholds from: {args.thresholds}")
    thresholds = load_thresholds(args.thresholds)

    # Get device
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
    models = load_models(config, device)

    if not models:
        print("No trained models found!")
        return

    # Filter by category if specified
    if args.category:
        if args.category in models:
            models = {args.category: models[args.category]}
        else:
            print(f"Category not found: {args.category}")
            return

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Samples per defect type: {args.samples_per_type}")

    # Process each category
    for key, model in models.items():
        dataset_name, category = key.split("/")

        # Get threshold for this category
        threshold = thresholds["categories"].get(key, thresholds["global"])

        process_category(
            model=model,
            config=config,
            dataset_name=dataset_name,
            category=category,
            threshold=threshold,
            output_dir=output_dir,
            samples_per_type=args.samples_per_type,
            device=device,
        )

    print(f"\n{'='*60}")
    print(f"Done! Comparison images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
