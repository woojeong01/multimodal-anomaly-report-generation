"""Run anomaly detection inference and generate JSON output for LLM evaluation.

This script processes images using ONNX anomaly detection models and generates
a JSON file that can be used with eval_llm_baseline.py --with-ad option.

Usage:
    # Run inference on all images in mmad.json
    python scripts/run_ad_inference.py \
        --models-dir models/onnx \
        --data-root /Volumes/T7/Dataset/MMAD \
        --mmad-json /Volumes/T7/Dataset/MMAD/mmad_10classes.json \
        --output output/ad_predictions.json

    # Run inference with custom threshold
    python scripts/run_ad_inference.py \
        --models-dir models/onnx \
        --data-root /Volumes/T7/Dataset/MMAD \
        --mmad-json /Volumes/T7/Dataset/MMAD/mmad_10classes.json \
        --threshold 0.3 \
        --output output/ad_predictions.json

    # Test with max images
    python scripts/run_ad_inference.py \
        --models-dir models/onnx \
        --data-root /Volumes/T7/Dataset/MMAD \
        --mmad-json /Volumes/T7/Dataset/MMAD/mmad_10classes.json \
        --max-images 10 \
        --output output/ad_predictions_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.anomaly import EfficientADOnnx, EfficientADModelManager, AnomalyResult


def parse_image_path(image_path: str) -> tuple[str, str]:
    """Parse dataset and class name from image path.

    Args:
        image_path: Image path like "GoodsAD/cigarette_box/test/good/001.jpg"

    Returns:
        Tuple of (dataset_name, class_name)
    """
    parts = image_path.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def compute_defect_location(anomaly_map: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute defect location information from anomaly map.

    Args:
        anomaly_map: Anomaly heatmap (H, W) with values 0-1
        threshold: Threshold for defect detection

    Returns:
        Dictionary with location information
    """
    h, w = anomaly_map.shape

    # Find defect region
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
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

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
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "center": [float(center_x), float(center_y)],
        "area_ratio": round(area_ratio, 4),
    }


def result_to_dict(
    result: AnomalyResult,
    image_path: str,
    include_map_stats: bool = True
) -> Dict[str, Any]:
    """Convert AnomalyResult to dictionary for JSON serialization.

    Args:
        result: AnomalyResult from model inference
        image_path: Relative path to the image
        include_map_stats: Whether to include anomaly map statistics

    Returns:
        Dictionary representation suitable for LLM input
    """
    output = {
        "image_path": image_path,
        "anomaly_score": round(float(result.anomaly_score), 4),
        "is_anomaly": result.is_anomaly,
        "threshold": result.threshold,
    }

    # Add location information if anomaly map is available
    if result.anomaly_map is not None:
        location_info = compute_defect_location(result.anomaly_map, result.threshold)
        output["defect_location"] = location_info

        if include_map_stats:
            output["map_stats"] = {
                "max": round(float(result.anomaly_map.max()), 4),
                "mean": round(float(result.anomaly_map.mean()), 4),
                "std": round(float(result.anomaly_map.std()), 4),
            }

    # Add metadata
    if result.metadata:
        output["metadata"] = result.metadata

    return output


def main():
    parser = argparse.ArgumentParser(description="Run AD inference and generate JSON for LLM evaluation")

    # Model settings
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/onnx",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="efficientad",
        choices=["efficientad", "patchcore", "uniad"],
        help="Type of anomaly detection model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly threshold for binary prediction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device",
    )

    # Data settings
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing images",
    )
    parser.add_argument(
        "--mmad-json",
        type=str,
        required=True,
        help="Path to mmad.json or filtered subset",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default="output/ad_predictions.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--include-map-stats",
        action="store_true",
        default=True,
        help="Include anomaly map statistics in output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )

    args = parser.parse_args()

    # Validate paths
    models_dir = Path(args.models_dir)
    data_root = Path(args.data_root)
    mmad_json = Path(args.mmad_json)
    output_path = Path(args.output)

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)

    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)

    if not mmad_json.exists():
        print(f"Error: MMAD JSON not found: {mmad_json}")
        sys.exit(1)

    # Load MMAD data
    print(f"Loading MMAD data from: {mmad_json}")
    with open(mmad_json, "r", encoding="utf-8") as f:
        mmad_data = json.load(f)

    image_paths = list(mmad_data.keys())
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Total images to process: {len(image_paths)}")

    # Load existing results if resuming
    existing_results = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_list = json.load(f)
            existing_results = {r["image_path"]: r for r in existing_list}
        print(f"Loaded {len(existing_results)} existing results")

    # Initialize model manager
    print(f"Initializing {args.model_type} model manager from: {models_dir}")

    if args.model_type == "efficientad":
        # Check available models
        available_classes = EfficientADOnnx.list_available_classes(models_dir)
        if not available_classes:
            # Try subdirectories (models/onnx/GoodsAD/cigarette_box/model.onnx)
            for dataset_dir in models_dir.iterdir():
                if dataset_dir.is_dir():
                    sub_classes = EfficientADOnnx.list_available_classes(dataset_dir)
                    if sub_classes:
                        available_classes.extend([f"{dataset_dir.name}/{c}" for c in sub_classes])

        print(f"Available models: {available_classes}")

        # Create model cache
        model_cache: Dict[str, EfficientADOnnx] = {}
    else:
        print(f"Error: Model type '{args.model_type}' not yet supported")
        sys.exit(1)

    # Process images
    results = list(existing_results.values())
    processed = len(existing_results)
    skipped = 0
    errors = 0

    print()
    print("=" * 60)
    print("Running inference")
    print("=" * 60)

    pbar = tqdm(image_paths, desc="Processing", ncols=100)
    for image_path in pbar:
        # Skip if already processed
        if image_path in existing_results:
            continue

        # Parse dataset and class
        dataset, class_name = parse_image_path(image_path)

        # Find model for this class
        model_key = f"{dataset}/{class_name}"

        # Try to get or load model
        if model_key not in model_cache:
            # Try different paths
            possible_paths = [
                models_dir / dataset / class_name / "model.onnx",
                models_dir / dataset / class_name / "efficientad.onnx",
                models_dir / f"{class_name}.onnx",
            ]

            model_path = None
            for p in possible_paths:
                if p.exists():
                    model_path = p
                    break

            if model_path is None:
                skipped += 1
                pbar.set_postfix({"done": processed, "skip": skipped, "err": errors})
                continue

            try:
                model = EfficientADOnnx(
                    model_path=model_path,
                    threshold=args.threshold,
                    device=args.device,
                )
                model.load_model()
                model_cache[model_key] = model
            except Exception as e:
                errors += 1
                pbar.set_postfix({"done": processed, "skip": skipped, "err": errors})
                continue

        model = model_cache[model_key]

        # Load image
        image_full_path = data_root / image_path
        if not image_full_path.exists():
            errors += 1
            pbar.set_postfix({"done": processed, "skip": skipped, "err": errors})
            continue

        try:
            image = cv2.imread(str(image_full_path))
            if image is None:
                errors += 1
                continue

            # Run inference
            result = model.predict(image)

            # Add class info to metadata
            result.metadata["dataset"] = dataset
            result.metadata["class_name"] = class_name

            # Convert to dict
            result_dict = result_to_dict(
                result,
                image_path,
                include_map_stats=args.include_map_stats,
            )
            results.append(result_dict)
            processed += 1

        except Exception as e:
            errors += 1

        pbar.set_postfix({"done": processed, "skip": skipped, "err": errors})

        # Save periodically
        if processed % 100 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    pbar.close()

    # Save final results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("Inference complete")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (no model): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_path}")

    # Print sample output
    if results:
        print()
        print("Sample output:")
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
