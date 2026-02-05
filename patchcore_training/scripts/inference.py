#!/usr/bin/env python
"""Run PatchCore inference and generate JSON for LLM evaluation.

This script processes all images in mmad_10classes.json and outputs
anomaly detection results in a format suitable for LLM evaluation.

Usage:
    # Run inference on all images
    python patchcore_training/scripts/inference.py

    # Run with custom config and output
    python patchcore_training/scripts/inference.py \
        --config patchcore_training/config/config.yaml \
        --output output/patchcore_predictions.json

    # Test with limited images
    python patchcore_training/scripts/inference.py --max-images 100

    # Resume from existing output
    python patchcore_training/scripts/inference.py --resume
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add patchcore_training to path
SCRIPT_DIR = Path(__file__).resolve().parent
PATCHCORE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PATCHCORE_ROOT))

from src.model import PatchCore
from src.utils import load_config, setup_seed, get_device, normalize_anomaly_map, compute_defect_location
from src.dataset import InferenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore inference for LLM evaluation")

    parser.add_argument(
        "--config",
        type=str,
        default=str(PATCHCORE_ROOT / "config" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (overrides config)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly score threshold (overrides config). Use score statistics to tune.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_models(config: Dict, device: torch.device) -> Dict[str, PatchCore]:
    """Load all trained PatchCore models.

    Args:
        config: Configuration dictionary
        device: Device to load models to

    Returns:
        Dictionary mapping "dataset/category" to models
    """
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    datasets_config = config["data"].get("datasets", {})

    # Count total categories
    all_categories = [(d, c) for d, cats in datasets_config.items() for c in cats]

    models = {}

    for dataset_name, category in tqdm(all_categories, desc="Loading models"):
        pt_path = checkpoint_dir / dataset_name / category / "model.pt"

        if pt_path.exists():
            model = PatchCore.load(str(pt_path), device)
            model.eval()
            key = f"{dataset_name}/{category}"
            models[key] = model
        # Silently skip missing models (will be reported during inference)

    print(f"Loaded {len(models)}/{len(all_categories)} models")
    return models


def process_batch(
    model: PatchCore,
    images: torch.Tensor,
    original_sizes: List[tuple],
    score_threshold: float,
    device: torch.device,
) -> List[Dict]:
    """Process a batch of images.

    Args:
        model: PatchCore model
        images: Batch of images (B, C, H, W)
        original_sizes: List of original image sizes
        score_threshold: Anomaly score threshold (raw score 기준)
        device: Device

    Returns:
        List of result dictionaries
    """
    images = images.to(device)

    with torch.no_grad():
        scores, maps = model.predict(images)

    scores = scores.cpu().numpy()
    maps = maps.cpu().numpy()

    results = []
    for i in range(len(scores)):
        anomaly_map = maps[i]
        score = float(scores[i])

        # Raw map stats (정규화 전)
        raw_max = float(anomaly_map.max())
        raw_mean = float(anomaly_map.mean())
        raw_std = float(anomaly_map.std())

        # Normalize map for visualization (0-1)
        anomaly_map_norm = normalize_anomaly_map(anomaly_map)

        # Resize map to original size
        orig_h, orig_w = original_sizes[i]
        if orig_h > 0 and orig_w > 0:
            anomaly_map_resized = cv2.resize(
                anomaly_map_norm,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            anomaly_map_resized = anomaly_map_norm

        # is_anomaly는 raw score 기준으로 판단 (threshold는 나중에 조정 가능)
        is_anomaly = score > score_threshold

        # Defect location은 normalized map에서 상위 영역 찾기
        # threshold를 높게 설정 (0.7)해서 확실한 영역만 검출
        location_info = compute_defect_location(anomaly_map_norm, 0.7)

        result = {
            "anomaly_score": round(score, 6),
            "is_anomaly": is_anomaly,
            "threshold_used": score_threshold,
            "defect_location": location_info,
            "map_stats": {
                "raw_max": round(raw_max, 4),
                "raw_mean": round(raw_mean, 4),
                "raw_std": round(raw_std, 4),
                "norm_max": round(float(anomaly_map_norm.max()), 4),
            },
        }

        results.append(result)

    return results


def main():
    args = parse_args()

    # Set seed
    setup_seed(args.seed)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Get device
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    # Paths
    data_root = Path(config["data"]["root"])
    mmad_json_path = Path(config["data"]["mmad_json"])
    output_path = Path(args.output or config["output"]["inference_output"])

    # Threshold (command line > config > default)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = config.get("evaluation", {}).get("threshold", 0.5)

    print(f"Anomaly score threshold: {threshold}")

    if not mmad_json_path.exists():
        print(f"Error: mmad.json not found: {mmad_json_path}")
        return

    # Load MMAD data
    print(f"Loading MMAD data from: {mmad_json_path}")
    with open(mmad_json_path, "r", encoding="utf-8") as f:
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

    # Filter out already processed
    remaining_paths = [p for p in image_paths if p not in existing_results]
    print(f"Remaining images: {len(remaining_paths)}")

    if not remaining_paths:
        print("All images already processed!")
        return

    # Load models
    print("\nLoading models...")
    models = load_models(config, device)

    if not models:
        print("No trained models found!")
        return

    # Group images by category
    images_by_category = {}
    for path in remaining_paths:
        parts = path.split("/")
        if len(parts) >= 2:
            key = f"{parts[0]}/{parts[1]}"
            if key not in images_by_category:
                images_by_category[key] = []
            images_by_category[key].append(path)

    # Process images
    results = list(existing_results.values())
    processed = len(existing_results)
    skipped = 0
    errors = 0

    print(f"\n{'='*60}")
    print("Running inference")
    print(f"{'='*60}")

    category_pbar = tqdm(images_by_category.items(), desc="Categories", position=0)
    for category_key, paths in category_pbar:
        if category_key not in models:
            skipped += len(paths)
            category_pbar.set_postfix({"status": "skipped (no model)"})
            continue

        model = models[category_key]
        dataset_name, category = category_key.split("/")

        category_pbar.set_description(f"Processing {category_key}")

        # Create dataset and dataloader
        dataset = InferenceDataset(
            data_root=data_root,
            image_paths=paths,
            image_size=config["data"].get("image_size", 224),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        for batch in tqdm(dataloader, desc=f"  {category_key}", position=1, leave=False, mininterval=1.0):
            valid_mask = batch["valid"]

            if not valid_mask.any():
                errors += (~valid_mask).sum().item()
                continue

            # Get valid images only
            valid_indices = torch.where(valid_mask)[0]
            images = batch["image"][valid_indices]

            # original_size는 DataLoader가 (heights_tensor, widths_tensor)로 collate함
            heights, widths = batch["original_size"]
            original_sizes = [
                (int(heights[i]), int(widths[i]))
                for i in valid_indices.tolist()
            ]

            # Process batch
            try:
                batch_results = process_batch(
                    model=model,
                    images=images,
                    original_sizes=original_sizes,
                    score_threshold=threshold,
                    device=device,
                )

                # Add metadata and append results
                for idx, result in zip(valid_indices.tolist(), batch_results):
                    result["image_path"] = batch["image_path"][idx]
                    result["metadata"] = {
                        "dataset": dataset_name,
                        "class_name": category,
                        "model_type": "patchcore",
                    }
                    results.append(result)
                    processed += 1

            except Exception as e:
                print(f"\nError: {e}")
                errors += len(valid_indices)

            # Handle invalid images
            invalid_count = (~valid_mask).sum().item()
            errors += invalid_count

        category_pbar.set_postfix({"done": processed, "skip": skipped, "err": errors})

        # Save periodically
        if processed % 500 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Save final results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Inference complete")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Skipped (no model): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_path}")

    # Score 분포 통계 출력
    if results:
        all_scores = [r["anomaly_score"] for r in results]
        scores_array = np.array(all_scores)

        print(f"\n{'='*60}")
        print("Score Statistics (for threshold tuning)")
        print(f"{'='*60}")
        print(f"  Min:    {scores_array.min():.4f}")
        print(f"  Max:    {scores_array.max():.4f}")
        print(f"  Mean:   {scores_array.mean():.4f}")
        print(f"  Std:    {scores_array.std():.4f}")
        print(f"  Median: {np.median(scores_array):.4f}")
        print(f"\n  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"    {p}%: {np.percentile(scores_array, p):.4f}")

        # 현재 threshold로 분류된 결과
        n_anomaly = sum(1 for r in results if r["is_anomaly"])
        print(f"\n  Current threshold ({threshold}): {n_anomaly}/{len(results)} classified as anomaly")

        print(f"\n  Tip: 정상 데이터가 anomaly로 분류되면 threshold를 높이세요.")
        print(f"       예: --threshold 값을 median~75% 사이로 설정")

    # Print sample output
    if results:
        print("\nSample output:")
        print(json.dumps(results[-1], indent=2))


if __name__ == "__main__":
    main()
