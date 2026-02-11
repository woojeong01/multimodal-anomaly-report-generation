"""Run PatchCore anomaly detection inference using checkpoints (not ONNX).

Uses anomalib Patchcore model directly for inference.

Usage:
    python scripts/run_ad_inference_ckpt.py \
        --checkpoint-dir /path/to/checkpoints \
        --data-root /path/to/MMAD \
        --mmad-json /path/to/mmad_10classes.json \
        --output output/ad_predictions.json \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Disable HuggingFace online checks (prevents slow downloads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.utils.loaders import load_config
from src.utils.device import get_device


def parse_image_path(image_path: str) -> Tuple[str, str]:
    """Parse dataset and class name from image path."""
    parts = image_path.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def compute_defect_location(anomaly_map: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute defect location information from anomaly map."""
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

    coords = np.where(defect_mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    center_y = (y_min + y_max) / 2 / h
    center_x = (x_min + x_max) / 2 / w

    region_y = "top" if center_y < 1/3 else ("bottom" if center_y > 2/3 else "middle")
    region_x = "left" if center_x < 1/3 else ("right" if center_x > 2/3 else "center")

    if region_y == "middle" and region_x == "center":
        region = "center"
    else:
        region = f"{region_y}-{region_x}"

    area_ratio = float(defect_mask.sum()) / (h * w)
    confidence = float(anomaly_map[defect_mask].max())

    return {
        "has_defect": True,
        "region": region,
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "center": [round(center_x, 3), round(center_y, 3)],
        "area_ratio": round(area_ratio, 4),
        "confidence": round(confidence, 4),
    }


class PatchCoreCheckpointManager:
    """Manager for PatchCore checkpoint models."""

    def __init__(
        self,
        checkpoint_dir: Path,
        version: Optional[int] = None,
        threshold: float = 0.5,
        device: str = "cpu",
        input_size: Tuple[int, int] = (700, 700),
    ):
        self.checkpoint_dir = checkpoint_dir
        self.version = version
        self.threshold = threshold
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self._models: Dict[str, Any] = {}
        self._warmup_done: set = set()

    def _find_checkpoint(self, dataset: str, category: str) -> Optional[Path]:
        """Find checkpoint path for dataset/category."""
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if not patchcore_dir.exists():
            patchcore_dir = self.checkpoint_dir

        category_dir = patchcore_dir / dataset / category
        if not category_dir.exists():
            return None

        if self.version is not None:
            ckpt = category_dir / f"v{self.version}" / "model.ckpt"
            if ckpt.exists():
                return ckpt
            return None
        else:
            # Find latest version
            versions = []
            for v_dir in category_dir.iterdir():
                if v_dir.is_dir() and v_dir.name.startswith("v"):
                    try:
                        versions.append((int(v_dir.name[1:]), v_dir))
                    except ValueError:
                        continue
            if versions:
                latest = max(versions, key=lambda x: x[0])[1]
                ckpt = latest / "model.ckpt"
                if ckpt.exists():
                    return ckpt
        return None

    def get_model(self, dataset: str, category: str):
        """Get or load model for dataset/category."""
        from anomalib.models import Patchcore

        key = f"{dataset}/{category}"
        if key not in self._models:
            ckpt_path = self._find_checkpoint(dataset, category)
            if ckpt_path is None:
                raise FileNotFoundError(f"Checkpoint not found for {key}")

            model = Patchcore.load_from_checkpoint(str(ckpt_path), map_location="cpu")
            model.eval()
            model.to(self.device)
            self._models[key] = model

        return self._models[key]

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        h, w = self.input_size

        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return tensor.to(self.device)

    def warmup_model(self, dataset: str, category: str):
        """Warmup model with dummy input."""
        model = self.get_model(dataset, category)
        key = f"{dataset}/{category}"

        if key not in self._warmup_done:
            dummy = torch.randn(1, 3, self.input_size[0], self.input_size[1]).to(self.device)
            with torch.no_grad():
                _ = model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._warmup_done.add(key)

    def predict(self, dataset: str, category: str, image: np.ndarray) -> Dict[str, Any]:
        """Run inference on image."""
        model = self.get_model(dataset, category)

        # Preprocess
        input_tensor = self._preprocess(image)
        original_size = image.shape[:2]

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)

        # Extract results
        anomaly_map = getattr(outputs, "anomaly_map", None)
        pred_score = getattr(outputs, "pred_score", None)

        if anomaly_map is not None:
            # Resize to original size
            if anomaly_map.shape[-2:] != (original_size[0], original_size[1]):
                anomaly_map = interpolate(
                    anomaly_map,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False
                )
            anomaly_map = anomaly_map[0, 0].cpu().numpy()

        anomaly_score = float(pred_score[0].cpu()) if pred_score is not None else float(anomaly_map.max())
        is_anomaly = anomaly_score > self.threshold

        return {
            "anomaly_score": anomaly_score,
            "anomaly_map": anomaly_map,
            "is_anomaly": is_anomaly,
            "threshold": self.threshold,
        }

    def list_available_models(self) -> List[Tuple[str, str]]:
        """List available dataset/category pairs."""
        available = []
        patchcore_dir = self.checkpoint_dir / "Patchcore"
        if not patchcore_dir.exists():
            patchcore_dir = self.checkpoint_dir

        if not patchcore_dir.exists():
            return available

        for dataset_dir in patchcore_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            if dataset_dir.name in ["eval", "predictions"]:
                continue

            for category_dir in dataset_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                ckpt_path = self._find_checkpoint(dataset_dir.name, category_dir.name)
                if ckpt_path:
                    available.append((dataset_dir.name, category_dir.name))

        return sorted(available)

    def clear_cache(self):
        """Clear loaded models."""
        self._models.clear()
        self._warmup_done.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run PatchCore inference using checkpoints")

    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml", help="Config file for version/categories")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Inference device")

    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--mmad-json", type=str, required=True, help="Path to mmad.json")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")

    parser.add_argument("--output", type=str, default="output/ad_predictions.json", help="Output JSON path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    mmad_json = Path(args.mmad_json)
    output_path = Path(args.output)

    # Load config
    config = load_config(args.config)
    config_version = config.get("predict", {}).get("version")
    config_datasets = config.get("data", {}).get("datasets")
    config_categories = config.get("data", {}).get("categories")
    input_size = tuple(config.get("data", {}).get("image_size", [700, 700]))

    print(f"Config: {args.config}")
    print(f"  Version: v{config_version}" if config_version else "  Version: latest")
    print(f"  Input size: {input_size}")
    print(f"  Device: {args.device}")
    print()

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    if not mmad_json.exists():
        print(f"Error: MMAD JSON not found: {mmad_json}")
        sys.exit(1)

    # Load MMAD data
    print(f"Loading MMAD data from: {mmad_json}")
    with open(mmad_json, "r", encoding="utf-8") as f:
        mmad_data = json.load(f)

    image_paths = list(mmad_data.keys())

    # Filter by config datasets/categories
    if config_datasets or config_categories:
        filtered_paths = []
        for path in image_paths:
            dataset, category = parse_image_path(path)
            if config_datasets and dataset not in config_datasets:
                continue
            if config_categories and category not in config_categories:
                continue
            filtered_paths.append(path)
        image_paths = filtered_paths

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
    print(f"Initializing PatchCore from checkpoints: {checkpoint_dir}")
    model_manager = PatchCoreCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        version=config_version,
        threshold=args.threshold,
        device=args.device,
        input_size=input_size,
    )

    available_models = model_manager.list_available_models()
    print(f"Available models: {len(available_models)}")
    for dataset, category in available_models:
        print(f"  - {dataset}/{category}")

    # Group images by category for efficient processing (one model at a time)
    print()
    print("Grouping images by category...")
    images_by_category: Dict[Tuple[str, str], List[str]] = {}
    for image_path in image_paths:
        if image_path in existing_results:
            continue
        dataset, category = parse_image_path(image_path)
        if (dataset, category) not in available_models:
            continue
        key = (dataset, category)
        if key not in images_by_category:
            images_by_category[key] = []
        images_by_category[key].append(image_path)

    total_to_process = sum(len(v) for v in images_by_category.values())
    print(f"Images to process: {total_to_process} across {len(images_by_category)} categories")

    # Process images category by category
    results = list(existing_results.values())
    processed = len(existing_results)
    skipped = 0
    errors = 0
    total_inference_time = 0.0

    print()
    print("=" * 60)
    print("Running inference (one model at a time)")
    print("=" * 60)

    for (dataset, category), cat_images in images_by_category.items():
        cat_key = f"{dataset}/{category}"
        print(f"\n[{cat_key}] Loading model and processing {len(cat_images)} images...")

        # Load and warmup this model only
        try:
            model_manager.get_model(dataset, category)
            model_manager.warmup_model(dataset, category)
        except Exception as e:
            print(f"  Failed to load model: {e}")
            skipped += len(cat_images)
            continue

        cat_start = time.perf_counter()
        pbar = tqdm(cat_images, desc=f"  {cat_key}", ncols=100, leave=False)

        for image_path in pbar:
            image_full_path = data_root / image_path
            if not image_full_path.exists():
                errors += 1
                continue

            try:
                image = cv2.imread(str(image_full_path))
                if image is None:
                    errors += 1
                    continue

                t0 = time.perf_counter()
                result = model_manager.predict(dataset, category, image)
                infer_time = time.perf_counter() - t0
                total_inference_time += infer_time

                result_dict = {
                    "image_path": image_path,
                    "anomaly_score": round(float(result["anomaly_score"]), 4),
                    "is_anomaly": result["is_anomaly"],
                    "threshold": result["threshold"],
                }

                if result["anomaly_map"] is not None:
                    location_info = compute_defect_location(result["anomaly_map"], result["threshold"])
                    result_dict["defect_location"] = location_info
                    result_dict["map_stats"] = {
                        "max": round(float(result["anomaly_map"].max()), 4),
                        "mean": round(float(result["anomaly_map"].mean()), 4),
                        "std": round(float(result["anomaly_map"].std()), 4),
                    }

                results.append(result_dict)
                processed += 1

            except Exception as e:
                errors += 1

            pbar.set_postfix({"done": processed, "err": errors})

        pbar.close()
        cat_elapsed = time.perf_counter() - cat_start
        cat_ms_per_img = (cat_elapsed / len(cat_images) * 1000) if cat_images else 0
        print(f"  Done: {len(cat_images)} images in {cat_elapsed:.1f}s ({cat_ms_per_img:.0f}ms/img)")

        # Unload model to free GPU memory
        model_manager.clear_cache()

    # Save final results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    ms_per_img = (total_inference_time / processed * 1000) if processed > 0 else 0

    print()
    print("=" * 60)
    print("Inference complete")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (no model): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average: {ms_per_img:.1f}ms/img")
    print(f"Output saved to: {output_path}")

    if results:
        print()
        print("Sample output:")
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
