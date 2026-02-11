"""Evaluate PatchCore ONNX models on test set.

Faster evaluation using ONNX Runtime instead of Lightning Engine.

Usage:
    python scripts/eval_onnx.py \
        --models-dir models/onnx \
        --data-root /path/to/MMAD \
        --config configs/anomaly.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.anomaly import PatchCoreOnnx
from src.utils.loaders import load_config


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC from scores and binary labels."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(preds: list, targets: list) -> float:
    """Compute pixel-level AUROC."""
    from sklearn.metrics import roc_auc_score

    all_preds = []
    all_targets = []

    for pred, target in zip(preds, targets):
        all_preds.extend(pred.flatten())
        all_targets.extend((target > 0).astype(int).flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(np.unique(all_targets)) < 2:
        return 0.0

    return float(roc_auc_score(all_targets, all_preds))


def compute_pro(preds: np.ndarray, targets: np.ndarray, num_thresholds: int = 50) -> float:
    """Compute Per-Region Overlap (PRO) score."""
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_scores = []

    for threshold in thresholds:
        region_overlaps = []

        for pred, target in zip(preds, targets):
            if target.max() == 0:
                continue

            pred_binary = (pred >= threshold).astype(np.uint8)
            target_binary = (target > 0).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(target_binary)

            for label_id in range(1, num_labels):
                region_mask = (labels == label_id)
                region_area = region_mask.sum()

                if region_area == 0:
                    continue

                overlap = (pred_binary & region_mask).sum()
                overlap_ratio = overlap / region_area
                region_overlaps.append(overlap_ratio)

        if region_overlaps:
            pro_scores.append(np.mean(region_overlaps))

    return float(np.mean(pro_scores)) if pro_scores else 0.0


def load_test_data(data_root: Path, dataset: str, category: str):
    """Load test images and ground truth."""
    if dataset == "GoodsAD":
        cat_path = data_root / dataset / category
        test_dir = cat_path / "test"
        gt_dir = cat_path / "ground_truth"
        img_ext = "jpg"
        mask_ext = "png"
    elif dataset == "MVTec-LOCO":
        cat_path = data_root / dataset / category
        test_dir = cat_path / "test"
        gt_dir = cat_path / "ground_truth"
        img_ext = "png"
        mask_ext = "png"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    samples = []

    if not test_dir.exists():
        return samples

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir() or defect_dir.name.startswith("."):
            continue

        defect_type = defect_dir.name
        is_normal = defect_type == "good"

        for img_path in sorted(defect_dir.glob(f"*.{img_ext}")):
            if img_path.name.startswith("."):
                continue

            mask_path = None
            if not is_normal:
                if dataset == "GoodsAD":
                    mask_file = gt_dir / defect_type / f"{img_path.stem}.{mask_ext}"
                    if mask_file.exists():
                        mask_path = mask_file
                elif dataset == "MVTec-LOCO":
                    mask_dir = gt_dir / defect_type / img_path.stem
                    if mask_dir.exists() and mask_dir.is_dir():
                        mask_files = sorted(mask_dir.glob(f"*.{mask_ext}"))
                        if mask_files:
                            mask_path = mask_files[0]

            samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "is_anomaly": not is_normal,
                "defect_type": defect_type,
            })

    return samples


def evaluate_category(
    model: PatchCoreOnnx,
    samples: list,
    compute_pro_metric: bool = True,
) -> dict:
    """Evaluate model on samples."""
    scores = []
    labels = []
    pred_maps = []
    gt_masks = []

    for sample in tqdm(samples, desc="Inference", leave=False):
        # Load image
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            continue

        # Run inference
        result = model.predict(image)

        scores.append(result.anomaly_score)
        labels.append(1 if sample["is_anomaly"] else 0)

        # Load GT mask if exists
        if sample["mask_path"] is not None:
            gt_mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # Resize to match anomaly map
                if result.anomaly_map is not None:
                    gt_mask = cv2.resize(gt_mask, (result.anomaly_map.shape[1], result.anomaly_map.shape[0]))
                gt_masks.append(gt_mask)
                pred_maps.append(result.anomaly_map)
        elif not sample["is_anomaly"] and result.anomaly_map is not None:
            # Normal sample - use zero mask
            gt_masks.append(np.zeros_like(result.anomaly_map))
            pred_maps.append(result.anomaly_map)

    # Compute metrics
    scores = np.array(scores)
    labels = np.array(labels)

    metrics = {
        "image_AUROC": compute_auroc(scores, labels),
        "num_samples": len(samples),
        "num_anomaly": int(labels.sum()),
    }

    # Pixel-level metrics
    if pred_maps and gt_masks:
        metrics["pixel_AUROC"] = compute_pixel_auroc(pred_maps, gt_masks)

        if compute_pro_metric:
            preds_arr = np.array(pred_maps)
            targets_arr = np.array(gt_masks)
            metrics["PRO"] = compute_pro(preds_arr, targets_arr, num_thresholds=50)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX models")
    parser.add_argument("--models-dir", type=str, default="models/onnx")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-pro", action="store_true", help="Skip PRO computation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_root = Path(args.data_root)
    config = load_config(args.config)

    # Get categories to evaluate
    if args.dataset and args.category:
        categories = [(args.dataset, args.category)]
    else:
        datasets = config["data"].get("datasets", [])
        cat_filter = config["data"].get("categories", None)

        categories = []
        for dataset in datasets:
            dataset_path = models_dir / dataset
            if not dataset_path.exists():
                continue
            for cat_dir in sorted(dataset_path.iterdir()):
                if not cat_dir.is_dir():
                    continue
                if (cat_dir / "model.onnx").exists():
                    if cat_filter is None or cat_dir.name in cat_filter:
                        categories.append((dataset, cat_dir.name))

    print(f"Evaluating {len(categories)} models (ONNX)")
    print(f"Device: {args.device}")
    print()

    all_results = {}

    for dataset, category in tqdm(categories, desc="Evaluating", unit="category"):
        key = f"{dataset}/{category}"
        model_path = models_dir / dataset / category / "model.onnx"

        if not model_path.exists():
            print(f"  {key}: Model not found")
            continue

        # Load model
        model = PatchCoreOnnx(
            model_path=model_path,
            threshold=args.threshold,
            device=args.device,
        )
        model.load_model()

        # Load test data
        samples = load_test_data(data_root, dataset, category)
        if not samples:
            print(f"  {key}: No test data found")
            continue

        # Evaluate
        start = time.time()
        metrics = evaluate_category(model, samples, compute_pro_metric=not args.no_pro)
        elapsed = time.time() - start

        metrics["time"] = elapsed
        all_results[key] = metrics

        img_auroc = metrics.get("image_AUROC", 0)
        pixel_auroc = metrics.get("pixel_AUROC", 0)
        pro = metrics.get("PRO", 0)
        print(f"  {key}: I-AUROC={img_auroc:.4f} P-AUROC={pixel_auroc:.4f} PRO={pro:.4f} ({elapsed:.1f}s)")

    # Print summary
    if all_results:
        print("\n" + "=" * 75)
        print(f"{'Category':<35} {'I-AUROC':>10} {'P-AUROC':>10} {'PRO':>10} {'Time':>8}")
        print("=" * 75)

        for key, metrics in all_results.items():
            print(f"{key:<35} {metrics.get('image_AUROC', 0):>10.4f} "
                  f"{metrics.get('pixel_AUROC', 0):>10.4f} {metrics.get('PRO', 0):>10.4f} "
                  f"{metrics.get('time', 0):>7.1f}s")

        print("=" * 75)

        # Average
        n = len(all_results)
        avg_img = sum(m.get("image_AUROC", 0) for m in all_results.values()) / n
        avg_pix = sum(m.get("pixel_AUROC", 0) for m in all_results.values()) / n
        avg_pro = sum(m.get("PRO", 0) for m in all_results.values()) / n
        avg_time = sum(m.get("time", 0) for m in all_results.values()) / n
        print(f"{'Average':<35} {avg_img:>10.4f} {avg_pix:>10.4f} {avg_pro:>10.4f} {avg_time:>7.1f}s")

        total_time = sum(m.get("time", 0) for m in all_results.values())
        print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
