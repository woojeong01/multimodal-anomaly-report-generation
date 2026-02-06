"""PatchCore evaluator module."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from .dataset import get_dataloader
from .model import PatchCore


class PatchCoreEvaluator:
    """Evaluator for PatchCore models.

    Handles evaluation metrics computation for anomaly detection.
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device = None,
    ):
        """Initialize evaluator.

        Args:
            config: Configuration dictionary
            device: Device to use
        """
        self.config = config

        if device is None:
            device_str = config.get("device", "cuda")
            if device_str == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            elif device_str == "mps" and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # Data settings
        self.data_root = Path(config["data"]["root"])
        self.image_size = config["data"].get("image_size", 224)

        # Evaluation settings
        eval_config = config.get("evaluation", {})
        self.batch_size = eval_config.get("batch_size", 32)
        self.threshold = eval_config.get("threshold", 0.5)
        self.num_workers = config.get("training", {}).get("num_workers", 4)

        # Output settings
        output_config = config.get("output", {})
        self.checkpoint_dir = Path(output_config.get("checkpoint_dir", "checkpoints"))

    def evaluate_category(
        self,
        model: PatchCore,
        dataset_name: str,
        category: str,
        verbose: bool = True,
        threshold: float = None,
    ) -> Dict:
        """Evaluate model on a single category.

        Args:
            model: PatchCore model
            dataset_name: Dataset name
            category: Category name
            verbose: Print progress
            threshold: Optional category-specific threshold (overrides self.threshold)

        Returns:
            Dictionary with evaluation metrics
        """
        # Use provided threshold or fall back to self.threshold
        eval_threshold = threshold if threshold is not None else self.threshold

        if verbose:
            print(f"\nEvaluating: {dataset_name}/{category} (threshold={eval_threshold:.4f})")

        model.to(self.device)
        model.eval()

        # Create test dataloader
        dataloader = get_dataloader(
            root=self.data_root,
            dataset_name=dataset_name,
            category=category,
            split="test",
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            include_mask=True,
        )

        if len(dataloader.dataset) == 0:
            print(f"Warning: No test samples found for {dataset_name}/{category}")
            return {}

        # Collect predictions
        all_scores = []
        all_labels = []
        all_maps = []
        all_masks = []
        all_paths = []
        all_defect_types = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"  {dataset_name}/{category}", leave=False, mininterval=1.0):
                images = batch["image"].to(self.device)
                labels = batch["label"].numpy()
                masks = batch["mask"].numpy()
                paths = batch["image_path"]
                defect_types = batch["defect_type"]

                scores, maps = model.predict(images)

                all_scores.append(scores.cpu().numpy())
                all_labels.append(labels)
                all_maps.append(maps.cpu().numpy())
                all_masks.append(masks)
                all_paths.extend(paths)
                all_defect_types.extend(defect_types)

        # Concatenate results
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        all_maps = np.concatenate(all_maps)
        all_masks = np.concatenate(all_masks).squeeze(1)  # Remove channel dim

        # Store predictions for CSV export
        self._last_predictions = {
            "paths": all_paths,
            "scores": all_scores,
            "labels": all_labels,
            "defect_types": all_defect_types,
            "dataset": dataset_name,
            "category": category,
        }

        # Compute image-level metrics
        metrics = self._compute_image_metrics(all_scores, all_labels, eval_threshold)

        # Compute pixel-level metrics if masks available
        if all_masks.max() > 0:
            pixel_metrics = self._compute_pixel_metrics(all_maps, all_masks, eval_threshold)
            metrics.update(pixel_metrics)

        metrics["threshold_used"] = eval_threshold

        metrics["n_samples"] = len(all_labels)
        metrics["n_anomaly"] = int(all_labels.sum())
        metrics["n_normal"] = int((1 - all_labels).sum())

        if verbose:
            print(f"  Image AUROC: {metrics.get('image_auroc', 0):.4f}")
            print(f"  Image F1: {metrics.get('image_f1', 0):.4f}")
            if "pixel_auroc" in metrics:
                print(f"  Pixel AUROC: {metrics['pixel_auroc']:.4f}")

        return metrics

    def _compute_image_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float = None,
    ) -> Dict:
        """Compute image-level metrics.

        Args:
            scores: Anomaly scores (N,)
            labels: Ground truth labels (N,)
            threshold: Threshold for binary classification (raw score, not normalized)

        Returns:
            Dictionary with metrics
        """
        # Use raw score threshold (not normalized)
        eval_threshold = threshold if threshold is not None else self.threshold
        preds = (scores > eval_threshold).astype(int)

        metrics = {
            "image_auroc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0,
            "image_accuracy": accuracy_score(labels, preds),
            "image_precision": precision_score(labels, preds, zero_division=0),
            "image_recall": recall_score(labels, preds, zero_division=0),
            "image_f1": f1_score(labels, preds, zero_division=0),
        }

        return metrics

    def _compute_pixel_metrics(
        self,
        maps: np.ndarray,
        masks: np.ndarray,
        threshold: float = None,
    ) -> Dict:
        """Compute pixel-level metrics.

        Args:
            maps: Anomaly maps (N, H, W)
            masks: Ground truth masks (N, H, W)
            threshold: Raw score threshold (will be normalized for pixel-level)

        Returns:
            Dictionary with pixel metrics
        """
        # Flatten for pixel-level evaluation
        maps_flat = maps.flatten()
        masks_flat = (masks > 0).astype(int).flatten()

        # Only compute if there are both positive and negative pixels
        if len(np.unique(masks_flat)) < 2:
            return {}

        # For pixel-level, normalize maps and threshold
        # Use 0.5 as default normalized threshold for pixel metrics
        maps_norm = (maps_flat - maps_flat.min()) / (maps_flat.max() - maps_flat.min() + 1e-8)
        pixel_threshold = 0.5  # Fixed normalized threshold for pixel-level
        preds_flat = (maps_norm > pixel_threshold).astype(int)

        metrics = {
            "pixel_auroc": roc_auc_score(masks_flat, maps_flat),
            "pixel_f1": f1_score(masks_flat, preds_flat, zero_division=0),
            "pixel_precision": precision_score(masks_flat, preds_flat, zero_division=0),
            "pixel_recall": recall_score(masks_flat, preds_flat, zero_division=0),
        }

        return metrics

    def evaluate_all(
        self,
        models: Dict[str, PatchCore],
        verbose: bool = True,
        save_predictions_csv: str = None,
        per_category_thresholds: Dict[str, float] = None,
    ) -> Dict[str, Dict]:
        """Evaluate all loaded models.

        Args:
            models: Dictionary mapping "dataset/category" to models
            verbose: Print progress
            save_predictions_csv: Path to save per-sample predictions CSV
            per_category_thresholds: Optional dict mapping "dataset/category" to threshold

        Returns:
            Dictionary mapping "dataset/category" to metrics
        """
        results = {}
        all_predictions = []  # For CSV export
        total = len(models)

        if per_category_thresholds:
            print(f"\nEvaluating {total} models with per-category thresholds")
        else:
            print(f"\nEvaluating {total} models (threshold={self.threshold})")

        pbar = tqdm(models.items(), desc="Evaluating", total=total)
        for key, model in pbar:
            # Get category-specific threshold if available
            category_threshold = None
            if per_category_thresholds:
                category_threshold = per_category_thresholds.get(key)

            pbar.set_description(f"Evaluating {key}")

            parts = key.split("/")
            dataset_name, category = parts[0], parts[1]

            metrics = self.evaluate_category(
                model=model,
                dataset_name=dataset_name,
                category=category,
                verbose=False,  # Suppress individual category output for cleaner progress
                threshold=category_threshold,
            )

            results[key] = metrics

            # Collect predictions for CSV
            if hasattr(self, '_last_predictions') and self._last_predictions:
                pred = self._last_predictions
                # Use the threshold that was actually used for this category
                used_threshold = metrics.get("threshold_used", self.threshold)
                for i in range(len(pred["paths"])):
                    score = float(pred["scores"][i])
                    pred_label = int(score > used_threshold)
                    all_predictions.append({
                        "image_path": pred["paths"][i],
                        "dataset": pred["dataset"],
                        "category": pred["category"],
                        "defect_type": pred["defect_types"][i],
                        "gt_label": int(pred["labels"][i]),  # 0=normal, 1=anomaly
                        "anomaly_score": score,
                        "threshold_used": used_threshold,
                        "pred_label": pred_label,
                        "correct": int(pred["labels"][i]) == pred_label,
                    })

            # Show AUROC in progress bar
            auroc = metrics.get('image_auroc', 0)
            pbar.set_postfix({"AUROC": f"{auroc:.4f}"})

        # Save predictions CSV
        if save_predictions_csv and all_predictions:
            self._save_predictions_csv(all_predictions, save_predictions_csv)

        # Compute average metrics
        if results:
            avg_metrics = self._compute_average_metrics(results)
            results["average"] = avg_metrics

            print(f"\n{'='*60}")
            print("Average Metrics:")
            print(f"  Image AUROC: {avg_metrics.get('image_auroc', 0):.4f}")
            print(f"  Image F1: {avg_metrics.get('image_f1', 0):.4f}")
            if "pixel_auroc" in avg_metrics:
                print(f"  Pixel AUROC: {avg_metrics['pixel_auroc']:.4f}")
            print(f"{'='*60}")

        return results

    def _save_predictions_csv(self, predictions: List[Dict], output_path: str) -> None:
        """Save per-sample predictions to CSV.

        Args:
            predictions: List of prediction dictionaries
            output_path: Output CSV path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "image_path", "dataset", "category", "defect_type",
            "gt_label", "anomaly_score", "threshold_used", "pred_label", "correct"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

        print(f"Predictions saved to: {output_path}")

    def _compute_average_metrics(self, results: Dict[str, Dict]) -> Dict:
        """Compute average metrics across all categories.

        Args:
            results: Dictionary of per-category metrics

        Returns:
            Dictionary with averaged metrics
        """
        metric_keys = [
            "image_auroc", "image_accuracy", "image_precision",
            "image_recall", "image_f1", "pixel_auroc", "pixel_f1",
        ]

        avg_metrics = {}
        total_samples = 0

        for key, metrics in results.items():
            if key == "average":
                continue
            total_samples += metrics.get("n_samples", 0)

        for metric_key in metric_keys:
            values = []
            for key, metrics in results.items():
                if key == "average":
                    continue
                if metric_key in metrics:
                    values.append(metrics[metric_key])

            if values:
                avg_metrics[metric_key] = np.mean(values)

        avg_metrics["total_samples"] = total_samples

        return avg_metrics

    def save_results(self, results: Dict[str, Dict], output_path: str) -> None:
        """Save evaluation results to JSON.

        Args:
            results: Evaluation results dictionary
            output_path: Output file path
        """
        import json

        # Convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        results_serializable = convert_to_serializable(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
