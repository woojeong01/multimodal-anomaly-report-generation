#!/usr/bin/env python
"""Find optimal threshold from predictions CSV.

Usage:
    python patchcore_training/scripts/find_optimal_threshold.py \
        --csv /path/to/patchcore_predictions.csv

    # Per-category thresholds
    python patchcore_training/scripts/find_optimal_threshold.py \
        --csv /path/to/patchcore_predictions.csv --per-category
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def load_predictions(csv_path: str) -> dict:
    """Load predictions from CSV file.

    Returns:
        Dictionary with scores, labels, and category info
    """
    data = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['dataset']}/{row['category']}"
            data["scores"].append(float(row["anomaly_score"]))
            data["labels"].append(int(row["gt_label"]))
            data["categories"].append(key)

    data["scores"] = np.array(data["scores"])
    data["labels"] = np.array(data["labels"])

    return data


def find_optimal_threshold_f1(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 100) -> dict:
    """Find optimal threshold that maximizes F1 score.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels (0=normal, 1=anomaly)
        n_thresholds: Number of thresholds to try

    Returns:
        Dictionary with optimal threshold and metrics
    """
    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score, max_score, n_thresholds)

    best_threshold = None
    best_f1 = 0
    best_metrics = {}

    for threshold in thresholds:
        preds = (scores > threshold).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "threshold": float(threshold),
                "f1": float(f1),
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
            }

    return best_metrics


def find_optimal_threshold_youden(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Find optimal threshold using Youden's J statistic (maximizes TPR - FPR).

    This is the point on ROC curve farthest from the diagonal.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels

    Returns:
        Dictionary with optimal threshold and metrics
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    best_threshold = thresholds[best_idx]
    preds = (scores > best_threshold).astype(int)

    return {
        "threshold": float(best_threshold),
        "youden_j": float(j_scores[best_idx]),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def find_percentile_threshold(scores: np.ndarray, labels: np.ndarray, percentile: float = 95) -> dict:
    """Find threshold based on percentile of normal samples.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        percentile: Percentile of normal scores to use as threshold

    Returns:
        Dictionary with threshold and metrics
    """
    normal_scores = scores[labels == 0]
    threshold = np.percentile(normal_scores, percentile)

    preds = (scores > threshold).astype(int)

    return {
        "threshold": float(threshold),
        "percentile": percentile,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def analyze_score_distribution(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Analyze score distribution for normal and anomaly samples."""
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    return {
        "normal": {
            "count": int(len(normal_scores)),
            "min": float(normal_scores.min()),
            "max": float(normal_scores.max()),
            "mean": float(normal_scores.mean()),
            "std": float(normal_scores.std()),
            "median": float(np.median(normal_scores)),
            "p95": float(np.percentile(normal_scores, 95)),
            "p99": float(np.percentile(normal_scores, 99)),
        },
        "anomaly": {
            "count": int(len(anomaly_scores)),
            "min": float(anomaly_scores.min()),
            "max": float(anomaly_scores.max()),
            "mean": float(anomaly_scores.mean()),
            "std": float(anomaly_scores.std()),
            "median": float(np.median(anomaly_scores)),
            "p5": float(np.percentile(anomaly_scores, 5)),
            "p25": float(np.percentile(anomaly_scores, 25)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Find optimal threshold from predictions CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--per-category", action="store_true", help="Compute per-category thresholds")
    parser.add_argument("--output-yaml", type=str, default=None,
                        help="Output path for thresholds YAML file (e.g., config/thresholds.yaml)")
    args = parser.parse_args()

    print(f"Loading predictions from: {args.csv}")
    data = load_predictions(args.csv)

    scores = data["scores"]
    labels = data["labels"]

    print(f"\nTotal samples: {len(scores)}")
    print(f"  Normal: {(labels == 0).sum()}")
    print(f"  Anomaly: {(labels == 1).sum()}")

    # Score distribution analysis
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION")
    print("=" * 70)

    dist = analyze_score_distribution(scores, labels)

    print(f"\nNormal samples (n={dist['normal']['count']}):")
    print(f"  Range: [{dist['normal']['min']:.4f}, {dist['normal']['max']:.4f}]")
    print(f"  Mean ± Std: {dist['normal']['mean']:.4f} ± {dist['normal']['std']:.4f}")
    print(f"  Median: {dist['normal']['median']:.4f}")
    print(f"  95th percentile: {dist['normal']['p95']:.4f}")
    print(f"  99th percentile: {dist['normal']['p99']:.4f}")

    print(f"\nAnomaly samples (n={dist['anomaly']['count']}):")
    print(f"  Range: [{dist['anomaly']['min']:.4f}, {dist['anomaly']['max']:.4f}]")
    print(f"  Mean ± Std: {dist['anomaly']['mean']:.4f} ± {dist['anomaly']['std']:.4f}")
    print(f"  Median: {dist['anomaly']['median']:.4f}")
    print(f"  5th percentile: {dist['anomaly']['p5']:.4f}")
    print(f"  25th percentile: {dist['anomaly']['p25']:.4f}")

    # Optimal thresholds
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS (Global)")
    print("=" * 70)

    # Method 1: F1 maximization
    f1_result = find_optimal_threshold_f1(scores, labels)
    print(f"\n1. F1 Score Maximization:")
    print(f"   Threshold: {f1_result['threshold']:.4f}")
    print(f"   F1: {f1_result['f1']:.4f}")
    print(f"   Accuracy: {f1_result['accuracy']:.4f}")
    print(f"   Precision: {f1_result['precision']:.4f}")
    print(f"   Recall: {f1_result['recall']:.4f}")

    # Method 2: Youden's J
    youden_result = find_optimal_threshold_youden(scores, labels)
    print(f"\n2. Youden's J Statistic (TPR - FPR):")
    print(f"   Threshold: {youden_result['threshold']:.4f}")
    print(f"   Youden's J: {youden_result['youden_j']:.4f}")
    print(f"   TPR (Recall): {youden_result['tpr']:.4f}")
    print(f"   FPR: {youden_result['fpr']:.4f}")
    print(f"   F1: {youden_result['f1']:.4f}")
    print(f"   Accuracy: {youden_result['accuracy']:.4f}")

    # Method 3: Percentile-based
    for pct in [90, 95, 99]:
        pct_result = find_percentile_threshold(scores, labels, pct)
        print(f"\n3-{pct}. Normal {pct}th Percentile:")
        print(f"   Threshold: {pct_result['threshold']:.4f}")
        print(f"   F1: {pct_result['f1']:.4f}")
        print(f"   Precision: {pct_result['precision']:.4f}")
        print(f"   Recall: {pct_result['recall']:.4f}")

    # Compare with threshold 3.0
    print("\n" + "-" * 70)
    print("Comparison with threshold = 3.0:")
    preds_3 = (scores > 3.0).astype(int)
    print(f"   F1: {f1_score(labels, preds_3, zero_division=0):.4f}")
    print(f"   Accuracy: {accuracy_score(labels, preds_3):.4f}")
    print(f"   Precision: {precision_score(labels, preds_3, zero_division=0):.4f}")
    print(f"   Recall: {recall_score(labels, preds_3, zero_division=0):.4f}")

    # Per-category analysis
    if args.per_category:
        print("\n" + "=" * 70)
        print("PER-CATEGORY OPTIMAL THRESHOLDS (F1 Maximization)")
        print("=" * 70)

        categories = list(set(data["categories"]))
        categories.sort()

        for cat in categories:
            mask = np.array([c == cat for c in data["categories"]])
            cat_scores = scores[mask]
            cat_labels = labels[mask]

            if len(np.unique(cat_labels)) < 2:
                print(f"\n{cat}: Skipped (single class)")
                continue

            cat_result = find_optimal_threshold_f1(cat_scores, cat_labels)
            cat_dist = analyze_score_distribution(cat_scores, cat_labels)

            print(f"\n{cat}:")
            print(f"  Samples: {len(cat_scores)} (normal: {(cat_labels == 0).sum()}, anomaly: {(cat_labels == 1).sum()})")
            print(f"  Normal range: [{cat_dist['normal']['min']:.4f}, {cat_dist['normal']['max']:.4f}]")
            print(f"  Anomaly range: [{cat_dist['anomaly']['min']:.4f}, {cat_dist['anomaly']['max']:.4f}]")
            print(f"  Optimal threshold: {cat_result['threshold']:.4f}")
            print(f"  F1: {cat_result['f1']:.4f}, Precision: {cat_result['precision']:.4f}, Recall: {cat_result['recall']:.4f}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Choose the method with best F1
    best_method = "F1 Maximization"
    best_threshold = f1_result['threshold']
    best_f1 = f1_result['f1']

    if youden_result['f1'] > best_f1:
        best_method = "Youden's J"
        best_threshold = youden_result['threshold']
        best_f1 = youden_result['f1']

    print(f"\nBest method: {best_method}")
    print(f"Recommended threshold: {best_threshold:.4f}")
    print(f"Expected F1: {best_f1:.4f}")

    print("\nNote: If you prioritize different goals:")
    print("  - High Precision (fewer false alarms): Use higher threshold")
    print("  - High Recall (catch more anomalies): Use lower threshold")
    print("  - Balance (Youden's J): Use TPR-FPR optimal point")

    # Save YAML if output path provided
    if args.output_yaml:
        import yaml

        # Compute per-category thresholds
        categories = list(set(data["categories"]))
        categories.sort()

        category_thresholds = {}
        for cat in categories:
            mask = np.array([c == cat for c in data["categories"]])
            cat_scores = scores[mask]
            cat_labels = labels[mask]

            if len(np.unique(cat_labels)) >= 2:
                cat_result = find_optimal_threshold_f1(cat_scores, cat_labels)
                category_thresholds[cat] = round(cat_result['threshold'], 2)

        thresholds_config = {
            "global": round(best_threshold, 2),
            "categories": category_thresholds,
        }

        output_path = Path(args.output_yaml)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Per-category optimal thresholds (F1 maximization)\n")
            f.write("# Auto-generated from predictions CSV\n\n")
            yaml.dump(thresholds_config, f, default_flow_style=False, allow_unicode=True)

        print(f"\nThresholds saved to: {output_path}")


if __name__ == "__main__":
    main()
