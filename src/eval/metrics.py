"""
Evaluation Metrics for Anomaly Detection.

1. MMAD Evaluation: calculate_accuracy_mmad()
2. Anomaly Detection Metrics: compute_anomaly_metrics()
3. Threshold Optimization: find_optimal_threshold()
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    precision_recall_curve,
)
from scipy.ndimage import label as connected_components


# ============================================================
# Anomaly Detection Metrics
# ============================================================

def find_optimal_threshold(
    predictions: dict,
    level: str = "image",
    num_thresholds: int = 200,
) -> dict:
    """카테고리별 F1 최대화 기반 optimal threshold 탐색.

    Args:
        predictions: {category: [batch, ...]} 형태의 예측 결과
        level: "image" (이미지 단위) 또는 "pixel" (픽셀 단위)
        num_thresholds: threshold 탐색 개수

    Returns:
        {category: {"threshold": float, "f1": float}} 딕셔너리
    """
    thresholds = {}

    for category, batches in predictions.items():
        if level == "image":
            y_true = np.concatenate([b.gt_label.cpu().numpy() for b in batches])
            y_score = np.concatenate([b.pred_score.cpu().numpy() for b in batches])
        elif level == "pixel":
            gt_masks = torch.cat([b.gt_mask for b in batches])
            anomaly_maps = torch.cat([b.anomaly_map for b in batches])
            y_true = gt_masks.flatten().cpu().numpy().astype(int)
            y_score = anomaly_maps.flatten().cpu().numpy().astype(float)
        else:
            raise ValueError(f"Unknown level: {level}. Use 'image' or 'pixel'.")

        if len(np.unique(y_true)) < 2:
            thresholds[category] = {"threshold": 0.5, "f1": 0.0}
            continue

        precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns n+1 precision/recall but n thresholds
        f1_arr = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
        best_idx = np.argmax(f1_arr)

        thresholds[category] = {
            "threshold": float(thresh_arr[best_idx]),
            "f1": float(f1_arr[best_idx]),
        }

    return thresholds


def compute_pro(gt_masks_np, anomaly_maps_np, num_thresholds=200):
    """Per-Region Overlap (PRO) 계산.

    각 GT connected component별로 overlap을 계산하고,
    FPR에 대해 적분하여 정규화한 값을 반환.

    Args:
        gt_masks_np: (N, H, W) numpy array, 0/1 GT 마스크
        anomaly_maps_np: (N, H, W) numpy array, anomaly score map
        num_thresholds: threshold 개수

    Returns:
        PRO score (float)
    """
    thresholds = np.linspace(anomaly_maps_np.max(), anomaly_maps_np.min(), num_thresholds)

    pro_values = []
    fpr_values = []

    for thresh in thresholds:
        pred_binary = (anomaly_maps_np >= thresh).astype(int)

        gt_neg = (gt_masks_np == 0)
        fp = np.sum(pred_binary[gt_neg])
        fpr = fp / max(np.sum(gt_neg), 1)

        region_overlaps = []
        for i in range(len(gt_masks_np)):
            labeled, num_regions = connected_components(gt_masks_np[i])
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled == region_id)
                region_size = np.sum(region_mask)
                if region_size == 0:
                    continue
                overlap = np.sum(pred_binary[i][region_mask]) / region_size
                region_overlaps.append(overlap)

        if region_overlaps:
            pro_values.append(np.mean(region_overlaps))
            fpr_values.append(fpr)

    if len(pro_values) < 2:
        return 0.0

    fpr_limit = 0.3
    filtered = [(f, p) for f, p in zip(fpr_values, pro_values) if f <= fpr_limit]
    if len(filtered) < 2:
        return 0.0

    fpr_filtered, pro_filtered = zip(*sorted(filtered))
    pro_auc = np.trapz(pro_filtered, fpr_filtered) / fpr_limit
    return pro_auc


def compute_anomaly_metrics(
    predictions: dict,
    image_thresholds: dict = None,
    pixel_thresholds: dict = None,
    pro_num_thresholds: int = 200,
) -> pd.DataFrame:
    """Anomaly detection 전체 지표 산출.

    Args:
        predictions: {category: [batch, ...]} 형태의 예측 결과
        image_thresholds: {category: {"threshold": float}} image-level threshold.
                          None이면 0.5 사용.
        pixel_thresholds: {category: {"threshold": float}} pixel-level threshold.
                          None이면 0.5 사용.
        pro_num_thresholds: PRO 계산 시 threshold 개수

    Returns:
        카테고리별 지표 DataFrame
    """
    results = []

    for category, batches in predictions.items():
        y_true = np.concatenate([b.gt_label.cpu().numpy() for b in batches])
        y_score = np.concatenate([b.pred_score.cpu().numpy() for b in batches])

        metrics = {
            "Category": category,
            "Image_AUROC": round(roc_auc_score(y_true, y_score), 4),
            "N_samples": len(y_true),
        }

        # Pixel-level
        if batches[0].gt_mask is not None:
            gt_masks = torch.cat([b.gt_mask for b in batches])
            anomaly_maps = torch.cat([b.anomaly_map for b in batches])

            gt_masks_np = gt_masks.cpu().numpy().astype(int)
            anomaly_maps_np = anomaly_maps.cpu().numpy().astype(float)

            # Pixel AUROC
            gt_flat = gt_masks_np.flatten()
            pred_flat = anomaly_maps_np.flatten()
            if len(np.unique(gt_flat)) > 1:
                metrics["Pixel_AUROC"] = round(roc_auc_score(gt_flat, pred_flat), 4)
            else:
                metrics["Pixel_AUROC"] = float("nan")

            # PRO
            gt_for_pro = gt_masks_np.squeeze(1) if gt_masks_np.ndim == 4 else gt_masks_np
            amap_for_pro = anomaly_maps_np.squeeze(1) if anomaly_maps_np.ndim == 4 else anomaly_maps_np
            metrics["PRO"] = round(compute_pro(gt_for_pro, amap_for_pro, pro_num_thresholds), 4)

            # Pixel-level threshold
            px_thresh = 0.5
            if pixel_thresholds and category in pixel_thresholds:
                px_thresh = pixel_thresholds[category]["threshold"]

            pred_binary = (anomaly_maps > px_thresh).int()
            metrics["Dice"] = round(
                f1_score(gt_flat, pred_binary.flatten().cpu().numpy(), zero_division=0), 4
            )
            metrics["IoU"] = round(
                jaccard_score(gt_flat, pred_binary.flatten().cpu().numpy(), average="binary", zero_division=0), 4
            )

        results.append(metrics)

    metrics_df = pd.DataFrame(results).set_index("Category")
    avg_row = metrics_df.drop(columns=["N_samples"], errors="ignore").mean().round(4)
    avg_row["N_samples"] = metrics_df["N_samples"].sum()
    metrics_df.loc["Average"] = avg_row
    col_order = [c for c in metrics_df.columns if c != "N_samples"] + ["N_samples"]
    metrics_df = metrics_df[col_order]

    return metrics_df


# ============================================================
# MMAD Evaluation Metrics
# ============================================================


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset names (merge DS-MVTec and MVTec-AD)."""
    if dataset_name in ['DS-MVTec', 'MVTec-AD']:
        return 'MVTec-AD'
    return dataset_name


def calculate_accuracy_mmad(
    answers_json_path: str,
    normal_flag: str = 'good',
    show_overkill_miss: bool = False,
    save_csv: bool = True,
    show_plot: bool = False,
) -> pd.DataFrame:
    """Calculate MMAD evaluation metrics - matches paper's caculate_accuracy_mmad().

    Args:
        answers_json_path: Path to answers JSON file
        normal_flag: String to identify normal samples in image path
        show_overkill_miss: Whether to show overkill/miss rates
        save_csv: Whether to save accuracy CSV
        show_plot: Whether to show heatmap plot

    Returns:
        DataFrame with accuracy metrics
    """
    if not os.path.exists(answers_json_path):
        raise FileNotFoundError(f"Answers file not found: {answers_json_path}")

    with open(answers_json_path, "r", encoding="utf-8") as f:
        all_answers = json.load(f)

    if not all_answers:
        print("No answers found in file")
        return pd.DataFrame()

    # Collect dataset names and question types
    dataset_names = []
    type_list = []

    for answer in all_answers:
        dataset_name = normalize_dataset_name(answer['image'].split('/')[0])
        question_type = answer.get('question_type', 'unknown')

        # Merge Object Structure and Object Details into Object Analysis
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"

        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        if question_type not in type_list:
            type_list.append(question_type)

    # Initialize statistics
    question_stats: Dict[str, Dict] = {ds: {} for ds in dataset_names}
    detection_stats: Dict[str, Dict] = {ds: {} for ds in dataset_names}

    for dataset_name in dataset_names:
        detection_stats[dataset_name]['normal'] = {'total': 0, 'correct': 0}
        detection_stats[dataset_name]['abnormal'] = {'total': 0, 'correct': 0}
        for question_type in type_list:
            question_stats[dataset_name][question_type] = {'total': 0, 'correct': 0}

    # Process answers
    valid_answers = ['A', 'B', 'C', 'D', 'E']
    removed_count = 0

    for answer in all_answers:
        dataset_name = normalize_dataset_name(answer['image'].split('/')[0])
        question_type = answer.get('question_type', 'unknown')

        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"

        gpt_answer = answer.get('gpt_answer', '')
        correct_answer = answer.get('correct_answer', '')

        # Validate answers
        if correct_answer not in valid_answers or gpt_answer not in valid_answers:
            removed_count += 1
            continue

        # Update question stats
        question_stats[dataset_name][question_type]['total'] += 1
        if correct_answer == gpt_answer:
            question_stats[dataset_name][question_type]['correct'] += 1

        # Update detection stats for Anomaly Detection questions
        if question_type == "Anomaly Detection":
            if normal_flag in answer['image']:
                detection_stats[dataset_name]['normal']['total'] += 1
                if correct_answer == gpt_answer:
                    detection_stats[dataset_name]['normal']['correct'] += 1
            else:
                detection_stats[dataset_name]['abnormal']['total'] += 1
                if correct_answer == gpt_answer:
                    detection_stats[dataset_name]['abnormal']['correct'] += 1

    if removed_count > 0:
        print(f"Removed {removed_count} invalid answers")

    # Create accuracy DataFrame
    accuracy_df = pd.DataFrame(index=dataset_names)

    for dataset_name in dataset_names:
        for question_type in type_list:
            total = question_stats[dataset_name][question_type]['total']
            correct = question_stats[dataset_name][question_type]['correct']
            accuracy = (correct / total * 100) if total > 0 else 0
            accuracy_df.at[dataset_name, question_type] = accuracy

        # Special handling for Anomaly Detection (use balanced accuracy)
        if 'Anomaly Detection' in type_list:
            normal_total = detection_stats[dataset_name]['normal']['total']
            normal_correct = detection_stats[dataset_name]['normal']['correct']
            abnormal_total = detection_stats[dataset_name]['abnormal']['total']
            abnormal_correct = detection_stats[dataset_name]['abnormal']['correct']

            normal_acc = (normal_correct / normal_total) if normal_total > 0 else 0
            anomaly_acc = (abnormal_correct / abnormal_total) if abnormal_total > 0 else 0

            # Balanced accuracy (average of normal and anomaly accuracy)
            accuracy_df.at[dataset_name, 'Anomaly Detection'] = (normal_acc + anomaly_acc) / 2 * 100

            # Calculate TP, FP, FN, TN for other metrics
            TP = abnormal_correct
            FP = normal_total - normal_correct
            FN = abnormal_total - abnormal_correct
            TN = normal_correct

            Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

            accuracy_df.at[dataset_name, 'Recall'] = Recall * 100
            accuracy_df.at[dataset_name, 'Precision'] = Precision * 100
            accuracy_df.at[dataset_name, 'F1'] = F1 * 100

        # Calculate average for this dataset
        metric_cols = [c for c in accuracy_df.columns if c not in ['Recall', 'Precision', 'F1', 'Overkill', 'Miss']]
        accuracy_df.at[dataset_name, 'Average'] = accuracy_df.loc[dataset_name, metric_cols].mean()

    # Add overkill/miss if requested
    if show_overkill_miss:
        for dataset_name in dataset_names:
            normal_total = detection_stats[dataset_name]['normal']['total']
            normal_correct = detection_stats[dataset_name]['normal']['correct']
            abnormal_total = detection_stats[dataset_name]['abnormal']['total']
            abnormal_correct = detection_stats[dataset_name]['abnormal']['correct']

            normal_acc = (normal_correct / normal_total) if normal_total > 0 else 0
            anomaly_acc = (abnormal_correct / abnormal_total) if abnormal_total > 0 else 0

            accuracy_df.at[dataset_name, 'Overkill'] = (1 - normal_acc) * 100
            accuracy_df.at[dataset_name, 'Miss'] = (1 - anomaly_acc) * 100

    # Add average row
    accuracy_df.loc['Average'] = accuracy_df.mean()

    # Print results
    print(f"\n=== Accuracy Statistics for {os.path.basename(answers_json_path)} ===\n")
    print(accuracy_df.to_string())
    print(f"\n=== Overall Average ===")
    print(accuracy_df.loc['Average'])

    # Save CSV
    if save_csv:
        csv_path = answers_json_path.replace('.json', '_accuracy.csv')
        accuracy_df.to_csv(csv_path)
        print(f"\nSaved accuracy CSV to: {csv_path}")

    # Show plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(10, 7))
            sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
            plt.title(f'Accuracy: {os.path.basename(answers_json_path).replace(".json", "")}')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Warning: matplotlib/seaborn not available for plotting")

    return accuracy_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--answers-json', type=str, required=True,
                        help='Path to answers JSON file')
    parser.add_argument('--normal-flag', type=str, default='good',
                        help='String to identify normal samples')
    parser.add_argument('--show-overkill-miss', action='store_true',
                        help='Show overkill/miss rates')
    parser.add_argument('--show-plot', action='store_true',
                        help='Show heatmap plot')

    args = parser.parse_args()

    calculate_accuracy_mmad(
        args.answers_json,
        normal_flag=args.normal_flag,
        show_overkill_miss=args.show_overkill_miss,
        show_plot=args.show_plot,
    )
