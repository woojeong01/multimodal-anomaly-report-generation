"""
MMAD Evaluation Metrics - matches paper's helper/summary.py implementation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


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
