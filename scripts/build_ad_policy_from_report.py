#!/usr/bin/env python3
"""Build data-driven AD policy/calibration from a full AD report JSON.

Input:
    - JSON produced by `scripts/run_ad_inference.py` full-report mode.

Output:
    1) policy JSON for `--policy-json`
    2) calibration JSON for `--calibration-json`
    3) diagnostic summary JSON (per-class evidence)

Ground-truth reconstruction rule:
    - `.../test/good/...` -> normal (label=0)
    - `.../test/<anything_else>/...` -> anomaly (label=1)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_gt_from_image_path(image_path: str) -> Optional[int]:
    """Infer binary GT from path convention.

    Returns:
        0 for normal, 1 for anomaly, None if path format is unsupported.
    """
    if not image_path:
        return None
    parts = image_path.replace("\\", "/").split("/")
    if "test" not in parts:
        return None
    idx = parts.index("test")
    if idx + 1 >= len(parts):
        return None
    sub = parts[idx + 1]
    return 0 if sub == "good" else 1


def build_class_key(pred: Dict[str, Any]) -> Optional[str]:
    class_key = pred.get("class_key")
    if isinstance(class_key, str) and class_key:
        return class_key
    dataset = pred.get("dataset")
    category = pred.get("category")
    if isinstance(dataset, str) and dataset and isinstance(category, str) and category:
        return f"{dataset}/{category}"
    return None


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_pos = scores > float(threshold)
    gt_pos = labels == 1
    gt_neg = ~gt_pos

    tp = int(np.logical_and(pred_pos, gt_pos).sum())
    fp = int(np.logical_and(pred_pos, gt_neg).sum())
    tn = int(np.logical_and(~pred_pos, gt_neg).sum())
    fn = int(np.logical_and(~pred_pos, gt_pos).sum())

    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    precision = _safe_div(tp, tp + fp)
    recall = tpr
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(scores))
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
    }


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_scores = np.unique(scores.astype(np.float64))
    if unique_scores.size == 0:
        return evaluate_threshold(scores, labels, 0.5)

    eps = 1e-8
    candidates = np.concatenate(
        (
            np.array([float(unique_scores.min() - eps)]),
            unique_scores,
            np.array([float(unique_scores.max() + eps)]),
        )
    )

    best: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, float, float]] = None
    for threshold in candidates:
        stats = evaluate_threshold(scores, labels, float(threshold))
        key = (
            stats["balanced_accuracy"],
            stats["f1"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best_key:
            best = stats
            best_key = key
    return best if best is not None else evaluate_threshold(scores, labels, 0.5)


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    compare = pos_scores[:, None] - neg_scores[None, :]
    wins = float((compare > 0).sum())
    ties = float((compare == 0).sum())
    auc = (wins + 0.5 * ties) / float(len(pos_scores) * len(neg_scores))
    return float(auc)


def compute_review_band(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    mis_q: float,
    min_band: float,
    max_band: float,
    max_review_rate: float,
    target_outside_accuracy: float,
) -> Dict[str, float]:
    deltas = np.abs(scores - float(threshold))
    pred_pos = scores > float(threshold)
    misclassified = pred_pos != (labels == 1)

    if misclassified.any():
        raw_band = float(np.quantile(deltas[misclassified], mis_q))
    else:
        raw_band = float(np.quantile(deltas, 0.10))

    # Prefer the smallest uncertainty band that reaches target outside-band accuracy.
    target_outside_accuracy = _clip(float(target_outside_accuracy), 0.50, 0.99)
    selected_band: Optional[float] = None
    max_review_rate = _clip(float(max_review_rate), 0.05, 0.95)
    quantiles = np.linspace(0.05, max_review_rate, num=17)
    for q in quantiles:
        cand_band = float(np.quantile(deltas, float(q)))
        inside_cand = deltas <= cand_band
        outside_cand = ~inside_cand
        if int(outside_cand.sum()) == 0:
            continue
        outside_acc_cand = float((pred_pos[outside_cand] == (labels[outside_cand] == 1)).mean())
        if outside_acc_cand >= target_outside_accuracy:
            selected_band = cand_band
            break

    # Prevent over-wide uncertainty bands that would collapse most samples into review_needed.
    max_review_rate = _clip(float(max_review_rate), 0.05, 0.95)
    rate_cap_band = float(np.quantile(deltas, max_review_rate))
    if selected_band is not None:
        raw_band = min(raw_band, selected_band)
    else:
        raw_band = min(raw_band, rate_cap_band)

    review_band = _clip(raw_band, min_band, max_band)

    inside = deltas <= review_band
    outside = ~inside

    inside_count = int(inside.sum())
    outside_count = int(outside.sum())
    review_rate = _safe_div(inside_count, len(scores))

    if outside_count > 0:
        outside_acc = float((pred_pos[outside] == (labels[outside] == 1)).mean())
    else:
        outside_acc = float((pred_pos == (labels == 1)).mean())

    if inside_count > 0:
        inside_error_rate = float((pred_pos[inside] != (labels[inside] == 1)).mean())
    else:
        inside_error_rate = 0.0

    return {
        "review_band": float(review_band),
        "review_rate": float(review_rate),
        "outside_accuracy": float(outside_acc),
        "inside_error_rate": float(inside_error_rate),
    }


def decide_reliability(
    *,
    auroc: float,
    balanced_accuracy: float,
    outside_accuracy: float,
    high_auroc: float,
    high_bal_acc: float,
    high_outside_acc: float,
    medium_auroc: float,
    medium_bal_acc: float,
) -> str:
    if (
        auroc >= high_auroc
        and balanced_accuracy >= high_bal_acc
        and outside_accuracy >= high_outside_acc
    ):
        return "high"
    if auroc >= medium_auroc and balanced_accuracy >= medium_bal_acc:
        return "medium"
    return "low"


def reliability_policy_defaults(reliability: str) -> Dict[str, Any]:
    if reliability == "high":
        return {
            "reliability": "high",
            "ad_weight": 0.70,
            "location_mode": "normal",
            "min_location_confidence": 0.25,
            "use_bbox": True,
        }
    if reliability == "medium":
        return {
            "reliability": "medium",
            "ad_weight": 0.45,
            "location_mode": "normal",
            "min_location_confidence": 0.35,
            "use_bbox": True,
        }
    return {
        "reliability": "low",
        "ad_weight": 0.20,
        "location_mode": "off",
        "min_location_confidence": 0.45,
        "use_bbox": False,
    }


def reliability_scale(reliability: str) -> float:
    return {
        "high": 0.9,
        "medium": 1.0,
        "low": 1.3,
    }.get(reliability, 1.0)


def round4(value: float) -> float:
    return round(float(value), 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        type=Path,
        required=True,
        help="Input full AD report JSON path.",
    )
    parser.add_argument(
        "--output-policy-json",
        type=Path,
        default=Path("configs/ad_policy_reproducible.json"),
        help="Output policy JSON path.",
    )
    parser.add_argument(
        "--output-calibration-json",
        type=Path,
        default=Path("configs/ad_calibration_reproducible.json"),
        help="Output calibration JSON path.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("output/ad_policy_reproducible_summary.json"),
        help="Output diagnostics summary JSON path.",
    )
    parser.add_argument(
        "--misclassified-quantile",
        type=float,
        default=0.85,
        help="Quantile of |score-threshold| over misclassified samples for review band.",
    )
    parser.add_argument(
        "--min-review-band",
        type=float,
        default=0.02,
        help="Minimum review band.",
    )
    parser.add_argument(
        "--max-review-band",
        type=float,
        default=0.30,
        help="Maximum review band.",
    )
    parser.add_argument(
        "--max-review-rate",
        type=float,
        default=0.45,
        help="Upper bound for fraction of samples inside uncertainty band.",
    )
    parser.add_argument(
        "--target-outside-accuracy",
        type=float,
        default=0.85,
        help="Preferred minimum accuracy for samples outside uncertainty band.",
    )
    parser.add_argument("--high-auroc", type=float, default=0.90)
    parser.add_argument("--high-balanced-acc", type=float, default=0.82)
    parser.add_argument("--high-outside-acc", type=float, default=0.83)
    parser.add_argument("--medium-auroc", type=float, default=0.78)
    parser.add_argument("--medium-balanced-acc", type=float, default=0.68)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_path: Path = args.report_json
    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    predictions = report.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Invalid report JSON: 'predictions' must be a list.")

    class_scores: Dict[str, List[float]] = defaultdict(list)
    class_labels: Dict[str, List[int]] = defaultdict(list)
    skipped_count = 0

    for pred in predictions:
        if not isinstance(pred, dict):
            skipped_count += 1
            continue
        class_key = build_class_key(pred)
        score = _safe_float(pred.get("anomaly_score"))
        gt = infer_gt_from_image_path(str(pred.get("image_path", "")))
        if class_key is None or score is None or gt is None:
            skipped_count += 1
            continue
        class_scores[class_key].append(float(score))
        class_labels[class_key].append(int(gt))

    class_keys = sorted(class_scores.keys())
    if not class_keys:
        raise ValueError("No usable samples found. Check report format/path convention.")

    generated_at = datetime.now(timezone.utc).isoformat()
    classes_policy: Dict[str, Any] = {}
    classes_calibration: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    all_thresholds: List[float] = []
    all_stored_bands: List[float] = []

    for class_key in class_keys:
        scores = np.array(class_scores[class_key], dtype=np.float64)
        labels = np.array(class_labels[class_key], dtype=np.int64)

        n_samples = int(len(scores))
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_samples == 0 or n_pos == 0 or n_neg == 0:
            continue

        best = find_best_threshold(scores, labels)
        auroc = compute_auroc(scores, labels)
        if auroc is None:
            auroc = 0.5

        review_stats = compute_review_band(
            scores,
            labels,
            best["threshold"],
            mis_q=float(args.misclassified_quantile),
            min_band=float(args.min_review_band),
            max_band=float(args.max_review_band),
            max_review_rate=float(args.max_review_rate),
            target_outside_accuracy=float(args.target_outside_accuracy),
        )

        reliability = decide_reliability(
            auroc=float(auroc),
            balanced_accuracy=float(best["balanced_accuracy"]),
            outside_accuracy=float(review_stats["outside_accuracy"]),
            high_auroc=float(args.high_auroc),
            high_bal_acc=float(args.high_balanced_acc),
            high_outside_acc=float(args.high_outside_acc),
            medium_auroc=float(args.medium_auroc),
            medium_bal_acc=float(args.medium_balanced_acc),
        )
        reliability_defaults = reliability_policy_defaults(reliability)

        center_threshold = float(best["threshold"])
        effective_review_band = float(review_stats["review_band"])
        scale = reliability_scale(reliability)
        stored_review_band = _clip(
            effective_review_band / max(scale, 1e-6),
            float(args.min_review_band),
            float(args.max_review_band),
        )
        runtime_uncertainty_band = _clip(
            stored_review_band * scale,
            0.02,
            0.35,
        )

        t_low = _clip(center_threshold - runtime_uncertainty_band, 0.0, 1.0)
        t_high = _clip(center_threshold + runtime_uncertainty_band, 0.0, 1.0)

        classes_policy[class_key] = {
            **reliability_defaults,
            "review_band": round4(stored_review_band),
            "t_low": round4(t_low),
            "t_high": round4(t_high),
        }

        classes_calibration[class_key] = {
            "center_threshold": round4(center_threshold),
            "review_band": round4(stored_review_band),
            "auroc": round4(auroc),
            "fpr": round4(best["fpr"]),
            "fnr": round4(best["fnr"]),
            "n_samples": n_samples,
            "n_normal": n_neg,
            "n_anomaly": n_pos,
            "balanced_accuracy": round4(best["balanced_accuracy"]),
            "f1": round4(best["f1"]),
            "outside_accuracy": round4(review_stats["outside_accuracy"]),
            "review_rate": round4(review_stats["review_rate"]),
            "inside_error_rate": round4(review_stats["inside_error_rate"]),
        }

        summary_rows.append(
            {
                "class_key": class_key,
                "n_samples": n_samples,
                "n_normal": n_neg,
                "n_anomaly": n_pos,
                "auroc": round4(auroc),
                "threshold_opt": round4(center_threshold),
                "review_band_effective": round4(runtime_uncertainty_band),
                "review_band_stored": round4(stored_review_band),
                "t_low": round4(t_low),
                "t_high": round4(t_high),
                "f1": round4(best["f1"]),
                "balanced_accuracy": round4(best["balanced_accuracy"]),
                "fpr": round4(best["fpr"]),
                "fnr": round4(best["fnr"]),
                "outside_accuracy": round4(review_stats["outside_accuracy"]),
                "review_rate": round4(review_stats["review_rate"]),
                "inside_error_rate": round4(review_stats["inside_error_rate"]),
                "reliability": reliability,
                "location_mode": reliability_defaults["location_mode"],
                "ad_weight": reliability_defaults["ad_weight"],
                "use_bbox": reliability_defaults["use_bbox"],
            }
        )

        all_thresholds.append(center_threshold)
        all_stored_bands.append(stored_review_band)

    if not summary_rows:
        raise ValueError("No class summary rows produced. Verify report labels and score fields.")

    default_threshold = float(np.median(np.array(all_thresholds, dtype=np.float64)))
    default_stored_band = _clip(
        float(np.median(np.array(all_stored_bands, dtype=np.float64))),
        float(args.min_review_band),
        float(args.max_review_band),
    )
    default_effective_band = default_stored_band * reliability_scale("medium")
    default_t_low = _clip(default_threshold - default_effective_band, 0.0, 1.0)
    default_t_high = _clip(default_threshold + default_effective_band, 0.0, 1.0)

    default_policy = {
        **reliability_policy_defaults("medium"),
        "review_band": round4(default_stored_band),
        "t_low": round4(default_t_low),
        "t_high": round4(default_t_high),
    }

    policy_out = {
        "schema_version": "ad_context_rules_v2_data_driven_v1",
        "generated_at": generated_at,
        "source_report": str(report_path),
        "generator": {
            "script": "scripts/build_ad_policy_from_report.py",
            "threshold_objective": "maximize_balanced_accuracy_then_f1",
            "review_band_rule": (
                "review_band = quantile(|score-threshold| of misclassified samples, q=misclassified_quantile)"
            ),
            "misclassified_quantile": float(args.misclassified_quantile),
            "min_review_band": float(args.min_review_band),
            "max_review_band": float(args.max_review_band),
            "max_review_rate": float(args.max_review_rate),
            "target_outside_accuracy": float(args.target_outside_accuracy),
            "reliability_rule": {
                "high": {
                    "auroc": float(args.high_auroc),
                    "balanced_accuracy": float(args.high_balanced_acc),
                    "outside_accuracy": float(args.high_outside_acc),
                },
                "medium": {
                    "auroc": float(args.medium_auroc),
                    "balanced_accuracy": float(args.medium_balanced_acc),
                },
            },
        },
        "default": default_policy,
        "classes": {k: classes_policy[k] for k in sorted(classes_policy)},
    }

    calibration_out = {
        "schema_version": "ad_context_calibration_v1",
        "generated_at": generated_at,
        "source_report": str(report_path),
        "classes": {k: classes_calibration[k] for k in sorted(classes_calibration)},
    }

    summary_out = {
        "schema_version": "ad_policy_build_summary_v1",
        "generated_at": generated_at,
        "source_report": str(report_path),
        "input_prediction_count": len(predictions),
        "used_sample_count": int(sum(row["n_samples"] for row in summary_rows)),
        "skipped_sample_count": int(skipped_count),
        "class_count": len(summary_rows),
        "classes": sorted(summary_rows, key=lambda x: x["class_key"]),
    }

    for path in (
        args.output_policy_json,
        args.output_calibration_json,
        args.output_summary_json,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_policy_json, "w", encoding="utf-8") as f:
        json.dump(policy_out, f, ensure_ascii=False, indent=2)
    with open(args.output_calibration_json, "w", encoding="utf-8") as f:
        json.dump(calibration_out, f, ensure_ascii=False, indent=2)
    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print(f"Saved policy: {args.output_policy_json}")
    print(f"Saved calibration: {args.output_calibration_json}")
    print(f"Saved summary: {args.output_summary_json}")
    print(f"Classes processed: {len(summary_rows)}")
    print(f"Samples used: {summary_out['used_sample_count']}")
    print(f"Samples skipped: {summary_out['skipped_sample_count']}")


if __name__ == "__main__":
    main()
