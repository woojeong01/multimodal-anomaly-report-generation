#!/usr/bin/env python
"""Visualize bbox-vs-mask quality on sampled MMAD images.

This script is designed for quick qualitative checks after AD inference.
It samples images per (dataset, category, defect_type), uses bbox from an
existing prediction JSON, runs model inference only for sampled images to
generate heatmap/highlight, and saves a 4-panel image:

1) Original
2) Mask overlay (GT)
3) BBox overlay (pred bbox + GT contour + metrics)
4) Model highlight (heatmap + thresholded region)

Example:
    python scripts/visualize_bbox_mask_subset.py \
      --checkpoint-dir /content/drive/MyDrive/.../checkpoints/patchcore_384 \
      --data-root /content/drive/MyDrive/.../MMAD \
      --mmad-json /content/drive/MyDrive/.../MMAD/mmad_10classes.json \
      --pred-json /content/drive/MyDrive/.../MMAD/checkpoints/patchcore_384/ad_predictions_v6.json \
      --output-dir output/bbox_mask_review \
      --samples-per-group 5 \
      --device cuda \
      --input-size 384 384
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
SCRIPTS_DIR = PROJ_ROOT / "scripts"
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_ad_inference import PatchCoreCheckpointRunner  # noqa: E402


DATASET_MARKERS = ("GoodsAD", "MVTec-LOCO", "MVTec-AD", "DS-MVTec", "VisA")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SampleRecord:
    rel_path: str
    dataset: str
    category: str
    defect_type: str
    mmad_meta: Dict[str, Any]
    pred_meta: Dict[str, Any]


def normalize_rel_path(path_like: str) -> str:
    normalized = str(path_like).strip().replace("\\", "/")
    for marker in DATASET_MARKERS:
        key = f"{marker}/"
        if key in normalized:
            return normalized[normalized.index(key) :]
    return normalized.lstrip("/")


def parse_rel_image_path(rel_path: str) -> Optional[Tuple[str, str, str, str, str]]:
    parts = Path(rel_path).parts
    if len(parts) < 5:
        return None
    dataset, category, split, defect_type = parts[0], parts[1], parts[2], parts[3]
    filename = parts[-1]
    return dataset, category, split, defect_type, filename


def extract_prediction_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        return [p for p in payload["predictions"] if isinstance(p, dict)]
    return []


def load_predictions_index(pred_json: Path) -> Dict[str, Dict[str, Any]]:
    with open(pred_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    pred_list = extract_prediction_list(payload)
    index: Dict[str, Dict[str, Any]] = {}
    for item in pred_list:
        image_path = item.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            continue
        rel = normalize_rel_path(image_path)
        index[rel] = item
    return index


def load_mmad_entries(mmad_json: Path) -> Dict[str, Dict[str, Any]]:
    with open(mmad_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("mmad json must be a dictionary keyed by image path")
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        rel = normalize_rel_path(key)
        out[rel] = value if isinstance(value, dict) else {}
    return out


def _none_or_set(values: Optional[List[str]]) -> Optional[set]:
    if not values:
        return None
    return {v for v in values if v}


def sample_records(
    mmad_entries: Dict[str, Dict[str, Any]],
    pred_index: Dict[str, Dict[str, Any]],
    *,
    datasets: Optional[List[str]],
    categories: Optional[List[str]],
    defect_types: Optional[List[str]],
    include_good: bool,
    samples_per_group: int,
    seed: int,
) -> List[SampleRecord]:
    random.seed(seed)
    ds_set = _none_or_set(datasets)
    cat_set = _none_or_set(categories)
    def_set = _none_or_set(defect_types)

    grouped: Dict[Tuple[str, str, str], List[SampleRecord]] = defaultdict(list)
    for rel_path, meta in mmad_entries.items():
        parsed = parse_rel_image_path(rel_path)
        if parsed is None:
            continue
        dataset, category, split, defect_type, filename = parsed
        if split != "test":
            continue
        if Path(filename).suffix.lower() not in IMAGE_EXTS:
            continue
        if ds_set and dataset not in ds_set:
            continue
        if cat_set and category not in cat_set:
            continue
        if not include_good and defect_type == "good":
            continue
        if def_set and defect_type not in def_set:
            continue

        pred = pred_index.get(rel_path)
        if pred is None:
            continue

        grouped[(dataset, category, defect_type)].append(
            SampleRecord(
                rel_path=rel_path,
                dataset=dataset,
                category=category,
                defect_type=defect_type,
                mmad_meta=meta,
                pred_meta=pred,
            )
        )

    selected: List[SampleRecord] = []
    for key, recs in sorted(grouped.items()):
        if not recs:
            continue
        k = min(max(1, int(samples_per_group)), len(recs))
        selected.extend(random.sample(recs, k))
    return selected


def resolve_image_path(sample: SampleRecord, data_root: Path) -> Optional[Path]:
    p = data_root / sample.rel_path
    if p.exists():
        return p

    meta_path = sample.mmad_meta.get("image_path")
    if isinstance(meta_path, str) and meta_path:
        rel = normalize_rel_path(meta_path)
        p2 = data_root / rel
        if p2.exists():
            return p2
    return None


def resolve_mask_ref(sample: SampleRecord, data_root: Path) -> Optional[Path]:
    if sample.defect_type == "good":
        return None

    value = sample.mmad_meta.get("mask_path")
    if isinstance(value, str) and value:
        rel = normalize_rel_path(value)
        m = data_root / rel
        if m.exists():
            return m

    stem = Path(sample.rel_path).stem
    gt_root = data_root / sample.dataset / sample.category / "ground_truth" / sample.defect_type

    cand = gt_root / f"{stem}.png"
    if cand.exists():
        return cand

    cand_dir = gt_root / stem
    if cand_dir.exists() and cand_dir.is_dir():
        return cand_dir

    cand2 = gt_root / f"{stem}_mask.png"
    if cand2.exists():
        return cand2

    return None


def load_mask(mask_ref: Optional[Path], image_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_ref is None:
        return None

    h, w = image_hw
    if mask_ref.is_dir():
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in sorted(mask_ref.glob("*.png")):
            if p.name.startswith("._"):
                continue
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, m)
        return mask

    if mask_ref.is_file():
        m = cv2.imread(str(mask_ref), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    return None


def sanitize_bbox(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [int(round(float(v))) for v in value]
    except Exception:
        return None


def extract_pred_bbox(pred_meta: Dict[str, Any]) -> Optional[List[int]]:
    loc = pred_meta.get("defect_location")
    if isinstance(loc, dict):
        bbox = sanitize_bbox(loc.get("bbox"))
        if bbox is not None:
            return bbox

    loc_raw = pred_meta.get("defect_location_raw")
    if isinstance(loc_raw, dict):
        bbox = sanitize_bbox(loc_raw.get("bbox"))
        if bbox is not None:
            return bbox
    return None


def extract_pred_threshold(pred_meta: Dict[str, Any], fallback: float) -> float:
    value = pred_meta.get("model_threshold", fallback)
    try:
        return float(value)
    except Exception:
        return float(fallback)


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_to_mask(shape: Tuple[int, int], bbox: Optional[List[int]]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        return m
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    if x1 <= x0 or y1 <= y0:
        return m
    m[y0 : y1 + 1, x0 : x1 + 1] = 1
    return m


def iou_binary(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a_bin = a > 0
    b_bin = b > 0
    union = int(np.logical_or(a_bin, b_bin).sum())
    if union <= 0:
        return None
    inter = int(np.logical_and(a_bin, b_bin).sum())
    return float(inter / union)


def compute_bbox_mask_metrics(pred_bbox: Optional[List[int]], gt_mask: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if gt_mask is None:
        return {
            "box_iou": None,
            "bbox_mask_iou": None,
            "bbox_mask_precision": None,
            "bbox_mask_recall": None,
        }

    gt_bin = (gt_mask > 0).astype(np.uint8)
    gt_bbox = mask_to_bbox(gt_bin)
    pred_rect = bbox_to_mask(gt_bin.shape, pred_bbox)

    box_iou = None
    if pred_bbox is not None and gt_bbox is not None:
        box_iou = iou_binary(bbox_to_mask(gt_bin.shape, pred_bbox), bbox_to_mask(gt_bin.shape, gt_bbox))

    bbox_mask_iou = iou_binary(pred_rect, gt_bin)
    pred_area = int(pred_rect.sum())
    gt_area = int(gt_bin.sum())
    inter = int(np.logical_and(pred_rect > 0, gt_bin > 0).sum())
    precision = float(inter / pred_area) if pred_area > 0 else None
    recall = float(inter / gt_area) if gt_area > 0 else None

    return {
        "box_iou": box_iou,
        "bbox_mask_iou": bbox_mask_iou,
        "bbox_mask_precision": precision,
        "bbox_mask_recall": recall,
    }


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    binary = mask > 0
    if not np.any(binary):
        return out
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[binary] = color
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)


def draw_mask_contour(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    out = image.copy()
    binary = (mask > 0).astype(np.uint8) * 255
    if binary.max() <= 0:
        return out
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def draw_bbox(image: np.ndarray, bbox: Optional[List[int]], color: Tuple[int, int, int], label: str = "") -> np.ndarray:
    out = image.copy()
    if bbox is None:
        return out
    h, w = out.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), w - 1))
    x1 = max(0, min(int(x1), w - 1))
    y0 = max(0, min(int(y0), h - 1))
    y1 = max(0, min(int(y1), h - 1))
    if x1 <= x0 or y1 <= y0:
        return out
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    if label:
        cv2.putText(out, label, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.3f}"


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def to_panel(image: np.ndarray, panel_size: int) -> np.ndarray:
    return cv2.resize(image, (panel_size, panel_size), interpolation=cv2.INTER_AREA)


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    m = anomaly_map.astype(np.float32)
    min_v = float(np.min(m))
    max_v = float(np.max(m))
    if max_v <= min_v:
        return np.zeros_like(m, dtype=np.float32)
    return (m - min_v) / (max_v - min_v)


def build_visual_row(
    *,
    image: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_bbox: Optional[List[int]],
    anomaly_map: np.ndarray,
    threshold: float,
    score: float,
    sample_name: str,
    metrics: Dict[str, Optional[float]],
    panel_size: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    heat_norm = normalize_map(anomaly_map)
    heat_u8 = (np.clip(heat_norm, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    pred_mask = (anomaly_map > float(threshold)).astype(np.uint8)
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    panel_orig = image.copy()
    cv2.putText(panel_orig, sample_name, (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(panel_orig, f"score={score:.4f}", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    panel_mask = image.copy()
    if gt_mask is not None:
        panel_mask = overlay_mask(panel_mask, gt_mask, (0, 140, 255), 0.45)
        panel_mask = draw_mask_contour(panel_mask, gt_mask, (0, 255, 0), 2)
    else:
        cv2.putText(panel_mask, "no gt mask (good)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    panel_bbox = image.copy()
    panel_bbox = draw_bbox(panel_bbox, pred_bbox, (0, 0, 255), "pred bbox")
    if gt_mask is not None:
        panel_bbox = draw_mask_contour(panel_bbox, gt_mask, (255, 255, 0), 2)
    cv2.putText(panel_bbox, f"IoU(mask): {_fmt(metrics['bbox_mask_iou'])}", (8, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(panel_bbox, f"P/R: {_fmt(metrics['bbox_mask_precision'])}/{_fmt(metrics['bbox_mask_recall'])}", (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(panel_bbox, f"IoU(box): {_fmt(metrics['box_iou'])}", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    panel_highlight = cv2.addWeighted(image, 0.55, heat, 0.45, 0.0)
    if pred_mask.max() > 0:
        red_overlay = np.zeros_like(panel_highlight, dtype=np.uint8)
        red_overlay[pred_mask > 0] = (0, 0, 255)
        panel_highlight = cv2.addWeighted(panel_highlight, 1.0, red_overlay, 0.25, 0.0)
        contours, _ = cv2.findContours((pred_mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(panel_highlight, contours, -1, (0, 0, 255), 2)
    cv2.putText(panel_highlight, f"thr={float(threshold):.4f}", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    p1 = add_title(to_panel(panel_orig, panel_size), "Original")
    p2 = add_title(to_panel(panel_mask, panel_size), "Mask Overlay")
    p3 = add_title(to_panel(panel_bbox, panel_size), "BBox Overlay")
    p4 = add_title(to_panel(panel_highlight, panel_size), "Model Highlight")
    return np.hstack([p1, p2, p3, p4])


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return float(sum(v) / len(v))


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['dataset']}/{row['category']}/{row['defect_type']}"
        by_group[key].append(row)

    group_summary: Dict[str, Any] = {}
    for key, items in sorted(by_group.items()):
        group_summary[key] = {
            "num_samples": len(items),
            "mean_box_iou": _mean(x.get("box_iou") for x in items),
            "mean_bbox_mask_iou": _mean(x.get("bbox_mask_iou") for x in items),
            "mean_bbox_mask_precision": _mean(x.get("bbox_mask_precision") for x in items),
            "mean_bbox_mask_recall": _mean(x.get("bbox_mask_recall") for x in items),
        }

    return {
        "num_samples": len(rows),
        "mean_box_iou": _mean(x.get("box_iou") for x in rows),
        "mean_bbox_mask_iou": _mean(x.get("bbox_mask_iou") for x in rows),
        "mean_bbox_mask_precision": _mean(x.get("bbox_mask_precision") for x in rows),
        "mean_bbox_mask_recall": _mean(x.get("bbox_mask_recall") for x in rows),
        "by_group": group_summary,
    }


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bbox-vs-mask quality on sampled AD outputs")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="PatchCore checkpoint root")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root")
    parser.add_argument("--mmad-json", type=str, required=True, help="MMAD json path")
    parser.add_argument("--pred-json", type=str, required=True, help="AD prediction json (ad_report_v1)")
    parser.add_argument("--output-dir", type=str, default="output/bbox_mask_review", help="Output root")
    parser.add_argument("--samples-per-group", type=int, default=4, help="Samples per dataset/category/defect_type")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Dataset filter")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Category filter")
    parser.add_argument("--defect-types", type=str, nargs="*", default=None, help="Defect-type filter (e.g., good logical_anomaly)")
    parser.add_argument("--include-good", action="store_true", help="Include good samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=int, default=None, help="Checkpoint version override")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--input-size", type=int, nargs=2, default=[384, 384], help="Model input size [H W]")
    parser.add_argument("--default-threshold", type=float, default=0.5, help="Fallback threshold")
    parser.add_argument("--allow-online-backbone", action="store_true", help="Allow online backbone fallback")
    parser.add_argument("--postprocess-map", type=str, default="input", choices=["input", "original"], help="Map scale for sampled inference")
    parser.add_argument("--panel-size", type=int, default=420, help="Single panel size")
    parser.add_argument("--amp", action="store_true", dest="amp", help="Enable AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    mmad_json = Path(args.mmad_json)
    pred_json = Path(args.pred_json)
    output_dir = Path(args.output_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint-dir not found: {checkpoint_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")
    if not mmad_json.exists():
        raise FileNotFoundError(f"mmad-json not found: {mmad_json}")
    if not pred_json.exists():
        raise FileNotFoundError(f"pred-json not found: {pred_json}")

    mmad_entries = load_mmad_entries(mmad_json)
    pred_index = load_predictions_index(pred_json)
    sampled = sample_records(
        mmad_entries=mmad_entries,
        pred_index=pred_index,
        datasets=args.datasets,
        categories=args.categories,
        defect_types=args.defect_types,
        include_good=bool(args.include_good),
        samples_per_group=int(args.samples_per_group),
        seed=int(args.seed),
    )
    if not sampled:
        print("No samples matched the filters.")
        return

    print(f"Sampled images: {len(sampled)}")
    cnt = defaultdict(int)
    for s in sampled:
        cnt[f"{s.dataset}/{s.category}/{s.defect_type}"] += 1
    for key, n in sorted(cnt.items()):
        print(f"  - {key}: {n}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failures = 0

    grouped: Dict[Tuple[str, str], List[SampleRecord]] = defaultdict(list)
    for rec in sampled:
        grouped[(rec.dataset, rec.category)].append(rec)

    runner = PatchCoreCheckpointRunner(
        checkpoint_dir=checkpoint_dir,
        version=args.version,
        threshold=float(args.default_threshold),
        device=args.device,
        input_size=(int(args.input_size[0]), int(args.input_size[1])),
        allow_online_backbone=bool(args.allow_online_backbone),
        sync_timing=False,
        postprocess_map=args.postprocess_map,
        use_amp=bool(args.amp),
    )

    for (dataset, category), recs in sorted(grouped.items()):
        try:
            runner.warmup_model(dataset, category)
        except Exception as exc:
            print(f"[{dataset}/{category}] model load failed: {exc}")
            failures += len(recs)
            continue

        for rec in recs:
            image_path = resolve_image_path(rec, data_root)
            if image_path is None:
                failures += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failures += 1
                continue

            try:
                infer = runner.predict(dataset, category, image)
            except Exception:
                failures += 1
                continue

            anomaly_map = infer.get("anomaly_map")
            if anomaly_map is None:
                failures += 1
                continue

            pred_bbox = extract_pred_bbox(rec.pred_meta)
            threshold = extract_pred_threshold(rec.pred_meta, float(infer.get("threshold", args.default_threshold)))
            score = float(rec.pred_meta.get("anomaly_score", infer.get("anomaly_score", 0.0)))

            mask_ref = resolve_mask_ref(rec, data_root)
            gt_mask = load_mask(mask_ref, image.shape[:2])
            metrics = compute_bbox_mask_metrics(pred_bbox, gt_mask)

            vis = build_visual_row(
                image=image,
                gt_mask=gt_mask,
                pred_bbox=pred_bbox,
                anomaly_map=anomaly_map,
                threshold=threshold,
                score=score,
                sample_name=f"{dataset}/{category}/{rec.defect_type}/{Path(rec.rel_path).name}",
                metrics=metrics,
                panel_size=int(args.panel_size),
            )

            out_group = output_dir / dataset / category / rec.defect_type
            out_group.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(rec.rel_path).stem}.png"
            cv2.imwrite(str(out_group / out_name), vis)

            row = {
                "dataset": dataset,
                "category": category,
                "defect_type": rec.defect_type,
                "image_path": rec.rel_path,
                "pred_bbox": json.dumps(pred_bbox) if pred_bbox is not None else "",
                "mask_path": str(mask_ref) if mask_ref is not None else "",
                "anomaly_score": round(float(score), 6),
                "threshold": round(float(threshold), 6),
                "box_iou": metrics["box_iou"],
                "bbox_mask_iou": metrics["bbox_mask_iou"],
                "bbox_mask_precision": metrics["bbox_mask_precision"],
                "bbox_mask_recall": metrics["bbox_mask_recall"],
            }
            rows.append(row)

        runner.clear_cache()

    summary = summarize_rows(rows)
    save_csv(output_dir / "bbox_mask_metrics.csv", rows)
    save_json(output_dir / "bbox_mask_summary.json", summary)

    print()
    print("=" * 70)
    print("Visualization complete")
    print("=" * 70)
    print(f"Saved rows: {len(rows)}")
    print(f"Failures: {failures}")
    print(f"Output dir: {output_dir}")
    print(f"Metrics CSV: {output_dir / 'bbox_mask_metrics.csv'}")
    print(f"Summary JSON: {output_dir / 'bbox_mask_summary.json'}")


if __name__ == "__main__":
    main()
