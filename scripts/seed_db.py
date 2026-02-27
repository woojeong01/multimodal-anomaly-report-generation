"""Seed PostgreSQL DB with sample inspection report data.

MMAD JSON에서 실제 카테고리/이미지 경로를 읽어 현실적인 더미 데이터를 생성하고
EC2 lion_db에 삽입한다.

Usage:
    # EC2 DB에 직접 삽입
    python scripts/seed_db.py --dsn "postgresql://lion_user:PASS@EC2_IP:5432/lion_db"

    # MMAD JSON 지정 (없으면 내장 카테고리 사용)
    python scripts/seed_db.py --dsn "..." --mmad-json dataset/MMAD/mmad.json

    # 샘플 수 조절
    python scripts/seed_db.py --dsn "..." --n 30 --seed 42

    # DB 삽입 없이 JSON 파일로만 저장 (확인용)
    python scripts/seed_db.py --output sample_data.json

    # 기존 데이터 삭제 후 삽입
    python scripts/seed_db.py --dsn "..." --clear
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJ_ROOT = SCRIPT_PATH.parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from src.service.settings import load_yaml

KST = timezone(timedelta(hours=9))

# ──────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────

DATASETS = ["DS-MVTec", "GoodsAD", "VisA"]

CATEGORIES = [
    "cigarette", "bottle", "capsules", "cashew",
    "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb", "pipe_fryum",
]

DEFECT_TYPES = {
    "cigarette": ["opened", "bent", "missing_filter", "color_stain"],
    "bottle":    ["broken_large", "broken_small", "contamination", "scratch"],
    "capsules":  ["crack", "faulty_imprint", "poke", "scratch"],
    "cashew":    ["hole", "cut", "scar"],
    "chewinggum":["color", "hole"],
    "fryum":     ["broken", "cut"],
    "macaroni1": ["broken", "cut", "hole"],
    "macaroni2": ["broken", "cut", "hole"],
    "pcb":       ["bent_lead", "damage", "missing_lead", "swap"],
    "pipe_fryum":["broken", "cut"],
}

REGIONS = [
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]

SEVERITIES = ["low", "medium", "high"]
RISK_LEVELS = ["low", "medium", "high"]

LINES = ["line-1", "line-2", "line-3"]

IMAGE_SIZE = 384  # 가상 이미지 크기 (bbox 계산용)


# ──────────────────────────────────────────────
# 데이터 생성 헬퍼
# ──────────────────────────────────────────────

def _rand_bbox(rng: random.Random) -> list[int]:
    x1 = rng.randint(30, IMAGE_SIZE // 2)
    y1 = rng.randint(30, IMAGE_SIZE // 2)
    x2 = rng.randint(x1 + 20, IMAGE_SIZE - 20)
    y2 = rng.randint(y1 + 20, IMAGE_SIZE - 20)
    return [x1, y1, x2, y2]


def _center_from_bbox(bbox: list[int]) -> list[float]:
    return [round((bbox[0] + bbox[2]) / 2, 1), round((bbox[1] + bbox[3]) / 2, 1)]


def _area_ratio(bbox: list[int]) -> float:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return round((w * h) / (IMAGE_SIZE ** 2), 4)


def _fake_path(category: str, is_anomaly: bool, defect_type: str, idx: int, ext: str = "png") -> str:
    split = defect_type if is_anomaly else "good"
    return f"dataset/MMAD/{category}/{split}/{category}_{idx:04d}.{ext}"


def _rand_ts(rng: random.Random, base: datetime) -> str:
    offset_s = rng.randint(-3600 * 24 * 7, 0)  # 최근 7일 내
    return (base + timedelta(seconds=offset_s)).isoformat()


def generate_record(
    rng: random.Random,
    category: str,
    idx: int,
    base_time: datetime,
    is_anomaly: bool | None = None,
) -> dict:
    """lion_db inspection_reports 스키마에 맞는 레코드 생성."""

    if is_anomaly is None:
        is_anomaly = rng.random() > 0.4  # 60% 이상 이상품

    defect_candidates = DEFECT_TYPES.get(category, ["scratch", "hole"])
    defect_type = rng.choice(defect_candidates) if is_anomaly else "good"
    dataset = rng.choice(DATASETS)
    line = rng.choice(LINES)

    # ── AD 점수 & 판정 ──────────────────────
    if is_anomaly:
        ad_score = round(rng.uniform(0.65, 1.0), 4)
    else:
        ad_score = round(rng.uniform(0.0, 0.45), 4)

    # 3단 판정 (t_low=0.5, t_high=0.8 기준)
    T_LOW, T_HIGH = 0.5, 0.8
    if ad_score >= T_HIGH:
        ad_decision = "anomaly"
        is_anomaly_ad = True
    elif ad_score >= T_LOW:
        ad_decision = "review_needed"
        is_anomaly_ad = True
    else:
        ad_decision = "normal"
        is_anomaly_ad = False

    # ── 결함 위치 정보 ──────────────────────
    has_defect = is_anomaly
    if has_defect:
        region = rng.choice(REGIONS)
        bbox = _rand_bbox(rng)
        area_ratio = _area_ratio(bbox)
    else:
        region = None
        area_ratio = 0.0

    # ── 경로 ──────────────────────────────
    image_path = _fake_path(category, is_anomaly, defect_type, idx)
    heatmap_path = image_path.replace("dataset/MMAD", "outputs/heatmaps").replace(".png", "_heat.png")
    mask_path    = image_path.replace("dataset/MMAD", "outputs/masks").replace(".png", "_mask.png")
    overlay_path = image_path.replace("dataset/MMAD", "outputs/overlay").replace(".png", "_overlay.png")
    similar_path = _fake_path(category, False, "good", rng.randint(1, 50))

    # ── 추론 시간 ──────────────────────────
    ad_dur  = round(rng.uniform(0.3, 3.5), 3)
    llm_dur = round(rng.uniform(1.5, 12.0), 3)

    # ── LLM 리포트 (TEXT로 저장) ────────────
    severity = rng.choice(SEVERITIES) if is_anomaly else "none"
    risk_level = severity if is_anomaly else "none"
    recommendation = (
        "출하 불가, 재검사 필요" if severity == "high"
        else "주의 관찰 후 출하 가능" if severity == "medium"
        else "정상 출하 가능"
    )

    if is_anomaly:
        description = (
            f"{category} 제품의 {region} 영역에서 {defect_type} 유형의 결함이 "
            f"감지되었습니다. 결함 면적 비율 {area_ratio:.1%}."
        )
        summary_text = f"{category} {defect_type} 결함 감지. {region} 영역."
        is_anomaly_llm = True
    else:
        description = f"{category} 제품이 정상 상태입니다. 결함이 감지되지 않았습니다."
        summary_text = f"{category} 정상 판정."
        is_anomaly_llm = False

    # llm_report, llm_summary는 DB에서 TEXT 타입 → JSON 문자열로 저장
    llm_report = json.dumps({
        "anomaly_type": defect_type if is_anomaly else "none",
        "severity": severity,
        "location": region or "none",
        "description": description,
        "recommendation": recommendation,
    }, ensure_ascii=False)

    llm_summary = json.dumps({
        "summary": summary_text,
        "risk_level": risk_level,
    }, ensure_ascii=False)

    return {
        "dataset":               dataset,
        "category":              category,
        "line":                  line,
        "ad_score":              ad_score,
        "ad_decision":           ad_decision,
        "is_anomaly_AD":         is_anomaly_ad,
        "has_defect":            has_defect,
        "region":                region,
        "area_ratio":            area_ratio,
        "image_path":            image_path,
        "heatmap_path":          heatmap_path,
        "mask_path":             mask_path,
        "overlay_path":          overlay_path,
        "similar_image_path":    similar_path,
        "ad_inference_duration": ad_dur,
        "is_anomaly_LLM":        is_anomaly_llm,
        "llm_report":            llm_report,
        "llm_summary":           llm_summary,
        "llm_inference_duration": llm_dur,
        "applied_policy":        {"t_low": T_LOW, "t_high": T_HIGH},
    }


def generate_records_from_mmad(
    mmad_json: str,
    n_per_folder: int,
    rng: random.Random,
    base_time: datetime,
) -> list[dict]:
    """MMAD JSON에서 실제 경로/카테고리를 읽어 레코드 생성."""
    with open(mmad_json, "r", encoding="utf-8") as f:
        mmad = json.load(f)

    # 폴더(category/split)별 그룹핑
    from collections import defaultdict
    folders: dict[str, list[str]] = defaultdict(list)
    for path in mmad.keys():
        parts = path.split("/")
        if len(parts) >= 4:
            key = f"{parts[0]}/{parts[1]}/{parts[3]}"
        elif len(parts) >= 2:
            key = f"{parts[0]}/{parts[1]}"
        else:
            key = "unknown"
        folders[key].append(path)

    sampled_paths = []
    for key in sorted(folders.keys()):
        imgs = folders[key]
        sampled_paths.extend(rng.sample(imgs, min(n_per_folder, len(imgs))))

    print(f"MMAD: {len(mmad)} 이미지 → {len(sampled_paths)}장 샘플링 ({n_per_folder}장/폴더)")

    records = []
    for i, path in enumerate(sampled_paths):
        parts = path.split("/")
        category = parts[1] if len(parts) > 1 else "unknown"
        dataset  = parts[0] if len(parts) > 0 else "MMAD"
        is_anomaly = "/good/" not in path and not path.endswith("/good")

        rec = generate_record(rng, category, i, base_time, is_anomaly=is_anomaly)
        rec["dataset"] = dataset
        rec["image_path"] = path
        records.append(rec)

    return records


def generate_records_dummy(
    n: int,
    rng: random.Random,
    base_time: datetime,
) -> list[dict]:
    """MMAD 없이 내장 카테고리로 더미 레코드 생성."""
    records = []
    per_cat = max(1, n // len(CATEGORIES))
    remainder = n - per_cat * len(CATEGORIES)

    idx = 0
    for cat in CATEGORIES:
        count = per_cat + (1 if remainder > 0 else 0)
        remainder -= 1
        for _ in range(count):
            records.append(generate_record(rng, cat, idx, base_time))
            idx += 1

    rng.shuffle(records)
    return records


# ──────────────────────────────────────────────
# DB 삽입
# ──────────────────────────────────────────────

def insert_to_db(dsn: str, records: list[dict], clear: bool = False) -> None:
    try:
        import psycopg2
        from psycopg2.extras import Json
    except ImportError:
        print("Error: psycopg2 미설치. pip install psycopg2-binary")
        sys.exit(1)

    conn = psycopg2.connect(dsn)

    if clear:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM inspection_reports")
        conn.commit()
        print("기존 데이터 삭제 완료")

    JSONB_COLS = {"applied_policy"}  # llm_report/llm_summary는 TEXT, bbox/center 컬럼 없음

    count = 0
    for rec in records:
        columns = list(rec.keys())
        values = []
        for col in columns:
            val = rec[col]
            if col in JSONB_COLS and val is not None:
                val = Json(val)
            values.append(val)

        placeholders = ", ".join(["%s"] * len(columns))
        sql = (
            f"INSERT INTO inspection_reports ({', '.join(columns)}) "
            f"VALUES ({placeholders}) RETURNING id"
        )

        with conn.cursor() as cur:
            cur.execute(sql, values)
            rid = cur.fetchone()[0]
        conn.commit()
        count += 1

        cat = rec.get("category", "?")
        dec = rec.get("ad_decision", "?")
        score = rec.get("ad_score", 0)
        print(f"  [{count:>3}] id={rid:<5} | {cat:<14} | {dec:<14} | score={score:.3f}")

    print(f"\n완료: {count}개 레코드 삽입")
    conn.close()


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed lion_db with sample inspection reports")

    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="experiment.yaml 경로 (기본: configs/experiment.yaml)")

    # CLI overrides (YAML보다 우선)
    parser.add_argument("--dsn", type=str, default=None,
                        help="PostgreSQL DSN (e.g. postgresql://lion_user:pass@IP/lion_db)")
    parser.add_argument("--mmad-json", type=str, default=None,
                        help="MMAD JSON 경로 (없으면 내장 카테고리 사용)")
    parser.add_argument("--n", type=int, default=None,
                        help="생성할 레코드 수 (mmad-json 없을 때)")
    parser.add_argument("--n-per-folder", type=int, default=None,
                        help="MMAD 사용 시 폴더당 샘플 수")
    parser.add_argument("--seed", type=int, default=None,
                        help="랜덤 시드")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON 파일로 저장할 경로 (DB 삽입과 병행 가능)")
    parser.add_argument("--clear", action="store_true", default=None,
                        help="삽입 전 기존 inspection_reports 데이터 삭제")

    args = parser.parse_args()

    # ── YAML 로드 (기존 필드만 읽음, 수정 없음) ──
    cfg_yaml = {}
    eval_cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        cfg_yaml = load_yaml(config_path)
        eval_cfg = cfg_yaml.get("eval", {})
    else:
        print(f"Config not found: {config_path}, CLI 인자만 사용")

    # ── 값 결정 (CLI > YAML 기존 필드 > 기본값) ──
    dsn          = args.dsn
    mmad_json    = args.mmad_json   or cfg_yaml.get("mmad_json")
    n            = args.n           
    n_per_folder = args.n_per_folder or eval_cfg.get("sample_per_folder", 3)
    rand_seed    = args.seed        or eval_cfg.get("sample_seed", 42)
    output       = args.output
    clear        = args.clear or False

    if not dsn and not output:
        print("Error: experiment.yaml의 seed.dsn 또는 --dsn / --output 필요")
        parser.print_help()
        sys.exit(1)

    print("=" * 50)
    print("DB Seed 설정")
    print("=" * 50)
    print(f"Config:       {config_path}")
    print(f"DSN:          {dsn.split('@')[-1] if dsn else '없음 (JSON만 저장)'}")
    print(f"MMAD JSON:    {mmad_json or '없음 (내장 카테고리 사용)'}")
    print(f"N / N/folder: {n} / {n_per_folder}")
    print(f"Seed:         {rand_seed}")
    print(f"Clear:        {clear}")
    print(f"Output JSON:  {output or '없음'}")
    print()

    rng = random.Random(rand_seed)
    base_time = datetime.now(KST)

    # ── 레코드 생성 ────────────────────────
    if mmad_json and Path(mmad_json).exists():
        records = generate_records_from_mmad(mmad_json, n_per_folder, rng, base_time)
    else:
        if mmad_json:
            print(f"MMAD JSON 없음 ({mmad_json}) → 내장 카테고리 사용")
        print(f"내장 카테고리로 {n}개 더미 데이터 생성")
        records = generate_records_dummy(n, rng, base_time)

    print(f"생성된 레코드: {len(records)}개\n")

    # ── JSON 저장 ──────────────────────────
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON 저장: {out}")

    # ── DB 삽입 ────────────────────────────
    if dsn:
        print(f"DB 삽입 시작 → {dsn.split('@')[-1]}")
        insert_to_db(dsn, records, clear=clear)


if __name__ == "__main__":
    main()
