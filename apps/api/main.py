from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import src.storage.pg as pg
from src.mllm.factory import get_llm_client, list_llm_models
from src.service.ad_service import AdService
from src.service.domain_rag_service import DomainKnowledgeRagService
from src.service.llm_service import LlmService
from src.service.visual_rag_service import RagService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("PG_DSN", "postgresql://son:1234@localhost/inspection"))
MODEL_NAME = os.getenv("LLM_MODEL", "internvl")
CHECKPOINT_DIR = Path(os.getenv("AD_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(PROJECT_ROOT / "rag")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "datasets")))
AD_POLICY_PATH = Path(os.getenv("AD_POLICY_PATH", str(PROJECT_ROOT / "configs" / "ad_policy.json")))
AD_CALIBRATION_PATH = Path(os.getenv("AD_CALIBRATION_PATH", str(PROJECT_ROOT / "configs" / "ad_calibration.json")))
DOMAIN_KNOWLEDGE_JSON_PATH = Path(
    os.getenv("DOMAIN_KNOWLEDGE_JSON_PATH", str(DATA_DIR / "domain_knowledge.json"))
)
DOMAIN_RAG_PERSIST_DIR = Path(
    os.getenv("DOMAIN_RAG_PERSIST_DIR", str(PROJECT_ROOT / "vectorstore" / "domain_knowledge"))
)
DOMAIN_RAG_TOP_K = max(1, int(os.getenv("DOMAIN_RAG_TOP_K", "3")))

INCOMING_ROOT = Path(os.getenv("INCOMING_ROOT", "/home/ubuntu/incoming"))
INCOMING_SCAN_INTERVAL_SEC = float(os.getenv("INCOMING_SCAN_INTERVAL_SEC", "5"))
INCOMING_STABLE_SECONDS = float(os.getenv("INCOMING_STABLE_SECONDS", "2"))
INCOMING_DEFAULT_DATASET = os.getenv("INCOMING_DEFAULT_DATASET", "incoming")
INCOMING_DEFAULT_CATEGORY = os.getenv("INCOMING_DEFAULT_CATEGORY", "")
INCOMING_DEFAULT_LINE = os.getenv("INCOMING_DEFAULT_LINE", "LINE-A-01")
INCOMING_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PIPELINE_WORKERS = max(1, int(os.getenv("PIPELINE_WORKERS", "2")))
LLM_LOCK_TIMEOUT_SEC = max(1.0, float(os.getenv("LLM_LOCK_TIMEOUT_SEC", "30")))
PIPELINE_WATCHDOG_INTERVAL_SEC = max(2.0, float(os.getenv("PIPELINE_WATCHDOG_INTERVAL_SEC", "10")))
PIPELINE_STALE_SECONDS = max(30, int(float(os.getenv("PIPELINE_STALE_SECONDS", "120"))))

LINE_PATTERN = re.compile(r"(?i)line[_-]?([a-z0-9]+)")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


INCOMING_WATCH_ENABLED = _env_bool("INCOMING_WATCH_ENABLED", True)
DOMAIN_RAG_ENABLED = _env_bool("DOMAIN_RAG_ENABLED", True) # False: RAG 사용 안함

LLM_MODEL_ALIASES = {
    "internv1": "internvl",
    "internv1-8b": "internvl-8b",
    "internv1-4b": "internvl-4b",
    "internv1-2b": "internvl-2b",
    "internv1-1b": "internvl-1b",
}


class LlmModelUpdateRequest(BaseModel):
    model: str


class LocalImageUpload:
    """UploadFile-like wrapper for filesystem image ingestion."""

    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self.file = path.open("rb")

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


class CorsStaticFiles(StaticFiles):
    """Static file mount that always exposes CORS headers for browser-side rendering."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code=status_code)
        if getattr(response, "status_code", 0) == 200:
            response.headers.setdefault("Access-Control-Allow-Origin", "*")
            response.headers.setdefault("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS")
            response.headers.setdefault("Access-Control-Allow-Headers", "*")
            response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        return response


class CachedStaticFiles(CorsStaticFiles):
    """Static file mount with cache headers for immutable output filenames."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code=status_code)
        if getattr(response, "status_code", 0) == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
        return response


def _load_json_doc(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        logger.warning("%s file not found: %s", label, path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        logger.exception("Failed to load %s file: %s", label, path)
    return {}


def _load_ad_policy_doc(path: Path) -> dict[str, Any]:
    return _load_json_doc(path, label="AD policy")


def _load_ad_calibration_doc(path: Path) -> dict[str, Any]:
    return _load_json_doc(path, label="AD calibration")


def _sync_category_metadata_from_policy_doc(conn, doc: dict[str, Any], *, default_line: str | None = None) -> int:
    """Seed/update category_metadata from ad_policy.json classes."""
    classes = doc.get("classes") if isinstance(doc, dict) else None
    if not isinstance(classes, dict):
        logger.warning("Skip category_metadata sync: AD policy has no valid classes object")
        return 0

    rows: list[dict[str, Any]] = []
    for class_key in classes.keys():
        if not isinstance(class_key, str):
            continue
        key = class_key.strip().strip("/")
        if not key:
            continue

        policy = _resolve_class_policy_from_json(doc, key)
        if not policy:
            continue

        dataset: str | None = None
        if "/" in key:
            dataset = key.split("/", 1)[0].strip() or None

        row: dict[str, Any] = {
            "category": key,
            "dataset": dataset,
            "line": default_line or None,
            "t_low": policy.get("t_low"),
            "t_high": policy.get("t_high"),
            "review_band": policy.get("review_band"),
            "reliability": policy.get("reliability"),
            "ad_weight": policy.get("ad_weight"),
            "location_mode": policy.get("location_mode"),
            "min_location_confidence": policy.get("min_location_confidence"),
            "use_bbox": policy.get("use_bbox"),
        }
        rows.append(row)

    if not rows:
        logger.warning("Skip category_metadata sync: no valid class policy rows")
        return 0

    try:
        return pg.upsert_category_metadata(conn, rows)
    except Exception:
        logger.exception("Failed to sync category_metadata from AD policy doc")
        return 0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _candidate_class_keys(dataset: str, model_category: str) -> list[str]:
    key = model_category.strip().strip("/")
    keys: list[str] = []
    if key:
        keys.append(key)

    ds = dataset.strip().strip("/")
    if key and ds and "/" not in key:
        keys.append(f"{ds}/{key}")

    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _resolve_class_policy_from_json(doc: dict[str, Any], class_key: str) -> dict[str, Any] | None:
    if not isinstance(doc, dict):
        return None

    merged: dict[str, Any] = {}
    source = "ad_policy_json"
    default_raw = doc.get("default")
    if isinstance(default_raw, dict):
        merged.update(default_raw)
    classes_raw = doc.get("classes")
    if isinstance(classes_raw, dict):
        class_raw = classes_raw.get(class_key)
        if isinstance(class_raw, dict):
            merged.update(class_raw)
            source = "ad_policy_json_class"

    if not merged:
        return None

    reliability = str(merged.get("reliability", merged.get("reliability_prior", "medium"))).lower()
    if reliability not in {"high", "medium", "low"}:
        reliability = "medium"

    ad_weight = _safe_float(merged.get("ad_weight"))
    if ad_weight is None:
        ad_weight = {"high": 0.70, "medium": 0.45, "low": 0.20}.get(reliability, 0.45)
    ad_weight = _clip(float(ad_weight), 0.0, 1.0)

    location_mode = str(merged.get("location_mode", "normal")).lower()
    if location_mode not in {"off", "normal", "strict"}:
        location_mode = "normal"

    min_location_confidence = _safe_float(merged.get("min_location_confidence"))
    if min_location_confidence is None:
        min_location_confidence = {"high": 0.25, "medium": 0.35, "low": 0.45}.get(reliability, 0.35)
    min_location_confidence = _clip(float(min_location_confidence), 0.0, 1.0)

    use_bbox_raw = merged.get("use_bbox")
    if isinstance(use_bbox_raw, bool):
        use_bbox = use_bbox_raw
    else:
        use_bbox = location_mode != "off"

    review_band = _safe_float(merged.get("review_band"))
    legacy_center = _safe_float(merged.get("decision_center"))
    t_low = _safe_float(merged.get("t_low"))
    t_high = _safe_float(merged.get("t_high"))
    if t_low is not None and t_high is not None and t_low > t_high:
        t_low, t_high = t_high, t_low

    if legacy_center is None and t_low is not None and t_high is not None:
        legacy_center = (t_low + t_high) / 2.0
    if review_band is None:
        if t_low is not None and t_high is not None:
            review_band = (t_high - t_low) / 2.0
        else:
            review_band = 0.08
    review_band = _clip(float(review_band), 0.02, 0.30)

    out: dict[str, Any] = {
        "class_key": class_key,
        "source": source,
        "reliability": reliability,
        "ad_weight": ad_weight,
        "location_mode": location_mode,
        "min_location_confidence": min_location_confidence,
        "use_bbox": use_bbox,
        "review_band": review_band,
        "legacy_center": legacy_center,
    }
    if t_low is not None:
        out["t_low"] = float(t_low)
    if t_high is not None:
        out["t_high"] = float(t_high)
    return out


def _resolve_class_policy_from_bounds(raw: dict[str, Any], class_key: str) -> dict[str, Any]:
    reliability = str(raw.get("reliability", "medium")).lower()
    if reliability not in {"high", "medium", "low"}:
        reliability = "medium"

    ad_weight = _safe_float(raw.get("ad_weight"))
    if ad_weight is None:
        ad_weight = {"high": 0.70, "medium": 0.45, "low": 0.20}.get(reliability, 0.45)
    ad_weight = _clip(float(ad_weight), 0.0, 1.0)

    location_mode = str(raw.get("location_mode", "normal")).lower()
    if location_mode not in {"off", "normal", "strict"}:
        location_mode = "normal"

    min_location_confidence = _safe_float(raw.get("min_location_confidence"))
    if min_location_confidence is None:
        min_location_confidence = {"high": 0.25, "medium": 0.35, "low": 0.45}.get(reliability, 0.35)
    min_location_confidence = _clip(float(min_location_confidence), 0.0, 1.0)

    use_bbox_raw = raw.get("use_bbox")
    if isinstance(use_bbox_raw, bool):
        use_bbox = use_bbox_raw
    else:
        use_bbox = location_mode != "off"

    t_low = _safe_float(raw.get("t_low"))
    t_high = _safe_float(raw.get("t_high"))
    if t_low is not None and t_high is not None and t_low > t_high:
        t_low, t_high = t_high, t_low

    if t_low is None:
        t_low = 0.5
    if t_high is None:
        t_high = 0.8
    if t_high <= t_low:
        center = _clip((t_low + t_high) / 2.0, 0.0, 1.0)
        t_low = _clip(center - 0.05, 0.0, 1.0)
        t_high = _clip(center + 0.05, 0.0, 1.0)

    return {
        "class_key": class_key,
        "source": str(raw.get("source", "default")),
        "reliability": reliability,
        "ad_weight": ad_weight,
        "location_mode": location_mode,
        "min_location_confidence": min_location_confidence,
        "use_bbox": use_bbox,
        "review_band": _clip((t_high - t_low) / 2.0, 0.02, 0.30),
        "legacy_center": (t_low + t_high) / 2.0,
        "t_low": float(t_low),
        "t_high": float(t_high),
    }


def _resolve_class_calibration(doc: dict[str, Any], class_key: str) -> dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    classes = doc.get("classes")
    if not isinstance(classes, dict):
        return {}
    raw = classes.get(class_key)
    if not isinstance(raw, dict):
        return {}

    center = (
        _safe_float(raw.get("center_threshold"))
        or _safe_float(raw.get("decision_threshold"))
        or _safe_float(raw.get("threshold_opt"))
    )
    review_band = (
        _safe_float(raw.get("review_band"))
        or _safe_float(raw.get("uncertainty_band"))
        or _safe_float(raw.get("decision_band"))
    )

    out: dict[str, Any] = {}
    if center is not None:
        out["center_threshold"] = float(center)
    if review_band is not None:
        out["review_band"] = _clip(float(review_band), 0.02, 0.30)
    return out


def _decide_with_policy(
    anomaly_score: float,
    class_policy: dict[str, Any],
    *,
    model_threshold: float,
    class_calibration: dict[str, Any] | None = None,
) -> tuple[str, bool, dict[str, Any]]:
    _ = class_calibration  # Kept for API compatibility; decision now uses explicit t_low/t_high only.

    t_low = _safe_float(class_policy.get("t_low"))
    t_high = _safe_float(class_policy.get("t_high"))
    if t_low is None or t_high is None:
        center = _safe_float(class_policy.get("legacy_center"))
        if center is None:
            center = float(model_threshold)
        center = _clip(float(center), 0.0, 1.0)
        review_band = _safe_float(class_policy.get("review_band"))
        if review_band is None:
            review_band = 0.08
        band = _clip(float(review_band), 0.02, 0.35)
        t_low = _clip(center - band, 0.0, 1.0)
        t_high = _clip(center + band, 0.0, 1.0)
    else:
        t_low = _clip(float(t_low), 0.0, 1.0)
        t_high = _clip(float(t_high), 0.0, 1.0)
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        center = _clip((t_low + t_high) / 2.0, 0.0, 1.0)
        band = _clip((t_high - t_low) / 2.0, 0.02, 0.35)

    score = float(anomaly_score)
    score_delta = score - center
    if score >= t_high:
        decision = "anomaly"
        review_needed = False
    elif score <= t_low:
        decision = "normal"
        review_needed = False
    else:
        decision = "review_needed"
        review_needed = True

    if review_needed:
        # Confidence is lower near the center of [t_low, t_high], higher near boundaries.
        half_band = max(1e-6, (t_high - t_low) / 2.0)
        near_boundary = min(score - t_low, t_high - score)
        decision_confidence = round(_clip(near_boundary / half_band, 0.0, 1.0), 4)
    elif decision == "anomaly":
        denom = max(1e-6, 1.0 - t_high)
        decision_confidence = round(_clip((score - t_high) / denom, 0.0, 1.0), 4)
    else:
        denom = max(1e-6, t_low)
        decision_confidence = round(_clip((t_low - score) / denom, 0.0, 1.0), 4)

    return decision, review_needed, {
        "decision_confidence": decision_confidence,
        "basis": {
            "center_threshold": round(center, 4),
            "uncertainty_band": round(float(band), 4),
            "score_delta": round(float(score_delta), 4),
            "used_calibration": False,
            "t_low": round(float(t_low), 4),
            "t_high": round(float(t_high), 4),
        },
    }


def _to_bool_like(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"true", "1", "yes", "y", "anomaly", "ng", "defect", "bad"}:
        return True
    if s in {"false", "0", "no", "n", "normal", "ok", "good"}:
        return False
    return None


def _final_decision_from_llm(llm_response: dict[str, Any]) -> str:
    llm_flag = _to_bool_like(llm_response.get("is_anomaly_llm"))
    if llm_flag is True:
        return "anomaly"
    if llm_flag is False:
        return "normal"
    return "review_needed"


def _final_decision_with_guardrail(
    *,
    llm_response: dict[str, Any],
    ad_decision: str,
    ad_data: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    _ = ad_data
    _ = policy
    ad_decision_norm = str(ad_decision or "").strip().lower()
    # Outside [t_low, t_high] -> follow AD decision.
    if ad_decision_norm in {"anomaly", "normal"}:
        return ad_decision_norm
    # Inside [t_low, t_high] (review_needed) -> let LLM decide.
    return _final_decision_from_llm(llm_response)


def _has_llm_content(llm_response: dict[str, Any]) -> bool:
    summary = llm_response.get("llm_summary")
    report = llm_response.get("llm_report")
    if isinstance(summary, dict) and summary:
        return True
    if isinstance(summary, str) and summary.strip():
        return True
    if isinstance(report, dict):
        for key in ("summary", "description", "possible_cause", "recommendation", "raw_response"):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return bool(report)
    return False


def _safe_update_report(conn, report_id: int, payload: dict[str, Any], *, context: str) -> bool:
    """Best-effort DB update with one rollback+retry for aborted transactions."""
    try:
        pg.update_report(conn, report_id, payload)
        return True
    except Exception:
        logger.exception("Report update failed (%s) | report_id=%s", context, report_id)
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            pg.update_report(conn, report_id, payload)
            return True
        except Exception:
            logger.exception("Report update retry failed (%s) | report_id=%s", context, report_id)
            return False


def _mark_pipeline_failed(
    conn,
    *,
    report_id: int,
    stage: str,
    error: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = dict(extra or {})
    payload.update({
        "ad_decision": "review_needed",
        "pipeline_status": "failed",
        "pipeline_stage": stage,
        "pipeline_error": str(error)[:500],
    })
    _safe_update_report(conn, report_id, payload, context=f"mark_failed:{stage}")


def _finalize_rag_llm_pipeline(
    app: FastAPI,
    *,
    report_id: int,
    model_category: str,
    ad_result: dict[str, Any],
    ad_decision: str,
    policy: dict[str, Any],
    domain_rag_enabled: bool | None = None,
) -> None:
    effective_domain_rag = domain_rag_enabled if domain_rag_enabled is not None else DOMAIN_RAG_ENABLED
    conn = pg.connect_fast(DATABASE_URL)
    ref_path: str | None = None
    try:
        _safe_update_report(
            conn,
            report_id,
            {"pipeline_status": "processing", "pipeline_stage": "rag_running"},
            context="rag_running",
        )

        try:
            rag_results = app.state.rag_service.search_closest_normal(ad_result["original_path"], model_category)
            ref_path = rag_results[0]["path"] if rag_results else None
            _safe_update_report(
                conn,
                report_id,
                {
                    "similar_image_path": ref_path,
                    "pipeline_status": "processing",
                    "pipeline_stage": "rag_done",
                },
                context="rag_done",
            )
        except Exception as exc:
            logger.exception("RAG stage failed for report_id=%s", report_id)
            _safe_update_report(
                conn,
                report_id,
                {
                    "pipeline_status": "processing",
                    "pipeline_stage": "rag_failed",
                    "pipeline_error": f"RAG: {str(exc)[:500]}",
                },
                context="rag_failed",
            )

        _safe_update_report(
            conn,
            report_id,
            {"pipeline_status": "processing", "pipeline_stage": "llm_waiting_lock"},
            context="llm_waiting_lock",
        )
        llm_started = datetime.now()
        acquired = app.state.llm_lock.acquire(timeout=LLM_LOCK_TIMEOUT_SEC)
        if not acquired:
            _mark_pipeline_failed(
                conn,
                report_id=report_id,
                stage="llm_lock_timeout",
                error=f"LLM lock wait timeout ({LLM_LOCK_TIMEOUT_SEC:.1f}s)",
            )
            return

        try:
            _safe_update_report(
                conn,
                report_id,
                {"pipeline_status": "processing", "pipeline_stage": "llm_running"},
                context="llm_running",
            )
            llm_response = app.state.llm_service.generate_dynamic_report(
                ad_result["original_path"],
                ref_path,
                model_category,
                ad_result,
                policy,
                domain_rag_enabled=effective_domain_rag,
            )
        finally:
            app.state.llm_lock.release()

        if not _has_llm_content(llm_response):
            _mark_pipeline_failed(
                conn,
                report_id=report_id,
                stage="llm_empty",
                error="LLM returned empty/invalid content",
                extra=llm_response,
            )
            return

        llm_decision = _final_decision_with_guardrail(
            llm_response=llm_response,
            ad_decision=ad_decision,
            ad_data=ad_result,
            policy=policy,
        )
        _safe_update_report(
            conn,
            report_id,
            {
                **llm_response,
                "ad_decision": llm_decision,
                "llm_start_time": llm_started,
                "pipeline_status": "completed",
                "pipeline_stage": "llm_done",
                "pipeline_error": None,
            },
            context="llm_done",
        )
    except Exception as exc:
        logger.exception("LLM stage failed for report_id=%s", report_id)
        _mark_pipeline_failed(
            conn,
            report_id=report_id,
            stage="llm_failed",
            error=f"LLM: {str(exc)[:500]}",
        )
    finally:
        conn.close()


def _set_llm_model(app: FastAPI, model_name: str) -> None:
    selected = model_name.strip().lower()
    if not selected:
        raise ValueError("Model name must not be empty")
    normalized = LLM_MODEL_ALIASES.get(selected, selected)
    if normalized != selected:
        logger.info("Normalized LLM model alias: %s -> %s", selected, normalized)
    selected = normalized
    app.state.llm_client = get_llm_client(selected)
    app.state.llm_service = LlmService(
        client=app.state.llm_client,
        domain_rag_service=getattr(app.state, "domain_rag_service", None),
    )
    app.state.llm_model = selected


def _checkpoint_exists_for_category_key(category_key: str) -> bool:
    key = category_key.strip().strip("/")
    if not key:
        return False

    roots = [CHECKPOINT_DIR / "Patchcore", CHECKPOINT_DIR]
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        category_dir = root / key
        if category_dir.exists() and category_dir.is_dir():
            for v_dir in category_dir.iterdir():
                if not v_dir.is_dir() or not v_dir.name.startswith("v"):
                    continue
                if (v_dir / "model.ckpt").exists():
                    return True
    return False


def _discover_model_category_key(category: str) -> str | None:
    selected = category.strip().strip("/")
    if not selected or "/" in selected:
        return None

    roots = [CHECKPOINT_DIR / "Patchcore", CHECKPOINT_DIR]
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)

        dataset_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)
        for ds_dir in dataset_dirs:
            if ds_dir.name.startswith("."):
                continue
            candidate = f"{ds_dir.name}/{selected}"
            if _checkpoint_exists_for_category_key(candidate):
                return candidate
    return None


def _resolve_model_category(dataset: str, category: str) -> str:
    selected = category.strip()
    if not selected:
        raise ValueError("Category must not be empty")
    if "/" in selected:
        return selected

    ds = dataset.strip().strip("/")
    if ds:
        candidate = f"{ds}/{selected}"
        if _checkpoint_exists_for_category_key(candidate):
            return candidate
    discovered = _discover_model_category_key(selected)
    if discovered:
        return discovered
    return selected


def _normalize_line(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    up = s.upper()
    if up.startswith("LINE-"):
        return up

    m = LINE_PATTERN.search(s)
    if not m:
        if len(s) == 1 and s.isalpha():
            return f"LINE-{s.upper()}-01"
        return None

    token = m.group(1).upper()
    if len(token) == 1 and token.isalpha():
        return f"LINE-{token}-01"
    return f"LINE-{token}"


def _resolve_incoming_context(image_path: Path) -> tuple[str, str, str] | None:
    try:
        rel_parts = image_path.relative_to(INCOMING_ROOT).parts
    except ValueError:
        rel_parts = image_path.parts

    # rel_parts includes filename as the last element.
    dir_parts = rel_parts[:-1] if len(rel_parts) >= 1 else ()

    dataset = INCOMING_DEFAULT_DATASET
    category = INCOMING_DEFAULT_CATEGORY

    if len(dir_parts) >= 2:
        dataset = dir_parts[0] or INCOMING_DEFAULT_DATASET
        category = dir_parts[1] or INCOMING_DEFAULT_CATEGORY
    elif len(dir_parts) == 1:
        # incoming/{batch}/image.png 형태: dataset/category는 default env 사용.
        dataset = INCOMING_DEFAULT_DATASET or dir_parts[0]
        category = INCOMING_DEFAULT_CATEGORY

    line: str | None = None
    # 권장 구조: incoming/{dataset}/{category}/{line}/{batch}/image.png
    if len(dir_parts) >= 3:
        line = _normalize_line(dir_parts[2])

    # line 디렉토리가 없으면 하위 경로(배치명 등)에서 line token 추출
    if line is None:
        if len(dir_parts) >= 3:
            tokens = list(dir_parts[2:])
        else:
            tokens = list(dir_parts)
        for token in reversed(tokens):
            line = _normalize_line(token)
            if line is not None:
                break

    line = line or INCOMING_DEFAULT_LINE

    if not category:
        return None
    return dataset, category, line


def _collect_incoming_images(root: Path) -> list[Path]:
    items: list[tuple[float, Path]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in INCOMING_IMAGE_EXTS:
            continue
        try:
            items.append((p.stat().st_mtime, p))
        except OSError:
            continue
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def _is_stable_file(path: Path) -> bool:
    try:
        age = datetime.now().timestamp() - path.stat().st_mtime
        return age >= INCOMING_STABLE_SECONDS
    except OSError:
        return False


def _run_inspection_pipeline(
    app: FastAPI,
    *,
    conn,
    file_obj,
    category: str,
    dataset: str,
    line: str,
    ingest_source_path: str | None,
    domain_rag_enabled: bool | None = None,
) -> dict[str, Any]:
    model_category = _resolve_model_category(dataset=dataset, category=category)
    class_keys = _candidate_class_keys(dataset=dataset, model_category=model_category)

    policy: dict[str, Any] | None = None
    for key in class_keys:
        policy = _resolve_class_policy_from_json(getattr(app.state, "ad_policy_doc", {}), key)
        if policy is not None:
            break
    if policy is None:
        db_policy = pg.get_category_policy(conn, model_category)
        policy = _resolve_class_policy_from_bounds(db_policy, model_category)
    policy_source = str(policy.get("source", "default"))

    if hasattr(file_obj, "file"):
        file_obj.file.seek(0)

    ad_start_time = datetime.now()
    # GPU/model access is serialized for stability; downstream RAG/LLM stays concurrent.
    with app.state.inspect_lock:
        ad_results = app.state.ad_service.predict_batch(
            [file_obj],
            category=model_category,
            dataset=dataset,
            threshold=None,
        )
    ad_result = ad_results[0]
    score = float(ad_result["ad_score"])
    model_thr = float(ad_result.get("model_threshold", 0.5))
    model_is_anomaly = bool(ad_result.get("model_is_anomaly", score > model_thr))

    calibration: dict[str, Any] = {}
    for key in class_keys:
        calibration = _resolve_class_calibration(getattr(app.state, "ad_calibration_doc", {}), key)
        if calibration:
            break

    ad_decision, review_needed, decision_meta = _decide_with_policy(
        score,
        policy,
        model_threshold=model_thr,
        class_calibration=calibration,
    )
    basis = decision_meta.get("basis", {})
    basis_t_low = _safe_float(basis.get("t_low"))
    basis_t_high = _safe_float(basis.get("t_high"))
    if basis_t_low is not None and basis_t_high is not None:
        t_low = _clip(float(basis_t_low), 0.0, 1.0)
        t_high = _clip(float(basis_t_high), 0.0, 1.0)
        if t_low > t_high:
            t_low, t_high = t_high, t_low
        center = _clip((t_low + t_high) / 2.0, 0.0, 1.0)
        band = max(1e-6, (t_high - t_low) / 2.0)
    else:
        center = _clip(float(_safe_float(basis.get("center_threshold")) or model_thr), 0.0, 1.0)
        band = _clip(float(_safe_float(basis.get("uncertainty_band")) or policy.get("review_band", 0.08)), 0.0, 0.35)
        t_low = _clip(center - band, 0.0, 1.0)
        t_high = _clip(center + band, 0.0, 1.0)
    applied_policy = {
        "class_key": str(policy.get("class_key", model_category)),
        "source": policy_source,
        "reliability": str(policy.get("reliability", "medium")),
        "review_band": round(float(policy.get("review_band", band)), 4),
        "t_low": round(float(t_low), 4),
        "t_high": round(float(t_high), 4),
        "decision_basis": basis,
        "used_calibration": bool(calibration),
    }

    logger.info(
        "AD decision | category=%s score=%.4f center=%.4f band=%.4f range=[%.4f, %.4f] "
        "source=%s model_thr=%.4f model_is_anomaly=%s review_needed=%s decision=%s",
        model_category,
        score,
        center,
        band,
        t_low,
        t_high,
        policy_source,
        model_thr,
        model_is_anomaly,
        review_needed,
        ad_decision,
    )
    ad_result["ad_decision"] = ad_decision
    ad_result["review_needed"] = review_needed
    ad_result["decision_confidence"] = decision_meta.get("decision_confidence")
    ad_result["decision_basis"] = basis
    ad_result["ingest_source_path"] = ingest_source_path

    initial_data = {
        "dataset": dataset,
        "category": model_category,
        "line": line,
        "ad_score": score,
        # 최종 판정 기준은 LLM이다. LLM 완료 전에는 review_needed로 유지한다.
        "ad_decision": "review_needed",
        "is_anomaly_ad": ad_decision == "anomaly",
        "has_defect": ad_result.get("has_defect", False),
        "region": ad_result.get("region"),
        "area_ratio": ad_result.get("area_ratio"),
        "bbox": ad_result.get("bbox"),
        "center": ad_result.get("center"),
        "confidence": ad_result.get("confidence"),
        "ingest_source_path": ingest_source_path,
        "image_path": ad_result["original_path"],
        "heatmap_path": ad_result["heatmap_path"],
        "mask_path": ad_result["mask_path"],
        "overlay_path": ad_result["overlay_path"],
        "created_at": ad_start_time,
        "ad_inference_duration": (datetime.now() - ad_start_time).total_seconds(),
        "pipeline_status": "processing",
        "pipeline_stage": "ad_done",
        "applied_policy": applied_policy,
    }
    report_id = pg.insert_report(conn, initial_data)

    try:
        app.state.pipeline_executor.submit(
            _finalize_rag_llm_pipeline,
            app,
            report_id=report_id,
            model_category=model_category,
            ad_result=ad_result,
            ad_decision=ad_decision,
            policy=policy,
            domain_rag_enabled=domain_rag_enabled,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue RAG/LLM pipeline for report_id=%s", report_id)
        pg.update_report(
            conn,
            report_id,
            {
                "pipeline_status": "failed",
                "pipeline_stage": "enqueue_failed",
                "pipeline_error": f"QUEUE: {str(exc)[:500]}",
            },
        )

    return {
        "status": "accepted",
        "report_id": report_id,
        "ad_decision": "review_needed",
        "category": model_category,
        "source_image": ingest_source_path,
        "pipeline_status": "processing",
        "pipeline_stage": "ad_done",
    }


def _run_single_inspection(
    app: FastAPI,
    file_obj,
    category: str,
    dataset: str,
    line: str,
    ingest_source_path: str | None,
    domain_rag_enabled: bool | None = None,
) -> dict[str, Any]:
    conn = pg.connect_fast(DATABASE_URL)
    try:
        return _run_inspection_pipeline(
            app,
            conn=conn,
            file_obj=file_obj,
            category=category,
            dataset=dataset,
            line=line,
            ingest_source_path=ingest_source_path,
            domain_rag_enabled=domain_rag_enabled,
        )
    finally:
        conn.close()


def _scan_incoming_once(app: FastAPI) -> int:
    if not INCOMING_ROOT.exists():
        return 0

    conn = pg.connect_fast(DATABASE_URL)
    processed = 0
    skipped_existing = 0
    skipped_unresolved = 0
    failed = 0
    try:
        for image_path in _collect_incoming_images(INCOMING_ROOT):
            if not _is_stable_file(image_path):
                continue

            source_path = str(image_path.resolve())
            if pg.has_ingest_source_path(conn, source_path):
                skipped_existing += 1
                continue

            resolved = _resolve_incoming_context(image_path)
            if resolved is None:
                skipped_unresolved += 1
                unresolved: set[str] = app.state.incoming_unresolved
                if source_path not in unresolved:
                    logger.warning(
                        "Skipping incoming image without category mapping: %s "
                        "(use incoming/{dataset}/{category}/... structure or set INCOMING_DEFAULT_CATEGORY)",
                        source_path,
                    )
                    unresolved.add(source_path)
                continue
            dataset, category, line = resolved

            local_file = LocalImageUpload(image_path)
            try:
                _run_inspection_pipeline(
                    app,
                    conn=conn,
                    file_obj=local_file,
                    category=category,
                    dataset=dataset,
                    line=line,
                    ingest_source_path=source_path,
                )
                processed += 1
                app.state.incoming_unresolved.discard(source_path)
                logger.info("Incoming image processed: %s", source_path)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                failed += 1
                logger.exception("Incoming image processing failed: %s", source_path)
            finally:
                local_file.close()

        if processed > 0 or failed > 0 or skipped_unresolved > 0:
            logger.info(
                "Incoming scan summary | processed=%d skipped_existing=%d unresolved=%d failed=%d",
                processed,
                skipped_existing,
                skipped_unresolved,
                failed,
            )
        return processed
    finally:
        conn.close()


def _mark_stale_pipelines_once() -> int:
    conn = pg.connect_fast(DATABASE_URL)
    try:
        updated = pg.mark_stale_processing_reports(
            conn,
            stale_seconds=PIPELINE_STALE_SECONDS,
            limit=1000,
        )
        return int(updated)
    finally:
        conn.close()


async def _incoming_watch_loop(app: FastAPI) -> None:
    logger.info(
        "Incoming watcher enabled | root=%s | interval=%.1fs | stable=%.1fs",
        INCOMING_ROOT,
        INCOMING_SCAN_INTERVAL_SEC,
        INCOMING_STABLE_SECONDS,
    )
    while True:
        try:
            processed = await run_in_threadpool(_scan_incoming_once, app)
            if processed > 0:
                logger.info("Incoming watcher processed %d image(s)", processed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Incoming watcher loop failed")

        await asyncio.sleep(max(1.0, INCOMING_SCAN_INTERVAL_SEC))


async def _pipeline_watchdog_loop() -> None:
    logger.info(
        "Pipeline watchdog enabled | interval=%.1fs | stale_after=%ds",
        PIPELINE_WATCHDOG_INTERVAL_SEC,
        PIPELINE_STALE_SECONDS,
    )
    while True:
        try:
            updated = await run_in_threadpool(_mark_stale_pipelines_once)
            if updated > 0:
                logger.warning(
                    "Pipeline watchdog marked %d stale report(s) as failed",
                    updated,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pipeline watchdog loop failed")
        await asyncio.sleep(PIPELINE_WATCHDOG_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot-time DB connectivity and additive migrations
    bootstrap_conn = pg.connect(DATABASE_URL, ensure_schema=True)
    ad_policy_doc = _load_ad_policy_doc(AD_POLICY_PATH)
    class_count = len(ad_policy_doc.get("classes", {})) if isinstance(ad_policy_doc, dict) else 0
    logger.info(
        "Boot policy sync start | ad_policy_path=%s classes=%d",
        AD_POLICY_PATH,
        class_count,
    )
    synced = _sync_category_metadata_from_policy_doc(
        bootstrap_conn,
        ad_policy_doc,
        default_line=INCOMING_DEFAULT_LINE,
    )
    if synced > 0:
        logger.info("Synced %d category_metadata row(s) from AD policy doc", synced)
    else:
        logger.warning("No category_metadata rows synced from AD policy doc")
    bootstrap_conn.close()

    app.state.ad_service = AdService(
        checkpoint_dir=str(CHECKPOINT_DIR),
        output_root=str(OUTPUT_DIR),
    )
    app.state.ad_policy_doc = ad_policy_doc
    app.state.ad_calibration_doc = _load_ad_calibration_doc(AD_CALIBRATION_PATH)
    app.state.rag_service = RagService(index_dir=str(RAG_INDEX_DIR))
    app.state.domain_rag_service = DomainKnowledgeRagService(
        json_path=str(DOMAIN_KNOWLEDGE_JSON_PATH),
        persist_dir=str(DOMAIN_RAG_PERSIST_DIR),
        enabled=DOMAIN_RAG_ENABLED,
        top_k=DOMAIN_RAG_TOP_K,
    )
    _set_llm_model(app, MODEL_NAME)
    app.state.inspect_lock = threading.Lock()
    app.state.llm_lock = threading.Lock()
    app.state.pipeline_executor = ThreadPoolExecutor(max_workers=PIPELINE_WORKERS)
    app.state.incoming_unresolved = set()

    incoming_task: asyncio.Task | None = None
    watchdog_task: asyncio.Task | None = asyncio.create_task(_pipeline_watchdog_loop())
    if INCOMING_WATCH_ENABLED:
        INCOMING_ROOT.mkdir(parents=True, exist_ok=True)
        incoming_task = asyncio.create_task(_incoming_watch_loop(app))

    try:
        yield
    finally:
        if incoming_task is not None:
            incoming_task.cancel()
            try:
                await incoming_task
            except asyncio.CancelledError:
                pass
        if watchdog_task is not None:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
        try:
            app.state.pipeline_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            logger.exception("Failed to shutdown pipeline executor")


app = FastAPI(title="Industrial AI Inspection System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", CachedStaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="outputs")
app.mount("/home/ubuntu/dataset", CorsStaticFiles(directory=str(DATA_DIR), check_dir=False), name="datasets")


@app.get("/settings/llm-model")
async def get_llm_model_settings():
    return {
        "active_model": str(getattr(app.state, "llm_model", MODEL_NAME)),
        "available_models": list_llm_models(),
    }


@app.put("/settings/llm-model")
async def update_llm_model_settings(payload: LlmModelUpdateRequest):
    model_name = payload.model.strip()
    try:
        _set_llm_model(app, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to switch LLM model")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "active_model": str(app.state.llm_model),
        "available_models": list_llm_models(),
    }


@app.get("/incoming/status")
async def get_incoming_status():
    return {
        "enabled": INCOMING_WATCH_ENABLED,
        "root": str(INCOMING_ROOT),
        "scan_interval_sec": INCOMING_SCAN_INTERVAL_SEC,
        "stable_seconds": INCOMING_STABLE_SECONDS,
        "pipeline_workers": PIPELINE_WORKERS,
        "llm_lock_timeout_sec": LLM_LOCK_TIMEOUT_SEC,
        "pipeline_watchdog_interval_sec": PIPELINE_WATCHDOG_INTERVAL_SEC,
        "pipeline_stale_seconds": PIPELINE_STALE_SECONDS,
        "domain_rag_enabled": DOMAIN_RAG_ENABLED,
        "domain_rag_json_path": str(DOMAIN_KNOWLEDGE_JSON_PATH),
        "domain_rag_persist_dir": str(DOMAIN_RAG_PERSIST_DIR),
        "domain_rag_top_k": DOMAIN_RAG_TOP_K,
        "default_dataset": INCOMING_DEFAULT_DATASET,
        "default_category": INCOMING_DEFAULT_CATEGORY,
        "default_line": INCOMING_DEFAULT_LINE,
    }


@app.post("/inspect")
async def run_inspection(
    file: UploadFile = File(...),
    category: str = Form(...),
    dataset: str = Form("default"),
    line: str = Form("line_1"),
    domain_rag_enabled: bool | None = Form(None),
):
    """Run AD immediately and enqueue RAG/LLM stages for asynchronous completion."""
    try:
        file.file.seek(0)
        return await run_in_threadpool(
            _run_single_inspection,
            app,
            file,
            category,
            dataset,
            line,
            None,
            domain_rag_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inspection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reports")
async def get_reports(
    category: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(500, ge=1, le=5000),
    limit: Optional[int] = Query(None, ge=1, le=5000),
    offset: Optional[int] = Query(None, ge=0),
    include_full: bool = Query(False),
):
    """Fetch reports in the same shape expected by current frontend."""
    if limit is not None or offset is not None:
        effective_limit = int(limit or page_size)
        effective_offset = int(offset or 0)
        effective_page = (effective_offset // max(1, effective_limit)) + 1
    else:
        effective_limit = int(page_size)
        effective_offset = (page - 1) * page_size
        effective_page = int(page)

    conn = pg.connect_fast(DATABASE_URL)
    try:
        total = pg.count_filtered_reports(
            conn,
            category=category,
            decision=decision,
        )
        reports = pg.get_filtered_reports(
            conn,
            category=category,
            decision=decision,
            limit=effective_limit,
            offset=effective_offset,
            include_full=include_full,
        )
    except Exception as exc:
        logger.exception("Report list fetch failed")
        raise HTTPException(status_code=500, detail="데이터 조회 중 오류가 발생했습니다.") from exc
    finally:
        conn.close()

    return {
        "page": effective_page,
        "page_size": effective_limit,
        "offset": effective_offset,
        "total_count": total,
        "total": total,
        "items": reports,
    }
