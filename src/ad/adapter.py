"""Adapters between AD outputs and LLM-friendly input schema."""

from __future__ import annotations

from typing import Any, Dict


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "anomaly", "defect", "bad"}
    return False


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_llm_ad_info(prediction: Dict[str, Any] | None) -> Dict[str, Any]:
    """Convert AD prediction payload to the shape expected by LLM prompts.

    Keeps only LLM-relevant fields and normalizes key names from both legacy and
    report-style AD outputs.
    """
    if not isinstance(prediction, dict):
        return {}

    decision = str(prediction.get("decision", "")).strip().lower()
    anomaly_score = _to_float(prediction.get("anomaly_score"))

    # Decision precedence:
    # 1) explicit policy/context decision (normal/anomaly),
    # 2) model flag for review-needed/legacy payloads,
    # 3) score-threshold fallback.
    if decision == "anomaly":
        is_anomaly = True
    elif decision == "normal":
        is_anomaly = False
    elif "is_anomaly" in prediction:
        is_anomaly = _to_bool(prediction.get("is_anomaly"))
    elif "model_is_anomaly" in prediction:
        is_anomaly = _to_bool(prediction.get("model_is_anomaly"))
    else:
        threshold = _to_float(prediction.get("model_threshold"))
        if threshold is None:
            threshold = _to_float(prediction.get("threshold"))
        if anomaly_score is not None and threshold is not None:
            is_anomaly = anomaly_score > threshold
        else:
            is_anomaly = False

    defect_location = prediction.get("defect_location")
    if not isinstance(defect_location, dict):
        defect_location = {}
    if not defect_location and isinstance(prediction.get("defect_location_raw"), dict):
        defect_location = prediction["defect_location_raw"]

    out: Dict[str, Any] = {
        "is_anomaly": is_anomaly,
        "defect_location": defect_location,
    }
    if anomaly_score is not None:
        out["anomaly_score"] = anomaly_score

    # Optional fields useful for LLM-side trust control.
    if isinstance(prediction.get("context_version"), str):
        out["context_version"] = prediction["context_version"]
    if decision:
        out["decision"] = decision
    if "review_needed" in prediction:
        out["review_needed"] = _to_bool(prediction.get("review_needed"))
    if "decision_confidence" in prediction:
        decision_conf = _to_float(prediction.get("decision_confidence"))
        if decision_conf is not None:
            out["decision_confidence"] = decision_conf
    if isinstance(prediction.get("decision_basis"), dict):
        out["decision_basis"] = prediction["decision_basis"]
    if isinstance(prediction.get("class_key"), str):
        out["class_key"] = prediction["class_key"]
    if isinstance(prediction.get("reason_codes"), list):
        out["reason_codes"] = prediction["reason_codes"]
    if isinstance(prediction.get("report_guidance"), dict):
        out["report_guidance"] = prediction["report_guidance"]
    if isinstance(prediction.get("confidence"), dict):
        out["confidence"] = prediction["confidence"]
    if isinstance(prediction.get("map_stats"), dict):
        out["map_stats"] = prediction["map_stats"]
    if "use_location_for_report" in prediction:
        out["use_location_for_report"] = _to_bool(prediction.get("use_location_for_report"))

    return out
