from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..common.io import imread_bgr
from ..structure.defect import structure_from_heatmap
from ..structure.render import save_heatmap_and_overlay
from ..report.schema import load_schema, validate_report

logger = logging.getLogger(__name__)


def _stem(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


class InspectionPipeline:
    """End-to-end pipeline: AD inference -> LLM report -> PostgreSQL storage."""

    def __init__(
        self,
        *,
        anomaly_model,
        mllm_client,
        runtime_cfg,
        pg_conn=None,
    ):
        self.anomaly_model = anomaly_model
        self.mllm = mllm_client
        self.cfg = runtime_cfg
        self.schema = load_schema(self.cfg.report["schema_path"])
        self.pg_conn = pg_conn

    # ── main entry point ───────────────────────────────────────────

    def inspect(
        self,
        image_abs: str,
        *,
        dataset: str = "",
        category: str = "",
        line: str | None = None,
        templates_abs: list[str] | None = None,
        mask_path: str | None = None,
        similar_image_path: str | None = None,
        save_to_db: bool = True,
    ) -> dict:
        """Run full inspection and optionally store the result in PostgreSQL.

        Returns:
            A dict matching the inspection_reports schema.
        """
        # ── 1. AD inference ────────────────────────────────────────
        img = imread_bgr(image_abs)
        templates_bgr = [imread_bgr(p) for p in (templates_abs or [])]

        ad_start = datetime.now(timezone.utc)
        t0 = time.time()
        ar = self.anomaly_model.infer(img, templates_bgr=templates_bgr)
        ad_duration = round(time.time() - t0, 3)

        is_anomaly_ad = bool(ar.score > 0.5)

        # Save heatmap artifact
        art_dir = Path(self.cfg.paths.artifact_root) / "artifacts"
        artifacts = save_heatmap_and_overlay(img, ar.heatmap, art_dir, _stem(image_abs))

        # ── 2. AD info for LLM context ────────────────────────────
        structured = structure_from_heatmap(ar.heatmap)
        ad_info = {
            "anomaly_score": float(ar.score),
            "is_anomaly": is_anomaly_ad,
            "defect_location": structured,
        }

        # ── 3. LLM report generation ─────────────────────────────
        llm_start = datetime.now(timezone.utc)
        llm_result = self.mllm.generate_report(
            image_path=image_abs,
            category=category or "unknown",
            ad_info=ad_info,
        )

        # ── 4. Build DB row ───────────────────────────────────────
        report = {
            "dataset": dataset,
            "category": category,
            "line": line,
            "image_path": image_abs,
            "heatmap_path": artifacts.get("heatmap_path"),
            "mask_path": mask_path,
            "similar_image_path": similar_image_path,
            "ad_score": float(ar.score),
            "is_anomaly_AD": is_anomaly_ad,
            "AD_start_time": ad_start.isoformat(),
            "AD_inference_duration": ad_duration,
            "is_anomaly_LLM": llm_result.get("is_anomaly_LLM"),
            "llm_report": llm_result.get("llm_report"),
            "llm_summary": llm_result.get("llm_summary"),
            "llm_start_time": llm_start.isoformat(),
            "llm_inference_duration": llm_result.get("llm_inference_duration"),
        }

        # ── 5. Validate & persist ─────────────────────────────────
        validate_report(report, self.schema)

        if save_to_db and self.pg_conn is not None:
            from ..storage.pg import insert_report
            report_id = insert_report(self.pg_conn, report)
            report["id"] = report_id
            logger.info("Report saved to PostgreSQL with id=%d", report_id)

        return report
