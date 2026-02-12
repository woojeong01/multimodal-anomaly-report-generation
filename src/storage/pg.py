"""PostgreSQL CRUD for inspection_reports table."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, Json

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspection_reports (
  id                     SERIAL PRIMARY KEY,
  dataset                VARCHAR(50),
  category               VARCHAR(50),
  line                   VARCHAR(50),
  image_path             TEXT,
  heatmap_path           TEXT,
  mask_path              TEXT,
  similar_image_path     TEXT,
  ad_score               FLOAT,
  is_anomaly_AD          BOOLEAN,
  AD_start_time          TIMESTAMP,
  AD_inference_duration  FLOAT,
  is_anomaly_LLM         BOOLEAN,
  llm_report             JSONB,
  llm_summary            JSONB,
  llm_start_time         TIMESTAMP,
  llm_inference_duration FLOAT
);
"""


def connect(dsn: str) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL and ensure tables exist."""
    conn = psycopg2.connect(dsn)
    create_tables(conn)
    return conn


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create inspection_reports table if not exists."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def insert_report(conn: psycopg2.extensions.connection, data: dict) -> int:
    """Insert a report row and return its id.

    Args:
        conn: psycopg2 connection
        data: Dict with keys matching inspection_reports columns.
              llm_report and llm_summary are automatically wrapped with Json().

    Returns:
        The auto-generated report id.
    """
    columns = [
        "dataset", "category", "line",
        "image_path", "heatmap_path", "mask_path", "similar_image_path",
        "ad_score", "is_anomaly_AD", "AD_start_time", "AD_inference_duration",
        "is_anomaly_LLM", "llm_report", "llm_summary",
        "llm_start_time", "llm_inference_duration",
    ]

    values = []
    for col in columns:
        val = data.get(col)
        if col in ("llm_report", "llm_summary") and val is not None:
            val = Json(val)
        values.append(val)

    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)

    sql = f"INSERT INTO inspection_reports ({col_names}) VALUES ({placeholders}) RETURNING id"

    with conn.cursor() as cur:
        cur.execute(sql, values)
        report_id = cur.fetchone()[0]
    conn.commit()
    return report_id


def list_reports(
    conn: psycopg2.extensions.connection,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return the most recent N reports."""
    sql = "SELECT * FROM inspection_reports ORDER BY id DESC LIMIT %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_report(
    conn: psycopg2.extensions.connection,
    report_id: int,
) -> Optional[Dict[str, Any]]:
    """Return a single report by id, or None."""
    sql = "SELECT * FROM inspection_reports WHERE id = %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (report_id,))
        row = cur.fetchone()
    return dict(row) if row else None
