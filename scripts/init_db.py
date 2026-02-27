"""Initialize PostgreSQL DB schema for inspection_reports.

Usage:
    python scripts/init_db.py --dsn "postgresql://son:1234@localhost/inspection"
    python scripts/init_db.py --dsn "postgresql://lion_user:lion123@localhost/lion_db"
"""
from __future__ import annotations

import argparse
import sys

import psycopg2


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspection_reports (
    id                     SERIAL PRIMARY KEY,
    dataset                VARCHAR(50),
    category               VARCHAR(100),
    line                   VARCHAR(50),

    ad_score               FLOAT,
    ad_decision            VARCHAR(20),
    is_anomaly_AD          BOOLEAN,

    has_defect             BOOLEAN,
    region                 TEXT,
    area_ratio             FLOAT,

    image_path             TEXT,
    heatmap_path           TEXT,
    mask_path              TEXT,
    overlay_path           TEXT,
    similar_image_path     TEXT,

    is_anomaly_LLM         BOOLEAN,
    llm_report             TEXT,
    llm_summary            TEXT,

    applied_policy         JSONB DEFAULT '{}'::jsonb,
    ad_inference_duration  FLOAT,
    llm_inference_duration FLOAT,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reports_category ON inspection_reports(category);
CREATE INDEX IF NOT EXISTS idx_reports_decision  ON inspection_reports(ad_decision);
CREATE INDEX IF NOT EXISTS idx_reports_created   ON inspection_reports(created_at);
"""


def main():
    parser = argparse.ArgumentParser(description="Initialize inspection_reports table")
    parser.add_argument("--dsn", type=str, default="postgresql://son:1234@localhost/inspection",
                        help="PostgreSQL DSN")
    parser.add_argument("--drop", action="store_true",
                        help="기존 테이블 삭제 후 재생성")
    args = parser.parse_args()

    try:
        conn = psycopg2.connect(args.dsn)
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        sys.exit(1)

    with conn.cursor() as cur:
        if args.drop:
            cur.execute("DROP TABLE IF EXISTS inspection_reports CASCADE;")
            print("기존 테이블 삭제 완료")

        cur.execute(SCHEMA_SQL)

    conn.commit()
    conn.close()
    print(f"완료: inspection_reports 테이블 생성 ({args.dsn.split('@')[-1]})")


if __name__ == "__main__":
    main()
