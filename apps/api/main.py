from __future__ import annotations
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.storage.pg import connect, insert_report, list_reports, get_report

app = FastAPI(title="MMAD Inspector API")

# CORS — 프론트엔드에서 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL 연결
PG_DSN = os.environ.get("PG_DSN", "postgresql://son:1234@localhost/inspection")
conn = connect(PG_DSN)


# ── API Endpoints ──────────────────────────────────────────────

@app.get("/reports")
def reports(limit: int = 50):
    """최근 N개 리포트 목록 조회."""
    return {"items": list_reports(conn, limit=limit)}


@app.get("/reports/{report_id}")
def report_detail(report_id: int):
    """리포트 단건 조회."""
    r = get_report(conn, report_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return r
