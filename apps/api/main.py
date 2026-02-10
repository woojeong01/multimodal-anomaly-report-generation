from __future__ import annotations
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from mmad_inspector.service.settings import load_runtime_config
from mmad_inspector.anomaly.dummy_edge import DummyEdgeAnomaly
from mmad_inspector.mllm.echo import EchoMLLM
from mmad_inspector.service.pipeline import InspectionPipeline
from mmad_inspector.storage.db import connect, insert_report, list_reports, get_report

app = FastAPI(title="MMAD Inspector API")

cfg_path = os.environ.get("RUNTIME_CFG", "configs/anomaly.yaml")
cfg = load_runtime_config(cfg_path)

# Step 1 components
anomaly = DummyEdgeAnomaly(**cfg.anomaly.get("dummy", {}))
mllm = EchoMLLM(**cfg.mllm.get("echo", {}))
pipe = InspectionPipeline(anomaly_model=anomaly, mllm_client=mllm, runtime_cfg=cfg)

conn = connect(cfg.paths.db_path)
UPLOAD_DIR = Path(cfg.paths.artifact_root) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class InspectResponse(BaseModel):
    report_id: int
    report: dict

@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".png"
    dst = UPLOAD_DIR / f"upload_{Path(file.filename).stem}{suffix}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    report = pipe.inspect(str(dst))
    report_id = insert_report(conn, report)
    return InspectResponse(report_id=report_id, report=report)

@app.get("/reports")
def reports(limit: int = 50):
    return {"items": list_reports(conn, limit=limit)}

@app.get("/reports/{report_id}")
def report_detail(report_id: int):
    r = get_report(conn, report_id)
    if r is None:
        return {"error": "not_found"}
    return r
