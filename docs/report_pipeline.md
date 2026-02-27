# LLM Report Generation & Storage Pipeline

## 개요

이미지를 멀티모달 LLM에 전달해 구조화된 JSON 결함 리포트를 생성하고,
PostgreSQL에 저장한 뒤 FastAPI로 외부에 제공하는 엔드투엔드 파이프라인.

```
이미지
  │
  ▼
[LLM (Gemini / InternVL)]
  │  프롬프트 + 이미지 → JSON 응답
  ▼
[JSON 파싱 & 정규화]
  │  is_anomaly, report, summary 추출
  ▼
[PostgreSQL]
  │  inspection_reports 테이블에 JSONB로 저장
  ▼
[FastAPI]
  │  GET /reports, GET /reports/{id}
  ▼
프론트엔드 / 대시보드
```

---

## 1. LLM 리포트 생성 (`src/mllm/base.py`)

### 프롬프트 구조

LLM에게 이미지와 함께 아래 형식의 JSON 응답을 요청한다.

```
You are an expert industrial quality inspector.
Product category: {category}

Respond in JSON format ONLY:
{
  "is_anomaly": true or false,
  "report": {
    "anomaly_type": "...",
    "severity": "low/medium/high/none",
    "location": "...",
    "description": "...",
    "confidence": 0.0~1.0,
    "recommendation": "..."
  },
  "summary": {
    "summary": "한 줄 요약",
    "risk_level": "low/medium/high/none"
  }
}
```

AD 모델 결과가 있을 경우 `REPORT_PROMPT_WITH_AD`를 사용해 AD 스코어·위치 정보를 함께 주입한다.

### `generate_report()` 동작 흐름

```python
result = client.generate_report(
    image_path="image.jpg",
    category="cigarette_box",
    ad_info={"anomaly_score": 0.82, "is_anomaly": True}  # 선택
)
```

| 단계 | 내용 |
|------|------|
| 1 | 프롬프트 생성 (`build_report_payload`) |
| 2 | LLM API 호출 (`send_request`) |
| 3 | JSON 파싱 (`_parse_llm_json`) — 마크다운 펜스, 이스케이프 문자 등 처리 |
| 4 | `is_anomaly` 정규화 — bool / 문자열("true","anomaly","불량" 등) 모두 처리 |
| 5 | 결과 반환 |

**반환값 키:**

| 키 | 타입 | 설명 |
|----|------|------|
| `is_anomaly_LLM` | bool | 이상 여부 |
| `llm_report` | dict | anomaly_type, severity, location, description, confidence, recommendation |
| `llm_summary` | dict | summary, risk_level |
| `llm_inference_duration` | float | 추론 시간 (초) |

---

## 2. PostgreSQL 저장 (`src/storage/pg.py`)

### 왜 PostgreSQL인가

LLM이 반환하는 리포트는 중첩 구조의 JSON이다. PostgreSQL의 **JSONB** 타입을 사용하면 JSON을 문자열이 아닌 바이너리로 저장해 내부 필드를 SQL로 바로 쿼리할 수 있다.

```sql
-- JSONB 내부 필드 직접 조회 예시
SELECT llm_report->>'severity' FROM inspection_reports WHERE is_anomaly_LLM = true;
```

### 테이블 스키마 자동 생성

서버 시작 시 `connect()`를 호출하면 테이블이 없을 경우 자동으로 생성된다.

```python
# src/storage/pg.py

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspection_reports (
  id                     SERIAL PRIMARY KEY,   -- 자동 증가 ID
  dataset                VARCHAR(50),          -- 데이터셋명 (GoodsAD, DS-MVTec 등)
  category               VARCHAR(50),          -- 제품 카테고리 (cigarette_box 등)
  line                   VARCHAR(50),          -- 생산 라인 (선택)
  image_path             TEXT,                 -- 원본 이미지 경로
  heatmap_path           TEXT,                 -- AD 히트맵 경로 (선택)
  mask_path              TEXT,                 -- AD 마스크 경로 (선택)
  similar_image_path     TEXT,                 -- Visual RAG 유사 이미지 경로 (선택)
  ad_score               FLOAT,                -- AD 이상 점수
  is_anomaly_AD          BOOLEAN,              -- AD 모델 판정
  AD_start_time          TIMESTAMP,            -- AD 추론 시작 시각
  AD_inference_duration  FLOAT,                -- AD 추론 시간 (초)
  is_anomaly_LLM         BOOLEAN,              -- LLM 판정
  llm_report             JSONB,                -- 상세 리포트 (중첩 JSON)
  llm_summary            JSONB,                -- 요약 (중첩 JSON)
  llm_start_time         TIMESTAMP,            -- LLM 추론 시작 시각
  llm_inference_duration FLOAT                 -- LLM 추론 시간 (초)
);
"""

def connect(dsn: str) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(dsn)
    create_tables(conn)   # 테이블 없으면 자동 생성
    return conn
```

### 데이터 삽입 (`insert_report`)

일반 Python dict를 받아서 JSONB 컬럼만 자동으로 래핑 후 INSERT한다.

```python
def insert_report(conn, data: dict) -> int:
    columns = ["dataset", "category", "image_path", "is_anomaly_LLM",
               "llm_report", "llm_summary", "llm_inference_duration", ...]

    values = []
    for col in columns:
        val = data.get(col)
        # JSONB 컬럼은 psycopg2.extras.Json()으로 래핑
        # → Python dict를 PostgreSQL이 이해하는 JSON으로 직렬화
        if col in ("llm_report", "llm_summary") and val is not None:
            val = Json(val)
        values.append(val)

    sql = "INSERT INTO inspection_reports (...) VALUES (...) RETURNING id"
    cur.execute(sql, values)
    return cur.fetchone()[0]   # 저장된 행의 id 반환
```

**`Json()` 래핑이 필요한 이유:**
psycopg2는 Python dict를 그냥 넘기면 문자열로 처리한다. `Json()`으로 감싸야 PostgreSQL JSONB 타입으로 올바르게 저장된다.

### 데이터 조회 (`get_report`, `list_reports`)

`RealDictCursor`를 사용해 컬럼명이 키인 dict로 바로 반환한다.

```python
def get_report(conn, report_id: int) -> dict:
    sql = "SELECT * FROM inspection_reports WHERE id = %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (report_id,))
        row = cur.fetchone()
    return dict(row) if row else None
    # → {"id": 1, "category": "cigarette_box", "llm_report": {...}, ...}

def list_reports(conn, limit: int = 50) -> list:
    sql = "SELECT * FROM inspection_reports ORDER BY id DESC LIMIT %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return [dict(r) for r in cur.fetchall()]
```

**`RealDictCursor` 없이 기본 cursor를 쓰면:**
```python
row = (1, "GoodsAD", "cigarette_box", ...)   # 튜플 → 컬럼명 없음
# RealDictCursor 사용 시:
row = {"id": 1, "dataset": "GoodsAD", "category": "cigarette_box", ...}  # dict
```

### 실제 사용 예시

```python
from src.storage.pg import connect, insert_report, get_report

conn = connect("postgresql://son:1234@localhost/inspection")

# LLM 결과 저장
report_id = insert_report(conn, {
    "dataset": "GoodsAD",
    "category": "cigarette_box",
    "image_path": "/data/MMAD/GoodsAD/cigarette_box/test/bad/001.jpg",
    "is_anomaly_LLM": True,
    "llm_report": {
        "anomaly_type": "dent",
        "severity": "high",
        "location": "top-right corner",
        "description": "Visible dent on the package corner",
        "confidence": 0.91,
        "recommendation": "Reject"
    },
    "llm_summary": {
        "summary": "High severity dent detected on cigarette box corner",
        "risk_level": "high"
    },
    "llm_inference_duration": 1.83,
})

# 저장 확인
saved = get_report(conn, report_id)
print(saved["llm_report"]["severity"])  # "high"
```

---

## 3. FastAPI 서버 (`apps/api/main.py`)

### 구조와 역할

PostgreSQL에 저장된 리포트를 HTTP로 외부에 제공하는 서버다. EC2에서 상시 실행되며, 프론트엔드나 다른 서비스가 HTTP 요청으로 리포트를 가져간다.

```python
# apps/api/main.py

app = FastAPI(title="MMAD Inspector API")

# CORS 설정 — 어떤 출처(프론트엔드)에서든 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 실제 운영 시 특정 도메인으로 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 PostgreSQL 연결 (이후 요청마다 재사용)
PG_DSN = os.environ.get("PG_DSN", "postgresql://son:1234@localhost/inspection")
conn = connect(PG_DSN)
```

### 엔드포인트 상세

**`GET /reports`** — 최근 리포트 목록

```python
@app.get("/reports")
def reports(limit: int = 50):
    return {"items": list_reports(conn, limit=limit)}
```

응답 예시:
```json
{
  "items": [
    {
      "id": 42,
      "category": "cigarette_box",
      "is_anomaly_LLM": true,
      "llm_report": {"severity": "high", "anomaly_type": "dent", ...},
      "llm_summary": {"risk_level": "high", ...},
      "llm_inference_duration": 1.83
    },
    ...
  ]
}
```

**`GET /reports/{id}`** — 단건 조회

```python
@app.get("/reports/{report_id}")
def report_detail(report_id: int):
    r = get_report(conn, report_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return r
```

응답 예시:
```json
{
  "id": 42,
  "dataset": "GoodsAD",
  "category": "cigarette_box",
  "image_path": "/data/MMAD/.../001.jpg",
  "is_anomaly_AD": true,
  "ad_score": 0.82,
  "is_anomaly_LLM": true,
  "llm_report": {
    "anomaly_type": "dent",
    "severity": "high",
    "location": "top-right corner",
    "description": "Visible dent on the package corner",
    "confidence": 0.91,
    "recommendation": "Reject"
  },
  "llm_summary": {
    "summary": "High severity dent detected",
    "risk_level": "high"
  },
  "llm_inference_duration": 1.83
}
```

### 서버 실행 (AWS EC2)

```bash
# SSH 접속
ssh -i C:/Users/son/Downloads/report.pem ubuntu@54.146.98.103

# 가상환경 활성화 & 서버 실행
cd ~/multimodal-anomaly-report-generation
source .venv/bin/activate

# 포그라운드 (터미널 유지 필요)
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

# 백그라운드 (터미널 닫아도 유지)
nohup uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 &
```

### 동작 원리

```
프론트엔드                    EC2 서버                   PostgreSQL
    │                           │                            │
    │  GET /reports?limit=10    │                            │
    │ ─────────────────────────►│                            │
    │                           │  SELECT * FROM             │
    │                           │  inspection_reports        │
    │                           │  ORDER BY id DESC LIMIT 10 │
    │                           │ ──────────────────────────►│
    │                           │                            │
    │                           │◄── rows (RealDictCursor) ──│
    │                           │                            │
    │◄── JSON response ─────────│                            │
```

> EC2 보안그룹에서 포트 8000 인바운드 허용 필요. (22, 80, 443, 8000)

---

## 4. 단일 이미지 테스트 (`scripts/test_report_pipeline.py`)

LLM 리포트 생성 → PostgreSQL 저장까지 한 번에 테스트하는 스크립트.

```bash
python scripts/test_report_pipeline.py \
  --image /path/to/image.jpg \
  --category cigarette_box \
  --dataset GoodsAD \
  --model gemini \
  --dsn "postgresql://son:1234@localhost/inspection"
```

실행 순서:
1. LLM 클라이언트 로드 (`get_llm_client`)
2. `generate_report()` 호출 → JSON 응답 파싱
3. `insert_report()` 로 PostgreSQL 저장
4. `get_report()` 로 저장 확인 출력

---

## 5. 전체 흐름 요약

```
[Colab / 로컬]                    [AWS EC2]
  이미지                           PostgreSQL DB
    │                                  │
    ├─ LLM 추론 (Gemini/InternVL)      │
    │   └─ JSON 리포트 생성             │
    │                                  │
    └─ insert_report() ───────────────►│ inspection_reports 테이블
                                       │
                              FastAPI (port 8000)
                                       │
                              GET /reports
                              GET /reports/{id}
                                       │
                              프론트엔드 / 대시보드
```
