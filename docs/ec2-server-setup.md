# EC2 서버 셋업 가이드

백엔드(FastAPI) + 프론트엔드(Vite/React) + PostgreSQL을 EC2에서 실행하는 과정 정리.

---

## 디렉터리 구조

```
/home/ubuntu/apps/
├── app/
│   ├── main.py              # FastAPI 백엔드 진입점
│   └── services/
│       ├── ad_service.py
│       ├── llm_service.py
│       └── visual_rag_service.py
├── src/storage/
│   ├── pg.py                # PostgreSQL CRUD
│   └── db.py                # 테이블 생성 스크립트
├── Frontend/                # Vite/React 프론트엔드
│   └── .env.local           # API 주소 설정
├── init_db.py               # DB 스키마 초기화
├── upload_reports_to_pg.py  # JSON → DB 삽입
└── .venv/                   # uv 가상환경
```

---

## 1. 가상환경 생성 및 의존성 설치

```bash
cd /home/ubuntu/apps

# uv 가상환경 생성
uv venv
source .venv/bin/activate

# PyTorch (CUDA 12.4)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 기타 의존성
uv pip install opencv-python-headless scikit-learn
uv pip install python-multipart
uv pip install pyyaml psycopg2-binary
```

---

## 2. DB 초기화

```bash
# lion_db 스키마 생성 (테이블이 이미 있으면 --drop으로 재생성)
python init_db.py --dsn "postgresql://lion_user:lion123@localhost/lion_db"

# 기존 테이블 초기화 후 재생성
python init_db.py --dsn "postgresql://lion_user:lion123@localhost/lion_db" --drop
```

### DB 접속 확인

```bash
# -h localhost: 소켓 인증 우회 (TCP 연결)
psql -h localhost -U lion_user -d lion_db -c "\conninfo"

# 컬럼 확인
psql -h localhost -U lion_user -d lion_db \
  -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name='inspection_reports';"
```

---

## 3. 샘플 데이터 삽입

로컬에서 JSON 생성 후 서버에서 삽입하거나, 서버에서 직접 실행.

```bash
# JSON 파일로 DB에 삽입
python upload_reports_to_pg.py \
  --input new_sample_data_99imgs.json \
  --dsn "postgresql://lion_user:lion123@localhost/lion_db"
```

> 여러 인자는 반드시 **한 줄**로 실행 (줄바꿈 시 파싱 오류 발생)

---

## 4. 백엔드 실행 (FastAPI)

```bash
cd /home/ubuntu/apps
source .venv/bin/activate

uvicorn app.main:app --host 0.0.0.0 --port 8000 or
uvicorn api.main:app --host 0.0.0.0 --port 8000 or
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 ## leehw/pipeline
```

백그라운드 실행:
```bash
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
```

API 확인:
```bash
curl http://localhost:8000/reports
```

---

## 5. 프론트엔드 실행 (Vite)

### 환경변수 설정

`/home/ubuntu/apps/Frontend/.env.local`:
```
VITE_API_BASE_URL=http://<EC2_PUBLIC_IP>:8000
VITE_REPORTS_PATH=/reports
```

### 실행

```bash
cd /home/ubuntu/apps/Frontend
npm install   # 최초 1회
npm run dev -- --host 0.0.0.0
```

백그라운드 실행:
```bash
nohup npm run dev -- --host 0.0.0.0 > ~/apps/logs/frontend.log 2>&1 &
```

브라우저 접속: `http://<EC2_PUBLIC_IP>:5173`

---

## 6. EC2 보안 그룹 포트 설정

AWS 콘솔 → EC2 → 보안 그룹 → 인바운드 규칙에 아래 포트 허용:

| 포트 | 용도 |
|------|------|
| 22 | SSH |
| 8000 | FastAPI 백엔드 |
| 5173 | Vite 프론트엔드 |

---

## 트러블슈팅

| 오류 | 원인 | 해결 |
|------|------|------|
| `Peer authentication failed` | psql 소켓 인증 | `-h localhost` 추가 |
| `--input: command not found` | 멀티라인 명령어 파싱 오류 | 한 줄로 실행 |
| `ModuleNotFoundError: torch` | 가상환경 미활성화 or 미설치 | `source .venv/bin/activate` 후 재설치 |
| `Form data requires python-multipart` | 패키지 누락 | `uv pip install python-multipart` |
| `UndefinedColumn` (init_db) | 구버전 테이블 존재 | `--drop` 플래그로 재생성 |
