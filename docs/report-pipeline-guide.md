# LLM 리포트 생성 + PostgreSQL 저장 파이프라인 가이드

## 전체 흐름

```
[InternVL] → JSON 리포트 생성 → [PostgreSQL] ← FastAPI → [프론트엔드]
```

---

## 1. `src/storage/pg.py` — DB 계층

```python
def connect(dsn: str):
    conn = psycopg2.connect(dsn)  # DSN 문자열로 PostgreSQL 접속
    create_tables(conn)            # 테이블 없으면 자동 생성
    return conn
```
- `dsn` = `"postgresql://son:1234@localhost/inspection"` → 유저/비밀번호/호스트/DB명

```python
def insert_report(conn, data: dict) -> int:
    # data dict에서 컬럼명과 매칭되는 값을 꺼냄
    # llm_report, llm_summary는 dict → Json()으로 감싸서 JSONB 타입으로 저장
    # INSERT ... RETURNING id → 삽입된 행의 id를 반환
```

```python
def list_reports(conn, limit=50):
    # RealDictCursor → 결과를 dict로 반환 (컬럼명: 값)
    # ORDER BY id DESC → 최신순
```

---

## 2. `src/mllm/base.py` — LLM 리포트 생성

```python
REPORT_PROMPT = '''... Product category: {category} ...
Respond in JSON format ONLY:
{{ "is_anomaly": true/false, "report": {{...}}, "summary": {{...}} }}'''
```
- `{category}` → 실행 시 실제 카테고리로 치환
- `{{`, `}}` → Python f-string이 아니라 `.format()` 용이라 중괄호 이스케이프

```python
def generate_report(self, image_path, category, ad_info=None):
    payload = self.build_report_payload(...)  # 프롬프트 + 이미지 → 모델 입력 형태로
    response = self.send_request(payload)     # 모델 추론 (서브클래스가 구현)
    text = self.extract_response_text(response)
    parsed = _parse_llm_json(text)            # 응답에서 JSON 추출
    return {"is_anomaly_LLM": ..., "llm_report": ..., "llm_summary": ...}
```

### JSON 파싱 로직

```python
def _parse_llm_json(text):
    cleaned = text.replace("\\_", "_")          # LLaVA가 \_를 출력하는 버그 수정
    json_match = re.search(r'\{[\s\S]*\}', cleaned)  # 텍스트에서 {...} 부분만 추출
    return json.loads(json_match.group())       # 문자열 → dict
```
- LLM이 JSON 앞뒤에 설명 텍스트를 붙여도 정규식으로 JSON 부분만 잘라냄

---

## 3. `apps/api/main.py` — FastAPI

```python
conn = connect(PG_DSN)  # 서버 시작 시 DB 연결 1회

@app.get("/reports")
def reports(limit=50):
    return {"items": list_reports(conn, limit)}
    # 프론트가 GET /reports 호출 → DB에서 조회 → JSON 응답
```

### CORS 미들웨어

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
- 브라우저 보안 정책상 다른 도메인에서 API 호출 못하게 막는데, `"*"`로 모든 도메인 허용

---

## 4. `sudo $(which uvicorn)` 원리

```bash
which uvicorn          # → /home/ubuntu/anaconda3/envs/report/bin/uvicorn
sudo $(which uvicorn)  # → sudo /home/ubuntu/anaconda3/envs/report/bin/uvicorn
```
- `sudo`는 root 권한으로 실행하는데, root의 PATH에는 conda 환경이 없음
- `$(which uvicorn)`이 현재 환경에서 uvicorn 절대경로를 찾아서 sudo에 넘겨줌
- 포트 80은 root 권한이 필요해서 `sudo`가 필수

---

## 실행 방법

### 리포트 생성 (단일 이미지)
```bash
python scripts/test_report_pipeline.py --image ~/data/test_images/000.png --category cigarette_box --dataset GoodsAD
```

### JSON 파일 → DB 업로드
```bash
python scripts/upload_reports_to_pg.py --input ~/data/reports.json
```

### API 서버 실행
```bash
sudo $(which uvicorn) apps.api.main:app --host 0.0.0.0 --port 80
```

### API 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/reports` | 최근 N개 리포트 목록 |
| GET | `/reports/{id}` | 리포트 단건 조회 |

### DB 직접 확인
```bash
psql -U son -d inspection -h localhost
\dt
SELECT * FROM inspection_reports;
```
