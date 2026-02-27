# 프론트엔드-백엔드 연동 트러블슈팅 기록

---

## 1. DB 스키마 불일치

### 문제
EC2에 두 개의 pg.py 파일이 존재하며 스키마가 달랐음:

| 파일 | llm_report 타입 | bbox/center | created_at |
|------|------|------|------|
| `app/services/pg.py` | JSONB | 있음 | 없음 |
| `src/storage/db.py` | TEXT | 없음 | 있음 (DEFAULT NOW()) |

### 해결
실제 lion_db 스키마 직접 조회로 확인:
```bash
psql -h localhost -U lion_user -d lion_db \
  -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name='inspection_reports';"
```
> `psql` 연결 시 `-h localhost` 필수 (소켓 인증 우회)

실제 컬럼: `id, dataset, category, line, ad_score, ad_decision, is_anomaly_ad, has_defect, region, area_ratio, image_path, heatmap_path, mask_path, overlay_path, similar_image_path, is_anomaly_llm, llm_report(TEXT), llm_summary(TEXT), applied_policy(JSONB), ad_inference_duration, llm_inference_duration, created_at`

---

## 2. 로컬 DB 스키마 초기화

### 문제
로컬 `inspection` DB가 구버전 스키마라 새 데이터 INSERT 불가

### 해결
`scripts/init_db.py` 생성 — lion_db 스키마와 동일한 테이블 생성:
```bash
python scripts/init_db.py --dsn "postgresql://son:1234@localhost/inspection" --drop
```
> `--drop`: 기존 테이블 삭제 후 재생성

---

## 3. 샘플 데이터 생성 및 삽입

### 문제
프론트에서 확인할 테스트 데이터가 없었음

### 해결
`scripts/seed_db.py` 생성 — MMAD 카테고리 기반 더미 데이터 생성:
- `experiment.yaml`의 기존 필드(`mmad_json`, `eval.sample_per_folder`, `eval.sample_seed`) 재활용
- `llm_report` / `llm_summary`는 TEXT 컬럼이므로 `json.dumps()`로 문자열 저장

```bash
python scripts/seed_db.py --output sample_data.json          # JSON만 확인
python scripts/seed_db.py --dsn "postgresql://..." --clear   # DB 삽입
```

JSON → DB 삽입은 기존 스크립트 활용:
```bash
python scripts/upload_reports_to_pg.py \
  --input new_sample_data_99imgs.json \
  --dsn "postgresql://lion_user:lion123@localhost/lion_db"
```
> 명령어는 **한 줄**로 실행 (줄바꿈 시 `--input: command not found` 오류 발생)

---

## 4. llm_report 이중/삼중 인코딩

### 문제
`upload_reports_to_pg.py` → `src/storage/pg.py`의 `insert_report()`가 `llm_report`를 JSONB 컬럼으로 처리해 `Json()`으로 감쌈.
그런데 `llm_report`는 이미 `json.dumps()`된 문자열 → **이중 인코딩** 발생.

DB 저장값:
```
"{\"anomaly_type\": \"hole\", \"severity\": \"high\", ...}"
```

### 해결
프론트 `reportsApi.ts`의 `parseJsonField()`에서 반복 파싱:
```typescript
function parseJsonField(raw: any): any {
  try {
    let val = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (typeof val === "string") val = JSON.parse(val);  // 이중 인코딩 처리
    if (typeof val === "string") val = JSON.parse(val);  // 삼중 인코딩 처리
    return val && typeof val === "object" ? val : {};
  } catch { return {}; }
}
```

---

## 5. 프론트엔드 필드 매핑 오류

### 문제
`reportsApi.ts`의 `normalizeRemoteToReportDTO`가 raw DB 응답을 `ReportDTO`로 변환하고,
`reportMapper.ts`의 `mapReportsToAnomalyCases`가 `ReportDTO`를 `AnomalyCase`로 변환.

**두 단계 변환인데 각 단계가 서로 다른 필드명을 기대해서 데이터가 모두 틀렸음.**

| 증상 | 원인 |
|------|------|
| 결함 타입이 항상 "anomaly" 또는 "-" | `defect_type`을 `r.has_defect ? "anomaly" : "none"`으로 하드코딩 |
| 판정이 항상 "REVIEW" | `r.ad_decision`이 undefined → 기본값 REVIEW |
| 심각도 고정 | `r.severity` 컬럼 없음 (llm_report 안에 있음) |

### 해결

**`reportsApi.ts` - `normalizeRemoteToReportDTO`:**
- `llm_report`, `llm_summary` JSON 파싱 후 `defect_type`, `severity`, `location`, `recommendation` 추출
- `ad_decision` (anomaly/normal/review_needed) → `decision` (ng/ok/review) 변환
- `ad_score` → `confidence` 매핑

```typescript
const llm = parseJsonField(x?.llm_report);
const sum = parseJsonField(x?.llm_summary);
const decision = normDecision(x?.ad_decision ?? x?.decision);

defect_type: String(llm?.anomaly_type ?? x?.defect_type ?? "none"),
location:    normLocation(x?.region ?? llm?.location),
severity:    normSeverity(llm?.severity ?? sum?.risk_level),
confidence:  typeof x?.ad_score === "number" ? x.ad_score : null,
datetime:    String(x?.created_at ?? new Date().toISOString()),
```

**`reportMapper.ts` - `mapReportsToAnomalyCases`:**
- raw DB 필드명 → `ReportDTO` 필드명으로 교체

| 변경 전 (raw DB) | 변경 후 (ReportDTO) |
|------|------|
| `r.created_at` | `r.datetime` |
| `r.ad_decision` | `r.decision` |
| `r.ad_score` | `r.confidence` |
| `r.region` | `r.location` |
| `r.ad_inference_duration` | `r.inference_time` |
| `llm_report` 재파싱 | `r.defect_type`, `r.severity` 직접 사용 |

---

## 6. 이미지 서빙 (미완)

### 시도
EC2 `app/main.py`에 FastAPI StaticFiles 마운트:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="/home/ubuntu"), name="static")
```

`reportsApi.ts`에서 절대경로 → URL 변환:
```typescript
function toStaticUrl(path?: string): string {
  const cleaned = path.startsWith("/home/ubuntu/")
    ? path.slice("/home/ubuntu/".length) : path;
  return `${API_BASE}/static/${cleaned}`;
}
```

### 현재 상태
적용 후 데이터 전체 미표시 문제 발생 → **롤백**.
원인 미확인, 추후 재시도 필요.

---

## 수정 파일 목록

| 파일 | 위치 |
|------|------|
| `init_db.py` | `scripts/init_db.py` |
| `seed_db.py` | `scripts/seed_db.py` |
| `reportsApi.ts` | `docs/reportsApi.ts` → EC2 `Frontend/src/app/api/reportsApi.ts` |
| `reportMapper.ts` | `docs/reportMapper.ts` → EC2 `Frontend/src/app/data/reportMapper.ts` |
| `ec2-server-setup.md` | `docs/ec2-server-setup.md` |
