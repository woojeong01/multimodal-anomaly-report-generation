// Mock data for industrial anomaly detection dashboard

export interface AnomalyCase {
  id: string;
  timestamp: Date;
  line_id: string;
  shift: string;
  product_group: "Food" | "Household";
  product_class: string;
  batch_id: string;
  image_id: string;
  decision: "OK" | "NG" | "REVIEW";
  anomaly_score: number;
  threshold: number;
  defect_type: string;
  defect_confidence: number;
  location: string;
  affected_area_pct: number;
  severity: "low" | "med" | "high";
  model_name: string;
  model_version: string;
  inference_time_ms: number;
  llm_summary: string;
  llm_structured_json: any;
  operator_note?: string;
  action_log: ActionLog[];
}

export interface ActionLog {
  who: string;
  when: Date;
  what: string;
}

export interface Alert {
  id: string;
  type: "spike" | "pattern" | "line_issue";
  severity: "low" | "med" | "high";
  title: string;
  description: string;
  timestamp: Date;
}

// Generate mock cases
export const mockCases: AnomalyCase[] = [
  {
    id: "CASE-2026-0204-001",
    timestamp: new Date("2026-02-04T14:32:15"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "Food",
    product_class: "상자",
    batch_id: "BATCH-20260204-A01-012",
    image_id: "IMG-A01-20260204-143215",
    decision: "NG",
    anomaly_score: 0.87,
    threshold: 0.65,
    defect_type: "seal_issue",
    defect_confidence: 0.91,
    location: "top-right",
    affected_area_pct: 12.3,
    severity: "high",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 142,
    llm_summary:
      "포장 상단 우측에서 실링 불량이 감지되었습니다. 열접착 불완전으로 인한 미세 틈새가 관찰됩니다.",
    llm_structured_json: {
      defect_location: { x: 0.78, y: 0.15 },
      defect_size_mm: { width: 8.2, height: 3.1 },
      probable_cause: "실링 온도 저하",
      recommendation: "실링 헤드 온도 점검 필요",
    },
    operator_note: "라인 정지 후 재검사 완료",
    action_log: [
      {
        who: "김민수",
        when: new Date("2026-02-04T14:33:00"),
        what: "불량 확정",
      },
      {
        who: "이영희",
        when: new Date("2026-02-04T14:35:12"),
        what: "메모 추가",
      },
    ],
  },
  {
    id: "CASE-2026-0204-002",
    timestamp: new Date("2026-02-04T13:45:22"),
    line_id: "LINE-B-02",
    shift: "주간",
    product_group: "Household",
    product_class: "병",
    batch_id: "BATCH-20260204-B02-008",
    image_id: "IMG-B02-20260204-134522",
    decision: "REVIEW",
    anomaly_score: 0.68,
    threshold: 0.65,
    defect_type: "contamination",
    defect_confidence: 0.73,
    location: "center",
    affected_area_pct: 3.2,
    severity: "med",
    model_name: "WinCLIP",
    model_version: "v1.8.2",
    inference_time_ms: 218,
    llm_summary:
      "병 중앙부에 미세한 이물질이 의심됩니다. 육안 재확인이 필요한 경계 케이스입니다.",
    llm_structured_json: {
      defect_location: { x: 0.51, y: 0.48 },
      defect_size_mm: { width: 2.1, height: 1.8 },
      probable_cause: "제조 과정 이물 혼입",
      recommendation: "육안 검사 재확인",
    },
    action_log: [
      {
        who: "박철수",
        when: new Date("2026-02-04T13:46:00"),
        what: "재검 요청",
      },
    ],
  },
  {
    id: "CASE-2026-0204-003",
    timestamp: new Date("2026-02-04T13:21:08"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "Food",
    product_class: "과자",
    batch_id: "BATCH-20260204-A01-009",
    image_id: "IMG-A01-20260204-132108",
    decision: "NG",
    anomaly_score: 0.92,
    threshold: 0.65,
    defect_type: "crack",
    defect_confidence: 0.95,
    location: "bottom-left",
    affected_area_pct: 18.7,
    severity: "high",
    model_name: "EfficientAD",
    model_version: "v3.1.0",
    inference_time_ms: 95,
    llm_summary:
      "포장재 하단 좌측에 명확한 파손이 확인되었습니다. 제품 품질에 직접적인 영향을 미칩니다.",
    llm_structured_json: {
      defect_location: { x: 0.18, y: 0.82 },
      defect_size_mm: { width: 15.2, height: 12.3 },
      probable_cause: "취급 중 충격",
      recommendation: "즉시 폐기 필요",
    },
    action_log: [
      {
        who: "김민수",
        when: new Date("2026-02-04T13:22:00"),
        what: "불량 확정",
      },
    ],
  },
  {
    id: "CASE-2026-0204-004",
    timestamp: new Date("2026-02-04T12:58:33"),
    line_id: "LINE-C-03",
    shift: "주간",
    product_group: "Household",
    product_class: "도시락",
    batch_id: "BATCH-20260204-C03-015",
    image_id: "IMG-C03-20260204-125833",
    decision: "OK",
    anomaly_score: 0.42,
    threshold: 0.65,
    defect_type: "none",
    defect_confidence: 0.0,
    location: "none",
    affected_area_pct: 0,
    severity: "low",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 138,
    llm_summary:
      "정상 제품으로 판정되었습니다. 이상 징후가 발견되지 않았습니다.",
    llm_structured_json: {
      quality_score: 0.98,
      notes: "정상",
    },
    action_log: [
      {
        who: "System",
        when: new Date("2026-02-04T12:58:34"),
        what: "자동 승인",
      },
    ],
  },
  {
    id: "CASE-2026-0204-005",
    timestamp: new Date("2026-02-04T12:15:47"),
    line_id: "LINE-B-02",
    shift: "주간",
    product_group: "Food",
    product_class: "우유",
    batch_id: "BATCH-20260204-B02-005",
    image_id: "IMG-B02-20260204-121547",
    decision: "NG",
    anomaly_score: 0.81,
    threshold: 0.65,
    defect_type: "dent",
    defect_confidence: 0.88,
    location: "top-left",
    affected_area_pct: 8.5,
    severity: "med",
    model_name: "WinCLIP",
    model_version: "v1.8.2",
    inference_time_ms: 203,
    llm_summary:
      "용기 상단 좌측에 찌그러짐이 발견되었습니다. 유통 및 보관에 영향을 줄 수 있습니다.",
    llm_structured_json: {
      defect_location: { x: 0.22, y: 0.18 },
      defect_size_mm: { width: 12.5, height: 9.2 },
      probable_cause: "운송 중 충격",
      recommendation: "포장 공정 점검",
    },
    action_log: [
      {
        who: "이영희",
        when: new Date("2026-02-04T12:16:30"),
        what: "불량 확정",
      },
    ],
  },
  {
    id: "CASE-2026-0204-006",
    timestamp: new Date("2026-02-04T11:42:19"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "Food",
    product_class: "라면",
    batch_id: "BATCH-20260204-A01-007",
    image_id: "IMG-A01-20260204-114219",
    decision: "REVIEW",
    anomaly_score: 0.67,
    threshold: 0.65,
    defect_type: "scratch",
    defect_confidence: 0.71,
    location: "center",
    affected_area_pct: 4.1,
    severity: "low",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 147,
    llm_summary:
      "포장 표면에 경미한 스크래치가 감지되었으나 품질 기준 경계선상입니다.",
    llm_structured_json: {
      defect_location: { x: 0.48, y: 0.52 },
      defect_size_mm: { width: 3.2, height: 0.8 },
      probable_cause: "컨베이어 접촉",
      recommendation: "육안 확인 후 판정",
    },
    action_log: [
      {
        who: "박철수",
        when: new Date("2026-02-04T11:43:00"),
        what: "재검 요청",
      },
    ],
  },
  {
    id: "CASE-2026-0204-007",
    timestamp: new Date("2026-02-04T11:18:52"),
    line_id: "LINE-C-03",
    shift: "주간",
    product_group: "Household",
    product_class: "병",
    batch_id: "BATCH-20260204-C03-012",
    image_id: "IMG-C03-20260204-111852",
    decision: "OK",
    anomaly_score: 0.38,
    threshold: 0.65,
    defect_type: "none",
    defect_confidence: 0.0,
    location: "none",
    affected_area_pct: 0,
    severity: "low",
    model_name: "EfficientAD",
    model_version: "v3.1.0",
    inference_time_ms: 102,
    llm_summary:
      "정상 제품으로 판정되었습니다. 모든 품질 기준을 충족합니다.",
    llm_structured_json: {
      quality_score: 0.96,
      notes: "정상",
    },
    action_log: [
      {
        who: "System",
        when: new Date("2026-02-04T11:18:53"),
        what: "자동 승인",
      },
    ],
  },
  {
    id: "CASE-2026-0204-008",
    timestamp: new Date("2026-02-04T10:55:31"),
    line_id: "LINE-B-02",
    shift: "주간",
    product_group: "Food",
    product_class: "과자",
    batch_id: "BATCH-20260204-B02-003",
    image_id: "IMG-B02-20260204-105531",
    decision: "NG",
    anomaly_score: 0.89,
    threshold: 0.65,
    defect_type: "seal_issue",
    defect_confidence: 0.93,
    location: "bottom-right",
    affected_area_pct: 14.2,
    severity: "high",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 151,
    llm_summary:
      "포장 하단 우측의 실링이 불완전합니다. 제품 밀봉성에 문제가 있습니다.",
    llm_structured_json: {
      defect_location: { x: 0.82, y: 0.88 },
      defect_size_mm: { width: 11.3, height: 4.2 },
      probable_cause: "실링 압력 부족",
      recommendation: "장비 점검 필요",
    },
    action_log: [
      {
        who: "김민수",
        when: new Date("2026-02-04T10:56:15"),
        what: "불량 확정",
      },
    ],
  },
  {
    id: "CASE-2026-0204-009",
    timestamp: new Date("2026-02-04T10:32:44"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "Household",
    product_class: "샴푸",
    batch_id: "BATCH-20260204-A01-004",
    image_id: "IMG-A01-20260204-103244",
    decision: "OK",
    anomaly_score: 0.45,
    threshold: 0.65,
    defect_type: "none",
    defect_confidence: 0.0,
    location: "none",
    affected_area_pct: 0,
    severity: "low",
    model_name: "WinCLIP",
    model_version: "v1.8.2",
    inference_time_ms: 195,
    llm_summary:
      "품질 검사 통과. 정상 제품으로 확인되었습니다.",
    llm_structured_json: {
      quality_score: 0.95,
      notes: "정상",
    },
    action_log: [
      {
        who: "System",
        when: new Date("2026-02-04T10:32:45"),
        what: "자동 승인",
      },
    ],
  },
  {
    id: "CASE-2026-0204-010",
    timestamp: new Date("2026-02-04T10:08:17"),
    line_id: "LINE-C-03",
    shift: "주간",
    product_group: "Food",
    product_class: "우유",
    batch_id: "BATCH-20260204-C03-009",
    image_id: "IMG-C03-20260204-100817",
    decision: "REVIEW",
    anomaly_score: 0.7,
    threshold: 0.65,
    defect_type: "contamination",
    defect_confidence: 0.76,
    location: "top-left",
    affected_area_pct: 5.3,
    severity: "med",
    model_name: "EfficientAD",
    model_version: "v3.1.0",
    inference_time_ms: 108,
    llm_summary:
      "캡 부분에 미세 오염 가능성이 있습니다. 재검사를 통한 확인이 필요합니다.",
    llm_structured_json: {
      defect_location: { x: 0.25, y: 0.12 },
      defect_size_mm: { width: 4.2, height: 3.8 },
      probable_cause: "캡 조립 공정 이물",
      recommendation: "육안 재검사",
    },
    action_log: [
      {
        who: "이영희",
        when: new Date("2026-02-04T10:09:00"),
        what: "재검 요청",
      },
    ],
  },
  {
    id: "CASE-2026-0204-011",
    timestamp: new Date("2026-02-04T09:41:55"),
    line_id: "LINE-B-02",
    shift: "주간",
    product_group: "Household",
    product_class: "병",
    batch_id: "BATCH-20260204-B02-001",
    image_id: "IMG-B02-20260204-094155",
    decision: "NG",
    anomaly_score: 0.85,
    threshold: 0.65,
    defect_type: "dent",
    defect_confidence: 0.9,
    location: "center",
    affected_area_pct: 10.8,
    severity: "med",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 144,
    llm_summary:
      "병 중앙부에 찌그러짐이 관찰됩니다. 외관 품질 기준 미달입니다.",
    llm_structured_json: {
      defect_location: { x: 0.52, y: 0.58 },
      defect_size_mm: { width: 14.1, height: 11.5 },
      probable_cause: "취급 부주의",
      recommendation: "폐기 처리",
    },
    action_log: [
      {
        who: "박철수",
        when: new Date("2026-02-04T09:42:30"),
        what: "불량 확정",
      },
    ],
  },
  {
    id: "CASE-2026-0204-012",
    timestamp: new Date("2026-02-04T09:15:28"),
    line_id: "LINE-A-01",
    shift: "주간",
    product_group: "Food",
    product_class: "라면",
    batch_id: "BATCH-20260204-A01-001",
    image_id: "IMG-A01-20260204-091528",
    decision: "OK",
    anomaly_score: 0.35,
    threshold: 0.65,
    defect_type: "none",
    defect_confidence: 0.0,
    location: "none",
    affected_area_pct: 0,
    severity: "low",
    model_name: "WinCLIP",
    model_version: "v1.8.2",
    inference_time_ms: 210,
    llm_summary: "정상 제품입니다. 품질 이상이 없습니다.",
    llm_structured_json: {
      quality_score: 0.97,
      notes: "정상",
    },
    action_log: [
      {
        who: "System",
        when: new Date("2026-02-04T09:15:29"),
        what: "자동 승인",
      },
    ],
  },
  {
    id: "CASE-2026-0204-013",
    timestamp: new Date("2026-02-04T08:52:11"),
    line_id: "LINE-C-03",
    shift: "주간",
    product_group: "Household",
    product_class: "병",
    batch_id: "BATCH-20260204-C03-006",
    image_id: "IMG-C03-20260204-085211",
    decision: "NG",
    anomaly_score: 0.94,
    threshold: 0.65,
    defect_type: "crack",
    defect_confidence: 0.96,
    location: "bottom-left",
    affected_area_pct: 22.1,
    severity: "high",
    model_name: "EfficientAD",
    model_version: "v3.1.0",
    inference_time_ms: 99,
    llm_summary:
      "병 하단부에 심각한 균열이 발견되었습니다. 즉시 제거가 필요합니다.",
    llm_structured_json: {
      defect_location: { x: 0.15, y: 0.85 },
      defect_size_mm: { width: 18.7, height: 15.3 },
      probable_cause: "성형 불량",
      recommendation: "즉시 폐기 및 공정 점검",
    },
    action_log: [
      {
        who: "김민수",
        when: new Date("2026-02-04T08:53:00"),
        what: "불량 확정",
      },
    ],
  },
  {
    id: "CASE-2026-0204-014",
    timestamp: new Date("2026-02-04T08:28:36"),
    line_id: "LINE-B-02",
    shift: "주간",
    product_group: "Food",
    product_class: "과자",
    batch_id: "BATCH-20260203-B02-045",
    image_id: "IMG-B02-20260204-082836",
    decision: "REVIEW",
    anomaly_score: 0.66,
    threshold: 0.65,
    defect_type: "scratch",
    defect_confidence: 0.69,
    location: "top-right",
    affected_area_pct: 2.8,
    severity: "low",
    model_name: "PatchCore",
    model_version: "v2.3.1",
    inference_time_ms: 156,
    llm_summary:
      "포장 상단에 경미한 스크래치 흔적. 기준 경계선상으로 재검토가 필요합니다.",
    llm_structured_json: {
      defect_location: { x: 0.75, y: 0.2 },
      defect_size_mm: { width: 2.8, height: 1.2 },
      probable_cause: "자동화 장비 접촉",
      recommendation: "육안 확인",
    },
    action_log: [
      {
        who: "이영희",
        when: new Date("2026-02-04T08:29:15"),
        what: "재검 요청",
      },
    ],
  },
  {
    id: "CASE-2026-0204-015",
    timestamp: new Date("2026-02-04T08:05:42"),
    line_id: "LINE-A-01",
    shift: "야간",
    product_group: "Food",
    product_class: "우유",
    batch_id: "BATCH-20260203-A01-042",
    image_id: "IMG-A01-20260204-080542",
    decision: "OK",
    anomaly_score: 0.41,
    threshold: 0.65,
    defect_type: "none",
    defect_confidence: 0.0,
    location: "none",
    affected_area_pct: 0,
    severity: "low",
    model_name: "WinCLIP",
    model_version: "v1.8.2",
    inference_time_ms: 207,
    llm_summary: "정상 제품으로 판정. 이상 없음.",
    llm_structured_json: {
      quality_score: 0.96,
      notes: "정상",
    },
    action_log: [
      {
        who: "System",
        when: new Date("2026-02-04T08:05:43"),
        what: "자동 승인",
      },
    ],
  },
];

// Alerts
export const mockAlerts: Alert[] = [
  {
    id: "ALERT-001",
    type: "spike",
    severity: "high",
    title: "LINE-A-01 불량률 급증 감지",
    description:
      "최근 2시간 동안 불량률이 평균 대비 245% 증가했습니다. seal_issue 유형이 주요 원인입니다.",
    timestamp: new Date("2026-02-04T14:15:00"),
  },
  {
    id: "ALERT-002",
    type: "pattern",
    severity: "med",
    title: "새로운 결함 패턴 발견",
    description:
      "LINE-C-03에서 기존에 없던 crack 유형의 결함이 3건 연속 발생했습니다.",
    timestamp: new Date("2026-02-04T13:42:00"),
  },
  {
    id: "ALERT-003",
    type: "line_issue",
    severity: "med",
    title: "LINE-B-02 장비 점검 권고",
    description:
      "평균 추론 시간이 기준 대비 35% 증가. 카메라 또는 조명 상태를 확인하세요.",
    timestamp: new Date("2026-02-04T12:28:00"),
  },
  {
    id: "ALERT-004",
    type: "spike",
    severity: "low",
    title: "도시락 제품군 재검률 상승",
    description:
      "Household > 도시 카테고리의 REVIEW 판정이 평소보다 18% 증가했습니다.",
    timestamp: new Date("2026-02-04T11:05:00"),
  },
  {
    id: "ALERT-005",
    type: "pattern",
    severity: "low",
    title: "주간 교대 품질 안정",
    description:
      "주간 교대조의 불량률이 목표치(2.5%) 이하를 7일 연속 유지 중입니다.",
    timestamp: new Date("2026-02-04T10:00:00"),
  },
];

export const defectTypes = [
  "scratch",
  "contamination",
  "crack",
  "dent",
  "seal_issue",
];

export const productClasses = [
  "상자",
  "과자",
  "병",
  "캔",
  "도시락",
];

export const lines = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];