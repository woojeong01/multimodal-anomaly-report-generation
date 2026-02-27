// src/app/data/reportMapper.ts
import type { AnomalyCase, ActionLog } from "./mockData";
import type { ReportDTO } from "../api/reportsApi";

/**
 * ReportDTO → AnomalyCase 매핑
 * normalizeRemoteToReportDTO에서 이미 llm_report 파싱 및 필드 정규화가 완료된 상태
 */

const PACKAGING_CLASS_LABEL: Record<string, string> = {
  cigarette_box: "cigarette box",
  drink_bottle: "drink bottle",
  drink_can: "drink can",
  food_bottle: "food_bottle",
  food_box: "food box",
  food_package: "food package",
  breakfast_box: "breakfast box",
  juice_bottle: "juice bottle",
  pushpins: "pushpins",
  screw_bag: "screw bag",
};

const LINES = ["LINE-A-01", "LINE-B-02", "LINE-C-03"];

function hash01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

function parseDatetime(dt?: string): Date {
  if (!dt) return new Date();
  const cleaned = dt.replace(/(\.\d{3})\d+/, "$1");
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? new Date() : d;
}

function toProductGroup(category: string): string {
  const pureCat = category.includes("/") ? category.split("/").pop()! : category;
  return PACKAGING_CLASS_LABEL[pureCat] ?? pureCat.replace(/_/g, " ");
}

function toLineId(line: string, seed: string): string {
  if (line) {
    // DB line 값이 있으면 그대로 사용 (LINE-A-01 형식이면 그대로, 아니면 매핑)
    const found = LINES.find((l) => l.toLowerCase() === line.toLowerCase());
    if (found) return found;
  }
  const v = hash01(seed);
  return LINES[Math.floor(v * LINES.length) % LINES.length];
}

function toShift(d: Date): string {
  const hour = d.getHours();
  return hour >= 7 && hour < 19 ? "주간" : "야간";
}

/** ReportDTO의 decision (ng/ok/review) → AnomalyCase의 (NG/OK/REVIEW) */
function normalizeDecision(d?: string): "OK" | "NG" | "REVIEW" {
  const s = (d ?? "").trim().toLowerCase();
  if (s === "ng" || s === "anomaly") return "NG";
  if (s === "ok" || s === "normal") return "OK";
  return "REVIEW";
}

function normalizeSeverity(raw?: string, decision?: "OK" | "NG" | "REVIEW"): "low" | "med" | "high" {
  if (decision && decision !== "NG") return "low";
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "med" || s === "medium") return "med";
  return "low";
}

function toKoreanSummary(decision: "OK" | "NG" | "REVIEW", location: string): string {
  if (decision === "OK") return "정상 제품으로 판정되었습니다. 이상 징후가 발견되지 않았습니다.";
  if (decision === "REVIEW") return "경계 케이스입니다. 육안 재검토를 통해 판정을 확정해 주세요.";
  const locKo: Record<string, string> = {
    "top-left": "상단 좌측", "top-center": "상단 중앙", "top-right": "상단 우측",
    "center-left": "중앙 좌측", "center": "중앙", "center-right": "중앙 우측",
    "bottom-left": "하단 좌측", "bottom-center": "하단 중앙", "bottom-right": "하단 우측",
  };
  return `${locKo[location] ?? "해당"} 영역에서 결함이 감지되었습니다. 불량으로 분류됩니다.`;
}

function toActionLog(decision: "OK" | "NG" | "REVIEW", ts: Date): ActionLog[] {
  const base = ts.getTime();
  if (decision === "OK") return [{ who: "System", when: new Date(base + 1000), what: "자동 승인" }];
  if (decision === "REVIEW") return [{ who: "박철수", when: new Date(base + 60_000), what: "재검 요청" }];
  return [
    { who: "System", when: new Date(base + 1000), what: "AI 불량 감지" },
    { who: "Operator", when: new Date(base + 60_000), what: "불량 확정" },
  ];
}

export function mapReportsToAnomalyCases(raw: any[]): AnomalyCase[] {
  return raw.map((r: ReportDTO, idx) => {
    // r은 normalizeRemoteToReportDTO를 거친 ReportDTO
    const ts = parseDatetime(r.datetime);
    const decision = normalizeDecision(r.decision);
    const category = r.category ?? "unknown";
    const loc = decision === "OK" ? "none" : (r.location ?? "center");
    const severity = normalizeSeverity(r.severity, decision);
    const defectType = r.defect_type ?? "none";

    return {
      id: `CASE-${r.id ?? idx}`,
      timestamp: ts,
      line_id: toLineId(r.line ?? "", `${r.dataset}-${category}`),
      shift: toShift(ts),

      product_group: toProductGroup(category),
      image_id: r.filename || `img_${idx}.jpg`,

      image_path: r.image_path || undefined,
      heatmap_path: r.heatmap_path || undefined,
      overlay_path: r.overlay_path || undefined,

      decision,
      anomaly_score: typeof r.confidence === "number" ? r.confidence : 0,
      threshold: 0.8,

      defect_type: defectType,
      defect_confidence: 0,
      location: loc,
      affected_area_pct: 0,
      severity,

      model_name: "PatchCore",
      model_version: "v1.0.0",
      inference_time_ms: typeof r.inference_time === "number"
        ? Math.round(r.inference_time * 1000)
        : 0,

      llm_summary: r.summary || toKoreanSummary(decision, loc),
      llm_structured_json: { source: r },
      operator_note: r.recommendation || "",
      action_log: toActionLog(decision, ts),
    };
  });
}
