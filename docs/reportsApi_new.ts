// src/app/api/reportsApi.ts
import { apiRequest, QueryParams } from "./http";

const REPORTS_PATH =
  (import.meta.env.VITE_REPORTS_PATH as string | undefined) ?? "/llava/reports";

export type ReportDTO = {
  id: number;
  filename: string;
  image_path: string;
  dataset: string;
  category: string;
  line: string;
  ground_truth: string | null;
  decision: string;
  confidence: number | null;

  has_defect: number;
  defect_type: string;
  location: string;
  severity: string;

  defect_description: string;
  possible_cause: string;
  product_description: string;

  summary: string;
  impact: string;
  recommendation: string;

  inference_time: number | null;
  datetime: string;

  heatmap_path?: string | null;
  overlay_path?: string | null;
};

export type ReportListDTO = {
  items: ReportDTO[];
  total: number;
};

export type ReportListQuery = {
  limit?: number;
  offset?: number;
  dataset?: string;
  category?: string;
  decision?: string;
  date_from?: string;
  date_to?: string;
};

async function fetchReportsRaw(query: ReportListQuery, opts?: { signal?: AbortSignal }) {
  return apiRequest<unknown>(REPORTS_PATH, {
    query: query as QueryParams,
    signal: opts?.signal,
  });
}

export async function fetchReports(
  query: ReportListQuery,
  opts?: { signal?: AbortSignal }
): Promise<ReportListDTO> {
  const data = await fetchReportsRaw(query, opts);

  const rawItems = Array.isArray(data)
    ? data
    : ((data as any)?.items ?? []);

  const items = (rawItems as any[]).map((x, i) => normalizeRemoteToReportDTO(x, i));

  const total = Array.isArray(data)
    ? items.length
    : Number((data as any)?.total ?? items.length);

  return { items, total };
}

export async function fetchReportsAll(
  baseQuery?: Omit<ReportListQuery, "limit" | "offset">,
  opts?: { signal?: AbortSignal; pageSize?: number; maxItems?: number }
): Promise<ReportDTO[]> {
  const pageSize = opts?.pageSize ?? 500;
  const maxItems = opts?.maxItems ?? 5000;

  let offset = 0;
  let out: ReportDTO[] = [];
  let total = Infinity;

  while (offset < total && out.length < maxItems) {
    const { items, total: t } = await fetchReports(
      { ...(baseQuery ?? {}), limit: pageSize, offset },
      { signal: opts?.signal }
    );

    total = Number.isFinite(t) ? t : Infinity;
    out = out.concat(items);

    if (items.length === 0) break;
    offset += items.length;
  }

  return out.slice(0, maxItems);
}

// ── 헬퍼 ──────────────────────────────────────

/** llm_report / llm_summary TEXT 컬럼은 이중/삼중 JSON 인코딩 가능 → 반복 파싱 */
function parseJsonField(raw: any): any {
  if (raw === null || raw === undefined) return {};
  try {
    let val: any = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (typeof val === "string") val = JSON.parse(val);
    if (typeof val === "string") val = JSON.parse(val);
    return val && typeof val === "object" ? val : {};
  } catch {
    return {};
  }
}

function normLocation(raw?: string) {
  const s = (raw ?? "").trim().toLowerCase();
  if (!s || s === "none") return "none";
  return s.replace(/\s+/g, "-");
}

function normSeverity(raw?: string) {
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "high") return "high";
  if (s === "medium" || s === "med") return "med";
  return "low";
}

function normDecision(raw?: string): string {
  const s = (raw ?? "").trim().toLowerCase();
  if (s === "anomaly" || s === "ng") return "ng";
  if (s === "normal" || s === "ok") return "ok";
  if (s === "review_needed" || s === "review") return "review";
  return "ok";
}

function basename(p?: string) {
  const s = (p ?? "").trim();
  if (!s) return "";
  return s.split("/").pop() ?? s;
}

function normalizeRemoteToReportDTO(x: any, idx: number): ReportDTO {
  // llm_report, llm_summary는 TEXT(JSON 문자열) → 파싱
  const llm = parseJsonField(x?.llm_report);
  const sum = parseJsonField(x?.llm_summary);

  // ad_decision (anomaly / normal / review_needed) → ng / ok / review
  const decision = normDecision(x?.ad_decision ?? x?.decision);

  const rawImagePath = typeof x?.image_path === "string" ? x.image_path : "";

  return {
    id: Number(x?.id ?? idx + 1),
    filename: String(basename(rawImagePath) || `remote_${idx}.png`),
    image_path: rawImagePath,
    heatmap_path: x?.heatmap_path ?? null,
    overlay_path: x?.overlay_path ?? x?.mask_path ?? null,

    dataset: String(x?.dataset ?? "remote"),
    category: String(x?.category ?? "unknown"),
    line: String(x?.line ?? ""),
    ground_truth: x?.ground_truth ?? null,

    decision,
    confidence: typeof x?.ad_score === "number" ? x.ad_score : null,

    has_defect: x?.has_defect ? 1 : 0,

    // llm_report에서 파싱한 값 사용
    defect_type: String(llm?.anomaly_type ?? x?.defect_type ?? "none"),
    location: normLocation(x?.region ?? llm?.location ?? x?.location),
    severity: normSeverity(llm?.severity ?? sum?.risk_level ?? x?.severity),

    defect_description: String(llm?.description ?? x?.defect_description ?? ""),
    possible_cause: String(x?.possible_cause ?? ""),
    product_description: String(x?.product_description ?? ""),

    summary: String(sum?.summary ?? x?.summary ?? ""),
    impact: String(x?.impact ?? ""),
    recommendation: String(llm?.recommendation ?? x?.recommendation ?? ""),

    inference_time:
      typeof x?.inference_time === "number"
        ? x.inference_time
        : typeof x?.llm_inference_duration === "number"
          ? x.llm_inference_duration
          : null,

    datetime: String(x?.datetime ?? x?.created_at ?? new Date().toISOString()),
  };
}
