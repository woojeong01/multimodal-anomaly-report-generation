#!/usr/bin/env python3
"""샘플 PDF 리포트 생성기 — 단일 이미지 & 배치 (더미 데이터, DB 불필요).

Usage:
    python scripts/generate_sample_reports.py
    → sample_single.pdf, sample_batch.pdf 생성
"""

import io
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트 설정 (Windows: Malgun Gothic)
def _set_korean_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans KR"]:
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    # fallback: 시스템에서 한글 지원 폰트 탐색
    korean_fonts = [f.fname for f in fm.fontManager.ttflist
                    if "gothic" in f.name.lower() or "gulim" in f.name.lower()]
    if korean_fonts:
        fm.fontManager.addfont(korean_fonts[0])
        plt.rcParams["font.family"] = fm.FontProperties(fname=korean_fonts[0]).get_name()
        plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 한글 폰트 등록 (맑은 고딕) ─────────────────────────────────────────────
_FONT_REGULAR = "C:/Windows/Fonts/malgun.ttf"
_FONT_BOLD    = "C:/Windows/Fonts/malgunbd.ttf"
pdfmetrics.registerFont(TTFont("MalgunGothic",     _FONT_REGULAR))
pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", _FONT_BOLD))

KR       = "MalgunGothic"
KR_BOLD  = "MalgunGothic-Bold"

# ── 색상 팔레트 ─────────────────────────────────────────────────────────────
C_NAVY   = colors.HexColor("#1E3A5F")
C_BLUE   = colors.HexColor("#2980B9")
C_LIGHT  = colors.HexColor("#EBF5FB")
C_GREEN  = colors.HexColor("#27AE60")
C_RED    = colors.HexColor("#E74C3C")
C_ORANGE = colors.HexColor("#E67E22")
C_YELLOW = colors.HexColor("#F39C12")
C_GRAY   = colors.HexColor("#7F8C8D")
C_LGRAY  = colors.HexColor("#ECF0F1")
C_WHITE  = colors.white
C_DARK   = colors.HexColor("#2C3E50")

SEVERITY_COLOR = {
    "없음":   C_GRAY,
    "낮음":   C_YELLOW,
    "보통":   C_ORANGE,
    "높음":   C_RED,
}

PAGE_W, PAGE_H = A4


# ── 더미 데이터 ──────────────────────────────────────────────────────────────
SINGLE_DUMMY = {
    "id": 42,
    "dataset": "GoodsAD",
    "category": "담배갑 (cigarette_box)",
    "image_path": None,
    "heatmap_path": None,
    "ad_score": 0.82,
    "is_anomaly_AD": True,
    "is_anomaly_LLM": True,
    "llm_report": {
        "anomaly_type": "찌그러짐 (dent)",
        "severity": "높음",
        "location": "우측 상단 모서리",
        "description": (
            "패키지 우측 상단 모서리에 명확한 찌그러짐이 관찰됨. "
            "포장재 구조적 무결성이 손상되었으며, "
            "표면의 약 15% 범위에 걸쳐 주름이 형성된 상태임."
        ),
        "confidence": 0.91,
        "recommendation": "즉시 생산 라인에서 제거 후 불량 처리",
    },
    "llm_summary": {
        "summary": "담배갑 우측 상단 모서리에 심각한 찌그러짐 결함 감지됨.",
        "risk_level": "높음",
    },
    "llm_inference_duration": 1.83,
    "AD_inference_duration": 0.42,
    "llm_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

BATCH_DUMMY = {
    "total": 50,
    "anomaly": 18,
    "category_stats": {
        "담배갑":   {"total": 20, "anomaly": 8},
        "음료병":   {"total": 15, "anomaly": 6},
        "케이블":   {"total": 10, "anomaly": 2},
        "목재":     {"total": 5,  "anomaly": 2},
    },
    "severity_dist": {"없음": 32, "낮음": 5, "보통": 7, "높음": 6},
    "anomaly_type_dist": {"찌그러짐": 8, "스크래치": 5, "오염": 3, "개봉": 2},
    "confidences": [round(random.uniform(0.60, 0.97), 2) for _ in range(18)],
    "ad_vs_llm_mismatch": 4,
}


# ── 공통 유틸 ────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def add(name, **kw):
        base.add(ParagraphStyle(name=name, **kw))

    add("H1",      fontSize=18, textColor=C_WHITE,  fontName=KR_BOLD,
        alignment=TA_CENTER, spaceAfter=4)
    add("H2",      fontSize=13, textColor=C_NAVY,   fontName=KR_BOLD,
        spaceBefore=10, spaceAfter=4)
    add("H3",      fontSize=11, textColor=C_DARK,   fontName=KR_BOLD,
        spaceBefore=6,  spaceAfter=2)
    add("Body",    fontSize=9,  textColor=C_DARK,   fontName=KR,
        spaceAfter=2,  leading=14)
    add("Label",   fontSize=8,  textColor=C_GRAY,   fontName=KR,
        spaceAfter=1)
    add("Value",   fontSize=10, textColor=C_DARK,   fontName=KR_BOLD,
        spaceAfter=4)
    add("Center9", fontSize=9,  textColor=C_DARK,   fontName=KR,
        alignment=TA_CENTER)
    add("Footer",  fontSize=8,  textColor=C_GRAY,   fontName=KR,
        alignment=TA_CENTER)
    return base


def plt_to_image(fig, width_mm=160, height_mm=80):
    """matplotlib figure -> ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width_mm * mm, height=height_mm * mm)


def placeholder_box(label: str, w_mm: float, h_mm: float):
    """회색 placeholder 이미지 박스."""
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax.set_facecolor("#D5D8DC")
    ax.text(0.5, 0.5, label, ha="center", va="center",
            fontsize=10, color="#7F8C8D", transform=ax.transAxes)
    ax.axis("off")
    fig.patch.set_facecolor("#D5D8DC")
    return plt_to_image(fig, w_mm, h_mm)


def verdict_badge(is_bad: bool, label_bad="불량", label_ok="정상") -> str:
    color = "#E74C3C" if is_bad else "#27AE60"
    text  = label_bad if is_bad else label_ok
    return f'<font color="{color}"><b> {text} </b></font>'


def severity_badge(sev: str) -> str:
    hex_map = {"없음": "#7F8C8D", "낮음": "#F39C12", "보통": "#E67E22", "높음": "#E74C3C"}
    c = hex_map.get(sev, "#7F8C8D")
    return f'<font color="{c}"><b>[{sev}]</b></font>'


# ══════════════════════════════════════════════════════════════════════════════
# 단일 이미지 PDF
# ══════════════════════════════════════════════════════════════════════════════
def build_single_pdf(output_path: str = "sample_single.pdf"):
    d   = SINGLE_DUMMY
    rep = d["llm_report"]
    summ = d["llm_summary"]
    S   = make_styles()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    story = []

    # ── 헤더 배너 ─────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("MMAD INSPECTOR", S["H1"]),
        Paragraph("단일 이미지 검사 리포트", S["H1"]),
    ]]
    header_table = Table(header_data, colWidths=[95*mm, 85*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6*mm))

    # ── 기본 정보 바 ──────────────────────────────────────────────────────
    info_data = [[
        Paragraph(f"<b>리포트 번호</b><br/># {d['id']}", S["Center9"]),
        Paragraph(f"<b>데이터셋</b><br/>{d['dataset']}", S["Center9"]),
        Paragraph(f"<b>제품 카테고리</b><br/>{d['category']}", S["Center9"]),
        Paragraph(f"<b>검사 일시</b><br/>{d['llm_start_time']}", S["Center9"]),
    ]]
    info_table = Table(info_data, colWidths=[35*mm, 35*mm, 55*mm, 55*mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEAFTER",  (0, 0), (2, 0), 0.5, C_GRAY),
        ("BOX",        (0, 0), (-1, -1), 0.5, C_BLUE),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6*mm))

    # ── 이미지 섹션 ───────────────────────────────────────────────────────
    story.append(Paragraph("이미지 분석", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    img_orig = placeholder_box("원본 이미지\n(cigarette_box)", 85, 65)
    img_heat = placeholder_box("이상 히트맵\n(AD Score: 0.82)", 85, 65)

    img_table = Table([[img_orig, img_heat]], colWidths=[90*mm, 90*mm])
    img_table.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    img_label = Table([[
        Paragraph("원본 이미지", S["Center9"]),
        Paragraph("이상 히트맵", S["Center9"]),
    ]], colWidths=[90*mm, 90*mm])
    img_label.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))

    story.append(img_table)
    story.append(img_label)
    story.append(Spacer(1, 6*mm))

    # ── 판정 결과 ─────────────────────────────────────────────────────────
    story.append(Paragraph("검사 판정", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    is_anomaly = d["is_anomaly_LLM"]
    ad_match   = d["is_anomaly_AD"] == d["is_anomaly_LLM"]
    mismatch_note = "" if ad_match else '  <font color="#E74C3C">⚠ AD/LLM 불일치</font>'

    verdict_data = [
        [
            Paragraph("<b>AD 모델 판정</b>", S["Center9"]),
            Paragraph("<b>LLM 판정</b>",     S["Center9"]),
            Paragraph("<b>최종 판정</b>",     S["Center9"]),
        ],
        [
            Paragraph(verdict_badge(d["is_anomaly_AD"]), S["Center9"]),
            Paragraph(verdict_badge(d["is_anomaly_LLM"]), S["Center9"]),
            Paragraph(
                verdict_badge(is_anomaly, "출하 불가", "출하 가능") + mismatch_note,
                S["Center9"]
            ),
        ],
    ]
    verdict_table = Table(verdict_data, colWidths=[55*mm, 55*mm, 70*mm])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_WHITE),
        ("BACKGROUND", (0, 1), (-1, 1), C_LGRAY),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.5, C_GRAY),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 6*mm))

    # ── 결함 상세 분석 ────────────────────────────────────────────────────
    story.append(Paragraph("결함 상세 분석", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    detail_data = [
        ["항목", "내용"],
        ["결함 유형",   rep["anomaly_type"]],
        ["심각도",      rep["severity"]],
        ["결함 위치",   rep["location"]],
        ["상세 설명",   rep["description"]],
        ["신뢰도",      f"{rep['confidence'] * 100:.0f}%"],
        ["위험 등급",   summ["risk_level"]],
    ]
    col_w = [40*mm, 140*mm]
    detail_table = Table(detail_data, colWidths=col_w)
    detail_table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), KR),
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), KR_BOLD),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (0, -1), "RIGHT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",       (0, 0), (-1, -1), 0.4, C_GRAY),
        ("TEXTCOLOR",  (1, 2), (1, 2), SEVERITY_COLOR[rep["severity"]]),
        ("FONTNAME",   (1, 2), (1, 2), KR_BOLD),
        ("FONTNAME",   (0, 1), (0, -1), KR_BOLD),
        ("TEXTCOLOR",  (0, 1), (0, -1), C_NAVY),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 5*mm))

    # ── 조치 권고 박스 ────────────────────────────────────────────────────
    rec_data = [[
        Paragraph(
            f'<b>조치 권고사항</b><br/>{rep["recommendation"]}',
            S["Body"]
        )
    ]]
    rec_table = Table(rec_data, colWidths=[180*mm])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FDEDEC")),
        ("BOX",           (0, 0), (-1, -1), 1.5, C_RED),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 5*mm))

    # ── 추론 성능 정보 ────────────────────────────────────────────────────
    story.append(Paragraph("추론 성능 정보", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    perf_data = [
        ["항목", "값"],
        ["AD 추론 시간",  f"{d['AD_inference_duration']:.2f} 초"],
        ["LLM 추론 시간", f"{d['llm_inference_duration']:.2f} 초"],
        ["AD 이상 점수",  f"{d['ad_score']:.3f}"],
    ]
    perf_table = Table(perf_data, colWidths=[60*mm, 120*mm])
    perf_table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), KR),
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), KR_BOLD),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (0, -1), "RIGHT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",       (0, 0), (-1, -1), 0.4, C_GRAY),
        ("FONTNAME",   (0, 1), (0, -1), KR_BOLD),
        ("TEXTCOLOR",  (0, 1), (0, -1), C_NAVY),
    ]))
    story.append(perf_table)

    # ── 푸터 ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GRAY))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"MMAD Inspector  |  생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        S["Footer"]
    ))

    doc.build(story)
    print(f"[OK] {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 배치(복수 이미지) PDF
# ══════════════════════════════════════════════════════════════════════════════
def make_bar_chart(labels, values, title, color_list=None):
    fig, ax = plt.subplots(figsize=(7, 3))
    bar_colors = color_list or ["#2980B9"] * len(labels)
    bars = ax.barh(labels, values, color=bar_colors, height=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("건수", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_pie_chart(labels, values, title, colors_list):
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors_list,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def make_histogram(values, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(values, bins=8, color="#2980B9", edgecolor="white", rwidth=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("신뢰도 점수", fontsize=9)
    ax.set_ylabel("건수", fontsize=9)
    ax.set_xlim(0.5, 1.0)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def build_batch_pdf(output_path: str = "sample_batch.pdf"):
    d = BATCH_DUMMY
    S = make_styles()

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    story = []

    # ── 헤더 ──────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("MMAD INSPECTOR", S["H1"]),
        Paragraph("배치 검사 종합 리포트", S["H1"]),
    ]]
    header_table = Table(header_data, colWidths=[95*mm, 85*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"데이터셋: GoodsAD / DS-MVTec",
        S["Footer"]
    ))
    story.append(Spacer(1, 5*mm))

    # ── 요약 통계 카드 ────────────────────────────────────────────────────
    story.append(Paragraph("검사 요약 통계", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    total   = d["total"]
    anomaly = d["anomaly"]
    normal  = total - anomaly
    rate    = anomaly / total * 100

    stat_data = [[
        Paragraph(f'<b><font size="18">{total}</font></b><br/><font size="8">전체 이미지</font>', S["Center9"]),
        Paragraph(f'<b><font size="18" color="#27AE60">{normal}</font></b><br/><font size="8">정상</font>', S["Center9"]),
        Paragraph(f'<b><font size="18" color="#E74C3C">{anomaly}</font></b><br/><font size="8">불량</font>', S["Center9"]),
        Paragraph(f'<b><font size="18" color="#E74C3C">{rate:.1f}%</font></b><br/><font size="8">불량률</font>', S["Center9"]),
        Paragraph(f'<b><font size="18" color="#E67E22">{d["ad_vs_llm_mismatch"]}</font></b><br/><font size="8">AD/LLM 불일치</font>', S["Center9"]),
    ]]
    stat_table = Table(stat_data, colWidths=[36*mm] * 5)
    stat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEAFTER",  (0, 0), (3, 0), 0.5, C_GRAY),
        ("BOX",        (0, 0), (-1, -1), 1, C_BLUE),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 6*mm))

    # ── 카테고리별 불량률 차트 ────────────────────────────────────────────
    story.append(Paragraph("카테고리별 불량 현황", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    cats   = list(d["category_stats"].keys())
    totals = [d["category_stats"][c]["total"]   for c in cats]
    anoms  = [d["category_stats"][c]["anomaly"] for c in cats]
    rates  = [a / t * 100 for a, t in zip(anoms, totals)]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    x = np.arange(len(cats))
    w = 0.35
    bars1 = ax.bar(x - w/2, totals, w, label="전체",   color="#2980B9", alpha=0.85)
    bars2 = ax.bar(x + w/2, anoms,  w, label="불량",   color="#E74C3C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("건수", fontsize=9)
    ax.set_title("카테고리별 불량 건수", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    for b, r in zip(bars2, rates):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
                f"{r:.0f}%", ha="center", fontsize=8, color="#E74C3C", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    story.append(plt_to_image(fig, 170, 70))
    story.append(Spacer(1, 4*mm))

    # ── 카테고리 상세 표 ──────────────────────────────────────────────────
    cat_table_data = [["카테고리", "전체", "정상", "불량", "불량률"]]
    for c, v in d["category_stats"].items():
        r = v["anomaly"] / v["total"] * 100
        cat_table_data.append([
            c,
            str(v["total"]),
            str(v["total"] - v["anomaly"]),
            str(v["anomaly"]),
            f"{r:.1f}%",
        ])
    cat_tbl = Table(cat_table_data, colWidths=[60*mm, 28*mm, 28*mm, 28*mm, 36*mm])
    cat_tbl.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), KR),
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), KR_BOLD),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",       (0, 0), (-1, -1), 0.4, C_GRAY),
        ("TEXTCOLOR",  (4, 1), (4, -1), C_RED),
        ("FONTNAME",   (4, 1), (4, -1), KR_BOLD),
    ]))
    story.append(cat_tbl)

    # ── 페이지 2: 결함 분석 ───────────────────────────────────────────────
    story.append(PageBreak())

    story.append(Paragraph("결함 유형 분석", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 4*mm))

    # 심각도 파이 + 결함 유형 바 (나란히)
    sev = d["severity_dist"]
    sev_hex = {"없음": "#7F8C8D", "낮음": "#F39C12", "보통": "#E67E22", "높음": "#E74C3C"}
    fig_pie = make_pie_chart(
        list(sev.keys()), list(sev.values()),
        "심각도 분포",
        [sev_hex[k] for k in sev.keys()]
    )
    pie_img = plt_to_image(fig_pie, 80, 70)

    atype = d["anomaly_type_dist"]
    fig_bar = make_bar_chart(
        list(atype.keys()), list(atype.values()),
        "결함 유형 빈도",
        ["#E74C3C", "#E67E22", "#F39C12", "#2980B9"]
    )
    bar_img = plt_to_image(fig_bar, 90, 70)

    charts_row = Table([[pie_img, bar_img]], colWidths=[88*mm, 92*mm])
    charts_row.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(charts_row)
    story.append(Spacer(1, 5*mm))

    # ── LLM 신뢰도 분포 ───────────────────────────────────────────────────
    story.append(Paragraph("LLM 신뢰도 분포", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 3*mm))

    fig_hist = make_histogram(d["confidences"], "불량 판정 신뢰도 분포 (불량 이미지 기준)")
    story.append(plt_to_image(fig_hist, 160, 65))
    story.append(Spacer(1, 3*mm))

    confs = d["confidences"]
    conf_data = [
        ["항목", "값"],
        ["평균 신뢰도",          f"{np.mean(confs):.3f}"],
        ["최소 신뢰도",          f"{np.min(confs):.3f}"],
        ["최대 신뢰도",          f"{np.max(confs):.3f}"],
        ["고신뢰도 (>=0.85)",    f"{sum(c >= 0.85 for c in confs)}건 / {len(confs)}건"],
    ]
    conf_tbl = Table(conf_data, colWidths=[80*mm, 100*mm])
    conf_tbl.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), KR),
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), KR_BOLD),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (0, -1), "RIGHT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",       (0, 0), (-1, -1), 0.4, C_GRAY),
        ("FONTNAME",   (0, 1), (0, -1), KR_BOLD),
        ("TEXTCOLOR",  (0, 1), (0, -1), C_NAVY),
    ]))
    story.append(conf_tbl)

    # ── 페이지 3: 불량 이미지 갤러리 ─────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("불량 이미지 갤러리", S["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_NAVY))
    story.append(Spacer(1, 4*mm))

    gallery_items = [
        {"id": 3,  "category": "담배갑",  "severity": "높음", "confidence": 0.91, "type": "찌그러짐"},
        {"id": 7,  "category": "음료병",  "severity": "보통", "confidence": 0.78, "type": "스크래치"},
        {"id": 12, "category": "케이블",  "severity": "낮음", "confidence": 0.65, "type": "오염"},
        {"id": 18, "category": "담배갑",  "severity": "높음", "confidence": 0.88, "type": "개봉"},
        {"id": 24, "category": "음료병",  "severity": "보통", "confidence": 0.72, "type": "찌그러짐"},
        {"id": 31, "category": "목재",    "severity": "낮음", "confidence": 0.61, "type": "스크래치"},
    ]
    sev_hex_border = {"없음": "#7F8C8D", "낮음": "#F39C12", "보통": "#E67E22", "높음": "#E74C3C"}

    COLS = 3
    rows = [gallery_items[i:i+COLS] for i in range(0, len(gallery_items), COLS)]

    for row in rows:
        row_imgs = []
        row_caps = []
        for item in row:
            bcolor = sev_hex_border[item["severity"]]
            fig, ax = plt.subplots(figsize=(1.8, 1.8))
            ax.set_facecolor("#D5D8DC")
            ax.text(0.5, 0.55, f"#{item['id']}", ha="center", va="center",
                    fontsize=11, color="#2C3E50", fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.3, item["category"],
                    ha="center", va="center", fontsize=8, color="#7F8C8D", transform=ax.transAxes)
            for spine in ax.spines.values():
                spine.set_edgecolor(bcolor)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor("#D5D8DC")
            fig.tight_layout(pad=0.1)
            row_imgs.append(plt_to_image(fig, 50, 45))

            cap = (
                f'<b>#{item["id"]}</b>  {item["category"]}<br/>'
                f'유형: {item["type"]}  |  '
                f'<font color="{bcolor}"><b>{item["severity"]}</b></font><br/>'
                f'신뢰도: {item["confidence"]:.0%}'
            )
            row_caps.append(Paragraph(cap, S["Center9"]))

        while len(row_imgs) < COLS:
            row_imgs.append(Spacer(50*mm, 45*mm))
            row_caps.append(Spacer(1, 1))

        gallery_tbl = Table(
            [row_imgs, row_caps],
            colWidths=[60*mm] * COLS,
        )
        gallery_tbl.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(gallery_tbl)
        story.append(Spacer(1, 3*mm))

    # ── 푸터 ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_GRAY))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"MMAD Inspector  |  생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        S["Footer"]
    ))

    doc.build(story)
    print(f"[OK] {output_path}")


# ── 실행 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_single_pdf("sample_single.pdf")
    build_batch_pdf("sample_batch.pdf")
    print("\n완료! sample_single.pdf 와 sample_batch.pdf 를 열어서 확인하세요.")
