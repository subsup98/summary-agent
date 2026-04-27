from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

from src.classifiers.document_classifier import classify_document
from src.parsers.common.models import DocumentElement
from src.parsers.pdf.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generalized_infographic_probe"

YEAR_RE = re.compile(r"^20\d{2}$")
PERCENT_RE = re.compile(r"^\d+(?:\.\d+)?%$")
NUMBER_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$")
KOREAN_RE = re.compile(r"[가-힣]")


@dataclass
class Token:
    text: str
    bbox: tuple[float, float, float, float]
    element_type: str

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def resolve_pdf_path() -> Path:
    desktop_docs = Path.home() / "Desktop" / "all_docs"
    for candidate in sorted(desktop_docs.glob("*.pdf")):
        if "미래에셋" in candidate.name and "3분기" in candidate.name:
            return candidate
    raise FileNotFoundError("Mirae Asset Q3 PDF not found on Desktop/all_docs")


def build_tokens(page_elements: list[DocumentElement], *, top_limit: float = 310.0) -> list[Token]:
    tokens: list[Token] = []
    for element in page_elements:
        text = str(element.text or "").strip()
        bbox = element.bbox or []
        if not text or len(bbox) != 4:
            continue
        if bbox[1] < 0 or bbox[1] > top_limit:
            continue
        if text.startswith("/Users/"):
            continue
        tokens.append(
            Token(
                text=text,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                element_type=element.element_type,
            )
        )
    return sorted(tokens, key=lambda item: (item.cy, item.cx))


def dedupe_tokens(tokens: list[Token]) -> list[Token]:
    kept: list[Token] = []
    for token in tokens:
        duplicate = False
        for existing in kept:
            if (
                token.text == existing.text
                and abs(token.cx - existing.cx) <= 4
                and abs(token.cy - existing.cy) <= 4
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
    return kept


def cluster_rows(tokens: list[Token], tolerance: float = 10.0) -> list[list[Token]]:
    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda item: (item.cy, item.cx)):
        if not rows:
            rows.append([token])
            continue
        row = rows[-1]
        row_center = sum(item.cy for item in row) / len(row)
        if abs(token.cy - row_center) <= tolerance:
            row.append(token)
        else:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda item: item.x0)
    return rows


def union_bbox(tokens: list[Token]) -> list[float]:
    return [
        min(token.x0 for token in tokens),
        min(token.y0 for token in tokens),
        max(token.x1 for token in tokens),
        max(token.y1 for token in tokens),
    ]


def build_visual_blocks(tokens: list[Token], chart_right_edge: float) -> list[dict[str, object]]:
    candidate_tokens = [token for token in tokens if token.cy >= 120]
    chart_block = [token for token in candidate_tokens if token.x0 < chart_right_edge + 18]
    right_block = [token for token in candidate_tokens if token.x0 >= chart_right_edge + 18]
    right_top = [token for token in right_block if token.cy < 220]
    right_bottom = [token for token in right_block if token.cy >= 220]

    blocks = [block for block in [chart_block, right_top, right_bottom] if block]

    payload: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        bbox = union_bbox(block)
        payload.append(
            {
                "block_id": index,
                "bbox": bbox,
                "token_count": len(block),
                "tokens": [
                    {"text": token.text, "bbox": list(token.bbox), "type": token.element_type}
                    for token in sorted(block, key=lambda item: (item.cy, item.cx))
                ],
            }
        )
    return payload


def detect_chart_year_anchors(tokens: list[Token]) -> list[Token]:
    candidates = [token for token in tokens if YEAR_RE.match(token.text)]
    if not candidates:
        return []
    rows = cluster_rows(candidates, tolerance=12.0)
    rows = [row for row in rows if len(row) >= 3]
    if not rows:
        return []
    chosen = max(rows, key=lambda row: (len(row), sum(item.cy for item in row) / len(row)))
    return sorted(chosen, key=lambda item: item.cx)


def detect_legend_labels(tokens: list[Token], chart_right_edge: float) -> list[str]:
    candidates = [
        token
        for token in tokens
        if token.x0 < chart_right_edge
        and token.cy < 170
        and KOREAN_RE.search(token.text)
        and not YEAR_RE.match(token.text)
        and not PERCENT_RE.match(token.text)
    ]
    rows = cluster_rows(candidates, tolerance=10.0)
    if not rows:
        return []
    best_row = max(rows, key=len)
    labels: list[str] = []
    for token in best_row:
        value = token.text.replace("■", "").strip()
        if not value or len(value) == 1:
            continue
        if value in labels:
            continue
        labels.append(value)
    return labels


def build_chart_series(tokens: list[Token], anchors: list[Token]) -> list[dict[str, object]]:
    if not anchors:
        return []
    chart_left = min(anchor.x0 for anchor in anchors) - 40
    chart_right = max(anchor.x1 for anchor in anchors) + 40
    chart_tokens = [
        token
        for token in tokens
        if chart_left <= token.cx <= chart_right
        and 160 <= token.cy <= 285
        and (NUMBER_RE.match(token.text) or PERCENT_RE.match(token.text))
    ]

    bands: dict[int, list[Token]] = {}
    for token in chart_tokens:
        anchor_index = min(range(len(anchors)), key=lambda idx: abs(token.cx - anchors[idx].cx))
        if abs(token.cx - anchors[anchor_index].cx) <= 28:
            bands.setdefault(anchor_index, []).append(token)

    years = [anchor.text for anchor in anchors]
    series_names = ["rate_percent", "total_amount", "dividend_amount", "retirement_amount"]
    series_values: dict[str, list[str | None]] = {name: [] for name in series_names}

    for index in range(len(anchors)):
        column = sorted(bands.get(index, []), key=lambda item: (item.cy, abs(item.cx - anchors[index].cx)))
        percents = [item for item in column if PERCENT_RE.match(item.text)]
        percent = None
        if percents:
            percent = min(percents, key=lambda item: abs(item.cx - anchors[index].cx)).text

        number_tokens = [item for item in column if NUMBER_RE.match(item.text)]
        number_tokens.sort(key=lambda item: item.cy)
        total = number_tokens[0].text if len(number_tokens) >= 1 else None
        amount_top = number_tokens[1].text if len(number_tokens) >= 2 else None
        amount_bottom = number_tokens[2].text if len(number_tokens) >= 3 else None

        series_values["rate_percent"].append(percent)
        series_values["total_amount"].append(total)
        series_values["dividend_amount"].append(amount_top)
        series_values["retirement_amount"].append(amount_bottom)

    return [
        {"name": name, "years": years, "values": values}
        for name, values in series_values.items()
    ]


def _numeric_value(text: str) -> float:
    try:
        return float(text.replace(",", "").replace("%", ""))
    except ValueError:
        return -1.0


def detect_right_side_kpis(tokens: list[Token], chart_right_edge: float) -> dict[str, object]:
    right_tokens = [token for token in tokens if token.x0 >= chart_right_edge + 20]
    rows = cluster_rows(right_tokens, tolerance=14.0)
    title_parts: list[str] = []
    subtitle_parts: list[str] = []
    large_numbers: list[Token] = []
    label_value_pairs: list[dict[str, str]] = []

    for row in rows:
        row_text = "".join(token.text for token in row if token.text != "■").strip()
        if not row_text:
            continue
        row_korean = [token.text for token in row if KOREAN_RE.search(token.text) and len(token.text.strip()) >= 2]
        if not title_parts and 130 <= row[0].cy <= 165 and row_korean:
            title_parts.append("".join(row_korean))
            continue
        if "이후" in row_text:
            subtitle_parts.append("".join(row_korean) or row_text)
            continue
        row_numbers = [token for token in row if NUMBER_RE.match(token.text)]
        if row_numbers and max(token.height for token in row_numbers) >= 15 and row[0].cy < 230:
            large_numbers.extend(row_numbers)
        labels = [token.text for token in row if KOREAN_RE.search(token.text) and not NUMBER_RE.match(token.text) and len(token.text.strip()) >= 2]
        values = [token.text for token in row if NUMBER_RE.match(token.text)]
        if labels and values:
            label_value_pairs.append({"label": "".join(labels), "value": "".join(values)})

    large_numbers = sorted(large_numbers, key=lambda item: (-item.height, item.cy, item.cx))
    total_amount = None
    if large_numbers:
        total_amount = large_numbers[0].text

    equation_row_tokens = [
        token
        for token in right_tokens
        if 220 <= token.cy <= 275
    ]
    equation_rows = cluster_rows(equation_row_tokens, tolerance=20.0)
    equation_row = max(equation_rows, key=lambda row: sum(1 for item in row if KOREAN_RE.search(item.text) or NUMBER_RE.match(item.text))) if equation_rows else []
    equation = _compress_equation_row(equation_row)

    return {
        "title_candidates": title_parts,
        "subtitle_candidates": subtitle_parts,
        "large_total_candidate": total_amount,
        "label_value_pairs": label_value_pairs,
        "equation_tokens": equation,
    }


def _compress_equation_row(row: list[Token]) -> list[str]:
    if not row:
        return []
    ordered = sorted(row, key=lambda item: item.x0)
    groups: list[list[Token]] = [[ordered[0]]]
    for token in ordered[1:]:
        if token.x0 - groups[-1][-1].x1 <= 18:
            groups[-1].append(token)
        else:
            groups.append([token])

    values: list[str] = []
    for group in groups:
        text = "".join(item.text for item in group).strip()
        text = text.replace(" ", "")
        if not text:
            continue
        values.append(text)
    return values


def main() -> None:
    pdf_path = resolve_pdf_path()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))
    page = parsed.pages[15]

    tokens = dedupe_tokens(build_tokens(page.elements))
    anchors = detect_chart_year_anchors(tokens)
    chart_right_edge = max((anchor.x1 for anchor in anchors), default=380.0)
    legend_labels = detect_legend_labels(tokens, chart_right_edge)
    chart_series = build_chart_series(tokens, anchors)
    kpis = detect_right_side_kpis(tokens, chart_right_edge)
    visual_blocks = build_visual_blocks(tokens, chart_right_edge)

    payload = {
        "pdf_path": pdf_path.as_posix(),
        "page_number": 16,
        "top_region_token_count": len(tokens),
        "chart": {
            "year_anchors": [
                {"text": anchor.text, "bbox": list(anchor.bbox)}
                for anchor in anchors
            ],
            "legend_labels": legend_labels,
            "series": chart_series,
        },
        "right_panel": kpis,
        "visual_blocks": visual_blocks,
        "sample_tokens": [
            {"text": token.text, "bbox": list(token.bbox), "type": token.element_type}
            for token in tokens[:120]
        ],
    }

    output_path = output_dir / "miraeasset_page16_top_probe.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path = output_dir / "miraeasset_page16_top_probe.html"
    html_path.write_text(build_html_report(payload), encoding="utf-8")
    print(json.dumps({"output_path": output_path.as_posix(), "html_path": html_path.as_posix()}, ensure_ascii=False))


def build_html_report(payload: dict[str, object]) -> str:
    blocks = payload.get("visual_blocks") or []
    cards: list[str] = []
    overlays: list[str] = []
    colors = ["#ff7a18", "#0f4c81", "#3f8f5f", "#a855f7", "#d9485f", "#1f9d8b", "#b7791f"]

    for index, block in enumerate(blocks):
        bbox = block["bbox"]
        color = colors[index % len(colors)]
        left = bbox[0]
        top = bbox[1]
        width = max(1.0, bbox[2] - bbox[0])
        height = max(1.0, bbox[3] - bbox[1])
        overlays.append(
            f"<div class='overlay' style='left:{left}px;top:{top}px;width:{width}px;height:{height}px;border-color:{color};'>"
            f"<span style='background:{color};'>B{block['block_id']}</span></div>"
        )
        token_lines = "".join(
            f"<li><code>{escape(token['text'])}</code> <span>{token['bbox']}</span></li>"
            for token in block["tokens"][:20]
        )
        cards.append(
            "<section class='card'>"
            f"<h3>Block {block['block_id']}</h3>"
            f"<p>bbox={block['bbox']} token_count={block['token_count']}</p>"
            f"<ul>{token_lines}</ul>"
            "</section>"
        )

    summary = payload.get("chart") or {}
    right_panel = payload.get("right_panel") or {}
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mirae Asset Page 16 Top Probe</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #f4f1ea;
      color: #1f2933;
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    .layout {{
      display: grid;
      grid-template-columns: 820px 1fr;
      gap: 20px;
      align-items: start;
    }}
    .canvas {{
      position: relative;
      width: 800px;
      height: 360px;
      background: linear-gradient(180deg, #fffaf2, #fff);
      border: 1px solid #e2d7c6;
      box-shadow: 0 16px 40px rgba(31, 41, 51, 0.08);
      overflow: hidden;
    }}
    .overlay {{
      position: absolute;
      border: 2px solid;
      background: rgba(255,255,255,0.08);
      box-sizing: border-box;
    }}
    .overlay span {{
      position: absolute;
      left: -2px;
      top: -22px;
      color: white;
      font-size: 12px;
      padding: 2px 6px;
      border-radius: 999px;
    }}
    .sidebar {{
      display: grid;
      gap: 12px;
    }}
    .summary, .card {{
      background: white;
      border: 1px solid #eadfce;
      padding: 14px 16px;
      box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06);
    }}
    .summary pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      margin: 8px 0 0;
    }}
    .card ul {{
      margin: 10px 0 0;
      padding-left: 18px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .card li span {{
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <h1>미래에셋 3분기 16페이지 상단 구조화 프로브</h1>
  <p>상단 인포그래픽 토큰을 일반화된 블록 군집으로 나눈 결과입니다.</p>
  <div class="layout">
    <div class="canvas">
      {''.join(overlays)}
    </div>
    <div class="sidebar">
      <section class="summary">
        <h2>Chart Summary</h2>
        <pre>{escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
      </section>
      <section class="summary">
        <h2>Right Panel Summary</h2>
        <pre>{escape(json.dumps(right_panel, ensure_ascii=False, indent=2))}</pre>
      </section>
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
