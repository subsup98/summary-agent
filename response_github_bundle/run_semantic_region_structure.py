from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any

import fitz

import run_region_type_classifier as type_classifier
import run_x_coordinate_region_structure as x_structure


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drawing_guided_pages_04_16"
PDF_NAME = "\ubbf8\ub798\uc5d0\uc14b\uc99d\uad8c 3\ubd84\uae30 \uc2e4\uc801\ubcf4\uace0\uc11c.pdf"
PDF_PATH = Path.home() / "Desktop" / "all_docs" / PDF_NAME
PAGE_JSON_PATHS = [
    OUTPUT_DIR / "page_04_drawing_guided_structure.json",
    OUTPUT_DIR / "page_16_drawing_guided_structure.json",
]


NUMBER_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$|^\d+(?:\.\d+)?%$|^\(\d+%\)$")
UNIT_RE = re.compile(r"^(?:억원|억|원|조원|조|%|%p|만주|만|주)$")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    output_json_paths: list[Path] = []

    with fitz.open(PDF_PATH) as document:
        for path in PAGE_JSON_PATHS:
            payload = json.loads(path.read_text(encoding="utf-8"))
            type_classifier.classify_page_regions(payload)
            x_structure.add_x_coordinate_structure(document, payload)
            add_semantic_structures(payload)
            page_number = int(payload["page_number"])
            payload["image_file"] = f"page_{page_number:02d}_render.png"
            payload["json_file"] = f"page_{page_number:02d}_drawing_guided_semantic_structure.json"
            render_page_image(document, page_number, OUTPUT_DIR / payload["image_file"])
            out_path = OUTPUT_DIR / payload["json_file"]
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output_json_paths.append(out_path)
            pages.append(payload)

    summary_path = OUTPUT_DIR / "semantic_structure_summary.md"
    html_path = OUTPUT_DIR / "semantic_structure_viewer.html"
    summary_path.write_text(render_summary(pages), encoding="utf-8")
    html_path.write_text(build_html(pages), encoding="utf-8")
    print(json.dumps({
        "json": [path.as_posix() for path in output_json_paths],
        "markdown": summary_path.as_posix(),
        "html": html_path.as_posix(),
    }, ensure_ascii=False, indent=2))


def render_page_image(document: fitz.Document, page_number: int, output_path: Path) -> None:
    page = document[page_number - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(output_path)


def add_semantic_structures(payload: dict[str, Any]) -> None:
    for region in payload.get("regions", []):
        predicted_type = (region.get("type_classification") or {}).get("predicted_type") or region.get("type") or "unknown"
        region["semantic_structure"] = build_semantic_structure(region, predicted_type)


def build_semantic_structure(region: dict[str, Any], predicted_type: str) -> dict[str, Any]:
    markdown = str(region.get("markdown") or "")
    x_coord = region.get("x_coordinate_structure") or {}
    if predicted_type == "table":
        return table_structure(markdown, x_coord)
    if predicted_type in {"chart", "mixed_chart_panel"}:
        return chart_structure(markdown, x_coord, predicted_type)
    if predicted_type in {"kpi_panel", "kpi_pair_panel"}:
        return kpi_structure(markdown, x_coord, predicted_type)
    if predicted_type == "highlight_card":
        return text_card_structure(markdown, kind="highlight_card")
    if predicted_type == "notes":
        return text_card_structure(markdown, kind="notes")
    return {
        "kind": "reading_order_text",
        "lines": clean_lines(markdown),
        "fallback_reason": f"no_semantic_builder_for_{predicted_type}",
    }


def table_structure(markdown: str, x_coord: dict[str, Any]) -> dict[str, Any]:
    rows = parse_markdown_table(markdown)
    if rows:
        return {
            "kind": "table",
            "strategy": "markdown_table_preserved",
            "header": rows[0],
            "rows": rows[1:],
            "row_count": max(0, len(rows) - 1),
            "column_count": len(rows[0]),
        }
    columns = semantic_columns(x_coord)
    return {
        "kind": "table",
        "strategy": "x_columns_fallback",
        "columns": columns,
        "warning": "No parseable markdown table found; using region-local x columns.",
    }


def chart_structure(markdown: str, x_coord: dict[str, Any], predicted_type: str) -> dict[str, Any]:
    columns = semantic_columns(x_coord)
    axis_columns = [column for column in columns if looks_like_axis_label(column.get("anchor", ""))]
    period_anchors = infer_period_anchors(columns)
    line_points = infer_line_chart_points(columns)
    callout = infer_numeric_callout(columns)
    series_columns = axis_columns or columns
    return {
        "kind": predicted_type,
        "strategy": "line_points_then_axis_or_x_column_series",
        "title_candidates": title_candidates(markdown),
        "period_anchors": period_anchors,
        "line_chart_points": line_points,
        "current_value": callout,
        "series_columns": [
            {
                "label": column.get("anchor"),
                "values": [item["text"] for item in column.get("items", []) if has_digit(item["text"])],
                "text": column.get("joined_text", ""),
            }
            for column in series_columns
        ],
        "bands": x_coord.get("y_bands") or [],
        "warnings": chart_warnings(columns, axis_columns),
    }


def kpi_structure(markdown: str, x_coord: dict[str, Any], predicted_type: str) -> dict[str, Any]:
    candidates = []
    for column in semantic_columns(x_coord):
        items = column.get("items", [])
        numeric_items = [item for item in items if NUMBER_RE.match(normalize(item.get("text", "")))]
        if not numeric_items:
            continue
        for numeric in numeric_items:
            label_tokens = nearest_label_tokens(items, numeric)
            unit_tokens = following_unit_tokens(items, numeric)
            candidates.append({
                "label": " ".join(label_tokens).strip(),
                "value": numeric["text"],
                "unit": " ".join(unit_tokens).strip(),
                "column_anchor": column.get("anchor"),
                "column_text": column.get("joined_text", ""),
            })
    return {
        "kind": predicted_type,
        "strategy": "numeric_value_candidates_from_region_columns",
        "title_candidates": title_candidates(markdown),
        "items": dedupe_kpi_items(candidates),
        "warnings": [] if candidates else ["No numeric KPI candidates found."],
    }


def nearest_label_tokens(items: list[dict[str, Any]], value_item: dict[str, Any]) -> list[str]:
    value_cy = float(value_item.get("cy") or 0.0)
    label_rows: dict[float, list[dict[str, Any]]] = {}
    for item in items:
        text = str(item.get("text") or "")
        cy = float(item.get("cy") or 0.0)
        if cy >= value_cy or value_cy - cy > 48.0:
            continue
        if has_digit(text) or UNIT_RE.match(normalize(text)) or normalize(text) in {"(", ")", "&", ","}:
            continue
        row_key = round(cy / 4.0) * 4.0
        label_rows.setdefault(row_key, []).append(item)
    if not label_rows:
        return []
    nearest_row = max(label_rows)
    return [item["text"] for item in sorted(label_rows[nearest_row], key=lambda token: token.get("cx") or 0.0)]


def following_unit_tokens(items: list[dict[str, Any]], value_item: dict[str, Any]) -> list[str]:
    value_cx = float(value_item.get("cx") or 0.0)
    value_cy = float(value_item.get("cy") or 0.0)
    units = []
    for item in sorted(items, key=lambda token: token.get("cx") or 0.0):
        text = str(item.get("text") or "")
        cx = float(item.get("cx") or 0.0)
        cy = float(item.get("cy") or 0.0)
        if cx <= value_cx or abs(cy - value_cy) > 10.0:
            continue
        if UNIT_RE.match(normalize(text)):
            units.append(text)
    return units[:2]


def infer_line_chart_points(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    period_anchors = infer_period_anchors(columns)
    plot_values = []
    for column in columns:
        if column.get("band_index") != 1:
            continue
        for item in column.get("items") or []:
            text = str(item.get("text") or "")
            if not NUMBER_RE.match(normalize(text)):
                continue
            bbox = item.get("bbox") or []
            height = float(bbox[3] - bbox[1]) if len(bbox) == 4 else 0.0
            if height >= 15.0:
                continue
            plot_values.append(item)
    plot_values = sorted(plot_values, key=lambda item: (float(item.get("cx") or 0.0), float(item.get("cy") or 0.0)))
    points = []
    last_index = len(plot_values) - 1
    for index, item in enumerate(plot_values):
        period = matched_period_label(index, last_index, period_anchors)
        points.append({
            "period": period,
            "value": item.get("text"),
            "cx": item.get("cx"),
            "cy": item.get("cy"),
            "bbox": item.get("bbox"),
        })
    return points


def infer_numeric_callout(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    period_anchors = infer_period_anchors(columns)
    candidates = []
    for column in columns:
        if column.get("band_index") != 1:
            continue
        for item in column.get("items") or []:
            text = str(item.get("text") or "")
            if not NUMBER_RE.match(normalize(text)):
                continue
            bbox = item.get("bbox") or []
            height = float(bbox[3] - bbox[1]) if len(bbox) == 4 else 0.0
            candidates.append((height, item, column))
    if not candidates:
        return None
    _, item, column = max(candidates, key=lambda candidate: candidate[0])
    return {
        "period": period_anchors[-1]["label"] if period_anchors else None,
        "value": item.get("text"),
        "unit": " ".join(following_unit_tokens(column.get("items") or [], item)),
        "cx": item.get("cx"),
        "cy": item.get("cy"),
        "bbox": item.get("bbox"),
        "reason": "largest numeric label in plot band, treated as callout/current value",
    }


def infer_period_anchors(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for column in columns:
        for item in column.get("items") or []:
            clone = dict(item)
            clone["band_index"] = column.get("band_index")
            items.append(clone)
    merged = merge_adjacent_period_tokens(items)
    anchors = [
        anchor
        for anchor in merged
        if looks_like_period_anchor(anchor.get("label", ""))
    ]
    return sorted(anchors, key=lambda anchor: (anchor.get("band_index") or 0, anchor.get("cy") or 0.0, anchor.get("cx") or 0.0))


def merge_adjacent_period_tokens(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=lambda item: (item.get("band_index") or 0, round(float(item.get("cy") or 0.0) / 4.0), item.get("cx") or 0.0))
    anchors: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, item in enumerate(sorted_items):
        if index in used:
            continue
        text = normalize(str(item.get("text") or ""))
        if re.fullmatch(r"\d", text):
            neighbor = next_period_neighbor(sorted_items, index)
            if neighbor is not None:
                used.add(index)
                used.add(neighbor[0])
                anchors.append(period_anchor(text + normalize(neighbor[1].get("text", "")), [item, neighbor[1]]))
                continue
        anchors.append(period_anchor(text, [item]))
    return anchors


def next_period_neighbor(items: list[dict[str, Any]], index: int) -> tuple[int, dict[str, Any]] | None:
    base = items[index]
    base_band = base.get("band_index")
    base_cy = float(base.get("cy") or 0.0)
    base_cx = float(base.get("cx") or 0.0)
    for next_index in range(index + 1, min(index + 4, len(items))):
        candidate = items[next_index]
        if candidate.get("band_index") != base_band:
            continue
        if abs(float(candidate.get("cy") or 0.0) - base_cy) > 6.0:
            continue
        if float(candidate.get("cx") or 0.0) - base_cx > 28.0:
            continue
        if re.fullmatch(r"Q\d{2}", normalize(str(candidate.get("text") or "")), flags=re.IGNORECASE):
            return next_index, candidate
    return None


def period_anchor(label: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "band_index": min(item.get("band_index") or 0 for item in items),
        "cx": sum(float(item.get("cx") or 0.0) for item in items) / max(1, len(items)),
        "cy": sum(float(item.get("cy") or 0.0) for item in items) / max(1, len(items)),
        "bbox": union_item_bboxes(items),
    }


def union_item_bboxes(items: list[dict[str, Any]]) -> list[float] | None:
    bboxes = [item.get("bbox") for item in items if len(item.get("bbox") or []) == 4]
    if not bboxes:
        return None
    return [
        round(min(float(bbox[0]) for bbox in bboxes), 3),
        round(min(float(bbox[1]) for bbox in bboxes), 3),
        round(max(float(bbox[2]) for bbox in bboxes), 3),
        round(max(float(bbox[3]) for bbox in bboxes), 3),
    ]


def matched_period_label(index: int, last_index: int, period_anchors: list[dict[str, Any]]) -> str | None:
    if not period_anchors:
        return None
    if len(period_anchors) == len(range(last_index + 1)):
        return str(period_anchors[index].get("label") or "")
    if index == 0:
        return str(period_anchors[0].get("label") or "")
    if index == last_index:
        return str(period_anchors[-1].get("label") or "")
    return None


def text_card_structure(markdown: str, *, kind: str) -> dict[str, Any]:
    lines = clean_lines(markdown)
    bullets = [strip_bullet(line) for line in lines if is_bullet(line)]
    headings = [line for line in lines if not is_bullet(line)][:2]
    return {
        "kind": kind,
        "strategy": "reading_order_lines",
        "headings": headings,
        "bullets": bullets,
        "lines": lines,
    }


def parse_markdown_table(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def semantic_columns(x_coord: dict[str, Any]) -> list[dict[str, Any]]:
    columns = []
    for column in x_coord.get("columns") or []:
        columns.append({
            "band_index": column.get("band_index"),
            "column_index": column.get("column_index"),
            "anchor": column.get("anchor"),
            "x_range": column.get("x_range"),
            "items": column.get("items_top_to_bottom") or [],
            "joined_text": column.get("joined_text") or "",
        })
    return columns


def title_candidates(markdown: str) -> list[str]:
    candidates = []
    for line in clean_lines(markdown):
        if len(candidates) >= 3:
            break
        if len(line) <= 60 and not NUMBER_RE.match(normalize(line)):
            candidates.append(line)
    return candidates


def chart_warnings(columns: list[dict[str, Any]], axis_columns: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if not axis_columns:
        warnings.append("No explicit axis anchors found; all x columns are shown as fallback series.")
    if len(columns) > 8:
        warnings.append("Many x columns detected; region may contain mixed chart, legend, or notes content.")
    return warnings


def dedupe_kpi_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for item in items:
        key = (item.get("label", ""), item.get("value", ""), item.get("unit", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def clean_lines(markdown: str) -> list[str]:
    return [line.strip() for line in markdown.replace("\r\n", "\n").splitlines() if line.strip()]


def is_bullet(line: str) -> bool:
    return bool(re.match(r"^\s*[-*•]\s+", line))


def strip_bullet(line: str) -> str:
    return re.sub(r"^\s*[-*•]\s+", "", line).strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", str(text or "")))


def looks_like_axis_label(text: str) -> bool:
    return bool(re.fullmatch(r"(?:20\d{2}|3\s?Q\d{2}|3Q\d{2}|Q\d{2}|YTD|QoQ)", normalize(text), flags=re.IGNORECASE))


def looks_like_period_anchor(text: str) -> bool:
    return bool(re.fullmatch(r"(?:20\d{2}|3\s?Q\d{2}|3Q\d{2}|Q\d{2})", normalize(text), flags=re.IGNORECASE))


def render_summary(pages: list[dict[str, Any]]) -> str:
    lines = [
        "# Semantic Region Structure Test",
        "",
        "> Pages: Mirae Asset Q3 report page 4 and page 16",
        "> Strategy: region type routing, then semantic builders; x-coordinate columns are kept as fallback evidence.",
        "",
        "| page | region | predicted | semantic kind | strategy | preview |",
        "|---:|---|---|---|---|---|",
    ]
    for page in pages:
        for region in page.get("regions", []):
            semantic = region.get("semantic_structure") or {}
            predicted = (region.get("type_classification") or {}).get("predicted_type") or region.get("type")
            lines.append(
                "| {page} | {region} | {predicted} | {kind} | {strategy} | {preview} |".format(
                    page=page.get("page_number"),
                    region=region.get("id"),
                    predicted=predicted,
                    kind=semantic.get("kind", ""),
                    strategy=semantic.get("strategy", ""),
                    preview=escape_markdown_table_cell(preview_semantic(semantic)),
                )
            )
    lines.extend(["", "## Region Details", ""])
    for page in pages:
        lines.append(f"### Page {page.get('page_number')}")
        lines.append("")
        for region in page.get("regions", []):
            semantic = region.get("semantic_structure") or {}
            predicted = (region.get("type_classification") or {}).get("predicted_type") or region.get("type")
            lines.append(f"#### {region.get('id')} - {predicted}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(semantic, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preview_semantic(semantic: dict[str, Any]) -> str:
    if semantic.get("kind") == "table":
        if semantic.get("header"):
            return "header=" + " / ".join(semantic.get("header") or [])
        return "columns=" + ", ".join(str(column.get("anchor")) for column in semantic.get("columns", [])[:4])
    if "items" in semantic:
        return "; ".join(
            " ".join(part for part in [item.get("label"), item.get("value"), item.get("unit")] if part).strip()
            for item in semantic.get("items", [])[:3]
        )
    if semantic.get("line_chart_points"):
        points = semantic.get("line_chart_points") or []
        endpoint_text = "; ".join(
            f"{point.get('period')}: {point.get('value')}"
            for point in points
            if str(point.get("period") or "").startswith("3Q")
        )
        current = semantic.get("current_value") or {}
        current_text = f"current {current.get('period')}: {current.get('value')} {current.get('unit')}".strip() if current else ""
        return "; ".join(part for part in [endpoint_text, current_text] if part)
    if "series_columns" in semantic:
        return "; ".join(
            f"{column.get('label')}: {'/'.join(column.get('values') or [])}"
            for column in semantic.get("series_columns", [])[:3]
        )
    return " / ".join((semantic.get("headings") or semantic.get("lines") or [])[:2])


def escape_markdown_table_cell(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def build_html(pages: list[dict[str, Any]]) -> str:
    data_json = json.dumps(pages, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mirae Asset Semantic Structure Test</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #fff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #f57c17;
      --blue: #1769e0;
      --green: #07844f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 "Segoe UI", Arial, sans-serif; }}
    header {{ height: 54px; display: flex; align-items: center; gap: 12px; padding: 0 18px; background: var(--panel); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 10; }}
    h1 {{ margin: 0; font-size: 16px; }}
    button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 6px; padding: 7px 10px; cursor: pointer; font: inherit; }}
    button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    .tabs {{ display: flex; gap: 6px; margin-left: auto; }}
    main {{ display: grid; grid-template-columns: minmax(620px, 1fr) 460px; gap: 14px; padding: 14px; min-height: calc(100vh - 54px); }}
    .canvasPanel, .detailPanel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .canvasToolbar {{ display: flex; gap: 10px; align-items: center; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .stageWrap {{ padding: 12px; overflow: auto; max-height: calc(100vh - 112px); }}
    .stage {{ position: relative; width: min(100%, 1180px); margin: 0 auto; background: #fff; box-shadow: 0 1px 10px rgba(16, 24, 40, 0.08); }}
    .stage img {{ display: block; width: 100%; height: auto; }}
    .regionBox {{ position: absolute; border: 2px solid var(--blue); background: rgba(23, 105, 224, 0.07); cursor: pointer; }}
    .regionBox:nth-of-type(3n + 1) {{ border-color: #ef3b2d; background: rgba(239, 59, 45, 0.07); }}
    .regionBox:nth-of-type(3n + 2) {{ border-color: var(--blue); background: rgba(23, 105, 224, 0.07); }}
    .regionBox:nth-of-type(3n) {{ border-color: var(--green); background: rgba(7, 132, 79, 0.07); }}
    .regionBox.active {{ border-color: var(--accent); background: rgba(245, 124, 23, 0.16); box-shadow: 0 0 0 3px rgba(245, 124, 23, 0.28); z-index: 3; }}
    .regionLabel {{ position: absolute; left: 4px; top: -22px; padding: 2px 5px; border-radius: 4px; background: rgba(255, 255, 255, 0.95); border: 1px solid var(--line); font-size: 11px; white-space: nowrap; pointer-events: none; }}
    .detailPanel {{ display: flex; flex-direction: column; min-width: 0; max-height: calc(100vh - 82px); }}
    .detailHeader {{ padding: 12px; border-bottom: 1px solid var(--line); }}
    .detailHeader h2 {{ margin: 0 0 4px; font-size: 16px; }}
    .meta {{ display: grid; grid-template-columns: 82px 1fr; gap: 4px 8px; color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .regionList {{ padding: 10px 12px; border-bottom: 1px solid var(--line); display: grid; gap: 6px; max-height: 220px; overflow: auto; }}
    .regionItem {{ display: grid; grid-template-columns: 72px 1fr; gap: 6px; text-align: left; padding: 8px; border-radius: 6px; }}
    .regionItem.active {{ border-color: var(--accent); background: #fff8f2; }}
    .regionId {{ font-weight: 700; color: var(--accent); }}
    .regionType {{ color: var(--ink); overflow-wrap: anywhere; }}
    .content {{ padding: 12px; overflow: auto; flex: 1; }}
    .sectionTitle {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0 0 14px; padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: #fbfcfe; color: #26323f; font: 12px/1.45 Consolas, "Courier New", monospace; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns: 1fr; }} .detailPanel {{ max-height: none; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Mirae Asset Semantic Structure Test</h1>
    <span class="hint">Page 4 and 16, routed by region type.</span>
    <div class="tabs" id="tabs"></div>
  </header>
  <main>
    <section class="canvasPanel">
      <div class="canvasToolbar"><strong id="pageTitle"></strong><span class="hint" id="strategy"></span></div>
      <div class="stageWrap"><div class="stage" id="stage"></div></div>
    </section>
    <aside class="detailPanel">
      <div class="detailHeader"><h2 id="regionTitle">Region</h2><div class="meta" id="regionMeta"></div></div>
      <div class="regionList" id="regionList"></div>
      <div class="content">
        <h3 class="sectionTitle">Semantic Structure</h3><pre id="semanticText"></pre>
        <h3 class="sectionTitle">Assigned Text / Markdown</h3><pre id="regionText"></pre>
        <h3 class="sectionTitle">X Coordinate Evidence</h3><pre id="xText"></pre>
      </div>
    </aside>
  </main>
  <script>
    const DATA = {data_json};
    let pageIndex = 0;
    let regionIndex = 0;
    const tabs = document.getElementById("tabs");
    const stage = document.getElementById("stage");
    const pageTitle = document.getElementById("pageTitle");
    const strategy = document.getElementById("strategy");
    const regionList = document.getElementById("regionList");
    const regionTitle = document.getElementById("regionTitle");
    const regionMeta = document.getElementById("regionMeta");
    const semanticText = document.getElementById("semanticText");
    const regionText = document.getElementById("regionText");
    const xText = document.getElementById("xText");
    function pct(value, total) {{ return (value / total * 100).toFixed(4) + "%"; }}
    function html(value) {{ return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;"); }}
    function initTabs() {{
      DATA.forEach((page, index) => {{
        const btn = document.createElement("button");
        btn.textContent = `Page ${{page.page_number}}`;
        btn.onclick = () => {{ pageIndex = index; regionIndex = 0; render(); }};
        tabs.appendChild(btn);
      }});
    }}
    function render() {{
      const page = DATA[pageIndex];
      const [pageW, pageH] = page.page_size;
      pageTitle.textContent = `Page ${{page.page_number}}`;
      strategy.textContent = page.strategy;
      [...tabs.children].forEach((btn, index) => btn.classList.toggle("active", index === pageIndex));
      stage.innerHTML = "";
      const img = document.createElement("img");
      img.src = page.image_file;
      img.alt = `Page ${{page.page_number}}`;
      stage.appendChild(img);
      page.regions.forEach((region, index) => {{
        const [x0, y0, x1, y1] = region.bbox;
        const box = document.createElement("div");
        box.className = "regionBox" + (index === regionIndex ? " active" : "");
        box.style.left = pct(x0, pageW);
        box.style.top = pct(y0, pageH);
        box.style.width = pct(x1 - x0, pageW);
        box.style.height = pct(y1 - y0, pageH);
        box.onclick = () => {{ regionIndex = index; renderRegion(); renderActive(); }};
        const label = document.createElement("div");
        label.className = "regionLabel";
        label.textContent = `${{region.id}} - ${{region.type_classification?.predicted_type || region.type}}`;
        box.appendChild(label);
        stage.appendChild(box);
      }});
      renderList();
      renderRegion();
    }}
    function renderList() {{
      const page = DATA[pageIndex];
      regionList.innerHTML = "";
      page.regions.forEach((region, index) => {{
        const item = document.createElement("button");
        item.className = "regionItem" + (index === regionIndex ? " active" : "");
        item.onclick = () => {{ regionIndex = index; renderRegion(); renderActive(); }};
        item.innerHTML = `<span class="regionId">${{html(region.id)}}</span><span class="regionType">${{html(region.semantic_structure?.kind || region.type)}}</span>`;
        regionList.appendChild(item);
      }});
    }}
    function renderActive() {{
      [...stage.querySelectorAll(".regionBox")].forEach((box, index) => box.classList.toggle("active", index === regionIndex));
      [...regionList.children].forEach((item, index) => item.classList.toggle("active", index === regionIndex));
    }}
    function renderRegion() {{
      const page = DATA[pageIndex];
      const region = page.regions[regionIndex];
      const predicted = region.type_classification?.predicted_type || "";
      regionTitle.textContent = `${{region.id}} - ${{predicted || region.type}}`;
      regionMeta.innerHTML = `
        <span>semantic</span><span>${{html(region.semantic_structure?.kind || "")}}</span>
        <span>predicted</span><span>${{html(predicted || "n/a")}}</span>
        <span>previous</span><span>${{html(region.type || "")}}</span>
        <span>tokens</span><span>${{region.token_count ?? 0}}</span>
      `;
      semanticText.textContent = JSON.stringify(region.semantic_structure || null, null, 2);
      regionText.textContent = region.markdown || "_No content extracted._";
      xText.textContent = JSON.stringify(region.x_coordinate_structure || null, null, 2);
    }}
    initTabs();
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
