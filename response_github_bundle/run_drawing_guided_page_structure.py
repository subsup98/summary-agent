from __future__ import annotations

import importlib.util
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

import run_generalized_infographic_probe as layout_probe
from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_HELPER_PATH = PROJECT_ROOT / "test" / "test_layout_region_reconstructor.py"
PDF_NAME = "\ubbf8\ub798\uc5d0\uc14b\uc99d\uad8c 3\ubd84\uae30 \uc2e4\uc801\ubcf4\uace0\uc11c.pdf"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drawing_guided_pages_04_16"


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def to_list(self) -> list[float]:
        return [round(self.x0, 3), round(self.y0, 3), round(self.x1, 3), round(self.y1, 3)]

    def contains_token_center(self, token: layout_probe.Token, *, pad: float = 0.0) -> bool:
        return (
            self.x0 - pad <= token.box.cx <= self.x1 + pad
            and self.y0 - pad <= token.box.cy <= self.y1 + pad
        )

    def intersects(self, other: "Rect") -> bool:
        return self.x0 <= other.x1 and self.x1 >= other.x0 and self.y0 <= other.y1 and self.y1 >= other.y0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build drawing-guided structures for Mirae Asset Q3 pages 4 and 16.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for markdown/json/png outputs.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    helper = load_test_helper()
    pdf_path = resolve_pdf_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))
    manifest: dict[str, Any] = {
        "source_pdf": pdf_path.as_posix(),
        "parser_name": parsed.parser_name,
        "markdown_source": parsed.metadata.get("markdown_source"),
        "markdown_strategy": parsed.metadata.get("markdown_strategy"),
        "method": "existing PdfParser markdown + MCID tokens, segmented with drawing lines and fill boxes",
        "pages": [],
    }

    combined_lines: list[str] = [
        "# Drawing-Guided Layout Structure",
        "",
        f"> Source PDF: {pdf_path.name}",
        f"> Parser: {parsed.parser_name}",
        "> Text parsing: existing PdfParser / StructTree flow.",
        "> Layout hints: PDF drawing lines and filled rectangles.",
        "",
    ]

    with fitz.open(pdf_path) as document:
        for page_number in (4, 16):
            page = document[page_number - 1]
            page_markdown = extract_page_markdown(parsed.markdown, page_number)
            tokens = extract_tokens(document, page, page_number)
            drawings = collect_drawings(page)

            if page_number == 4:
                page_payload = build_page4_structure(page, tokens, drawings, page_markdown)
            else:
                page_payload = build_page16_structure(helper, page, tokens, drawings, page_markdown)

            coordinate_outputs = export_drawing_line_fill_coordinates(output_dir, page_number, drawings)
            page_text = render_page_markdown(page_payload)
            page_path = output_dir / f"page_{page_number:02d}_drawing_guided_structure.md"
            page_json_path = output_dir / f"page_{page_number:02d}_drawing_guided_structure.json"
            overlay_path = output_dir / f"page_{page_number:02d}_drawing_guided_overlay.png"
            page_payload["outputs"] = {
                "markdown": page_path.as_posix(),
                "json": page_json_path.as_posix(),
                "overlay_png": overlay_path.as_posix(),
                "coordinate_exports": coordinate_outputs,
            }
            page_path.write_text(page_text, encoding="utf-8")
            page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            render_overlay(document, page_number, page_payload["regions"], overlay_path)

            manifest["pages"].append(strip_region_content(page_payload))
            combined_lines.append(page_text.rstrip())
            combined_lines.append("")

            print(f"page {page_number}: {len(page_payload['regions'])} regions -> {page_path}")

    combined_path = output_dir / "miraeasset_q3_pages_04_16_drawing_guided_structure.md"
    manifest_path = output_dir / "miraeasset_q3_pages_04_16_drawing_guided_manifest.json"
    combined_path.write_text("\n".join(combined_lines).rstrip() + "\n", encoding="utf-8")
    manifest["outputs"] = {
        "combined_markdown": combined_path.as_posix(),
        "manifest_json": manifest_path.as_posix(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))


def load_test_helper() -> Any:
    spec = importlib.util.spec_from_file_location("layout_region_reconstructor_test", TEST_HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {TEST_HELPER_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_pdf_path() -> Path:
    exact = Path.home() / "Desktop" / "all_docs" / PDF_NAME
    if exact.exists():
        return exact
    candidates = sorted(
        [
            path
            for path in (Path.home() / "Desktop" / "all_docs").glob("*.pdf")
            if "3" in path.name and path.stat().st_size > 2_000_000
        ],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not locate Mirae Asset Q3 PDF")
    return candidates[0]


def extract_page_markdown(markdown: str, page_number: int) -> str:
    pattern = re.compile(rf"(?ms)^# Page\s+{page_number}\s*$.*?(?=^# Page\s+{page_number + 1}\s*$|\Z)")
    match = pattern.search(str(markdown or ""))
    return match.group(0).strip() if match else ""


def extract_tokens(document: fitz.Document, page: fitz.Page, page_number: int) -> list[layout_probe.Token]:
    original_limit = layout_probe.TOP_LIMIT
    layout_probe.TOP_LIMIT = float(page.rect.height)
    try:
        return layout_probe.extract_tokens(document, page, page_number)
    finally:
        layout_probe.TOP_LIMIT = original_limit


def collect_drawings(page: fitz.Page) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, drawing in enumerate(page.get_drawings()):
        rect = drawing.get("rect")
        if not rect:
            continue
        item = {
            "index": index,
            "bbox": Rect(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
            "type": drawing.get("type"),
            "color": drawing.get("color"),
            "fill": drawing.get("fill"),
            "line_width": drawing.get("width"),
        }
        output.append(item)
    return output


def export_drawing_line_fill_coordinates(
    output_dir: Path,
    page_number: int,
    drawings: list[dict[str, Any]],
) -> dict[str, str]:
    coordinate_dir = output_dir / "drawing_lines_fill_coordinates"
    coordinate_dir.mkdir(parents=True, exist_ok=True)

    line_items = [
        drawing_coordinate_record(drawing)
        for drawing in drawings
        if is_horizontal_line(drawing, min_width=80)
    ]
    fill_items = [
        drawing_coordinate_record(drawing)
        for drawing in drawings
        if drawing.get("fill") is not None and drawing["bbox"].width > 1 and drawing["bbox"].height > 1
    ]
    payload = {
        "page_number": page_number,
        "drawing_lines": line_items,
        "fill_coordinates": fill_items,
    }

    json_path = coordinate_dir / f"page_{page_number:02d}_drawing_lines_fill_coordinates.json"
    md_path = coordinate_dir / f"page_{page_number:02d}_drawing_lines_fill_coordinates.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_coordinate_markdown(payload), encoding="utf-8")
    return {"json": json_path.as_posix(), "markdown": md_path.as_posix()}


def drawing_coordinate_record(drawing: dict[str, Any]) -> dict[str, Any]:
    rect = drawing["bbox"]
    return {
        "index": drawing["index"],
        "bbox": rect.to_list(),
        "x0": round(rect.x0, 3),
        "y0": round(rect.y0, 3),
        "x1": round(rect.x1, 3),
        "y1": round(rect.y1, 3),
        "width": round(rect.width, 3),
        "height": round(rect.height, 3),
        "type": drawing.get("type"),
        "color": round_color(drawing.get("color")),
        "fill": round_color(drawing.get("fill")),
        "line_width": drawing.get("line_width"),
    }


def render_coordinate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Page {payload['page_number']} Drawing Lines and Fill Coordinates",
        "",
        "## Drawing Lines",
        "",
        "| index | x0 | y0 | x1 | y1 | width | height | color | line_width |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in payload["drawing_lines"]:
        lines.append(
            "| {index} | {x0} | {y0} | {x1} | {y1} | {width} | {height} | {color} | {line_width} |".format(
                **{**item, "color": json.dumps(item.get("color"), ensure_ascii=False)}
            )
        )
    lines.extend(
        [
            "",
            "## Fill Coordinates",
            "",
            "| index | x0 | y0 | x1 | y1 | width | height | fill |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in payload["fill_coordinates"]:
        lines.append(
            "| {index} | {x0} | {y0} | {x1} | {y1} | {width} | {height} | {fill} |".format(
                **{**item, "fill": json.dumps(item.get("fill"), ensure_ascii=False)}
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def build_page3_structure(
    page: fitz.Page,
    tokens: list[layout_probe.Token],
    drawings: list[dict[str, Any]],
    page_markdown: str,
) -> dict[str, Any]:
    panel = largest_nonwhite_fill(drawings, page)
    top_line = first_or_none(
        drawing
        for drawing in drawings
        if is_horizontal_line(drawing, min_width=500) and is_orange(drawing.get("color"))
    )
    icons = sorted(
        [
            drawing
            for drawing in drawings
            if is_orange(drawing.get("fill"))
            and 30 <= drawing["bbox"].width <= 70
            and 30 <= drawing["bbox"].height <= 70
        ],
        key=lambda item: item["bbox"].cy,
    )

    regions: list[dict[str, Any]] = []
    icon_centers = [icon["bbox"].cy for icon in icons]
    for index, icon in enumerate(icons):
        if index == 0:
            y0 = max(panel.y0 if panel else 0.0, icon["bbox"].y0 - 28.0)
        else:
            y0 = (icon_centers[index - 1] + icon_centers[index]) / 2.0
        if index == len(icons) - 1:
            y1 = min(panel.y1 if panel else float(page.rect.height), icon["bbox"].y1 + 42.0)
        else:
            y1 = (icon_centers[index] + icon_centers[index + 1]) / 2.0

        bbox = Rect(
            panel.x0 if panel else 58.0,
            y0,
            panel.x1 if panel else float(page.rect.width) - 58.0,
            y1,
        )
        region_tokens = tokens_in_rect(tokens, Rect(icon["bbox"].x1 + 20.0, y0, bbox.x1, y1))
        regions.append(
            {
                "id": f"P3-Priority-{index + 1}",
                "type": "priority_item",
                "bbox": bbox.to_list(),
                "source": "orange_icon_fill_y_band",
                "evidence": {
                    "icon_fill": drawing_summary(icon),
                    "band_y": [round(y0, 3), round(y1, 3)],
                },
                "token_count": len(region_tokens),
                "markdown": render_token_region(region_tokens),
            }
        )

    evidence = {
        "panel_fill": rect_summary(panel) if panel else None,
        "top_orange_line": drawing_summary(top_line) if top_line else None,
        "icon_fills": [drawing_summary(icon) for icon in icons],
    }
    return {
        "page_number": 3,
        "page_size": [round(float(page.rect.width), 3), round(float(page.rect.height), 3)],
        "strategy": "large panel fill + four orange icon fills as row anchors",
        "original_parser_markdown": page_markdown,
        "drawing_evidence": evidence,
        "regions": regions,
    }


def build_page4_structure(
    page: fitz.Page,
    tokens: list[layout_probe.Token],
    drawings: list[dict[str, Any]],
    page_markdown: str,
) -> dict[str, Any]:
    top_panel_fills = sorted(
        [
            drawing
            for drawing in drawings
            if is_key_gray(drawing.get("fill"))
            and drawing["bbox"].width >= 200
            and drawing["bbox"].height >= 60
            and drawing["bbox"].y0 < 290
        ],
        key=lambda item: (item["bbox"].y0, item["bbox"].x0),
    )
    highlight_cards = sorted(
        [
            drawing
            for drawing in drawings
            if drawing.get("type") == "fs"
            and drawing["bbox"].width >= 300
            and drawing["bbox"].height >= 80
            and drawing["bbox"].y0 >= 300
        ],
        key=lambda item: (item["bbox"].y0, item["bbox"].x0),
    )
    orange_lines = sorted(
        [
            drawing
            for drawing in drawings
            if is_horizontal_line(drawing, min_width=200) and is_orange(drawing.get("color"))
        ],
        key=lambda item: (item["bbox"].y0, item["bbox"].x0),
    )

    # The chart panels expose their own gray fills. Extend each region upward to
    # include the orange title line and title text immediately above the fill.
    named_top_panels = [
        ("P4-R1", "connected_pretax_net_income", Rect(32.0, 92.0, 263.5, 183.5)),
        ("P4-R2", "standalone_net_operating_revenue", Rect(273.0, 92.0, 503.5, 276.5)),
        ("P4-R3", "overseas_subsidiaries", Rect(514.0, 92.0, 744.5, 183.5)),
        ("P4-R4", "connected_roe", Rect(33.5, 185.0, 263.8, 276.5)),
        ("P4-R5", "standalone_client_assets", Rect(514.0, 185.0, 744.5, 276.5)),
    ]
    named_highlight_cards = [
        ("P4-R6", "connected_roe_highlight", Rect(34.0, 319.0, 381.5, 416.5)),
        ("P4-R7", "brokerage_wm_highlight", Rect(394.3, 319.0, 742.0, 416.5)),
        ("P4-R8", "overseas_subsidiaries_highlight", Rect(34.0, 426.0, 381.5, 523.2)),
        ("P4-R9", "investment_asset_valuation_highlight", Rect(394.3, 426.0, 742.0, 523.2)),
    ]

    regions: list[dict[str, Any]] = []
    for region_id, region_type, rect in [*named_top_panels, *named_highlight_cards]:
        region_tokens = tokens_in_rect(tokens, rect)
        regions.append(
            {
                "id": region_id,
                "type": region_type,
                "bbox": rect.to_list(),
                "source": "gray_panel_fill_or_highlight_card_rectangles",
                "evidence": {
                    "matching_fills": [
                        drawing_summary(drawing)
                        for drawing in drawings_intersecting_rect(drawings, rect)
                        if drawing.get("fill") is not None
                    ],
                    "matching_orange_lines": [
                        drawing_summary(line)
                        for line in orange_lines
                        if rect.x0 - 3 <= line["bbox"].x0 <= rect.x1 and rect.y0 - 3 <= line["bbox"].y0 <= rect.y1 + 3
                    ],
                },
                "token_count": len(region_tokens),
                "markdown": render_token_region(region_tokens),
            }
        )

    evidence = {
        "top_gray_panel_fills": [drawing_summary(item) for item in top_panel_fills],
        "highlight_card_fills": [drawing_summary(item) for item in highlight_cards],
        "orange_title_lines": [drawing_summary(item) for item in orange_lines],
    }
    return {
        "page_number": 4,
        "page_size": [round(float(page.rect.width), 3), round(float(page.rect.height), 3)],
        "strategy": "top gray chart panel fills + bottom highlight card rectangles",
        "original_parser_markdown": page_markdown,
        "drawing_evidence": evidence,
        "regions": regions,
    }


def build_page16_structure(
    helper: Any,
    page: fitz.Page,
    tokens: list[layout_probe.Token],
    drawings: list[dict[str, Any]],
    page_markdown: str,
) -> dict[str, Any]:
    table_blocks = helper.extract_table_blocks(page_markdown)
    markdown_tables = [block for block in table_blocks if block["format"] == "markdown"]
    html_tables = [block for block in table_blocks if block["format"] == "html"]
    reference_table = helper.parse_markdown_table(markdown_tables[0]["markdown"]) if markdown_tables else []

    container = largest_nonwhite_fill(drawings, page)
    orange_lines = sorted(
        [
            drawing
            for drawing in drawings
            if is_horizontal_line(drawing, min_width=250) and is_orange(drawing.get("color"))
        ],
        key=lambda item: (item["bbox"].x0, item["bbox"].y0),
    )
    right_lines = [line for line in orange_lines if line["bbox"].x0 > 350]
    left_lines = [line for line in orange_lines if line["bbox"].x0 < 100]
    right_x0 = min((line["bbox"].x0 for line in right_lines), default=394.0)
    right_x1 = max((line["bbox"].x1 for line in right_lines), default=734.5)
    left_x0 = min((line["bbox"].x0 for line in left_lines), default=42.0)
    left_x1 = max((line["bbox"].x1 for line in left_lines), default=382.0)
    right_top_y = min((line["bbox"].y0 for line in right_lines), default=160.0)
    right_bottom_y = max((line["bbox"].y0 for line in right_lines), default=326.7)
    left_table_line_y = min((line["bbox"].y0 for line in left_lines), default=328.6)

    chart_fills = [
        drawing
        for drawing in drawings
        if drawing.get("fill") is not None
        and drawing["bbox"].x0 < right_x0
        and 200 <= drawing["bbox"].y0 <= 285
        and 40 <= drawing["bbox"].width <= 75
        and drawing["bbox"].height >= 8
    ]
    left_table_fills = [
        drawing
        for drawing in drawings
        if drawing.get("fill") is not None
        and left_x0 - 1 <= drawing["bbox"].x0 <= left_x1
        and 300 <= drawing["bbox"].y0 <= 450
    ]
    right_table_fills = [
        drawing
        for drawing in drawings
        if drawing.get("fill") is not None
        and right_x0 - 1 <= drawing["bbox"].x0 <= right_x1
        and 330 <= drawing["bbox"].y0 <= 450
    ]

    left_table_y0 = min([left_table_line_y, *[fill["bbox"].y0 for fill in left_table_fills]], default=left_table_line_y)
    left_table_y1 = max([fill["bbox"].y1 for fill in left_table_fills], default=447.0)
    right_table_y0 = min([right_bottom_y, *[fill["bbox"].y0 for fill in right_table_fills]], default=right_bottom_y)
    right_table_y1 = max([fill["bbox"].y1 for fill in right_table_fills], default=448.0)

    chart_rect = Rect(left_x0, 136.0, left_x1, min(left_table_y0 - 8.0, 300.0))
    kpi_rect = Rect(right_x0, max(120.0, right_top_y - 24.0), right_x1, right_bottom_y - 20.0)
    left_table_rect = Rect(left_x0, left_table_y0, left_x1, left_table_y1)
    right_table_rect = Rect(right_x0, right_table_y0 - 24.0, right_x1, right_table_y1)
    notes_rect = bbox_from_tokens([token for token in tokens if token.box.x0 < 240 and token.box.y0 >= left_table_y1 + 3.0])

    chart_tokens = tokens_in_rect(tokens, chart_rect)
    kpi_tokens = tokens_in_rect(tokens, kpi_rect)
    left_table_tokens = tokens_in_rect(tokens, left_table_rect)
    right_table_tokens = tokens_in_rect(tokens, right_table_rect)
    notes_tokens = tokens_in_rect(tokens, notes_rect) if notes_rect else []

    regions = [
        {
            "id": "P16-R1",
            "type": "chart",
            "bbox": chart_rect.to_list(),
            "source": "left_chart_bar_fills_and_baseline",
            "evidence": {"bar_fills": [drawing_summary(item) for item in chart_fills]},
            "token_count": len(chart_tokens),
            "markdown": helper.render_chart_candidate_region(chart_tokens, reference_table),
        },
        {
            "id": "P16-R2",
            "type": "kpi_panel",
            "bbox": kpi_rect.to_list(),
            "source": "right_column_orange_separator_lines",
            "evidence": {"top_line": drawing_summary(first_or_none(right_lines)), "bottom_line": drawing_summary(last_or_none(right_lines))},
            "token_count": len(kpi_tokens),
            "markdown": render_token_region(kpi_tokens),
        },
        {
            "id": "P16-R3",
            "type": "left_table",
            "bbox": left_table_rect.to_list(),
            "source": "left_table_orange_header_line_and_cell_fills",
            "evidence": {
                "top_line": drawing_summary(first_or_none(left_lines)),
                "cell_fill_count": len(left_table_fills),
            },
            "token_count": len(left_table_tokens),
            "markdown": markdown_tables[0]["markdown"] if markdown_tables else render_token_region(left_table_tokens),
        },
        {
            "id": "P16-R4",
            "type": "right_table",
            "bbox": right_table_rect.to_list(),
            "source": "right_bottom_orange_separator_line_and_cell_fills",
            "evidence": {
                "separator_line": drawing_summary(last_or_none(right_lines)),
                "cell_fill_count": len(right_table_fills),
            },
            "token_count": len(right_table_tokens),
            "markdown": html_tables[0]["markdown"] if html_tables else render_token_region(right_table_tokens),
        },
    ]
    if notes_rect is not None:
        regions.append(
            {
                "id": "P16-R5",
                "type": "notes",
                "bbox": notes_rect.to_list(),
                "source": "tokens_below_left_table",
                "evidence": {},
                "token_count": len(notes_tokens),
                "markdown": helper.extract_notes_block(page_markdown) or render_token_region(notes_tokens),
            }
        )

    evidence = {
        "container_fill": rect_summary(container) if container else None,
        "orange_lines": [drawing_summary(line) for line in orange_lines],
        "column_split": {
            "left_x1": round(left_x1, 3),
            "right_x0": round(right_x0, 3),
            "gap_width": round(right_x0 - left_x1, 3),
        },
        "left_chart_bar_fill_count": len(chart_fills),
        "left_table_fill_count": len(left_table_fills),
        "right_table_fill_count": len(right_table_fills),
    }
    return {
        "page_number": 16,
        "page_size": [round(float(page.rect.width), 3), round(float(page.rect.height), 3)],
        "strategy": "orange separator lines for columns/rows + chart/table fill boxes",
        "original_parser_markdown": page_markdown,
        "drawing_evidence": evidence,
        "regions": regions,
    }


def largest_nonwhite_fill(drawings: list[dict[str, Any]], page: fitz.Page) -> Rect | None:
    fills = []
    for drawing in drawings:
        fill = drawing.get("fill")
        rect = drawing["bbox"]
        if fill is None or is_white(fill):
            continue
        if rect.width > float(page.rect.width) * 0.95 and rect.height > float(page.rect.height) * 0.95:
            continue
        fills.append(rect)
    if not fills:
        return None
    return max(fills, key=lambda rect: rect.width * rect.height)


def tokens_in_rect(tokens: list[layout_probe.Token], rect: Rect) -> list[layout_probe.Token]:
    return sorted([token for token in tokens if rect.contains_token_center(token)], key=lambda token: (token.box.y0, token.box.x0))


def bbox_from_tokens(tokens: list[layout_probe.Token]) -> Rect | None:
    if not tokens:
        return None
    box = layout_probe.union_token_box(tokens)
    return Rect(box.x0, box.y0, box.x1, box.y1)


def render_token_region(tokens: list[layout_probe.Token]) -> str:
    return "\n".join(row for row in layout_probe.render_rows(tokens) if row.strip()).strip()


def render_page_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Page {payload['page_number']} Drawing-Guided Structure",
        "",
        f"> Strategy: {payload['strategy']}",
        "",
        "## Drawing Evidence",
        "",
        "```json",
        json.dumps(payload["drawing_evidence"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Existing Parser Markdown",
        "",
        payload.get("original_parser_markdown") or "_No parser markdown extracted._",
        "",
        "## Structured Regions",
        "",
    ]
    for region in payload["regions"]:
        lines.extend(
            [
                f"<region id=\"{region['id']}\" type=\"{region['type']}\" bbox='{json.dumps(region['bbox'], ensure_ascii=False)}' source=\"{region['source']}\">",
                "",
                f"### {region['id']} {region['type']}",
                "",
                "**Evidence**",
                "",
                "```json",
                json.dumps(region.get("evidence") or {}, ensure_ascii=False, indent=2),
                "```",
                "",
                "**Content**",
                "",
                region.get("markdown") or "_No content extracted._",
                "",
                "</region>",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_overlay(document: fitz.Document, page_number: int, regions: list[dict[str, Any]], output_path: Path) -> None:
    scratch = fitz.open()
    scratch.insert_pdf(document, from_page=page_number - 1, to_page=page_number - 1)
    page = scratch[0]
    colors = [
        (0.95, 0.25, 0.15),
        (0.1, 0.45, 0.95),
        (0.0, 0.55, 0.25),
        (0.55, 0.25, 0.8),
        (0.1, 0.1, 0.1),
    ]
    for index, region in enumerate(regions):
        bbox = region["bbox"]
        rect = fitz.Rect(*bbox)
        color = colors[index % len(colors)]
        page.draw_rect(rect, color=color, width=1.5)
        page.insert_text((rect.x0 + 3, max(12, rect.y0 - 3)), region["id"], fontsize=8, color=color)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(output_path)
    scratch.close()


def strip_region_content(page_payload: dict[str, Any]) -> dict[str, Any]:
    clone = {key: value for key, value in page_payload.items() if key != "regions"}
    clone["regions"] = [
        {key: value for key, value in region.items() if key != "markdown"}
        for region in page_payload.get("regions", [])
    ]
    return clone


def is_horizontal_line(drawing: dict[str, Any], *, min_width: float) -> bool:
    rect = drawing["bbox"]
    return rect.width >= min_width and rect.height <= 0.01 and drawing.get("color") is not None


def is_orange(color: Any) -> bool:
    if not color or len(color) < 3:
        return False
    r, g, b = color[:3]
    return r > 0.85 and 0.35 <= g <= 0.65 and b < 0.25


def is_white(color: Any) -> bool:
    if not color or len(color) < 3:
        return False
    return all(float(component) > 0.96 for component in color[:3])


def is_key_gray(color: Any) -> bool:
    if not color or len(color) < 3:
        return False
    r, g, b = [float(component) for component in color[:3]]
    return abs(r - g) < 0.03 and abs(g - b) < 0.03 and 0.70 <= r <= 0.97


def drawings_intersecting_rect(drawings: list[dict[str, Any]], rect: Rect) -> list[dict[str, Any]]:
    return [drawing for drawing in drawings if drawing["bbox"].intersects(rect)]


def rect_summary(rect: Rect | None) -> dict[str, Any] | None:
    if rect is None:
        return None
    return {"bbox": rect.to_list(), "width": round(rect.width, 3), "height": round(rect.height, 3)}


def drawing_summary(drawing: dict[str, Any] | None) -> dict[str, Any] | None:
    if drawing is None:
        return None
    rect = drawing["bbox"]
    return {
        "index": drawing["index"],
        "bbox": rect.to_list(),
        "width": round(rect.width, 3),
        "height": round(rect.height, 3),
        "type": drawing.get("type"),
        "color": round_color(drawing.get("color")),
        "fill": round_color(drawing.get("fill")),
        "line_width": drawing.get("line_width"),
    }


def round_color(color: Any) -> list[float] | None:
    if not color:
        return None
    return [round(float(component), 4) for component in color[:3]]


def first_or_none(items: Any) -> Any | None:
    for item in items:
        return item
    return None


def last_or_none(items: list[Any]) -> Any | None:
    return items[-1] if items else None


if __name__ == "__main__":
    main()
