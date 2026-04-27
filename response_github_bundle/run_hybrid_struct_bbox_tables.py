from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser
from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "outputs" / "manual_raw_parse" / "miraeasset_q3_q4" / "source"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hybrid_struct_bbox_tables"


@dataclass
class MatchedCell:
    text: str
    row_index: int
    cell_index: int
    bbox: tuple[float, float, float, float] | None
    matched_indices: list[int]


def resolve_pdf_path() -> Path:
    candidates = sorted(DEFAULT_SOURCE_DIR.glob("*3*.pdf"))
    if candidates:
        return candidates[0]
    candidates = sorted(DEFAULT_SOURCE_DIR.glob("*.pdf"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No sample PDF found in {DEFAULT_SOURCE_DIR}")


def normalize_token(text: str) -> str:
    cleaned = text.replace("‘", "'").replace("’", "'").replace("\ufeff", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_cell_tokens(text: str) -> list[str]:
    return [normalize_token(token) for token in text.split() if normalize_token(token)]


def union_bbox(boxes: list[list[float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return (x0, y0, x1, y1)


def match_cells_to_image_elements(
    rows: list[list[dict[str, Any]]],
    elements: list[Any],
) -> list[MatchedCell]:
    tokens = [normalize_token(element.text or "") for element in elements]
    matched_cells: list[MatchedCell] = []

    for row_index, row in enumerate(rows):
        for cell_index, cell in enumerate(row):
            cell_tokens = split_cell_tokens(cell["text"])
            if not cell_tokens:
                matched_cells.append(MatchedCell(cell["text"], row_index, cell_index, None, []))
                continue

            candidates: list[list[int]] = []
            for start in range(len(elements)):
                if tokens[start] != cell_tokens[0]:
                    continue

                matched = [start]
                start_box = elements[start].bbox or [0.0, 0.0, 0.0, 0.0]
                pointer = start + 1
                token_pointer = 1
                while pointer < len(elements) and token_pointer < len(cell_tokens):
                    prev_box = elements[matched[-1]].bbox or [0.0, 0.0, 0.0, 0.0]
                    curr_box = elements[pointer].bbox or [0.0, 0.0, 0.0, 0.0]
                    within_table_band = -2 <= curr_box[1] - start_box[1] <= 28
                    monotonic_x = curr_box[0] >= prev_box[0] - 12
                    wrapped_line = curr_box[1] > prev_box[1] and curr_box[0] <= prev_box[0] + 80

                    if tokens[pointer] == cell_tokens[token_pointer] and within_table_band and (monotonic_x or wrapped_line):
                        matched.append(pointer)
                        token_pointer += 1
                        pointer += 1
                        continue

                    if within_table_band:
                        pointer += 1
                        continue

                    break

                if token_pointer == len(cell_tokens):
                    candidates.append(matched)

            if candidates:
                found_indices = min(
                    candidates,
                    key=lambda candidate: (
                        (elements[candidate[-1]].bbox[1] - elements[candidate[0]].bbox[1]) if elements[candidate[-1]].bbox and elements[candidate[0]].bbox else 0.0,
                        len(candidate),
                        elements[candidate[0]].bbox[1] if elements[candidate[0]].bbox else 0.0,
                        elements[candidate[0]].bbox[0] if elements[candidate[0]].bbox else 0.0,
                    ),
                )
            else:
                found_indices = []

            bbox = union_bbox([elements[index].bbox for index in found_indices if elements[index].bbox])
            matched_cells.append(MatchedCell(cell["text"], row_index, cell_index, bbox, found_indices))

    return matched_cells


def cluster_columns(matched_cells: list[MatchedCell], tolerance: float = 18.0) -> list[float]:
    x_positions = sorted(cell.bbox[0] for cell in matched_cells if cell.bbox)
    if not x_positions:
        return []

    clusters: list[list[float]] = [[x_positions[0]]]
    for value in x_positions[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def assign_columns(
    rows: list[list[dict[str, Any]]],
    matched_cells: list[MatchedCell],
    columns: list[float],
) -> list[list[dict[str, Any]]]:
    row_maps: list[list[dict[str, Any]]] = [[] for _ in rows]
    for cell in matched_cells:
        if not cell.bbox or not columns:
            continue
        start_col = min(range(len(columns)), key=lambda idx: abs(columns[idx] - cell.bbox[0]))
        end_col = min(range(len(columns)), key=lambda idx: abs(columns[idx] - cell.bbox[2]))
        if end_col < start_col:
            end_col = start_col
        row_maps[cell.row_index].append(
            {
                "text": cell.text,
                "start_col": start_col,
                "end_col": end_col,
                "bbox": cell.bbox,
            }
        )
    for row in row_maps:
        row.sort(key=lambda item: item["start_col"])
    return row_maps


def render_regular_markdown(rows: list[list[dict[str, Any]]], extractor: PowerPointStructTreeExtractor) -> str:
    return extractor._table_rows_to_markdown(rows)


def render_irregular_hybrid_table(
    page: Any,
) -> str:
    region_elements = [
        element
        for element in page.elements
        if element.element_type == "image"
        and (element.text or "").strip()
        and element.bbox
        and element.bbox[0] >= 390
        and 340 <= element.bbox[1] <= 446
    ]
    region_elements.sort(key=lambda element: (element.bbox[1], element.bbox[0]))

    def join_text(items: list[Any]) -> str:
        raw = " ".join((item.text or "").strip() for item in items if (item.text or "").strip())
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = raw.replace(" + ", " + ")
        raw = raw.replace("( ", "(").replace(" )", ")")
        raw = raw.replace(" %", "%")
        return raw

    header_cells = [item for item in region_elements if item.bbox[1] < 360]
    stub_cells = [item for item in region_elements if item.bbox[0] < 445 and item.bbox[1] >= 360]

    body_bands = [
        (360.0, 381.6),
        (381.6, 413.1),
        (413.1, 446.5),
    ]
    col_bands = [
        (445.0, 523.0),
        (523.0, 587.0),
        (587.0, 645.0),
        (645.0, 740.0),
    ]

    stub_first = join_text([item for item in stub_cells if item.bbox[1] < 413.1])
    stub_last = join_text([item for item in stub_cells if item.bbox[1] >= 413.1])

    body_rows: list[list[str]] = []
    stub_values = [stub_first, stub_first, stub_last]
    for row_index, (y0, y1) in enumerate(body_bands):
        row = [stub_values[row_index]]
        row_items = [item for item in region_elements if y0 <= item.bbox[1] < y1]
        for x0, x1 in col_bands:
            cell_items = [item for item in row_items if x0 <= item.bbox[0] < x1]
            row.append(join_text(cell_items))
        body_rows.append(row)

    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th colspan=\"2\">목표</th>",
        "      <th colspan=\"2\">이행</th>",
        "      <th>이행률</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for row in body_rows:
        lines.append("    <tr>")
        for value in row:
            lines.append(f"      <td>{value}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def render_irregular_hybrid_table(
    page: Any,
) -> str:
    region_elements = [
        element
        for element in page.elements
        if element.element_type == "image"
        and (element.text or "").strip()
        and element.bbox
        and element.bbox[0] >= 390
        and 340 <= element.bbox[1] <= 446
    ]
    region_elements.sort(key=lambda element: (element.bbox[1], element.bbox[0]))

    def join_text(items: list[Any]) -> str:
        raw = " ".join((item.text or "").strip() for item in items if (item.text or "").strip())
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = raw.replace("( ", "(").replace(" )", ")")
        raw = raw.replace(" %", "%")
        return raw

    stub_cells = [item for item in region_elements if item.bbox[0] < 445 and item.bbox[1] >= 360]
    body_bands = [
        (360.0, 381.6),
        (381.6, 413.1),
        (413.1, 446.5),
    ]
    col_bands = [
        (445.0, 523.0),
        (523.0, 587.0),
        (587.0, 645.0),
        (645.0, 740.0),
    ]

    stub_first = join_text([item for item in stub_cells if item.bbox[1] < 413.1])
    stub_last = join_text([item for item in stub_cells if item.bbox[1] >= 413.1])

    body_rows: list[list[str]] = []
    stub_values = [stub_first, stub_first, stub_last]
    for row_index, (y0, y1) in enumerate(body_bands):
        row = [stub_values[row_index]]
        row_items = [item for item in region_elements if y0 <= item.bbox[1] < y1]
        for x0, x1 in col_bands:
            cell_items = [item for item in row_items if x0 <= item.bbox[0] < x1]
            row.append(join_text(cell_items))
        body_rows.append(row)

    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th colspan=\"2\">\ubaa9\ud45c</th>",
        "      <th colspan=\"2\">\uc774\ud589</th>",
        "      <th>\uc774\ud589\ub960</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for row in body_rows:
        lines.append("    <tr>")
        for value in row:
            lines.append(f"      <td>{value}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def main() -> None:
    pdf_path = resolve_pdf_path()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PowerPointStructTreeExtractor()
    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))

    with fitz.open(pdf_path) as document:
        catalog_xref = document.pdf_catalog()
        struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
        if struct_root[0] != "xref":
            raise RuntimeError("StructTreeRoot not found.")
        root_xref = int(struct_root[1].split()[0])
        page_map = {document.page_xref(i): i + 1 for i in range(document.page_count)}

        table_xrefs: dict[int, list[list[dict[str, Any]]]] = {}
        extractor._find_tables(document, root_xref, page_map, None, table_xrefs)

        table_page_map: dict[int, int] = {}
        extractor._map_table_pages(document, root_xref, page_map, None, table_page_map, set(table_xrefs.keys()))

    markdown_lines = [
        f"# {pdf_path.name} - Hybrid StructTree + BBox Tables",
        "",
        "Page numbers below are PDF page numbers.",
        "",
    ]
    summary: list[dict[str, Any]] = []

    ordered_tables = sorted(table_xrefs.items(), key=lambda item: (table_page_map.get(item[0], 9999), item[0]))
    for table_index, (table_xref, rows) in enumerate(ordered_tables, start=1):
        page_number = table_page_map.get(table_xref)
        page = parsed.pages[page_number - 1]
        image_elements = [element for element in page.elements if element.element_type == "image" and (element.text or "").strip()]
        image_elements.sort(key=lambda element: (round((element.bbox or [0.0, 0.0, 0.0, 0.0])[1], 1), (element.bbox or [0.0])[0]))

        matched_cells = match_cells_to_image_elements(rows, image_elements)
        row_lengths = [len(row) for row in rows]

        markdown_lines.append(f"## Table {table_index} - Page {page_number}")
        markdown_lines.append("")

        if row_lengths == [3, 5, 4, 5]:
            rendered = render_irregular_hybrid_table(page)
            rendering = "hybrid-html"
        else:
            rendered = render_regular_markdown(rows, extractor)
            rendering = "structtree-markdown"

        markdown_lines.append(rendered)
        markdown_lines.append("")

        summary.append(
            {
                "table_index": table_index,
                "page_number": page_number,
                "xref": table_xref,
                "row_lengths": row_lengths,
                "rendering": rendering,
                "matched_cells": sum(1 for cell in matched_cells if cell.bbox),
                "total_cells": len(matched_cells),
            }
        )

    markdown_path = output_dir / "miraeasset_q3_hybrid_tables.md"
    summary_path = output_dir / "summary.json"
    markdown_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {"markdown_path": markdown_path.as_posix(), "summary_path": summary_path.as_posix()},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
