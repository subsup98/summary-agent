from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_generalized_infographic_probe as layout_probe
from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser


PAGE_NUMBER = 16
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "miraeasset_page16_layout_auto"


def resolve_pdf_path() -> Path:
    docs = Path.home() / "Desktop" / "all_docs"
    exact = docs / "미래에셋증권 3분기 실적보고서.pdf"
    if exact.exists():
        return exact
    candidates = sorted(
        [path for path in docs.glob("*.pdf") if "3" in path.name and path.stat().st_size > 2_000_000],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Q3 PDF candidate under {docs}")
    return candidates[0]


def main() -> None:
    pdf_path = resolve_pdf_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))
    page_markdown = extract_page_markdown(parsed.markdown, PAGE_NUMBER)
    table_blocks = extract_table_blocks(page_markdown)

    with fitz.open(pdf_path) as document:
        page = document[PAGE_NUMBER - 1]
        original_limit = layout_probe.TOP_LIMIT
        layout_probe.TOP_LIMIT = float(page.rect.height)
        try:
            tokens = layout_probe.extract_tokens(document, page, PAGE_NUMBER)
        finally:
            layout_probe.TOP_LIMIT = original_limit

        geometry = infer_geometry(tokens, float(page.rect.width), float(page.rect.height))
        notes_markdown = extract_notes_block(page_markdown)
        regions = build_regions(tokens, geometry, table_blocks, notes_markdown)
        markdown = render_layout_markdown(pdf_path, parsed, regions, geometry)

    md_path = OUTPUT_DIR / "miraeasset_q3_page16_auto_layout.md"
    json_path = OUTPUT_DIR / "miraeasset_q3_page16_auto_layout_manifest.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "source_pdf": pdf_path.as_posix(),
                "page_number": PAGE_NUMBER,
                "parser_name": parsed.parser_name,
                "markdown_source": parsed.metadata.get("markdown_source"),
                "markdown_strategy": parsed.metadata.get("markdown_strategy"),
                "geometry": geometry,
                "region_count": len(regions),
                "regions": [
                    {
                        "id": region["id"],
                        "type": region["type"],
                        "bbox": region["bbox"],
                        "source": region["source"],
                        "token_count": len(region.get("tokens") or []),
                    }
                    for region in regions
                ],
                "markdown_path": md_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"markdown_path": md_path.as_posix(), "manifest_path": json_path.as_posix()}, ensure_ascii=False))


def extract_page_markdown(markdown: str, page_number: int) -> str:
    pattern = re.compile(rf"(?ms)^# Page\s+{page_number}\s*$.*?(?=^# Page\s+{page_number + 1}\s*$|\Z)")
    match = pattern.search(str(markdown or ""))
    return match.group(0).strip() if match else ""


def extract_table_blocks(page_markdown: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for match in re.finditer(r"(?is)<table\b.*?</table>", page_markdown):
        raw = match.group(0).strip()
        blocks.append({"format": "html", "raw": raw, "markdown": html_table_to_markdown(raw)})

    lines = page_markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("|"):
            start = index
            while index < len(lines) and lines[index].strip().startswith("|"):
                index += 1
            raw = "\n".join(lines[start:index]).strip()
            if raw and re.search(r"\|\s*---", raw):
                blocks.append({"format": "markdown", "raw": raw, "markdown": raw})
            continue
        index += 1
    return blocks


def extract_notes_block(page_markdown: str) -> str:
    notes: list[str] = []
    for raw_line in page_markdown.splitlines():
        line = raw_line.strip()
        if re.match(r"^-\s*\d+\)", line):
            notes.append(line.lstrip("- ").strip())
    if not notes:
        return ""
    normalized: list[str] = []
    for index, note in enumerate(notes):
        text = re.sub(r"^\d+\)\s*", "", note)
        normalized.append(f"{index + 1}. {text}")
    return "\n".join(normalized)


def html_table_to_markdown(html: str) -> str:
    rows: list[list[str]] = []
    for row_match in re.finditer(r"(?is)<tr\b.*?</tr>", html):
        row_html = row_match.group(0)
        cells: list[str] = []
        for cell_match in re.finditer(r"(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>", row_html):
            text = re.sub(r"(?is)<[^>]+>", " ", cell_match.group(1))
            text = re.sub(r"\s+", " ", text).strip()
            cells.append(text)
        if cells:
            rows.append(cells)
    if not rows:
        return html
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    if width == 5 and len(rows[0]) == 3 and {"목표", "이행", "이행률"}.issubset(set(rows[0])):
        header = ["목표 기간", "대상", "목표", "이행", "이행률"]
    output = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in normalized[1:]:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def parse_markdown_table(markdown: str) -> list[dict[str, str]]:
    rows: list[list[str]] = []
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return []
    header = rows[0]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        normalized = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, normalized)))
    return records


def infer_geometry(tokens: list[layout_probe.Token], page_width: float, page_height: float) -> dict[str, Any]:
    split_x = page_width / 2.0
    rows = cluster_token_rows(tokens)
    table_start = next(
        (
            row["cy"]
            for row in rows
            if row["cy"] >= 300 and len(row["tokens"]) >= 6 and min(token.box.x0 for token in row["tokens"]) < split_x
        ),
        310.0,
    )
    notes_start = next(
        (
            row["cy"]
            for row in rows
            if row["cy"] >= 440 and len(row["tokens"]) >= 5 and max(token.box.x1 for token in row["tokens"]) < split_x
        ),
        page_height * 0.83,
    )
    return {
        "page_width": round(page_width, 3),
        "page_height": round(page_height, 3),
        "split_x": round(split_x, 3),
        "content_top": 120.0,
        "table_start_y": round(float(table_start), 3),
        "notes_start_y": round(float(notes_start), 3),
    }


def cluster_token_rows(tokens: list[layout_probe.Token]) -> list[dict[str, Any]]:
    rows: list[list[layout_probe.Token]] = []
    for token in sorted(tokens, key=lambda item: (item.box.cy, item.box.x0)):
        if not rows:
            rows.append([token])
            continue
        row = rows[-1]
        row_center = sum(item.box.cy for item in row) / len(row)
        if abs(token.box.cy - row_center) <= 5.5:
            row.append(token)
        else:
            rows.append([token])
    return [{"cy": sum(token.box.cy for token in row) / len(row), "tokens": row} for row in rows]


def build_regions(
    tokens: list[layout_probe.Token],
    geometry: dict[str, Any],
    table_blocks: list[dict[str, str]],
    notes_markdown: str = "",
) -> list[dict[str, Any]]:
    split_x = float(geometry["split_x"])
    top_y = float(geometry["content_top"])
    table_start = float(geometry["table_start_y"])
    notes_start = float(geometry["notes_start_y"])

    top_region_end = table_start - 8.0
    chart_tokens = filter_tokens(tokens, x_max=split_x, y_min=top_y, y_max=top_region_end)
    kpi_tokens = filter_tokens(tokens, x_min=split_x, y_min=top_y, y_max=top_region_end)
    notes_tokens = filter_tokens(tokens, x_max=split_x, y_min=notes_start - 6.0)

    html_tables = [block for block in table_blocks if block["format"] == "html"]
    markdown_tables = [block for block in table_blocks if block["format"] == "markdown"]
    left_table_records = parse_markdown_table(markdown_tables[0]["markdown"]) if markdown_tables else []

    regions = [
        {
            "id": "R1",
            "type": "chart_candidate",
            "bbox": token_bbox(chart_tokens),
            "source": "mcid_xref_coordinates",
            "tokens": chart_tokens,
            "markdown": render_chart_candidate_region(chart_tokens, left_table_records),
        },
        {
            "id": "R2",
            "type": "kpi_panel_candidate",
            "bbox": token_bbox(kpi_tokens),
            "source": "mcid_xref_coordinates",
            "tokens": kpi_tokens,
            "markdown": render_token_region(kpi_tokens),
        },
        {
            "id": "R3",
            "type": "table",
            "bbox": [0, table_start, split_x, notes_start],
            "source": "original_pipeline_markdown_table",
            "tokens": filter_tokens(tokens, x_max=split_x, y_min=table_start, y_max=notes_start),
            "markdown": markdown_tables[0]["markdown"] if markdown_tables else render_token_region(filter_tokens(tokens, x_max=split_x, y_min=table_start, y_max=notes_start)),
        },
        {
            "id": "R4",
            "type": "table",
            "bbox": [split_x, table_start, geometry["page_width"], notes_start],
            "source": "original_pipeline_html_table",
            "tokens": filter_tokens(tokens, x_min=split_x, y_min=table_start, y_max=notes_start),
            "markdown": html_tables[0]["markdown"] if html_tables else render_token_region(filter_tokens(tokens, x_min=split_x, y_min=table_start, y_max=notes_start)),
        },
        {
            "id": "R5",
            "type": "notes",
            "bbox": token_bbox(notes_tokens),
            "source": "mcid_xref_coordinates",
            "tokens": notes_tokens,
            "markdown": notes_markdown or render_token_region(notes_tokens),
        },
    ]
    return regions


def filter_tokens(
    tokens: list[layout_probe.Token],
    *,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> list[layout_probe.Token]:
    result: list[layout_probe.Token] = []
    for token in tokens:
        if x_min is not None and token.box.cx < x_min:
            continue
        if x_max is not None and token.box.cx >= x_max:
            continue
        if y_min is not None and token.box.cy < y_min:
            continue
        if y_max is not None and token.box.cy >= y_max:
            continue
        result.append(token)
    return sorted(result, key=lambda item: (item.box.y0, item.box.x0))


def token_bbox(tokens: list[layout_probe.Token]) -> list[float]:
    if not tokens:
        return []
    return layout_probe.union_token_box(tokens).to_list()


def render_token_region(tokens: list[layout_probe.Token]) -> str:
    rows = layout_probe.render_rows(tokens)
    return "\n".join(row for row in rows if row.strip()).strip()


def render_chart_candidate_region(tokens: list[layout_probe.Token], reference_table: list[dict[str, str]] | None = None) -> str:
    if not tokens:
        return ""

    legend_tokens, body_tokens = split_legend_tokens(tokens)
    year_anchors = find_year_anchors(body_tokens)
    if len(year_anchors) < 2:
        return render_token_region(tokens)

    columns = assign_tokens_to_year_columns(body_tokens, year_anchors)
    records = [build_chart_column_record(year, column_tokens) for year, column_tokens in columns]
    normalized_records = build_chart_records_from_reference_table(reference_table or [])

    lines: list[str] = []
    if legend_tokens:
        lines.append("### Legend")
        for legend_row in layout_probe.render_rows(legend_tokens):
            if legend_row.strip():
                lines.append(f"- {legend_row.strip()}")
        lines.append("")

    lines.extend(
        [
            '### Normalized chart table',
            "",
        ]
    )
    if normalized_records:
        lines.extend(
            [
                "> Values are aligned from the original pipeline table in the same page region; raw coordinate columns are kept below as extraction evidence.",
                "",
                "| year | total_shareholder_return_rate | shareholder_return_total | treasury_stock_cancellation_total | dividend_total |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for record in normalized_records:
            lines.append(
                "| {year} | {rate} | {total} | {cancel} | {dividend} |".format(
                    year=record["year"],
                    rate=record["rate"],
                    total=record["total"],
                    cancel=record["cancel"],
                    dividend=record["dividend"],
                )
            )
        lines.append("")

    lines.extend(
        [
            "### Raw column-major reading",
            "",
            "| year_anchor | rate_or_top_percent | total | upper_bar_or_first_amount | lower_bar_or_second_amount | extra_values |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in records:
        lines.append(
            "| {year} | {rate} | {total} | {upper} | {lower} | {extra} |".format(
                year=record["year"],
                rate=record["rate"] or "",
                total=record["total"] or "",
                upper=record["upper"] or "",
                lower=record["lower"] or "",
                extra=", ".join(record["extra"]),
            )
        )

    lines.extend(["", "### Raw columns", ""])
    for record in records:
        raw = ", ".join(record["raw_values"])
        lines.append(f"- {record['year']}: {raw}")

    return "\n".join(lines).strip()


def build_chart_records_from_reference_table(records: list[dict[str, str]]) -> list[dict[str, str]]:
    if not records:
        return []
    year_columns = [key for key in records[0] if re.fullmatch(r"20\d{2}", key)]
    if len(year_columns) < 2:
        return []

    def label_for(row: dict[str, str]) -> str:
        return next(iter(row.values()), "")

    def find_row(pattern: str) -> dict[str, str]:
        for row in records:
            if re.search(pattern, label_for(row)):
                return row
        return {}

    rate_row = find_row(r"주주환원율")
    total_row = find_row(r"주주환원\s*총액")
    cancel_row = find_row(r"자기주식소각\s*총액")
    dividend_row = find_row(r"배당\s*총액")
    output: list[dict[str, str]] = []
    for year in year_columns:
        output.append(
            {
                "year": year,
                "rate": rate_row.get(year, ""),
                "total": total_row.get(year, ""),
                "cancel": cancel_row.get(year, ""),
                "dividend": dividend_row.get(year, ""),
            }
        )
    return output


def split_legend_tokens(tokens: list[layout_probe.Token]) -> tuple[list[layout_probe.Token], list[layout_probe.Token]]:
    top_y = min(token.box.y0 for token in tokens)
    legend_limit = top_y + 24.0
    legend_tokens = [token for token in tokens if token.box.cy <= legend_limit]
    body_tokens = [token for token in tokens if token not in legend_tokens]
    return legend_tokens, body_tokens


def find_year_anchors(tokens: list[layout_probe.Token]) -> list[layout_probe.Token]:
    candidates = [token for token in tokens if re.fullmatch(r"20\d{2}", token.text)]
    if len(candidates) <= 1:
        return sorted(candidates, key=lambda token: token.box.cx)
    max_y = max(token.box.cy for token in candidates)
    bottom_candidates = [token for token in candidates if max_y - token.box.cy <= 18.0]
    anchors = bottom_candidates if len(bottom_candidates) >= 2 else candidates
    return sorted(anchors, key=lambda token: token.box.cx)


def assign_tokens_to_year_columns(
    tokens: list[layout_probe.Token],
    anchors: list[layout_probe.Token],
) -> list[tuple[layout_probe.Token, list[layout_probe.Token]]]:
    columns: dict[int, list[layout_probe.Token]] = {index: [] for index in range(len(anchors))}
    left_bound = min(anchor.box.cx for anchor in anchors) - 55.0
    right_bound = max(anchor.box.cx for anchor in anchors) + 55.0
    for token in tokens:
        if token.text == "%":
            continue
        if token.box.cx < left_bound or token.box.cx > right_bound:
            continue
        index = min(range(len(anchors)), key=lambda idx: abs(token.box.cx - anchors[idx].box.cx))
        if abs(token.box.cx - anchors[index].box.cx) <= 62.0:
            columns[index].append(token)
    return [(anchor, sorted(columns[index], key=lambda token: (token.box.y0, token.box.x0))) for index, anchor in enumerate(anchors)]


def build_chart_column_record(year_anchor: layout_probe.Token, tokens: list[layout_probe.Token]) -> dict[str, Any]:
    values = [
        token
        for token in tokens
        if token is not year_anchor
        and token.text != year_anchor.text
        and not re.fullmatch(r"20\d{2}", token.text)
        and not re.fullmatch(r"\d+\)", token.text)
    ]
    values = sorted(values, key=lambda token: (token.box.y0, token.box.x0))
    raw_values = [token.text for token in values] + [year_anchor.text]
    percentages = [token.text for token in values if re.fullmatch(r"\d+(?:\.\d+)?%", token.text)]
    numbers = [token.text for token in values if re.fullmatch(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", token.text)]
    return {
        "year": year_anchor.text,
        "rate": percentages[0] if percentages else "",
        "total": numbers[0] if len(numbers) >= 1 else "",
        "upper": numbers[1] if len(numbers) >= 2 else "",
        "lower": numbers[2] if len(numbers) >= 3 else "",
        "extra": numbers[3:] + percentages[1:],
        "raw_values": raw_values,
    }


def render_layout_markdown(
    pdf_path: Path,
    parsed: Any,
    regions: list[dict[str, Any]],
    geometry: dict[str, Any],
) -> str:
    lines = [
        "# Page 16. Auto Layout Reconstruction",
        "",
        f"> Source PDF: {pdf_path.name}",
        f"> Parser: {parsed.parser_name}",
        f"> Method: original pipeline table blocks + MCID/xref coordinate regions; no LLM used.",
        f"> Geometry: split_x={geometry['split_x']}, table_start_y={geometry['table_start_y']}, notes_start_y={geometry['notes_start_y']}",
        "",
    ]
    for region in regions:
        bbox = json.dumps(region["bbox"], ensure_ascii=False)
        lines.extend(
            [
                f"<region id=\"{region['id']}\" type=\"{region['type']}\" bbox='{bbox}' source=\"{region['source']}\">",
                "",
                f"## {region['id']} {region['type']}",
                "",
            ]
        )
        markdown = str(region.get("markdown") or "").strip()
        if markdown:
            lines.append(markdown)
        else:
            lines.append("_No content extracted._")
        lines.extend(["", "</region>", ""])
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
