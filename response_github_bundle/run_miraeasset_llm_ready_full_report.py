from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "manual_compare"
    / "miraeasset_main_pipeline"
    / "structured"
    / "documents"
    / "미래에셋증권_3분기_실적보고서--59a0fab3.json"
)
SEMANTIC_REGION_DIR = PROJECT_ROOT / "outputs" / "drawing_guided_pages_04_16"
SEMANTIC_PAGE_JSONS = [
    SEMANTIC_REGION_DIR / "page_04_drawing_guided_semantic_structure.json",
    SEMANTIC_REGION_DIR / "page_16_drawing_guided_semantic_structure.json",
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    semantic_pages = load_semantic_pages()
    page_markdown = split_markdown_by_page(str(payload.get("markdown") or ""))

    chunks: list[dict[str, Any]] = []
    report = render_full_report(payload, page_markdown, semantic_pages, chunks)
    curated_chunks: list[dict[str, Any]] = []
    curated_report = render_curated_semantic_review(payload, page_markdown, semantic_pages, curated_chunks)

    md_path = OUTPUT_DIR / "miraeasset_q3_llm_ready_full_report.md"
    chunks_path = OUTPUT_DIR / "miraeasset_q3_llm_ready_chunks.json"
    curated_md_path = OUTPUT_DIR / "miraeasset_q3_semantic_chunking_review.md"
    curated_chunks_path = OUTPUT_DIR / "miraeasset_q3_semantic_chunks_curated.json"
    md_path.write_text(report, encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    curated_md_path.write_text(curated_report, encoding="utf-8")
    curated_chunks_path.write_text(json.dumps(curated_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "markdown": md_path.as_posix(),
                "chunks": chunks_path.as_posix(),
                "curated_markdown": curated_md_path.as_posix(),
                "curated_chunks": curated_chunks_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_semantic_pages() -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for path in SEMANTIC_PAGE_JSONS:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages[int(payload["page_number"])] = payload
    return pages


def split_markdown_by_page(markdown: str) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    current_page: int | None = None
    for line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = re.match(r"^# Page (\d+)\s*$", line.strip())
        if match:
            current_page = int(match.group(1))
            pages[current_page] = [line.strip()]
            continue
        if current_page is not None:
            pages[current_page].append(line.rstrip())
    return {page: "\n".join(lines).strip() for page, lines in pages.items()}


def render_full_report(
    payload: dict[str, Any],
    page_markdown: dict[int, str],
    semantic_pages: dict[int, dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> str:
    lines = [
        "# Mirae Asset Securities Q3 2025 LLM-Ready Report",
        "",
        f"- Source: {payload.get('source_name')}",
        f"- Document ID: {payload.get('document_id')}",
        f"- Parser: {payload.get('parser_name')}",
        f"- Page count: {len(payload.get('pages') or [])}",
        "",
        "## How To Use",
        "",
        "- Each page is rendered as an LLM-readable block.",
        "- Pages with visual semantic reconstruction include relation-first blocks before the raw parser text.",
        "- Relation blocks are intended for retrieval and answer synthesis; raw page text is retained as fallback evidence.",
        "",
    ]

    for page in payload.get("pages") or []:
        page_number = int(page.get("page_number") or 0)
        lines.extend(render_page(page_number, page_markdown.get(page_number, ""), semantic_pages.get(page_number), chunks))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_page(
    page_number: int,
    raw_markdown: str,
    semantic_page: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"## Page {page_number}",
        "",
    ]
    if semantic_page:
        lines.extend(
            [
                "### Visual Semantic Relations",
                "",
                "> These blocks are generated from page-region geometry and semantic matching.",
                "",
            ]
        )
        for region in semantic_page.get("regions") or []:
            block = render_region_block(page_number, region)
            if not block.strip():
                continue
            lines.append(block.rstrip())
            lines.append("")
            chunks.append(
                {
                    "chunk_id": f"miraeasset_q3_p{page_number:02d}_{str(region.get('id') or '').lower()}",
                    "chunk_type": "visual_semantic_region",
                    "page_number": page_number,
                    "region_id": region.get("id"),
                    "semantic_type": (region.get("semantic_structure") or {}).get("kind"),
                    "bbox": region.get("bbox"),
                    "text": block.rstrip(),
                }
            )

    if raw_markdown:
        cleaned = normalize_raw_page_markdown(raw_markdown)
        lines.extend(
            [
                "### Parser Page Text",
                "",
                cleaned,
                "",
            ]
        )
        chunks.append(
            {
                "chunk_id": f"miraeasset_q3_p{page_number:02d}_parser_text",
                "chunk_type": "parser_page_text",
                "page_number": page_number,
                "region_id": "",
                "semantic_type": "page_text",
                "bbox": [],
                "text": f"## Page {page_number}\n\n{cleaned}".strip(),
            }
        )
    return lines


def render_curated_semantic_review(
    payload: dict[str, Any],
    page_markdown: dict[int, str],
    semantic_pages: dict[int, dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> str:
    lines = [
        "# Mirae Asset Securities Q3 2025 Semantic Chunking Review",
        "",
        f"- Source: {payload.get('source_name')}",
        f"- Document ID: {payload.get('document_id')}",
        "",
        "## Chunking Policy",
        "",
        "- Use `structured_relation` chunks for numeric/chart/table relationships.",
        "- Use `clean_text` chunks for narrative text, bullets, and preserved parser tables.",
        "- Exclude debug coordinates, raw token dumps, isolated axis labels, and repeated page boilerplate from chunk text.",
        "- Keep page/region metadata in JSON rather than in the prose body.",
        "",
        "## Structured Relation Chunks",
        "",
    ]

    for page_number in sorted(semantic_pages):
        semantic_page = semantic_pages[page_number]
        page_header_written = False
        for region in semantic_page.get("regions") or []:
            block = render_curated_region_block(page_number, region)
            if not block.strip():
                continue
            if not page_header_written:
                lines.extend([f"### Page {page_number}", ""])
                page_header_written = True
            lines.extend([block.rstrip(), ""])
            chunks.append(
                {
                    "chunk_id": f"miraeasset_q3_p{page_number:02d}_{str(region.get('id') or '').lower()}_structured",
                    "chunk_type": "structured_relation",
                    "page_number": page_number,
                    "region_id": region.get("id"),
                    "semantic_type": (region.get("semantic_structure") or {}).get("kind"),
                    "text": block.rstrip(),
                    "metadata": {
                        "bbox": region.get("bbox") or [],
                        "source_name": payload.get("source_name"),
                    },
                }
            )

    lines.extend(["## Clean Text Chunks", ""])
    for page_number in sorted(page_markdown):
        text_blocks = build_clean_text_blocks(page_number, page_markdown[page_number])
        if not text_blocks:
            continue
        lines.extend([f"### Page {page_number}", ""])
        for index, block in enumerate(text_blocks, start=1):
            lines.extend([f"#### Text Block {index}", "", block.rstrip(), ""])
            chunks.append(
                {
                    "chunk_id": f"miraeasset_q3_p{page_number:02d}_text_{index:02d}",
                    "chunk_type": "clean_text",
                    "page_number": page_number,
                    "region_id": "",
                    "semantic_type": "clean_text",
                    "text": f"Page {page_number}\n\n{block.rstrip()}",
                    "metadata": {
                        "source_name": payload.get("source_name"),
                    },
                }
            )
    return "\n".join(lines).rstrip() + "\n"


def render_curated_region_block(page_number: int, region: dict[str, Any]) -> str:
    semantic = region.get("semantic_structure") or {}
    kind = str(semantic.get("kind") or "unknown")
    title = compact_title(semantic.get("title_candidates") or [], str(region.get("markdown") or ""))
    header = [f"#### {region.get('id')} - {kind}", ""]
    if title:
        header.extend([f"Topic: {title}", ""])

    if kind in {"kpi_panel", "kpi_pair_panel"}:
        body = render_kpi_table(semantic)
    elif kind in {"chart", "mixed_chart_panel"}:
        body = render_curated_chart(semantic)
    elif kind == "table":
        body = render_table_semantic(semantic)
    elif kind in {"highlight_card", "notes"}:
        body = render_text_card(semantic)
    else:
        body = ""
    if not body.strip() or body.strip().startswith("_No "):
        return ""
    footer = f"\n\nReference: page {page_number}, region {region.get('id')}, type {kind}"
    return "\n".join(header) + body.rstrip() + footer


def render_curated_chart(semantic: dict[str, Any]) -> str:
    lines: list[str] = []
    points = semantic.get("line_chart_points") or []
    labeled_points = [point for point in points if point.get("period")]
    if labeled_points:
        lines.extend(["Line chart values:", "", "| Period | Value |", "|---|---:|"])
        for point in labeled_points:
            lines.append(f"| {escape_cell(point.get('period'))} | {escape_cell(point.get('value'))} |")
        lines.append("")

    current = semantic.get("current_value")
    if current:
        lines.extend(
            [
                "Current/callout value:",
                "",
                "| Period | Value | Unit |",
                "|---|---:|---|",
                "| {period} | {value} | {unit} |".format(
                    period=escape_cell(current.get("period")),
                    value=escape_cell(current.get("value")),
                    unit=escape_cell(current.get("unit")),
                ),
                "",
            ]
        )

    series = semantic.get("series_columns") or []
    usable_series = [column for column in series if len(column.get("values") or []) >= 2]
    if usable_series:
        lines.extend(["Additional extracted series:", "", "| Label | Values |", "|---|---|"])
        for column in usable_series:
            lines.append(
                "| {label} | {values} |".format(
                    label=escape_cell(column.get("label")),
                    values=escape_cell(", ".join(str(value) for value in column.get("values") or [])),
                )
            )
    return "\n".join(lines).strip()


def compact_title(candidates: list[Any], fallback_markdown: str) -> str:
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and not looks_like_noise_line(text):
            return text
    return first_nonempty_line(fallback_markdown)


def build_clean_text_blocks(page_number: int, markdown: str) -> list[str]:
    body = normalize_raw_page_markdown(markdown)
    raw_lines = [line.rstrip() for line in body.splitlines()]
    blocks: list[str] = []
    current: list[str] = []
    table: list[str] = []
    in_html_table = False

    def flush_current() -> None:
        nonlocal current
        text = "\n".join(line for line in current if line.strip()).strip()
        current = []
        if is_useful_text_block(text):
            blocks.append(text)

    def flush_table() -> None:
        nonlocal table
        if table:
            blocks.append("\n".join(table).strip())
            table = []

    for line in raw_lines:
        stripped = line.strip()
        if re.match(r"^<table\b", stripped, flags=re.IGNORECASE):
            flush_current()
            flush_table()
            in_html_table = True
            continue
        if in_html_table:
            if re.search(r"</table>", stripped, flags=re.IGNORECASE):
                in_html_table = False
            continue
        if stripped.startswith("|"):
            flush_current()
            table.append(stripped)
            continue
        if table:
            flush_table()
        if not stripped:
            flush_current()
            continue
        if should_skip_clean_line(stripped):
            continue
        current.append(stripped)
    flush_current()
    flush_table()

    return merge_short_blocks(blocks)


def should_skip_clean_line(line: str) -> bool:
    normalized = re.sub(r"\s+", "", line)
    boilerplate = {
        "2025년3분기실적보고서",
        "KeyHighlights",
        "[3Q2025]KeyHighlights",
    }
    if normalized in boilerplate:
        return True
    if re.search(r"(?:/Users/|[A-Za-z]:\\).+\.(?:png|jpg|jpeg)$", line):
        return True
    if re.match(r"^</?[A-Za-z][^>]*>$", line):
        return True
    if looks_like_noise_line(line):
        return True
    return False


def looks_like_noise_line(line: str) -> bool:
    normalized = re.sub(r"\s+", "", str(line or ""))
    if not normalized:
        return True
    if re.fullmatch(r"\d{1,3}", normalized):
        return True
    if re.fullmatch(r"-?\(?\d+(?:\.\d+)?%?\)?", normalized):
        return True
    if re.fullmatch(r"(?:20\d{2}|[1-4]?Q\d{2}|YTD|QoQ)", normalized, flags=re.IGNORECASE):
        return True
    if len(normalized) <= 2 and not re.search(r"[가-힣A-Za-z]", normalized):
        return True
    return False


def is_useful_text_block(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    if len(normalized) < 12:
        return False
    if not re.search(r"[가-힣A-Za-z]", normalized):
        return False
    return True


def merge_short_blocks(blocks: list[str], target_size: int = 700) -> list[str]:
    merged: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= target_size:
            current = candidate
            continue
        if current:
            merged.append(current)
        current = block
    if current:
        merged.append(current)
    return merged


def render_region_block(page_number: int, region: dict[str, Any]) -> str:
    semantic = region.get("semantic_structure") or {}
    kind = str(semantic.get("kind") or "unknown")
    title = " / ".join(semantic.get("title_candidates") or []) or first_nonempty_line(str(region.get("markdown") or ""))
    header = [
        f"#### Page {page_number} / {region.get('id')} / {kind}",
        "",
        f"Title: {title}",
        "",
    ]
    if kind in {"kpi_panel", "kpi_pair_panel"}:
        body = render_kpi_table(semantic)
    elif kind in {"chart", "mixed_chart_panel"}:
        body = render_chart_tables(semantic)
    elif kind == "table":
        body = render_table_semantic(semantic)
    elif kind in {"highlight_card", "notes"}:
        body = render_text_card(semantic)
    else:
        body = "```json\n" + json.dumps(semantic, ensure_ascii=False, indent=2) + "\n```"

    source = [
        "",
        "Source:",
        f"- Page: {page_number}",
        f"- Region: {region.get('id')}",
        f"- Type: {kind}",
        f"- BBox: {json.dumps(region.get('bbox') or [], ensure_ascii=False)}",
    ]
    return "\n".join([*header, body.rstrip(), *source]).rstrip()


def render_kpi_table(semantic: dict[str, Any]) -> str:
    rows = semantic.get("items") or []
    if not rows:
        return "_No KPI items detected._"
    lines = [
        "| Metric | Value | Unit | Column Anchor |",
        "|---|---:|---|---|",
    ]
    for item in rows:
        lines.append(
            "| {label} | {value} | {unit} | {anchor} |".format(
                label=escape_cell(item.get("label")),
                value=escape_cell(item.get("value")),
                unit=escape_cell(item.get("unit")),
                anchor=escape_cell(item.get("column_anchor")),
            )
        )
    return "\n".join(lines)


def render_chart_tables(semantic: dict[str, Any]) -> str:
    lines: list[str] = []
    period_anchors = semantic.get("period_anchors") or []
    if period_anchors:
        lines.extend(
            [
                "Period anchors:",
                "",
                "| Period | X | Y |",
                "|---|---:|---:|",
            ]
        )
        for anchor in period_anchors:
            lines.append(f"| {escape_cell(anchor.get('label'))} | {anchor.get('cx', '')} | {anchor.get('cy', '')} |")
        lines.append("")

    points = semantic.get("line_chart_points") or []
    if points:
        lines.extend(
            [
                "Line chart values:",
                "",
                "| Period | Value | X | Y |",
                "|---|---:|---:|---:|",
            ]
        )
        for point in points:
            period = point.get("period") or ""
            lines.append(f"| {escape_cell(period)} | {escape_cell(point.get('value'))} | {point.get('cx', '')} | {point.get('cy', '')} |")
        lines.append("")

    current = semantic.get("current_value")
    if current:
        lines.extend(
            [
                "Current/callout value:",
                "",
                "| Period | Value | Unit | Reason |",
                "|---|---:|---|---|",
                "| {period} | {value} | {unit} | {reason} |".format(
                    period=escape_cell(current.get("period")),
                    value=escape_cell(current.get("value")),
                    unit=escape_cell(current.get("unit")),
                    reason=escape_cell(current.get("reason")),
                ),
                "",
            ]
        )

    series_columns = semantic.get("series_columns") or []
    if series_columns:
        lines.extend(
            [
                "Additional x-column evidence:",
                "",
                "| Label | Values | Text |",
                "|---|---|---|",
            ]
        )
        for column in series_columns:
            lines.append(
                "| {label} | {values} | {text} |".format(
                    label=escape_cell(column.get("label")),
                    values=escape_cell(", ".join(str(value) for value in column.get("values") or [])),
                    text=escape_cell(column.get("text")),
                )
            )
    return "\n".join(lines).strip() or "_No chart structure detected._"


def render_table_semantic(semantic: dict[str, Any]) -> str:
    header = semantic.get("header") or []
    rows = semantic.get("rows") or []
    if header and rows:
        lines = [
            "| " + " | ".join(escape_cell(cell) for cell in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in rows:
            normalized = list(row) + [""] * max(0, len(header) - len(row))
            lines.append("| " + " | ".join(escape_cell(cell) for cell in normalized[: len(header)]) + " |")
        return "\n".join(lines)
    columns = semantic.get("columns") or []
    if columns:
        lines = ["| Column | Text |", "|---|---|"]
        for column in columns:
            lines.append(f"| {escape_cell(column.get('anchor'))} | {escape_cell(column.get('joined_text'))} |")
        return "\n".join(lines)
    return "_No table structure detected._"


def render_text_card(semantic: dict[str, Any]) -> str:
    lines = []
    headings = semantic.get("headings") or []
    bullets = semantic.get("bullets") or []
    body_lines = semantic.get("lines") or []
    if headings:
        lines.append("Headings:")
        lines.extend(f"- {heading}" for heading in headings)
        lines.append("")
    if bullets:
        lines.append("Bullets:")
        lines.extend(f"- {bullet}" for bullet in bullets)
        lines.append("")
    if body_lines and not bullets:
        lines.append("Text:")
        lines.extend(f"- {line}" for line in body_lines)
    return "\n".join(lines).strip() or "_No text card content detected._"


def normalize_raw_page_markdown(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        if re.match(r"^# Page \d+\s*$", line.strip()):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def escape_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    main()
