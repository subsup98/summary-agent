from __future__ import annotations

import re
from typing import Any


PAGE_HEADING_RE = re.compile(r"^#\s*Page\s+(\d+)\b", re.IGNORECASE)


def build_llm_ready_artifacts(payload: dict[str, Any], semantic_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    original_markdown = str(payload.get("markdown") or "")
    page_blocks = _split_markdown_pages(original_markdown)
    page_summaries = {
        int(item.get("page_number") or 0): item
        for item in payload.get("page_summaries") or []
        if isinstance(item, dict) and item.get("page_number")
    }

    llm_ready_chunks: list[dict[str, Any]] = []
    report_parts = ["# LLM-ready document", ""]
    for page_number in sorted(page_blocks):
        page_text = _clean_page_markdown(page_blocks[page_number])
        if not page_text:
            continue
        summary = page_summaries.get(page_number) or {}
        section_title = _first_content_line(page_text) or f"Page {page_number}"
        chunk_text = _render_clean_text_chunk(page_number, section_title, summary, page_text)
        llm_ready_chunks.append(
            {
                "strategy": "llm_ready",
                "chunk_type": "clean_text",
                "semantic_type": "clean_text",
                "retrieval_lane": "clean_context",
                "chunk_index": len(llm_ready_chunks),
                "chunk_id": f"p{page_number:02d}_clean_text",
                "page_number": page_number,
                "section_hint": section_title,
                "text": chunk_text,
                "char_count": len(chunk_text),
            }
        )
        for table_index, table_context in enumerate(_extract_markdown_table_contexts(page_text), start=1):
            relation_text = _render_table_relation_chunk(
                page_number=page_number,
                table_index=table_index,
                page_title=section_title,
                table_text=table_context["table_text"],
                table_title=table_context.get("title", ""),
                unit=table_context.get("unit", ""),
            )
            llm_ready_chunks.append(
                {
                    "strategy": "llm_ready",
                    "chunk_type": "structured_relation",
                    "semantic_type": "markdown_table",
                    "retrieval_lane": "relation",
                    "chunk_index": len(llm_ready_chunks),
                    "chunk_id": f"p{page_number:02d}_table_{table_index:02d}",
                    "page_number": page_number,
                    "section_hint": table_context.get("title") or section_title,
                    "text": relation_text,
                    "char_count": len(relation_text),
                }
            )
        report_parts.append(chunk_text)
        report_parts.append("")

    if not llm_ready_chunks and original_markdown.strip():
        text = original_markdown.strip()
        llm_ready_chunks.append(
            {
                "strategy": "llm_ready",
                "chunk_type": "clean_text",
                "semantic_type": "clean_text",
                "retrieval_lane": "clean_context",
                "chunk_index": 0,
                "chunk_id": "document_clean_text",
                "page_number": "",
                "section_hint": _first_content_line(text),
                "text": text,
                "char_count": len(text),
            }
        )
        report_parts.extend([text, ""])

    hybrid_chunks = build_hybrid_retrieval_chunks(semantic_chunks, llm_ready_chunks)
    return {
        "llm_ready_markdown": "\n".join(report_parts).rstrip() + "\n",
        "llm_ready_chunks": llm_ready_chunks,
        "hybrid_retrieval_chunks": hybrid_chunks,
    }


def build_hybrid_retrieval_chunks(
    semantic_chunks: list[dict[str, Any]],
    llm_ready_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hybrid: list[dict[str, Any]] = []
    for index, chunk in enumerate(semantic_chunks):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        page_number = chunk.get("page_number") or _infer_page_number(text)
        hybrid.append(
            {
                **chunk,
                "strategy": "hybrid",
                "chunk_type": chunk.get("chunk_type") or "main_markdown_semantic",
                "semantic_type": chunk.get("semantic_type") or "markdown_semantic",
                "retrieval_lane": "legacy_context",
                "chunk_index": len(hybrid),
                "chunk_id": f"hybrid_legacy_{chunk.get('chunk_index', index)}",
                "page_number": page_number,
                "text": text,
                "char_count": len(text),
            }
        )
    for chunk in llm_ready_chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        hybrid.append(
            {
                **chunk,
                "strategy": "hybrid",
                "retrieval_lane": chunk.get("retrieval_lane") or "clean_context",
                "chunk_index": len(hybrid),
                "chunk_id": f"hybrid_llm_ready_{chunk.get('chunk_id') or len(hybrid)}",
                "text": text,
                "char_count": len(text),
            }
        )
    return hybrid


def _split_markdown_pages(markdown: str) -> dict[int, str]:
    pages: dict[int, str] = {}
    current_page = 0
    current_lines: list[str] = []
    for line in str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = PAGE_HEADING_RE.match(line.strip())
        if match:
            if current_page and current_lines:
                pages[current_page] = "\n".join(current_lines).strip()
            current_page = int(match.group(1))
            current_lines = [line]
            continue
        if current_page:
            current_lines.append(line)
    if current_page and current_lines:
        pages[current_page] = "\n".join(current_lines).strip()
    return pages


def _clean_page_markdown(page_markdown: str) -> str:
    lines: list[str] = []
    blank_seen = False
    for raw_line in str(page_markdown or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if not blank_seen:
                lines.append("")
            blank_seen = True
            continue
        blank_seen = False
        lines.append(line)
    return "\n".join(lines).strip()


def _render_clean_text_chunk(page_number: int, title: str, summary: dict[str, Any], page_text: str) -> str:
    lines = [f"## Page {page_number}", "", f"Title: {title}"]
    summary_text = str(summary.get("summary_text") or "").strip()
    if summary_text:
        lines.extend(["", f"Summary: {summary_text}"])
    key_points = [str(item).strip() for item in summary.get("key_points") or [] if str(item).strip()]
    if key_points:
        lines.extend(["", "Key points:"])
        lines.extend(f"- {item}" for item in key_points[:6])
    lines.extend(["", "Content:", "", page_text])
    return "\n".join(lines).strip()


def _render_table_relation_chunk(
    page_number: int,
    table_index: int,
    page_title: str,
    table_text: str,
    table_title: str = "",
    unit: str = "",
) -> str:
    fact_rows = _render_table_fact_rows(
        page_number=page_number,
        table_index=table_index,
        table_title=table_title or page_title,
        unit=unit,
        table_text=table_text,
    )
    lines = [
        f"## Page {page_number} Table {table_index}",
        "",
        f"Context: {page_title}",
    ]
    if table_title:
        lines.append(f"Table title: {table_title}")
    if unit:
        lines.append(f"Unit: {unit}")
    lines.extend(["Type: structured_relation / markdown_table", "", table_text.strip()])
    if fact_rows:
        lines.extend(["", "KPI facts:", "", fact_rows])
    return "\n".join(lines).strip()


def _extract_markdown_tables(page_text: str) -> list[str]:
    return [item["table_text"] for item in _extract_markdown_table_contexts(page_text)]


def _extract_markdown_table_contexts(page_text: str) -> list[dict[str, str]]:
    current: list[str] = []
    lines = str(page_text or "").splitlines()
    table_ranges: list[tuple[int, int, str]] = []
    start_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            if start_index is None:
                start_index = index
            current.append(stripped)
            continue
        if current:
            if len(current) >= 2:
                table_ranges.append((start_index or index, index - 1, "\n".join(current)))
            current = []
            start_index = None
    if current and len(current) >= 2:
        table_ranges.append((start_index or max(len(lines) - 1, 0), len(lines) - 1, "\n".join(current)))

    contexts: list[dict[str, str]] = []
    for start, end, table_text in table_ranges:
        title = _infer_table_title(lines, start, end)
        contexts.append(
            {
                "table_text": table_text,
                "title": title,
                "unit": _infer_table_unit(lines, start, end),
            }
        )
    return contexts


def _infer_table_title(lines: list[str], start: int, end: int) -> str:
    before = [
        _clean_context_line(lines[index])
        for index in range(max(0, start - 4), start)
        if _clean_context_line(lines[index])
    ]
    after = [
        _clean_context_line(lines[index])
        for index in range(end + 1, min(len(lines), end + 14))
        if _clean_context_line(lines[index])
    ]

    strong_after = [item for item in after if _looks_like_strong_table_title(item)]
    if strong_after:
        return strong_after[0]

    candidates = [*reversed(before), *after]
    meaningful = [item for item in candidates if _looks_like_table_title(item)]
    if meaningful:
        return meaningful[0]
    return candidates[0] if candidates else ""


def _infer_table_unit(lines: list[str], start: int, end: int) -> str:
    for index in range(max(0, start - 4), min(len(lines), end + 14)):
        candidate = _clean_context_line(lines[index])
        if re.search(r"^\(?\s*(단위|Unit)\s*[:：]", candidate, flags=re.IGNORECASE):
            return candidate.strip("() ")
    return ""


def _clean_context_line(line: str) -> str:
    stripped = re.sub(r"\s+", " ", str(line or "")).strip()
    if not stripped or stripped.startswith("|"):
        return ""
    if re.match(r"^#{1,6}\s+Page\s+\d+\b", stripped, flags=re.IGNORECASE):
        return ""
    if stripped == "Source:" or re.match(r"^-\s+(Page|Region|Type|BBox):", stripped, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(r"[-: ]+", stripped):
        return ""
    return stripped.lstrip("#").strip()


def _looks_like_table_title(text: str) -> bool:
    if not text or len(text) > 80:
        return False
    if re.search(r"(요약|손익계산서|재무|성과|실적|현황|내역|KPI|지표|Table|표)", text, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_strong_table_title(text: str) -> bool:
    return bool(
        text
        and len(text) <= 80
        and re.search(r"(손익계산서|재무상태표|현금흐름표|재무제표|KPI|주요 지표)", text, flags=re.IGNORECASE)
    )


def _render_table_fact_rows(
    *,
    page_number: int,
    table_index: int,
    table_title: str,
    unit: str,
    table_text: str,
) -> str:
    rows = _parse_markdown_table(table_text)
    if len(rows) < 2:
        return ""

    header = rows[0]
    body_rows = [row for row in rows[1:] if not _is_separator_row(row)]
    if len(header) < 2 or not body_rows:
        return ""

    period_columns = [
        (index, _normalize_period_label(label))
        for index, label in enumerate(header[1:], start=1)
        if _looks_like_period_label(label)
    ]
    if not period_columns:
        return ""

    facts: list[str] = []
    for row in body_rows:
        if not row:
            continue
        metric = _clean_table_cell(row[0])
        if not metric or _looks_like_period_label(metric):
            continue
        for column_index, period in period_columns:
            if column_index >= len(row):
                continue
            value = _clean_table_cell(row[column_index])
            if not value or not _looks_like_fact_value(value):
                continue
            facts.append(
                f"- Page {page_number} Table {table_index}: {table_title} / {metric} / {period} = {value}"
                + (f" ({unit})" if unit else "")
            )

    return "\n".join(facts[:80])


def _parse_markdown_table(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(table_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [_clean_table_cell(cell) for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def _clean_table_cell(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\\|", "|")).strip()


def _is_separator_row(row: list[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in row)


def _looks_like_period_label(value: str) -> bool:
    label = _normalize_period_label(value)
    if not label:
        return False
    return bool(
        re.fullmatch(r"(?:['’]?\d{2,4}\s*)?(?:[1-4]Q|Q[1-4]|FY|YTD|상반기|하반기)", label, flags=re.IGNORECASE)
        or re.fullmatch(r"[1-4]Q\s*['’]?\d{2,4}", label, flags=re.IGNORECASE)
        or re.fullmatch(r"['’]?\d{2,4}\s*(?:YTD|FY)", label, flags=re.IGNORECASE)
        or re.fullmatch(r"\d{4}", label)
    )


def _normalize_period_label(value: str) -> str:
    label = _clean_table_cell(value).replace("‘", "'").replace("’", "'")
    label = re.sub(r"\s+", "", label)
    return label


def _looks_like_fact_value(value: str) -> bool:
    cleaned = _clean_table_cell(value)
    if not cleaned:
        return False
    normalized = cleaned.replace(",", "").replace("%", "").replace("−", "-")
    normalized = re.sub(r"\s+", "", normalized)
    return bool(re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", normalized))


def _first_content_line(text: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:160]
    return ""


def _infer_page_number(text: str) -> int | str:
    match = re.search(r"(?:^|\n)#\s*Page\s+(\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return ""
