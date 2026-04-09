from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
TITLE_NUMERIC_RE = re.compile(r"^(?:\d+|[가-하])[\.\)](?:\s*\D.*)?$")
TITLE_CHAPTER_RE = re.compile(r"^제\s*\d+\s*[장절조항목](?:\s|$)")
LIST_BULLET_RE = re.compile(
    r"^(?:[-*•·※]\s*|[①-⑳]\s*|[ㄱ-ㅎ][\.\)]\s*|[가-하][\.\)]\s*|(?:\d+|[A-Za-z])[\.\)](?=\s*\D)\s*)"
)
DATE_VALUE_RE = re.compile(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$")
NUMERIC_VALUE_RE = re.compile(r"^[\d,]+(?:\.\d+)?%?$")
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?。！？:]$")
SENTENCE_ENDING_RE = re.compile(r"(?:다|니다|함|됨|음)\.?$")


@dataclass(frozen=True)
class NormalizedLine:
    line_index: int
    text: str
    indent: int = 0
    page: int | None = 1


@dataclass(frozen=True)
class LineClassification:
    line_index: int
    text: str
    block_type: str
    is_title: bool = False
    is_list_item: bool = False
    is_kv_label: bool = False
    looks_like_value: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HwpPostprocessRules:
    version: str = "hwp-postprocess-rules/v1"
    noise_min_chars: int = 2
    max_title_chars: int = 32
    max_short_label_chars: int = 24
    max_kv_labels: int = 6
    paragraph_chunk_block_limit: int = 3
    paragraph_chunk_char_limit: int = 700
    short_paragraph_chars: int = 40
    min_kv_labels: int = 2
    minimum_list_items: int = 2
    min_row_table_labels: int = 6
    min_row_table_values: int = 5
    allowed_short_titles: tuple[str, ...] = (
        "목차",
        "개요",
        "현황",
        "실적",
        "개최일자",
        "참석인원",
        "주요 의결사항",
    )


@dataclass
class HwpPostprocessResult:
    normalized_lines: list[NormalizedLine]
    classifications: list[LineClassification]
    blocks: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    logs: dict[str, Any] = field(default_factory=dict)


def normalize_lines(raw_text: str, rules: HwpPostprocessRules | None = None) -> tuple[list[NormalizedLine], dict[str, int]]:
    active_rules = rules or HwpPostprocessRules()
    page = 1
    raw_lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized: list[NormalizedLine] = []
    dropped_noise = 0
    dropped_blank = 0

    for line_index, raw_line in enumerate(raw_lines, start=1):
        if "\f" in raw_line:
            page += raw_line.count("\f")
            raw_line = raw_line.replace("\f", " ")

        indent = len(raw_line) - len(raw_line.lstrip(" \t"))
        cleaned = CONTROL_CHAR_RE.sub(" ", raw_line)
        cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()

        if not cleaned:
            dropped_blank += 1
            continue
        if _is_noise_line(cleaned, active_rules):
            dropped_noise += 1
            continue

        normalized.append(
            NormalizedLine(
                line_index=line_index,
                text=cleaned,
                indent=indent,
                page=page,
            )
        )

    return normalized, {
        "raw_line_count": len(raw_lines),
        "normalized_line_count": len(normalized),
        "dropped_blank_line_count": dropped_blank,
        "dropped_noise_line_count": dropped_noise,
    }


def classify_line(line: NormalizedLine, rules: HwpPostprocessRules | None = None) -> LineClassification:
    active_rules = rules or HwpPostprocessRules()
    reasons: list[str] = []
    text = line.text

    if _looks_like_title(text, active_rules):
        reasons.append("title_rule")
        return LineClassification(
            line_index=line.line_index,
            text=text,
            block_type="title",
            is_title=True,
            reasons=tuple(reasons),
        )

    if _looks_like_list_item(text):
        reasons.append("list_rule")
        return LineClassification(
            line_index=line.line_index,
            text=text,
            block_type="list",
            is_list_item=True,
            looks_like_value=True,
            reasons=tuple(reasons),
        )

    is_kv_label = _looks_like_kv_label(text, active_rules)
    looks_like_value = _looks_like_value(text)
    if is_kv_label:
        reasons.append("kv_label_rule")
    if looks_like_value:
        reasons.append("kv_value_rule")

    if _looks_unknown(text):
        return LineClassification(
            line_index=line.line_index,
            text=text,
            block_type="unknown",
            is_kv_label=is_kv_label,
            looks_like_value=looks_like_value,
            reasons=tuple(reasons or ["unknown_rule"]),
        )

    return LineClassification(
        line_index=line.line_index,
        text=text,
        block_type="paragraph",
        is_kv_label=is_kv_label,
        looks_like_value=looks_like_value,
        reasons=tuple(reasons or ["paragraph_default"]),
    )


def group_blocks(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    rules: HwpPostprocessRules | None = None,
) -> list[dict[str, Any]]:
    active_rules = rules or HwpPostprocessRules()
    blocks: list[dict[str, Any]] = []
    current_section: str | None = None
    index = 0

    while index < len(lines):
        line = lines[index]
        classification = classifications[index]

        row_table_block, row_table_end = parse_row_table(lines, classifications, index, current_section, active_rules)
        if row_table_block is not None:
            blocks.append(row_table_block)
            index = row_table_end
            continue

        kv_block, kv_end = parse_kv_table(lines, classifications, index, current_section, active_rules)
        if kv_block is not None:
            blocks.append(kv_block)
            index = kv_end
            continue

        if classification.is_title:
            if not line.text.startswith("("):
                current_section = line.text
            blocks.append(
                _build_title_block(
                    line=line,
                    section=current_section,
                )
            )
            index += 1
            continue

        if classification.is_list_item:
            list_block, next_index = _build_list_block(lines, classifications, index, current_section, active_rules)
            blocks.append(list_block)
            index = next_index
            continue

        if classification.block_type in {"paragraph", "unknown"}:
            paragraph_block, next_index = merge_paragraphs(
                lines=lines,
                classifications=classifications,
                start_index=index,
                current_section=current_section,
                rules=active_rules,
            )
            blocks.append(paragraph_block)
            index = next_index
            continue

        blocks.append(
            _build_unknown_block(
                line=line,
                section=current_section,
            )
        )
        index += 1

    return blocks


def parse_kv_table(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    start_index: int,
    current_section: str | None,
    rules: HwpPostprocessRules | None = None,
) -> tuple[dict[str, Any] | None, int]:
    active_rules = rules or HwpPostprocessRules()
    label_lines: list[NormalizedLine] = []
    cursor = start_index

    while cursor < len(lines) and len(label_lines) < active_rules.max_kv_labels:
        line = lines[cursor]
        if classifications[cursor].is_list_item or not _looks_like_kv_label(line.text, active_rules):
            break
        label_lines.append(line)
        cursor += 1

    if len(label_lines) < active_rules.min_kv_labels:
        return None, start_index
    if cursor >= len(lines):
        return None, start_index

    scalar_value_lines: list[NormalizedLine] = []
    while cursor < len(lines):
        classification = classifications[cursor]
        line = lines[cursor]
        if classification.is_title or classification.is_list_item:
            break
        if _looks_like_kv_label(line.text, active_rules) and not _looks_like_value(line.text):
            break
        if not _looks_like_scalar_kv_value(line.text, active_rules):
            break
        scalar_value_lines.append(line)
        cursor += 1

    list_items, next_cursor = _consume_list_value(lines, classifications, cursor)
    if not scalar_value_lines and not list_items:
        return None, start_index
    cursor = next_cursor if list_items else cursor

    fields = _assign_kv_fields(
        labels=label_lines,
        scalar_values=scalar_value_lines,
        list_items=list_items,
    )

    block = {
        "type": "kv_table",
        "section": current_section,
        "fields": fields,
        "source_format": "hwp",
        "page": label_lines[0].page,
        "line_start": label_lines[0].line_index,
        "line_end": lines[cursor - 1].line_index,
    }
    return block, cursor


def parse_row_table(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    start_index: int,
    current_section: str | None,
    rules: HwpPostprocessRules | None = None,
) -> tuple[dict[str, Any] | None, int]:
    active_rules = rules or HwpPostprocessRules()
    if start_index >= len(lines):
        return None, start_index

    label_lines: list[NormalizedLine] = []
    cursor = start_index

    while cursor < len(lines):
        line = lines[cursor]
        if classifications[cursor].is_list_item:
            break
        if not _looks_like_tabular_label(line.text, active_rules):
            break
        label_lines.append(line)
        cursor += 1

    if len(label_lines) < active_rules.min_row_table_labels:
        return None, start_index
    if not _has_repeated_label(label_lines):
        return None, start_index

    value_lines: list[NormalizedLine] = []
    while cursor < len(lines):
        classification = classifications[cursor]
        line = lines[cursor]
        if classification.is_title or classification.is_list_item:
            break
        if not _looks_like_row_table_value(line.text):
            break
        value_lines.append(line)
        cursor += 1

    if len(value_lines) < active_rules.min_row_table_values:
        return None, start_index

    header_rows = _build_row_table_headers(label_lines)
    row_values = [line.text for line in value_lines]
    block = {
        "type": "row_table",
        "section": current_section,
        "header_rows": header_rows,
        "rows": [row_values],
        "source_format": "hwp",
        "page": label_lines[0].page,
        "line_start": label_lines[0].line_index,
        "line_end": value_lines[-1].line_index,
    }
    return block, cursor


def merge_paragraphs(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    start_index: int,
    current_section: str | None,
    rules: HwpPostprocessRules | None = None,
) -> tuple[dict[str, Any], int]:
    active_rules = rules or HwpPostprocessRules()
    paragraph_lines: list[NormalizedLine] = []
    cursor = start_index

    while cursor < len(lines):
        classification = classifications[cursor]
        if cursor != start_index:
            if classification.is_title or classification.is_list_item:
                break
            kv_block, _ = parse_kv_table(lines, classifications, cursor, current_section, active_rules)
            if kv_block is not None:
                break
        paragraph_lines.append(lines[cursor])
        cursor += 1

        if cursor >= len(lines):
            break
        current_text = paragraph_lines[-1].text
        next_text = lines[cursor].text
        if not _should_merge_paragraph_lines(current_text, next_text, active_rules):
            break

    content = _join_paragraph_lines([line.text for line in paragraph_lines])
    block_type = "unknown" if all(classifications[start_index + offset].block_type == "unknown" for offset in range(len(paragraph_lines))) else "paragraph"
    block = {
        "type": block_type,
        "section": current_section,
        "content": content,
        "source_format": "hwp",
        "page": paragraph_lines[0].page if paragraph_lines else None,
        "line_start": paragraph_lines[0].line_index,
        "line_end": paragraph_lines[-1].line_index,
    }
    return block, cursor


def build_blocks(raw_text: str, rules: HwpPostprocessRules | None = None, source_format: str = "hwp") -> HwpPostprocessResult:
    active_rules = rules or HwpPostprocessRules()
    normalized_lines, normalization_log = normalize_lines(raw_text, active_rules)
    classifications = [classify_line(line, active_rules) for line in normalized_lines]
    blocks = group_blocks(normalized_lines, classifications, active_rules)
    if source_format != "hwp":
        for block in blocks:
            block["source_format"] = source_format
    chunks = chunk_blocks(blocks, active_rules, source_format)

    block_counter = Counter(block["type"] for block in blocks)
    classification_counter = Counter(classification.block_type for classification in classifications)
    kv_table_count = block_counter.get("kv_table", 0)
    unknown_count = block_counter.get("unknown", 0)
    unknown_ratio = (unknown_count / len(blocks)) if blocks else 0.0

    logs: dict[str, Any] = {
        **normalization_log,
        "rule_version": active_rules.version,
        "classified_line_type_counts": dict(classification_counter),
        "block_type_counts": dict(block_counter),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "kv_table_count": kv_table_count,
        "unknown_ratio": round(unknown_ratio, 4),
    }

    return HwpPostprocessResult(
        normalized_lines=normalized_lines,
        classifications=classifications,
        blocks=blocks,
        chunks=chunks,
        logs=logs,
    )


def chunk_blocks(blocks: list[dict[str, Any]], rules: HwpPostprocessRules | None = None, source_format: str = "hwp") -> list[dict[str, Any]]:
    active_rules = rules or HwpPostprocessRules()
    chunks: list[dict[str, Any]] = []
    index = 0

    while index < len(blocks):
        block = blocks[index]
        if block["type"] == "title":
            if index + 1 >= len(blocks):
                chunks.append(_make_chunk([block], len(chunks) + 1, source_format))
                index += 1
                continue
            next_block = blocks[index + 1]
            if next_block["type"] == "paragraph":
                grouped = [block, next_block]
                total_chars = len(next_block.get("content", ""))
                cursor = index + 2
                while cursor < len(blocks) and len([item for item in grouped if item["type"] == "paragraph"]) < active_rules.paragraph_chunk_block_limit:
                    candidate = blocks[cursor]
                    if candidate["type"] != "paragraph" or candidate.get("section") != next_block.get("section"):
                        break
                    candidate_chars = len(candidate.get("content", ""))
                    if total_chars + candidate_chars > active_rules.paragraph_chunk_char_limit:
                        break
                    grouped.append(candidate)
                    total_chars += candidate_chars
                    cursor += 1
                chunks.append(_make_chunk(grouped, len(chunks) + 1, source_format))
                index = cursor
                continue

            chunks.append(_make_chunk([block, next_block], len(chunks) + 1, source_format))
            index += 2
            continue

        if block["type"] == "paragraph":
            grouped = [block]
            total_chars = len(block.get("content", ""))
            cursor = index + 1
            while cursor < len(blocks) and len(grouped) < active_rules.paragraph_chunk_block_limit:
                candidate = blocks[cursor]
                if candidate["type"] != "paragraph" or candidate.get("section") != block.get("section"):
                    break
                candidate_chars = len(candidate.get("content", ""))
                if total_chars + candidate_chars > active_rules.paragraph_chunk_char_limit:
                    break
                grouped.append(candidate)
                total_chars += candidate_chars
                cursor += 1
            chunks.append(_make_chunk(grouped, len(chunks) + 1, source_format))
            index = cursor
            continue

        chunks.append(_make_chunk([block], len(chunks) + 1, source_format))
        index += 1

    return chunks


def serialize_for_embedding(payload: dict[str, Any]) -> str:
    if "blocks" in payload:
        return "\n\n".join(serialize_for_embedding(block) for block in payload["blocks"]).strip()

    block_type = payload["type"]
    section = payload.get("section")
    section_prefix = f"[{section}]\n" if section else ""

    if block_type == "title":
        return f"[{payload.get('content', section or '').strip()}]".strip()

    if block_type == "paragraph":
        return f"{section_prefix}{payload['content']}".strip()

    if block_type == "list":
        items = "\n".join(f"* {item}" for item in payload.get("items", []))
        return f"{section_prefix}{items}".strip()

    if block_type == "kv_table":
        lines = [section_prefix.rstrip()] if section_prefix else []
        for field, value in payload.get("fields", {}).items():
            if isinstance(value, list):
                lines.append(f"{field}:")
                lines.extend(f"* {item}" for item in value)
            else:
                lines.append(f"{field}: {value}")
        return "\n".join(line for line in lines if line).strip()

    if block_type == "row_table":
        lines = [section_prefix.rstrip()] if section_prefix else []
        header_rows = payload.get("header_rows", [])
        rows = payload.get("rows", [])
        for index, header_row in enumerate(header_rows, start=1):
            lines.append(f"header_row_{index}: " + " | ".join(header_row))
        for index, row in enumerate(rows, start=1):
            lines.append(f"row_{index}: " + " | ".join(row))
        return "\n".join(line for line in lines if line).strip()

    return f"{section_prefix}{payload.get('content', '')}".strip()


def render_blocks_as_markdown(blocks: list[dict[str, Any]]) -> str:
    markdown_parts: list[str] = []
    for block in blocks:
        block_type = block["type"]
        if block_type == "title":
            markdown_parts.append(f"## {block['content']}")
            continue
        if block_type == "paragraph":
            markdown_parts.append(block["content"])
            continue
        if block_type == "list":
            markdown_parts.append("\n".join(f"- {item}" for item in block.get("items", [])))
            continue
        if block_type == "kv_table":
            markdown_parts.append(serialize_for_embedding(block))
            continue
        if block_type == "row_table":
            markdown_parts.append(serialize_for_embedding(block))
            continue
        markdown_parts.append(block.get("content", ""))
    return "\n\n".join(part for part in markdown_parts if part).strip()


def _build_title_block(line: NormalizedLine, section: str | None) -> dict[str, Any]:
    return {
        "type": "title",
        "section": section,
        "content": line.text,
        "source_format": "hwp",
        "page": line.page,
        "line_start": line.line_index,
        "line_end": line.line_index,
    }


def _build_unknown_block(line: NormalizedLine, section: str | None) -> dict[str, Any]:
    return {
        "type": "unknown",
        "section": section,
        "content": line.text,
        "source_format": "hwp",
        "page": line.page,
        "line_start": line.line_index,
        "line_end": line.line_index,
    }


def _build_list_block(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    start_index: int,
    current_section: str | None,
    rules: HwpPostprocessRules,
) -> tuple[dict[str, Any], int]:
    items: list[str] = []
    list_lines: list[NormalizedLine] = []
    cursor = start_index
    baseline_indent = lines[start_index].indent

    while cursor < len(lines):
        classification = classifications[cursor]
        line = lines[cursor]
        if not classification.is_list_item:
            break
        if cursor != start_index and abs(line.indent - baseline_indent) > 4:
            break
        list_lines.append(line)
        items.append(_strip_bullet(line.text))
        cursor += 1

    block_type = "list" if len(items) >= rules.minimum_list_items else "paragraph"
    if block_type == "paragraph":
        return (
            {
                "type": "paragraph",
                "section": current_section,
                "content": " ".join(item for item in items if item),
                "source_format": "hwp",
                "page": list_lines[0].page,
                "line_start": list_lines[0].line_index,
                "line_end": list_lines[-1].line_index,
            },
            cursor,
        )

    return (
        {
            "type": "list",
            "section": current_section,
            "items": items,
            "source_format": "hwp",
            "page": list_lines[0].page,
            "line_start": list_lines[0].line_index,
            "line_end": list_lines[-1].line_index,
        },
        cursor,
    )


def _make_chunk(blocks: list[dict[str, Any]], sequence: int, source_format: str = "hwp") -> dict[str, Any]:
    unique_types = list(dict.fromkeys(block["type"] for block in blocks))
    section = next((block.get("section") for block in blocks if block.get("section")), None)
    serialized = serialize_for_embedding({"blocks": blocks})
    return {
        "chunk_id": f"{source_format}-chunk-{sequence}",
        "section": section,
        "block_types": unique_types,
        "source_format": source_format,
        "page": next((block.get("page") for block in blocks if block.get("page") is not None), None),
        "line_start": min(block["line_start"] for block in blocks),
        "line_end": max(block["line_end"] for block in blocks),
        "blocks": blocks,
        "serialized_text": serialized,
    }


def _consume_list_value(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    start_index: int,
) -> tuple[list[str], int]:
    items: list[str] = []
    cursor = start_index
    while cursor < len(lines) and classifications[cursor].is_list_item:
        items.append(_strip_bullet(lines[cursor].text))
        cursor += 1
    return items, cursor


def _consume_paragraph_value(
    lines: list[NormalizedLine],
    classifications: list[LineClassification],
    start_index: int,
    rules: HwpPostprocessRules,
) -> tuple[list[str], int]:
    values: list[str] = []
    cursor = start_index
    while cursor < len(lines):
        classification = classifications[cursor]
        if classification.is_title:
            break
        if cursor > start_index:
            if classification.is_list_item:
                break
            next_kv_block, _ = parse_kv_table(lines, classifications, cursor, None, rules)
            if next_kv_block is not None:
                break
        values.append(lines[cursor].text)
        cursor += 1
        if values and (_looks_like_value(values[-1]) or TERMINAL_PUNCTUATION_RE.search(values[-1])):
            if cursor >= len(lines) or classifications[cursor].is_title or classifications[cursor].is_list_item:
                break
    return values, cursor


def _is_noise_line(text: str, rules: HwpPostprocessRules) -> bool:
    if len(text) >= rules.noise_min_chars:
        return False
    if text in {"-", "*", "•", "·", "※"}:
        return False
    if text.isdigit():
        return False
    if text in rules.allowed_short_titles:
        return False
    return True


def _looks_like_title(text: str, rules: HwpPostprocessRules) -> bool:
    if text in rules.allowed_short_titles:
        return True
    if TITLE_NUMERIC_RE.match(text):
        return True
    if TITLE_CHAPTER_RE.match(text):
        return True
    if len(text) <= rules.max_title_chars and not _looks_like_list_item(text) and not _looks_like_value(text):
        if not TERMINAL_PUNCTUATION_RE.search(text) and not SENTENCE_ENDING_RE.search(text) and len(text.split()) <= 4:
            return True
    return False


def _looks_like_list_item(text: str) -> bool:
    return bool(LIST_BULLET_RE.match(text))


def _looks_like_kv_label(text: str, rules: HwpPostprocessRules) -> bool:
    normalized = _normalize_label_text(text)
    if _looks_like_list_item(normalized) or _looks_like_value(normalized):
        return False
    if SENTENCE_ENDING_RE.search(normalized):
        return False
    return len(normalized) <= rules.max_short_label_chars and len(normalized.split()) <= 3


def _looks_like_value(text: str) -> bool:
    if DATE_VALUE_RE.match(text):
        return True
    if NUMERIC_VALUE_RE.match(text):
        return True
    if re.search(r"\d", text) and len(text) <= 24:
        return True
    return False


def _looks_like_scalar_kv_value(text: str, rules: HwpPostprocessRules) -> bool:
    if _looks_like_value(text):
        return True
    if len(text) <= rules.max_title_chars and not _looks_like_kv_label(text, rules):
        return True
    return False


def _looks_like_tabular_label(text: str, rules: HwpPostprocessRules) -> bool:
    normalized = _normalize_label_text(text)
    if normalized.startswith("("):
        return False
    if _looks_like_list_item(normalized) or SENTENCE_ENDING_RE.search(normalized):
        return False
    if _looks_like_value(normalized) and not normalized.endswith("년"):
        return False
    return len(normalized) <= rules.max_short_label_chars


def _looks_like_row_table_value(text: str) -> bool:
    normalized = text.strip()
    if DATE_VALUE_RE.match(normalized):
        return True
    if NUMERIC_VALUE_RE.match(normalized):
        return True
    if re.match(r"^\d{1,2}\.\d{1,2}$", normalized):
        return True
    return False


def _has_repeated_label(label_lines: list[NormalizedLine]) -> bool:
    seen: set[str] = set()
    for line in label_lines:
        normalized = _normalize_label_text(line.text)
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def _build_row_table_headers(label_lines: list[NormalizedLine]) -> list[list[str]]:
    normalized = [_normalize_label_text(line.text) for line in label_lines]
    if len(normalized) >= 6 and len(set(normalized[-6:])) <= 2:
        return [normalized[:-6], normalized[-6:]]
    seen: dict[str, int] = {}
    split_index = len(normalized)
    for index, value in enumerate(normalized):
        if value in seen:
            split_index = index
            break
        seen[value] = index
    if split_index == len(normalized):
        return [normalized]
    return [normalized[:split_index], normalized[split_index:]]


def _looks_unknown(text: str) -> bool:
    return bool(re.fullmatch(r"[^\w가-힣]+", text))


def _should_merge_paragraph_lines(current_text: str, next_text: str, rules: HwpPostprocessRules) -> bool:
    if len(current_text) <= rules.short_paragraph_chars:
        return True
    if not TERMINAL_PUNCTUATION_RE.search(current_text):
        return True
    if len(next_text) <= rules.short_paragraph_chars:
        return True
    return False


def _join_paragraph_lines(lines: list[str]) -> str:
    return re.sub(r"\s{2,}", " ", " ".join(line.strip() for line in lines if line.strip())).strip()


def _strip_bullet(text: str) -> str:
    stripped = LIST_BULLET_RE.sub("", text, count=1).strip()
    if stripped.startswith("."):
        stripped = stripped[1:].strip()
    return stripped


def _normalize_label_text(text: str) -> str:
    tokens = text.split()
    if len(tokens) >= 2 and all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return text.strip()


def _assign_kv_fields(
    labels: list[NormalizedLine],
    scalar_values: list[NormalizedLine],
    list_items: list[str],
) -> dict[str, Any]:
    normalized_labels = [_normalize_label_text(label.text) for label in labels]
    scalar_texts = [value.text for value in scalar_values]
    fields: dict[str, Any] = {}

    if list_items:
        leading_value_count = min(len(scalar_texts), max(len(normalized_labels) - 1, 0))
        for index in range(leading_value_count):
            fields[normalized_labels[index]] = scalar_texts[index]

        if len(normalized_labels) > leading_value_count:
            last_label = normalized_labels[-1]
            fields[last_label] = list_items
        elif scalar_texts:
            fields[normalized_labels[-1]] = scalar_texts[-1]

        if len(scalar_texts) > leading_value_count and len(normalized_labels) > 0:
            remaining_scalars = scalar_texts[leading_value_count:]
            if normalized_labels[-1] in fields and isinstance(fields[normalized_labels[-1]], list):
                fields[normalized_labels[-1]] = remaining_scalars + fields[normalized_labels[-1]]
        return fields

    for index, label in enumerate(normalized_labels):
        if index < len(scalar_texts):
            fields[label] = scalar_texts[index]
        else:
            fields[label] = ""

    return fields
