from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict

from src.parsers.common.models import ParsedDocument


def parsed_document_to_dict(document: ParsedDocument) -> dict:
    return asdict(document)


def extract_markdown_sections(markdown: str) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if not match:
            continue
        sections.append(
            {
                "line_number": line_number,
                "level": len(match.group(1)),
                "title": match.group(2),
            }
        )
    return sections


def build_document_summary(document: ParsedDocument) -> dict[str, object]:
    element_counter: Counter[str] = Counter()
    total_text_characters = 0
    block_counter: Counter[str] = Counter(block["type"] for block in document.blocks)

    for page in document.pages:
        total_text_characters += page.text_length
        for element in page.elements:
            element_counter[element.element_type] += 1

    return {
        "page_count": len(document.pages),
        "element_counts": dict(element_counter),
        "section_count": len(document.sections),
        "issue_count": len(document.issues),
        "text_characters": total_text_characters,
        "block_count": len(document.blocks),
        "chunk_count": len(document.chunks),
        "block_type_counts": dict(block_counter),
    }
