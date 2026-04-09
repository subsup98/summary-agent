from __future__ import annotations

import re
from pathlib import Path

from src.parsers.common.models import DocumentClassification, DocumentElement, ParsedDocument, ParsedPage
from src.parsers.common.serialization import build_document_summary, extract_markdown_sections
from src.shared.io import iso_now, make_artifact_stem, read_text_with_fallback


class TextParser:
    parser_name = "plain-text-parser"

    def parse(self, path: Path, classification: DocumentClassification) -> ParsedDocument:
        raw_text, encoding = read_text_with_fallback(path)
        normalized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized_text) if part.strip()]

        elements = [
            DocumentElement(
                element_id=f"p1-e{index}",
                element_type="text",
                page_number=1,
                order=index,
                text=paragraph,
                metadata={"source": "txt-paragraph"},
            )
            for index, paragraph in enumerate(paragraphs, start=1)
        ]

        markdown = "\n\n".join(paragraphs)
        parsed = ParsedDocument(
            document_id=make_artifact_stem(path),
            source_name=path.name,
            source_path=path.as_posix(),
            extension=path.suffix.lower(),
            status="parsed",
            parser_name=self.parser_name,
            classification=classification,
            metadata={"encoding": encoding},
            sections=extract_markdown_sections(markdown),
            pages=[
                ParsedPage(
                    page_number=1,
                    width=0.0,
                    height=0.0,
                    text_length=len(normalized_text.strip()),
                    elements=elements,
                    metadata={"synthetic_page": True},
                )
            ],
            markdown=markdown,
            created_at=iso_now(),
        )
        parsed.summary = build_document_summary(parsed)
        return parsed
