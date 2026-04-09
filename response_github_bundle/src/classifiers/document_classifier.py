from __future__ import annotations

from pathlib import Path

import fitz

from src.parsers.common.models import DocumentClassification


def classify_document(path: Path) -> DocumentClassification:
    extension = path.suffix.lower()
    size_bytes = path.stat().st_size
    base_metadata = {
        "extension": extension,
        "size_bytes": size_bytes,
    }

    if extension == ".pdf":
        return _classify_pdf(path, base_metadata)
    if extension == ".docx":
        return DocumentClassification(
            document_type="docx",
            parser_route="docx",
            confidence=0.95,
            notes=["direct-docx-parser"],
            metadata=base_metadata,
        )
    if extension == ".txt":
        return DocumentClassification(
            document_type="txt",
            parser_route="txt",
            confidence=0.98,
            notes=["direct-text-parser"],
            metadata=base_metadata,
        )
    if extension == ".doc":
        return DocumentClassification(
            document_type="doc",
            parser_route="doc",
            confidence=0.75,
            recommended_action="convert_to_pdf",
            notes=["legacy-doc-best-effort-parser"],
            metadata=base_metadata,
        )
    if extension == ".hwp":
        return DocumentClassification(
            document_type="hwp",
            parser_route="hwp",
            confidence=0.82,
            recommended_action="convert_to_pdf",
            notes=["ole-based-hwp-parser"],
            metadata=base_metadata,
        )

    return DocumentClassification(
        document_type="unsupported",
        parser_route="conversion_fallback",
        confidence=0.1,
        recommended_action="unsupported-format",
        notes=["unsupported-extension"],
        metadata=base_metadata,
    )


def _classify_pdf(path: Path, base_metadata: dict[str, object]) -> DocumentClassification:
    document = fitz.open(path)
    try:
        page_count = document.page_count
        metadata = document.metadata or {}
        creator = (metadata.get("creator") or "").strip()
        producer = (metadata.get("producer") or "").strip()

        pages_with_text = 0
        pages_with_images = 0
        total_text_characters = 0

        for page in document:
            page_text = page.get_text("text", sort=True).strip()
            if len(page_text) >= 30:
                pages_with_text += 1
            total_text_characters += len(page_text)

            page_dict = page.get_text("dict", sort=True)
            image_blocks = sum(1 for block in page_dict.get("blocks", []) if block.get("type") == 1)
            if image_blocks > 0:
                pages_with_images += 1

        text_layer_present = pages_with_text > 0

        if pages_with_text == 0 and pages_with_images > 0:
            pdf_kind = "scanned"
            confidence = 0.85
        elif pages_with_text == page_count:
            pdf_kind = "digital"
            confidence = 0.92
        else:
            pdf_kind = "mixed"
            confidence = 0.78

        producer_lower = producer.lower()
        creator_lower = creator.lower()
        if "powerpoint" in producer_lower or "ppt" in producer_lower:
            origin_hint = "powerpoint-generated"
            pdf_parser_strategy = "structtree-actualtext"
        elif producer_lower:
            origin_hint = "other-generated"
            pdf_parser_strategy = "pymupdf4llm"
        elif "acrobat" in creator_lower or "adobe" in creator_lower:
            origin_hint = "acrobat-generated"
            pdf_parser_strategy = "pymupdf4llm"
        elif creator_lower:
            origin_hint = "other-generated"
            pdf_parser_strategy = "pymupdf4llm"
        else:
            origin_hint = "unknown"
            pdf_parser_strategy = "pymupdf4llm"

        notes = [
            "pdf-structure-parser",
            f"pages-with-text={pages_with_text}",
            f"pages-with-images={pages_with_images}",
            f"producer-strategy={pdf_parser_strategy}",
        ]

        return DocumentClassification(
            document_type="pdf",
            parser_route="pdf",
            confidence=confidence,
            pdf_kind=pdf_kind,
            origin_hint=origin_hint,
            pdf_parser_strategy=pdf_parser_strategy,
            text_layer_present=text_layer_present,
            notes=notes,
            metadata={
                **base_metadata,
                "page_count": page_count,
                "creator": creator,
                "producer": producer,
                "pages_with_text": pages_with_text,
                "pages_with_images": pages_with_images,
                "total_text_characters": total_text_characters,
            },
        )
    finally:
        document.close()
