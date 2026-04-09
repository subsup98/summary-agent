from __future__ import annotations

from pathlib import Path

from src.parsers.common.models import DocumentClassification, DocumentIssue, ParsedDocument
from src.parsers.common.serialization import build_document_summary
from src.shared.io import iso_now, make_artifact_stem


class ConversionFallbackParser:
    parser_name = "conversion-fallback"

    def parse(self, path: Path, classification: DocumentClassification) -> ParsedDocument:
        parsed = ParsedDocument(
            document_id=make_artifact_stem(path),
            source_name=path.name,
            source_path=path.as_posix(),
            extension=path.suffix.lower(),
            status="requires_conversion",
            parser_name=self.parser_name,
            classification=classification,
            metadata={
                "recommended_conversion_target": "pdf",
                "recommended_output_directory": "data/converted",
            },
            markdown=(
                f"{path.name} 은(는) 현재 직접 파싱 대상이 아니다.\n\n"
                "권장 조치: PDF로 변환 후 `data/converted` 또는 `data/raw/pdf`에서 다시 처리."
            ),
            issues=[
                DocumentIssue(
                    code="conversion_required",
                    message="Direct parsing is not enabled for this format in the current MVP.",
                    severity="warning",
                )
            ],
            created_at=iso_now(),
        )
        parsed.summary = build_document_summary(parsed)
        return parsed
