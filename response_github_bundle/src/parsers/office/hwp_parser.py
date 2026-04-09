from __future__ import annotations

import re
import zlib
from pathlib import Path

import olefile

from src.parsers.common.models import DocumentClassification, DocumentElement, DocumentIssue, ParsedDocument, ParsedPage
from src.parsers.common.serialization import build_document_summary, extract_markdown_sections
from src.shared.io import iso_now, make_artifact_stem
from src.structure.hwp_postprocessing import HwpPostprocessRules, build_blocks, render_blocks_as_markdown, serialize_for_embedding


class HwpParser:
    parser_name = "ole-hwp-parser"
    postprocess_rules = HwpPostprocessRules()

    def parse(self, path: Path, classification: DocumentClassification) -> ParsedDocument:
        issues: list[DocumentIssue] = []
        extracted_text = ""
        extraction_method = ""
        stream_names: list[str] = []
        extraction_metadata: dict[str, object] = {}

        if not olefile.isOleFile(path):
            return self._build_failure_document(
                path,
                classification,
                issues=[
                    DocumentIssue(
                        code="invalid-ole-container",
                        message="The HWP file is not a readable OLE container.",
                        severity="warning",
                    )
                ],
            )

        ole = olefile.OleFileIO(path)
        try:
            stream_names = ["/".join(item) for item in ole.listdir()]

            body_text, body_metadata = self._extract_body_text(ole, issues)
            if body_text.strip():
                extracted_text = body_text
                extraction_method = "BodyTextRecords"
                extraction_metadata = body_metadata
            elif ole.exists("PrvText"):
                try:
                    prv_bytes = ole.openstream("PrvText").read()
                    extracted_text = self._decode_prv_text(prv_bytes)
                    extraction_method = "PrvText"
                    extraction_metadata = {
                        "text_record_count": 0,
                        "body_section_count": 0,
                        "body_paragraph_count": 0,
                    }
                    if extracted_text.strip():
                        issues.append(
                            DocumentIssue(
                                code="bodytext_unavailable_fallback_prvtext",
                                message="BodyText extraction was unavailable, so the parser fell back to PrvText preview text.",
                                severity="warning",
                            )
                        )
                except Exception as error:
                    issues.append(
                        DocumentIssue(
                            code="prvtext_read_failed",
                            message=f"Failed to read PrvText stream: {error}",
                            severity="warning",
                        )
                    )
        finally:
            ole.close()

        if not extracted_text.strip():
            return self._build_failure_document(
                path,
                classification,
                issues=issues
                + [
                    DocumentIssue(
                        code="hwp_text_unavailable",
                        message="OLE-based HWP extraction did not yield usable text.",
                        severity="warning",
                    )
                ],
                extra_metadata={"available_streams": stream_names},
            )

        postprocess_result = build_blocks(extracted_text, self.postprocess_rules)
        blocks = postprocess_result.blocks
        chunks = postprocess_result.chunks
        markdown = render_blocks_as_markdown(blocks)
        elements = [
            DocumentElement(
                element_id=f"p1-b{index}",
                element_type=block["type"],
                page_number=1,
                order=index,
                text=serialize_for_embedding(block),
                metadata={
                    "source": extraction_method,
                    "section": block.get("section"),
                    "line_start": block.get("line_start"),
                    "line_end": block.get("line_end"),
                },
            )
            for index, block in enumerate(blocks, start=1)
        ]

        parsed = ParsedDocument(
            document_id=make_artifact_stem(path),
            source_name=path.name,
            source_path=path.as_posix(),
            extension=path.suffix.lower(),
            status="parsed",
            parser_name=self.parser_name,
            classification=classification,
            metadata={
                "extraction_method": extraction_method,
                "available_streams": stream_names,
                "postprocess_logs": postprocess_result.logs,
                "rule_version": self.postprocess_rules.version,
                **extraction_metadata,
            },
            sections=extract_markdown_sections(markdown),
            pages=[
                ParsedPage(
                    page_number=1,
                    width=0.0,
                    height=0.0,
                    text_length=len(extracted_text),
                    elements=elements,
                    metadata={"synthetic_page": True},
                )
            ],
            markdown=markdown,
            blocks=blocks,
            chunks=chunks,
            issues=issues,
            created_at=iso_now(),
        )
        parsed.summary = build_document_summary(parsed)
        return parsed

    def _decode_prv_text(self, raw_bytes: bytes) -> str:
        for encoding in ("utf-16le", "utf-8", "cp949"):
            try:
                decoded = raw_bytes.decode(encoding)
                cleaned = self._normalize_text(decoded)
                if cleaned:
                    return cleaned
            except UnicodeDecodeError:
                continue
        return ""

    def _extract_body_text(
        self, ole: olefile.OleFileIO, issues: list[DocumentIssue]
    ) -> tuple[str, dict[str, object]]:
        paragraphs: list[str] = []
        body_section_count = 0
        text_record_count = 0

        for entry in ole.listdir():
            stream_name = "/".join(entry)
            if not stream_name.startswith("BodyText/Section"):
                continue

            body_section_count += 1
            try:
                raw_bytes = ole.openstream(entry).read()
                try:
                    raw_bytes = zlib.decompress(raw_bytes, -15)
                except zlib.error:
                    pass

                section_paragraphs, section_record_count = self._extract_text_records(raw_bytes)
                text_record_count += section_record_count
                paragraphs.extend(section_paragraphs)
            except Exception as error:
                issues.append(
                    DocumentIssue(
                        code="bodytext_section_failed",
                        message=f"Failed to extract {stream_name}: {error}",
                        severity="warning",
                    )
                )

        cleaned_paragraphs = [paragraph for paragraph in paragraphs if paragraph.strip()]
        return "\n\n".join(cleaned_paragraphs), {
            "body_section_count": body_section_count,
            "text_record_count": text_record_count,
            "body_paragraph_count": len(cleaned_paragraphs),
        }

    def _extract_text_records(self, raw_bytes: bytes) -> tuple[list[str], int]:
        paragraphs: list[str] = []
        pos = 0
        text_record_count = 0

        while pos + 4 <= len(raw_bytes):
            header = int.from_bytes(raw_bytes[pos : pos + 4], "little")
            tag_id = header & 0x3FF
            size = (header >> 20) & 0xFFF
            pos += 4

            if size == 0xFFF:
                if pos + 4 > len(raw_bytes):
                    break
                size = int.from_bytes(raw_bytes[pos : pos + 4], "little")
                pos += 4

            if pos + size > len(raw_bytes):
                break

            payload = raw_bytes[pos : pos + size]
            pos += size

            if tag_id != 67:
                continue

            text_record_count += 1
            paragraph = self._decode_para_text_record(payload)
            if paragraph:
                paragraphs.append(paragraph)

        return paragraphs, text_record_count

    def _decode_para_text_record(self, payload: bytes) -> str:
        values = [
            int.from_bytes(payload[index : index + 2], "little")
            for index in range(0, len(payload) - (len(payload) % 2), 2)
        ]

        characters: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]

            if value == 0:
                index += 1
                continue
            if value in (10, 13):
                characters.append("\n")
                index += 1
                continue
            if value == 9:
                characters.append("\t")
                index += 1
                continue
            if 0 < value < 32:
                # HWP para-text embeds extended control records inline.
                # Skip the full control payload when present instead of decoding it as text.
                index += 8 if index + 7 < len(values) else 1
                continue

            characters.append(chr(value))
            index += 1

        return self._normalize_text("".join(characters))

    def _normalize_text(self, text: str) -> str:
        cleaned = text.replace("\x00", " ")
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_failure_document(
        self,
        path: Path,
        classification: DocumentClassification,
        issues: list[DocumentIssue],
        extra_metadata: dict | None = None,
    ) -> ParsedDocument:
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
                "source_location": path.as_posix(),
                **(extra_metadata or {}),
            },
            markdown=(
                f"{path.name} 에서 직접 추출 가능한 HWP 텍스트를 확보하지 못했다.\n\n"
                f"원본 위치: `{path.as_posix()}`\n"
                "권장 조치: PDF로 변환한 뒤 다시 파싱."
            ),
            issues=issues,
            created_at=iso_now(),
        )
        parsed.summary = build_document_summary(parsed)
        return parsed
