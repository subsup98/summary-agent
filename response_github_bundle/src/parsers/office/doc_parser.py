from __future__ import annotations

import re
import subprocess
from pathlib import Path

import olefile

from src.parsers.common.models import DocumentClassification, DocumentElement, DocumentIssue, ParsedDocument, ParsedPage
from src.parsers.common.serialization import build_document_summary, extract_markdown_sections
from src.shared.io import iso_now, make_artifact_stem
from src.structure.hwp_postprocessing import HwpPostprocessRules, build_blocks, render_blocks_as_markdown, serialize_for_embedding


NOISE_PATTERNS = (
    "MERGEFORMAT",
    "Default Paragraph Font",
    "Table Normal",
    "theme/",
    ".xml",
    "HYPERLINK",
    "pixel",
    "Times New Roman",
    "Malgun Gothic",
    "HYGothic",
    "HYSinMyeongJo",
    "CLP000",
)


class DocParser:
    parser_name = "doc-best-effort-parser"
    postprocess_rules = HwpPostprocessRules()

    def parse(self, path: Path, classification: DocumentClassification) -> ParsedDocument:
        issues: list[DocumentIssue] = []
        attempted_methods: list[str] = []

        com_text = self._try_powershell_word(path, issues)
        attempted_methods.append("powershell-word-com")
        if com_text:
            return self._build_parsed_document(path, classification, com_text, "powershell-word-com", issues)

        ole_text = self._try_ole_extraction(path, issues)
        attempted_methods.append("ole-best-effort")
        if ole_text:
            return self._build_parsed_document(path, classification, ole_text, "ole-best-effort", issues)

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
                "attempted_methods": attempted_methods,
            },
            markdown=(
                f"{path.name} 에서 신뢰 가능한 DOC 본문 추출을 확보하지 못했다.\n\n"
                f"원본 위치: `{path.as_posix()}`\n"
                "시도 경로: PowerShell Word COM, OLE best-effort\n"
                "권장 조치: PDF 또는 DOCX로 변환 후 다시 파싱."
            ),
            issues=issues
            + [
                DocumentIssue(
                    code="doc_text_unavailable",
                    message="Direct DOC extraction was attempted but did not yield reliable text.",
                    severity="warning",
                )
            ],
            created_at=iso_now(),
        )
        parsed.summary = build_document_summary(parsed)
        return parsed

    def _try_powershell_word(self, path: Path, issues: list[DocumentIssue]) -> str:
        resolved_path = str(path.resolve()).replace("'", "''")
        script = (
            "$ErrorActionPreference = 'Stop'\n"
            "$word = $null\n"
            "$doc = $null\n"
            "try {\n"
            "  $word = New-Object -ComObject Word.Application\n"
            "  $word.Visible = $false\n"
            "  $word.DisplayAlerts = 0\n"
            f"  $doc = $word.Documents.Open('{resolved_path}', $false, $true)\n"
            "  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
            "  $doc.Content.Text\n"
            "} catch {\n"
            "  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
            "  Write-Output $_.Exception.Message\n"
            "  exit 1\n"
            "} finally {\n"
            "  if ($doc -ne $null) { $doc.Close([ref]$false) }\n"
            "  if ($word -ne $null) { $word.Quit() }\n"
            "}\n"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode == 0:
            text = self._normalize_text(result.stdout)
            if len(text) >= 100:
                return text
            issues.append(
                DocumentIssue(
                    code="word_com_short_text",
                    message="Word COM opened the file but returned insufficient text.",
                    severity="warning",
                )
            )
            return ""

        issues.append(
            DocumentIssue(
                code="word_com_unavailable",
                message=f"PowerShell Word COM extraction failed: {result.stderr.strip() or result.stdout.strip()}",
                severity="warning",
            )
        )
        return ""

    def _try_ole_extraction(self, path: Path, issues: list[DocumentIssue]) -> str:
        if not olefile.isOleFile(path):
            issues.append(
                DocumentIssue(
                    code="doc_invalid_ole",
                    message="The DOC file is not a readable OLE container.",
                    severity="warning",
                )
            )
            return ""

        candidates: list[str] = []
        ole = olefile.OleFileIO(path)
        try:
            for stream_name in ("WordDocument", "1Table", "0Table", "Data"):
                if not ole.exists(stream_name):
                    continue
                stream_bytes = ole.openstream(stream_name).read()
                candidates.extend(self._extract_text_candidates(stream_bytes))
        finally:
            ole.close()

        filtered: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = self._normalize_text(candidate)
            if not self._is_useful_candidate(normalized):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(normalized)

        combined = "\n\n".join(filtered[:80]).strip()
        if len(combined) >= 120:
            return combined

        issues.append(
            DocumentIssue(
                code="doc_ole_insufficient_text",
                message="OLE best-effort extraction did not produce enough trustworthy text.",
                severity="warning",
            )
        )
        return ""

    def _extract_text_candidates(self, stream_bytes: bytes) -> list[str]:
        candidates: list[str] = []
        utf16_text = stream_bytes.decode("utf-16le", errors="ignore")
        candidates.extend(re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9\s\-\.,:%()/]{10,}", utf16_text))

        cp949_text = stream_bytes.decode("cp949", errors="ignore")
        candidates.extend(re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9\s\-\.,:%()/]{10,}", cp949_text))
        return candidates

    def _is_useful_candidate(self, text: str) -> bool:
        if len(text) < 10:
            return False
        if any(noise in text for noise in NOISE_PATTERNS):
            return False
        if len(set(text)) <= 3:
            return False

        hangul_count = len(re.findall(r"[가-힣]", text))
        alpha_count = len(re.findall(r"[A-Za-z]", text))
        digit_count = len(re.findall(r"\d", text))
        return (hangul_count + alpha_count + digit_count) >= max(8, int(len(text) * 0.35))

    def _normalize_text(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\x07", " ").replace("\x0b", "\n")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_parsed_document(
        self,
        path: Path,
        classification: DocumentClassification,
        text: str,
        extraction_method: str,
        issues: list[DocumentIssue],
    ) -> ParsedDocument:
        postprocess_result = build_blocks(text, self.postprocess_rules, source_format="doc")
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
                "postprocess_logs": postprocess_result.logs,
                "rule_version": self.postprocess_rules.version,
            },
            sections=extract_markdown_sections(markdown),
            pages=[
                ParsedPage(
                    page_number=1,
                    width=0.0,
                    height=0.0,
                    text_length=len(text),
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
