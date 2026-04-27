from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_MAX_WORKERS_HARD_LIMIT = min(os.cpu_count() or 4, 8)

from src.classifiers.document_classifier import classify_document
from src.ingest.document_loader import DocumentLoader
from src.parsers.common.serialization import extract_markdown_sections, parsed_document_to_dict
from src.parsers.office.doc_parser import DocParser
from src.parsers.office.docx_parser import DocxParser
from src.parsers.office.fallback_parser import ConversionFallbackParser
from src.parsers.office.hwp_parser import HwpParser
from src.parsers.pdf.pdf_parser import PdfParser
from src.parsers.text.text_parser import TextParser
from src.indexing.chunking import build_semantic_chunks
from src.indexing.embedding_backends import resolve_embedding_backend
from src.retrieval.document_summary import build_page_summaries, ensure_basic_summary
from src.shared.io import ensure_directory, iso_now, make_artifact_stem, write_json, write_text
from src.shared.office_pdf_converter import convert_office_source_to_pdf_with_diagnostics
from src.shared.versioning import load_version_info
from src.ui.pdf_overlay_report import PdfOverlaySiteBuilder
from src.ui.parsing_review_report import ParsingReviewSiteBuilder

INTERNAL_MARKER_LINE_PATTERNS = (
    re.compile(r"\[financial fact table\]", re.IGNORECASE),
    re.compile(r"\[row_path\]", re.IGNORECASE),
)
UNIT_LINE_PATTERN = re.compile(r"^\(Unit:\s*.*\)$", re.IGNORECASE)
ALIGNMENT_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+", re.UNICODE)


@dataclass
class ParsingPipelineConfig:
    source_root: Path
    interim_root: Path
    structured_root: Path
    outputs_root: Path
    comparisons_root: Path | None = None
    reports_root: Path | None = None
    enable_omitted_picture_ocr: bool = True
    on_document_ready: Callable[[dict[str, Any]], None] | None = field(default=None, compare=False)
    max_workers: int = 4


class ParsingPipeline:
    def __init__(self, config: ParsingPipelineConfig) -> None:
        self.config = config
        self.loader = DocumentLoader()
        self.embedding_backend = resolve_embedding_backend()
        self.version_info = load_version_info()
        self._ensure_directories()

    def _make_parsers(self) -> dict[str, Any]:
        """각 워커 스레드가 독립적인 파서 인스턴스를 사용하도록 매번 새로 생성."""
        return {
            "pdf": PdfParser(enable_omitted_picture_ocr=self.config.enable_omitted_picture_ocr),
            "docx": DocxParser(),
            "doc": DocParser(),
            "hwp": HwpParser(),
            "txt": TextParser(),
            "conversion_fallback": ConversionFallbackParser(),
        }

    def _sanitize_markdown_for_chunking(self, markdown: str) -> str:
        cleaned_lines: list[str] = []
        skip_unit_line = False

        for line in str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = line.strip()
            if any(pattern.search(stripped) for pattern in INTERNAL_MARKER_LINE_PATTERNS):
                skip_unit_line = True
                continue
            if skip_unit_line and UNIT_LINE_PATTERN.match(stripped):
                skip_unit_line = False
                continue
            skip_unit_line = False
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _build_supporting_pdf_assets(
        self,
        *,
        source_path: Path,
        artifact_stem: str,
        parsers: dict[str, Any],
    ) -> dict[str, Any] | None:
        converted_pdf_path = self.config.interim_root / "converted_pdf" / f"{artifact_stem}.pdf"
        conversion_started_perf = time.perf_counter()
        conversion_diagnostics = convert_office_source_to_pdf_with_diagnostics(source_path, converted_pdf_path)
        conversion: dict[str, Any] = {
            "attempted": True,
            "source_document_type": source_path.suffix.lower().lstrip("."),
            "source_extension": source_path.suffix.lower(),
            "converted_pdf_path": converted_pdf_path.as_posix(),
            "succeeded": bool(conversion_diagnostics.get("succeeded")) and converted_pdf_path.exists(),
            "elapsed_seconds": round(time.perf_counter() - conversion_started_perf, 3),
            "diagnostics": conversion_diagnostics,
        }
        if not conversion["succeeded"]:
            return {"conversion": conversion}

        try:
            converted_classification = classify_document(converted_pdf_path)
            pdf_document = parsers["pdf"].parse(converted_pdf_path, converted_classification)
            pdf_payload = parsed_document_to_dict(pdf_document)
        except Exception as error:
            return {
                "conversion": {
                    **conversion,
                    "asset_extraction_error": str(error),
                }
            }
        return {
            "conversion": {
                **conversion,
                "pdf_parser_name": pdf_document.parser_name,
                "pdf_classification": converted_classification.__dict__,
                "asset_page_count": len(pdf_payload.get("pages") or []),
            },
            "asset_markdown": str(pdf_payload.get("markdown") or ""),
            "asset_pages": pdf_payload.get("pages") or [],
            "asset_parser_name": pdf_document.parser_name,
            "asset_classification": converted_classification.__dict__,
            "asset_metadata": pdf_payload.get("metadata") or {},
            "asset_summary": pdf_payload.get("summary") or {},
        }

    def _extract_supporting_pdf_table_blocks(self, markdown: str) -> list[str]:
        lines = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[str] = []
        current: list[str] = []
        pending_context: list[str] = []
        in_table = False

        def flush_current() -> None:
            nonlocal current, in_table
            if not current:
                return
            block = "\n".join(line.rstrip() for line in current).strip()
            if block:
                blocks.append(block)
            current = []
            in_table = False

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            lowered = stripped.casefold()

            if not stripped:
                flush_current()
                pending_context = []
                continue

            if lowered.startswith("## [표") or lowered.startswith("## [table"):
                flush_current()
                pending_context = [line]
                continue

            if stripped.startswith("(단위:") and pending_context:
                pending_context.append(line)
                continue

            if stripped.startswith("|"):
                if not in_table:
                    flush_current()
                    current = list(pending_context)
                    pending_context = []
                    in_table = True
                current.append(line)
                continue

            if "[financial fact table]" in lowered:
                flush_current()
                current = list(pending_context)
                current.append(line)
                pending_context = []
                continue

            if "[row_path]" in lowered:
                if not current:
                    current = list(pending_context)
                    pending_context = []
                current.append(line)
                continue

            if current:
                flush_current()
            if lowered.startswith("## ") or stripped.startswith("# Page"):
                pending_context = []

        flush_current()
        return blocks

    def _merge_office_markdown_with_supporting_pdf(
        self,
        payload: dict[str, Any],
        supporting_pdf_assets: dict[str, Any] | None,
        classification: Any,
    ) -> dict[str, Any]:
        if classification.document_type not in {"doc", "hwp"}:
            return payload
        if not isinstance(supporting_pdf_assets, dict):
            return payload

        base_markdown = str(payload.get("markdown") or "").strip()
        asset_markdown = str(supporting_pdf_assets.get("asset_markdown") or "").strip()
        if not asset_markdown:
            return payload

        table_blocks = self._extract_supporting_pdf_table_blocks(asset_markdown)
        if not table_blocks:
            return payload

        existing_key = re.sub(r"\s+", " ", base_markdown).strip().casefold()
        unique_blocks: list[str] = []
        for block in table_blocks:
            normalized_block = re.sub(r"\s+", " ", block).strip().casefold()
            if normalized_block and normalized_block not in existing_key:
                unique_blocks.append(block.strip())

        if not unique_blocks:
            return payload

        merged_markdown = base_markdown
        if merged_markdown:
            merged_markdown += "\n\n"
        merged_markdown += "## PDF 보강 표 구조\n\n" + "\n\n".join(unique_blocks)

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        supporting_metadata = metadata.get("supporting_pdf") if isinstance(metadata.get("supporting_pdf"), dict) else {}
        supporting_metadata = {
            **supporting_metadata,
            "hybrid_markdown_applied": True,
            "hybrid_table_block_count": len(unique_blocks),
        }

        payload["markdown"] = merged_markdown.strip()
        payload["sections"] = extract_markdown_sections(payload["markdown"])
        payload["metadata"] = {
            **metadata,
            "supporting_pdf": supporting_metadata,
        }
        return payload

    def _format_alignment_text(self, value: object) -> str:
        text = str(value or "")
        return re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip().lower()

    def _alignment_tokens(self, value: object) -> set[str]:
        return {token for token in ALIGNMENT_TOKEN_RE.findall(self._format_alignment_text(value)) if token}

    def _build_asset_page_lookup_texts(self, asset_pages: list[dict[str, Any]]) -> list[tuple[int, str, set[str]]]:
        collected: list[tuple[int, str, set[str]]] = []
        for page in asset_pages:
            if not isinstance(page, dict):
                continue
            try:
                page_number = int(page.get("page_number") or 0)
            except (TypeError, ValueError):
                page_number = 0
            if page_number <= 0:
                continue

            parts: list[str] = []
            page_text = self._format_alignment_text(page.get("text") or "")
            if page_text:
                parts.append(page_text)

            elements = page.get("elements") if isinstance(page.get("elements"), list) else []
            for element in elements:
                if not isinstance(element, dict):
                    continue
                metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
                for candidate in (
                    element.get("text"),
                    element.get("markdown"),
                    metadata.get("mcid_text"),
                ):
                    normalized = self._format_alignment_text(candidate)
                    if normalized:
                        parts.append(normalized)

            combined = self._format_alignment_text("\n".join(parts))
            tokens = self._alignment_tokens(combined)
            if combined and tokens:
                collected.append((page_number, combined, tokens))
        return collected

    def _score_asset_page_alignment(
        self,
        *,
        chunk_text: str,
        chunk_tokens: set[str],
        section_hint: str,
        page_text: str,
        page_tokens: set[str],
    ) -> tuple[float, int]:
        shared_tokens = chunk_tokens & page_tokens
        score = float(len(shared_tokens) * 3)
        normalized_section = self._format_alignment_text(section_hint)
        if chunk_text and chunk_text in page_text:
            score += 1200.0
        elif chunk_text and page_text in chunk_text:
            score += 700.0
        if normalized_section:
            if normalized_section in page_text:
                score += 120.0
            else:
                score += len(self._alignment_tokens(normalized_section) & page_tokens) * 8.0
        numeric_chunk = {token for token in chunk_tokens if any(char.isdigit() for char in token)}
        if numeric_chunk:
            score += len(numeric_chunk & page_tokens) * 6.0
        return score, len(shared_tokens)

    def _annotate_chunks_with_asset_pages(self, payload: dict[str, Any]) -> None:
        asset_pages = payload.get("asset_pages") if isinstance(payload.get("asset_pages"), list) else []
        if not asset_pages:
            return

        asset_page_texts = self._build_asset_page_lookup_texts(asset_pages)
        if not asset_page_texts:
            return

        for field_name in ("chunks", "semantic_chunks"):
            chunks = payload.get(field_name)
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                raw_text = chunk.get("text") or chunk.get("serialized_text") or ""
                chunk_text = self._format_alignment_text(raw_text)
                chunk_tokens = self._alignment_tokens(chunk_text)
                if not chunk_text or not chunk_tokens:
                    continue
                section_hint = str(chunk.get("section_hint") or chunk.get("section") or "")
                best_page = 0
                best_score = 0.0
                best_shared_tokens = 0
                for page_number, page_text, page_tokens in asset_page_texts:
                    score, shared_tokens = self._score_asset_page_alignment(
                        chunk_text=chunk_text,
                        chunk_tokens=chunk_tokens,
                        section_hint=section_hint,
                        page_text=page_text,
                        page_tokens=page_tokens,
                    )
                    if score > best_score:
                        best_score = score
                        best_page = page_number
                        best_shared_tokens = shared_tokens
                if best_page <= 0:
                    continue
                if best_shared_tokens <= 0 and best_score < 100.0:
                    continue
                chunk["asset_page_number"] = best_page
                chunk["asset_page_score"] = round(best_score, 3)
                chunk["asset_page_shared_tokens"] = best_shared_tokens

    def _process_document(self, path: Path, parsers: dict[str, Any]) -> dict[str, Any]:
        """단일 문서를 분류 → 파싱 → 요약 → 청킹 → 저장 → 콜백 순서로 처리한다."""
        document_started_perf = time.perf_counter()
        artifact_stem = make_artifact_stem(path)
        relative_path = path.relative_to(self.config.source_root).as_posix()

        classify_started_perf = time.perf_counter()
        classification = classify_document(path)
        classify_seconds = round(time.perf_counter() - classify_started_perf, 3)

        write_json(
            self.config.interim_root / "classification" / f"{artifact_stem}.json",
            {
                "source_path": relative_path,
                "classification": classification.__dict__,
            },
        )

        parser = parsers.get(classification.parser_route, parsers["conversion_fallback"])
        supporting_pdf_assets: dict[str, Any] | None = None

        try:
            parser_started_perf = time.perf_counter()
            parsed_document = parser.parse(path, classification)
            parser_seconds = round(time.perf_counter() - parser_started_perf, 3)
        except Exception as error:
            failure_payload = {
                "source_path": relative_path,
                "status": "failed",
                "error": str(error),
                "parser_route": classification.parser_route,
                "timings": {
                    "classification_seconds": classify_seconds,
                    "parser_seconds": None,
                    "basic_summary_seconds": 0.0,
                    "semantic_chunk_seconds": 0.0,
                    "persist_seconds": 0.0,
                    "total_seconds": round(time.perf_counter() - document_started_perf, 3),
                },
            }
            write_json(self.config.outputs_root / "logs" / f"{artifact_stem}.error.json", failure_payload)
            return {"outcome": "failed", "record": failure_payload, "classify_seconds": classify_seconds}

        if parsed_document.status == "parsed" and classification.document_type in {"doc", "hwp"}:
            supporting_pdf_assets = self._build_supporting_pdf_assets(
                source_path=path,
                artifact_stem=artifact_stem,
                parsers=parsers,
            )
            if supporting_pdf_assets:
                parsed_document.metadata = {
                    **dict(parsed_document.metadata or {}),
                    "supporting_pdf": {
                        key: value
                        for key, value in supporting_pdf_assets.items()
                        if key not in {"asset_pages"}
                    },
                }

        basic_summary_started_perf = time.perf_counter()
        payload = ensure_basic_summary(parsed_document_to_dict(parsed_document))
        basic_summary_seconds = round(time.perf_counter() - basic_summary_started_perf, 3)

        if supporting_pdf_assets:
            payload = self._merge_office_markdown_with_supporting_pdf(payload, supporting_pdf_assets, classification)

        if supporting_pdf_assets:
            payload["asset_pages"] = supporting_pdf_assets.get("asset_pages") or []
            payload["asset_source"] = {
                key: value
                for key, value in supporting_pdf_assets.items()
                if key not in {"asset_pages", "asset_markdown"}
            }
            self._annotate_chunks_with_asset_pages(payload)

        markdown = str(payload.get("markdown") or "")
        chunking_markdown = self._sanitize_markdown_for_chunking(markdown)
        payload["page_summaries"] = build_page_summaries(payload)
        if payload.get("semantic_chunks"):
            payload["semantic_chunks"] = []
        semantic_chunk_seconds = 0.0
        if chunking_markdown.strip() and not payload.get("semantic_chunks"):
            semantic_chunk_started_perf = time.perf_counter()
            payload["semantic_chunks"] = build_semantic_chunks(chunking_markdown, embeddings=self.embedding_backend)
            semantic_chunk_seconds = round(time.perf_counter() - semantic_chunk_started_perf, 3)

        record: dict[str, Any] = {
            "source_path": relative_path,
            "status": parsed_document.status,
            "parser_name": parsed_document.parser_name,
            "document_type": parsed_document.classification.document_type,
            "pdf_kind": parsed_document.classification.pdf_kind,
            "issue_count": len(parsed_document.issues),
            "conversion": (supporting_pdf_assets or {}).get("conversion"),
        }

        persist_started_perf = time.perf_counter()
        if parsed_document.status == "parsed":
            structured_path = self.config.structured_root / "documents" / f"{artifact_stem}.json"
            output_json_path = self.config.outputs_root / "json" / f"{artifact_stem}.json"
            output_markdown_path = self.config.outputs_root / "markdown" / f"{artifact_stem}.md"

            write_json(structured_path, payload)
            write_json(output_json_path, payload)
            write_text(output_markdown_path, markdown.rstrip() + "\n")

            record["structured_path"] = structured_path.as_posix()
            record["markdown_path"] = output_markdown_path.as_posix()
            outcome = "parsed"
        else:
            fallback_path = self.config.interim_root / "fallbacks" / f"{artifact_stem}.json"
            write_json(fallback_path, payload)
            record["fallback_path"] = fallback_path.as_posix()
            outcome = "fallback"

        persist_seconds = round(time.perf_counter() - persist_started_perf, 3)
        record["timings"] = {
            "classification_seconds": classify_seconds,
            "parser_seconds": parser_seconds,
            "basic_summary_seconds": basic_summary_seconds,
            "semantic_chunk_seconds": semantic_chunk_seconds,
            "persist_seconds": persist_seconds,
            "total_seconds": round(time.perf_counter() - document_started_perf, 3),
        }

        # 파싱 완료 즉시 콜백 호출 (벡터 인덱싱 트리거)
        if self.config.on_document_ready is not None:
            self.config.on_document_ready(payload)

        return {
            "outcome": outcome,
            "record": record,
            "classify_seconds": classify_seconds,
            "parser_seconds": parser_seconds,
            "basic_summary_seconds": basic_summary_seconds,
            "semantic_chunk_seconds": semantic_chunk_seconds,
            "persist_seconds": persist_seconds,
        }

    def run(self) -> dict[str, Any]:
        started_at = iso_now()
        run_started_perf = time.perf_counter()
        discovery_started_perf = time.perf_counter()
        documents = self.loader.discover_documents(self.config.source_root)
        discovery_seconds = round(time.perf_counter() - discovery_started_perf, 3)
        summary: dict[str, Any] = {
            "version": self.version_info["version"],
            "scope": self.version_info.get("scope"),
            "started_at": started_at,
            "source_root": self.config.source_root.as_posix(),
            "total_documents": len(documents),
            "parsed_documents": 0,
            "fallback_documents": 0,
            "failed_documents": 0,
            "documents": [],
            "timings": {
                "document_discovery_seconds": discovery_seconds,
                "document_pipeline_seconds": 0.0,
                "classification_seconds": 0.0,
                "parser_seconds": 0.0,
                "basic_summary_seconds": 0.0,
                "semantic_chunk_seconds": 0.0,
                "persist_seconds": 0.0,
                "overlay_report_seconds": 0.0,
                "parsing_review_seconds": 0.0,
                "report_build_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }

        summary_lock = threading.Lock()
        max_workers = min(self.config.max_workers, _MAX_WORKERS_HARD_LIMIT, len(documents)) if documents else 1

        def process_one(path: Path) -> None:
            parsers = self._make_parsers()
            result = self._process_document(path, parsers)
            with summary_lock:
                outcome = result["outcome"]
                if outcome == "failed":
                    summary["failed_documents"] += 1
                elif outcome == "parsed":
                    summary["parsed_documents"] += 1
                else:
                    summary["fallback_documents"] += 1
                summary["documents"].append(result["record"])
                summary["timings"]["classification_seconds"] += result.get("classify_seconds", 0.0)
                summary["timings"]["parser_seconds"] += result.get("parser_seconds", 0.0)
                summary["timings"]["basic_summary_seconds"] += result.get("basic_summary_seconds", 0.0)
                summary["timings"]["semantic_chunk_seconds"] += result.get("semantic_chunk_seconds", 0.0)
                summary["timings"]["persist_seconds"] += result.get("persist_seconds", 0.0)

        document_pipeline_started_perf = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, path) for path in documents]
            for future in as_completed(futures):
                future.result()  # 예외 전파

        summary["timings"]["document_pipeline_seconds"] = round(time.perf_counter() - document_pipeline_started_perf, 3)

        summary["status"] = "completed"

        if self.config.reports_root:
            overlay_started_perf = time.perf_counter()
            overlay_manifest = PdfOverlaySiteBuilder(
                parsed_root=self.config.outputs_root / "json",
                site_root=self.config.reports_root / "pdf_overlay_review",
            ).build()
            summary["timings"]["overlay_report_seconds"] = round(time.perf_counter() - overlay_started_perf, 3)
            summary["overlay_index_path"] = overlay_manifest["index_path"]
            summary["overlay_manifest_path"] = (self.config.reports_root / "pdf_overlay_review" / "manifest.json").as_posix()
            summary["overlay_document_count"] = overlay_manifest["document_count"]

        if self.config.reports_root:
            parsing_review_started_perf = time.perf_counter()
            parsing_review_manifest = ParsingReviewSiteBuilder(
                parsed_root=self.config.outputs_root / "json",
                site_root=self.config.reports_root / "parsing_review",
                overlay_site_root=self.config.reports_root / "pdf_overlay_review",
            ).build(summary)
            summary["timings"]["parsing_review_seconds"] = round(time.perf_counter() - parsing_review_started_perf, 3)
            summary["parsing_review_index_path"] = parsing_review_manifest["index_path"]
            summary["parsing_review_manifest_path"] = (self.config.reports_root / "parsing_review" / "manifest.json").as_posix()
            summary["parsing_review_document_count"] = parsing_review_manifest["document_count"]

        summary["timings"]["report_build_seconds"] = round(
            summary["timings"]["overlay_report_seconds"] + summary["timings"]["parsing_review_seconds"],
            3,
        )
        summary["timings"]["total_seconds"] = round(time.perf_counter() - run_started_perf, 3)
        summary["finished_at"] = iso_now()

        write_json(self.config.outputs_root / "logs" / "latest_run.json", summary)
        write_text(self.config.outputs_root / "logs" / "latest_run.md", self._render_summary_markdown(summary))
        return summary

    def _ensure_directories(self) -> None:
        ensure_directory(self.config.interim_root / "classification")
        ensure_directory(self.config.interim_root / "fallbacks")
        ensure_directory(self.config.structured_root / "documents")
        ensure_directory(self.config.outputs_root / "json")
        ensure_directory(self.config.outputs_root / "markdown")
        ensure_directory(self.config.outputs_root / "logs")
        if self.config.comparisons_root:
            ensure_directory(self.config.comparisons_root)
        if self.config.reports_root:
            ensure_directory(self.config.reports_root)

    def _render_summary_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Parsing Run Summary",
            "",
            f"- Version: `{summary['version']}`",
            f"- Started At: `{summary['started_at']}`",
            f"- Finished At: `{summary.get('finished_at', '')}`",
            f"- Total Documents: `{summary['total_documents']}`",
            f"- Parsed Documents: `{summary['parsed_documents']}`",
            f"- Fallback Documents: `{summary['fallback_documents']}`",
            f"- Failed Documents: `{summary['failed_documents']}`",
        ]
        timings = summary.get("timings") or {}
        if timings:
            lines.extend(
                [
                    f"- Document Discovery Seconds: `{timings.get('document_discovery_seconds', 0.0)}`",
                    f"- Document Pipeline Seconds: `{timings.get('document_pipeline_seconds', 0.0)}`",
                    f"- Report Build Seconds: `{timings.get('report_build_seconds', 0.0)}`",
                    f"- Total Seconds: `{timings.get('total_seconds', 0.0)}`",
                ]
            )

        if summary.get("parsing_review_index_path"):
            lines.append(f"- Parsing Review UI: `{summary['parsing_review_index_path']}`")
        if summary.get("overlay_index_path"):
            lines.append(f"- Overlay UI: `{summary['overlay_index_path']}`")

        lines.extend(
            [
                "",
                "| Source | Status | Type | Parser | Issues |",
                "| --- | --- | --- | --- | --- |",
            ]
        )

        for document in summary["documents"]:
            lines.append(
                "| {source} | {status} | {doc_type} | {parser} | {issues} |".format(
                    source=document.get("source_path", ""),
                    status=document.get("status", ""),
                    doc_type=document.get("document_type", ""),
                    parser=document.get("parser_name", document.get("parser_route", "")),
                    issues=document.get("issue_count", 0),
                )
            )

        return "\n".join(lines) + "\n"
