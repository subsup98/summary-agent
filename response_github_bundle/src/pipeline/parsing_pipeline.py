from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_MAX_WORKERS_HARD_LIMIT = min(os.cpu_count() or 4, 8)

from src.classifiers.document_classifier import classify_document
from src.ingest.document_loader import DocumentLoader
from src.parsers.common.serialization import parsed_document_to_dict
from src.parsers.office.doc_parser import DocParser
from src.parsers.office.docx_parser import DocxParser
from src.parsers.office.fallback_parser import ConversionFallbackParser
from src.parsers.office.hwp_parser import HwpParser
from src.parsers.pdf.pdf_parser import PdfParser
from src.parsers.text.text_parser import TextParser
from src.indexing.chunking import build_semantic_chunks
from src.indexing.embedding_backends import resolve_embedding_backend
from src.retrieval.document_summary import ensure_basic_summary
from src.shared.io import ensure_directory, iso_now, make_artifact_stem, write_json, write_text
from src.shared.versioning import load_version_info
from src.ui.pdf_overlay_report import PdfOverlaySiteBuilder
from src.ui.parsing_review_report import ParsingReviewSiteBuilder


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

        basic_summary_started_perf = time.perf_counter()
        payload = ensure_basic_summary(parsed_document_to_dict(parsed_document))
        basic_summary_seconds = round(time.perf_counter() - basic_summary_started_perf, 3)

        markdown = str(payload.get("markdown") or "")
        semantic_chunk_seconds = 0.0
        if markdown.strip() and not payload.get("semantic_chunks"):
            semantic_chunk_started_perf = time.perf_counter()
            payload["semantic_chunks"] = build_semantic_chunks(markdown, embeddings=self.embedding_backend)
            semantic_chunk_seconds = round(time.perf_counter() - semantic_chunk_started_perf, 3)

        record: dict[str, Any] = {
            "source_path": relative_path,
            "status": parsed_document.status,
            "parser_name": parsed_document.parser_name,
            "document_type": parsed_document.classification.document_type,
            "pdf_kind": parsed_document.classification.pdf_kind,
            "issue_count": len(parsed_document.issues),
        }

        persist_started_perf = time.perf_counter()
        if parsed_document.status == "parsed":
            structured_path = self.config.structured_root / "documents" / f"{artifact_stem}.json"
            output_json_path = self.config.outputs_root / "json" / f"{artifact_stem}.json"
            output_markdown_path = self.config.outputs_root / "markdown" / f"{artifact_stem}.md"

            write_json(structured_path, payload)
            write_json(output_json_path, payload)
            write_text(output_markdown_path, parsed_document.markdown.rstrip() + "\n")

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
