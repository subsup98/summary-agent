from __future__ import annotations

import html
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import unquote
from uuid import uuid4

import fitz
import olefile

from src.classifiers.document_classifier import classify_document
from src.indexing.chroma_store import ChromaIndexManager
from src.indexing.embedding_backends import EmbeddingBackend
from src.parsers.pdf.markdown_extractor import PdfMarkdownExtractor
from src.pipeline.parsing_pipeline import ParsingPipeline, ParsingPipelineConfig
from src.retrieval.document_summary import ensure_basic_summary
from src.retrieval.openai_answerer import load_openai_settings
from src.shared.constants import SUPPORTED_EXTENSIONS
from src.shared.io import ensure_directory, iso_now, make_artifact_stem, read_text_with_fallback, write_bytes, write_json, write_text
from src.shared.office_pdf_converter import (
    LIBREOFFICE_CONVERTIBLE_EXTENSIONS,
    convert_office_source_to_pdf,
    find_libreoffice,
    iter_libreoffice_candidates,
    libreoffice_has_h2orestart,
)
from src.evaluation.embedding_model_comparison import TARGET_MODELS, EmbeddingModelComparisonRunner
from src.ui.parsing_review_report import ParsingReviewSiteBuilder


@dataclass(frozen=True)
class UploadedDocument:
    filename: str
    content: bytes


class ReviewSessionManager:
    def __init__(
        self,
        project_root: Path,
        embedding_backend: EmbeddingBackend | None = None,
        *,
        markdown_mode: str = "both",
        qa_mode: str = "hybrid",
    ) -> None:
        self.project_root = project_root
        self.markdown_mode = markdown_mode
        self.qa_mode = qa_mode
        self.project_review_root = project_root / "outputs" / "reports" / "pdf_overlay_review"
        self.project_review_manifest_path = self.project_review_root / "manifest.json"
        self.project_parsing_root = project_root / "outputs" / "reports" / "parsing_review"
        self.project_parsing_manifest_path = self.project_parsing_root / "manifest.json"
        self.project_compare_root = project_root / "outputs" / "document_compare_dashboard"
        self.project_process_compare_root = project_root / "outputs" / "process_compare_dashboard"
        self.project_latest_run_path = project_root / "outputs" / "parsing" / "logs" / "latest_run.json"
        self.runs_root = project_root / "outputs" / "ui_runs"
        self.preview_root = project_root / "outputs" / "ui_previews"
        self.latest_session_path = self.runs_root / "latest_session.json"
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_lock = threading.Lock()
        self._sse_clients: list[queue.Queue] = []
        self._sse_lock = threading.Lock()
        self._pending_source_hashes: set[str] = set()
        self.index_manager = ChromaIndexManager(project_root=project_root, embedding_backend=embedding_backend)
        ensure_directory(self.runs_root)
        ensure_directory(self.preview_root)
        self._ensure_project_parsing_review()

    def close(self) -> None:
        self.index_manager.close()

    def _ensure_project_parsing_review(self) -> None:
        parsed_root = self.project_root / "data" / "structured" / "documents"
        if not parsed_root.exists():
            return
        run_summary = self._read_json(self.project_latest_run_path) or {}
        ParsingReviewSiteBuilder(
            parsed_root=parsed_root,
            site_root=self.project_parsing_root,
            overlay_site_root=self.project_review_root,
        ).build(run_summary)

    def start_run(self, uploads: list[UploadedDocument]) -> dict[str, Any]:
        uploads_to_process, duplicate_uploads, reserved_hashes = self._reserve_uploads(uploads)
        job_id = uuid4().hex[:12]
        with self.job_lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "message": "작업을 준비 중입니다.",
                "created_at": iso_now(),
                "progress": {"current": 0, "total": max(len(uploads), 1)},
                "result": None,
                "error": None,
            }
        threading.Thread(
            target=self._run_job,
            args=(job_id, uploads_to_process, duplicate_uploads, reserved_hashes),
            daemon=True,
        ).start()
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.job_lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def create_run(
        self,
        uploads: list[UploadedDocument],
        progress_callback: Callable[[str, int, int], None] | None = None,
        *,
        duplicate_uploads: list[dict[str, Any]] | None = None,
        reserved_hashes: list[str] | None = None,
    ) -> dict[str, Any]:
        run_id = self._make_run_id()
        run_started_at = iso_now()
        run_started_perf = time.perf_counter()
        session_root = self.runs_root / run_id
        source_root = session_root / "source"
        duplicate_uploads = list(duplicate_uploads or [])
        reserved_hashes = list(reserved_hashes or [])
        stage_timings: dict[str, Any] = {
            "run_started_at": run_started_at,
            "source_write_started_at": iso_now(),
        }
        saved_uploads, skipped_uploads = self._write_uploads(source_root, uploads)
        stage_timings["source_write_finished_at"] = iso_now()
        stage_timings["source_write_seconds"] = round(time.perf_counter() - run_started_perf, 3)
        if not saved_uploads and not duplicate_uploads:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"처리 가능한 문서가 없습니다. 지원 확장자: {supported}")

        try:
            project_structured_documents = self.project_root / "data" / "structured" / "documents"
            if project_structured_documents.exists():
                if progress_callback:
                    progress_callback("기존 파싱 결과를 벡터 DB에 반영하는 중입니다.", 0, 1)
                stage_timings["project_sync_started_at"] = iso_now()
                sync_started_perf = time.perf_counter()
                self.index_manager.ingest_structured_documents(
                    structured_documents_root=project_structured_documents,
                    source_root=self.project_root / "data" / "raw",
                )
                stage_timings["project_sync_finished_at"] = iso_now()
                stage_timings["project_sync_seconds"] = round(time.perf_counter() - sync_started_perf, 3)

            unique_uploads = list(saved_uploads)

            if not unique_uploads:
                total_elapsed = round(time.perf_counter() - run_started_perf, 3)
                stage_timings["qa_ready_at"] = iso_now()
                stage_timings["qa_ready_seconds"] = total_elapsed
                session = {
                    "run_id": run_id,
                    "created_at": run_started_at,
                    "session_root": session_root.as_posix(),
                    "review_url": "/project-parsing/index.html" if self.project_parsing_root.joinpath("index.html").exists() else None,
                    "overlay_review_url": "/project-review/index.html" if self.project_review_root.joinpath("index.html").exists() else None,
                    "parsing_review_url": "/project-parsing/index.html" if self.project_parsing_root.joinpath("index.html").exists() else None,
                    "review_index_path": None,
                    "overlay_review_index_path": None,
                    "parsing_review_index_path": None,
                    "latest_run_markdown_path": None,
                    "upload_count": len(saved_uploads),
                    "uploads": [],
                    "duplicate_uploads": duplicate_uploads,
                    "content_duplicate_uploads": [],
                    "skipped_uploads": skipped_uploads,
                    "summary": {
                        "status": "duplicate_only",
                        "total_documents": 0,
                        "parsed_documents": 0,
                        "fallback_documents": 0,
                        "failed_documents": 0,
                        "duplicate_documents": len(duplicate_uploads),
                        "documents": [],
                    },
                    "vector_index": self.index_manager.get_status(),
                    "timings": stage_timings,
                }
                write_json(session_root / "session.json", session)
                write_json(self.latest_session_path, session)
                return session

            unique_names = {item["stored_name"] for item in unique_uploads}
            for path in source_root.iterdir():
                if path.is_file() and path.name not in unique_names:
                    path.unlink()

            # 파싱 완료된 문서를 즉시 벡터 인덱싱하기 위한 콜백과 결과 수집기
            index_results: list[dict[str, Any]] = []
            index_results_lock = threading.Lock()
            done_count = [0]

            def on_document_ready(payload: dict[str, Any]) -> None:
                result = self.index_manager.ingest_single_document(payload, source_root=source_root)
                with index_results_lock:
                    index_results.append(result)
                    done_count[0] += 1
                    current = done_count[0]
                self.broadcast_sse(
                    "documents_updated",
                    {
                        "document_id": str(payload.get("document_id") or ""),
                        "source_name": str(payload.get("source_name") or ""),
                        "current": current,
                        "total": len(unique_uploads),
                    },
                )
                if progress_callback:
                    progress_callback(
                        f"파싱·인덱싱 완료: {payload.get('source_name', '')} ({current}/{len(unique_uploads)})",
                        current,
                        len(unique_uploads),
                    )
            if progress_callback:
                progress_callback(
                    "신규 문서를 병렬 파싱·인덱싱하는 중입니다.",
                    0,
                    len(unique_uploads),
                )
            stage_timings["parse_started_at"] = iso_now()
            parse_started_perf = time.perf_counter()
            config = ParsingPipelineConfig(
                source_root=source_root,
                interim_root=session_root / "interim",
                structured_root=session_root / "structured",
                outputs_root=session_root / "parsing",
                comparisons_root=session_root / "comparisons",
                reports_root=session_root / "reports",
                enable_omitted_picture_ocr=False,
                markdown_mode=self.markdown_mode,
                qa_mode=self.qa_mode,
                on_document_ready=on_document_ready,
            )
            summary = ParsingPipeline(config).run()
            stage_timings["parse_finished_at"] = iso_now()
            stage_timings["parse_seconds"] = round(time.perf_counter() - parse_started_perf, 3)
            pipeline_timings = summary.get("timings") or {}
            stage_timings["document_pipeline_seconds"] = pipeline_timings.get("document_pipeline_seconds")
            stage_timings["classification_seconds"] = pipeline_timings.get("classification_seconds")
            stage_timings["parser_seconds"] = pipeline_timings.get("parser_seconds")
            stage_timings["basic_summary_seconds"] = pipeline_timings.get("basic_summary_seconds")
            stage_timings["semantic_chunk_seconds"] = pipeline_timings.get("semantic_chunk_seconds")
            stage_timings["persist_seconds"] = pipeline_timings.get("persist_seconds")
            stage_timings["overlay_report_seconds"] = pipeline_timings.get("overlay_report_seconds")
            stage_timings["parsing_review_seconds"] = pipeline_timings.get("parsing_review_seconds")
            stage_timings["report_build_seconds"] = pipeline_timings.get("report_build_seconds")
            stage_timings["vector_index_started_at"] = stage_timings["parse_started_at"]
            stage_timings["vector_index_finished_at"] = stage_timings["parse_finished_at"]
            stage_timings["vector_index_seconds"] = stage_timings["parse_seconds"]

            indexed_results = [r for r in index_results if r.get("status") == "indexed"]
            duplicate_results = [r for r in index_results if r.get("status") == "duplicate"]
            failed_results = [r for r in index_results if r.get("status") == "failed"]
            vector_summary: dict[str, Any] = {
                "started_at": stage_timings["parse_started_at"],
                "finished_at": stage_timings["parse_finished_at"],
                "total_documents": len(index_results),
                "indexed_documents": len(indexed_results),
                "duplicate_documents": len(duplicate_results),
                "failed_documents": len(failed_results),
                "documents": [r.get("record", r) for r in index_results],
                "comparisons": [r["comparison"] for r in indexed_results if r.get("comparison")],
            }

            total_vector_documents = len(index_results)
            ready_documents = len(indexed_results) + len(duplicate_results)
            failed_vector_documents = len(failed_results)
            qa_ready = total_vector_documents > 0 and ready_documents >= total_vector_documents and failed_vector_documents == 0
            stage_timings["qa_ready"] = qa_ready
            if qa_ready:
                stage_timings["qa_ready_at"] = stage_timings["vector_index_finished_at"]
                stage_timings["qa_ready_seconds"] = round(time.perf_counter() - run_started_perf, 3)
            else:
                stage_timings["qa_ready_at"] = None
                stage_timings["qa_ready_seconds"] = None
                stage_timings["qa_blocker"] = "vector_index_incomplete"
            content_duplicate_uploads = [
                document
                for document in vector_summary.get("documents", [])
                if document.get("status") == "duplicate"
            ]
            summary["duplicate_documents"] = len(duplicate_uploads) + len(content_duplicate_uploads)
            office_pdf_compare = self._build_office_pdf_strategy_compare(
                session_root=session_root,
                saved_uploads=unique_uploads,
            )

            session = {
                "run_id": run_id,
                "created_at": run_started_at,
                "session_root": session_root.as_posix(),
                "review_url": f"/runs/{run_id}/reports/parsing_review/index.html",
                "overlay_review_url": f"/runs/{run_id}/reports/pdf_overlay_review/index.html",
                "parsing_review_url": f"/runs/{run_id}/reports/parsing_review/index.html",
                "office_pdf_compare_url": (
                    f"/runs/{run_id}/reports/office_pdf_strategy_compare/index.html"
                    if office_pdf_compare and office_pdf_compare.get("document_count", 0) > 0
                    else None
                ),
                "review_index_path": (session_root / "reports" / "parsing_review" / "index.html").as_posix(),
                "overlay_review_index_path": (session_root / "reports" / "pdf_overlay_review" / "index.html").as_posix(),
                "parsing_review_index_path": (session_root / "reports" / "parsing_review" / "index.html").as_posix(),
                "office_pdf_compare_index_path": (
                    (session_root / "reports" / "office_pdf_strategy_compare" / "index.html").as_posix()
                    if office_pdf_compare and office_pdf_compare.get("document_count", 0) > 0
                    else None
                ),
                "latest_run_markdown_path": (session_root / "parsing" / "logs" / "latest_run.md").as_posix(),
                "upload_count": len(saved_uploads),
                "uploads": unique_uploads,
                "duplicate_uploads": duplicate_uploads,
                "content_duplicate_uploads": content_duplicate_uploads,
                "skipped_uploads": skipped_uploads,
                "summary": summary,
                "vector_index": vector_summary,
                "office_pdf_compare": office_pdf_compare,
                "timings": stage_timings,
            }
            write_json(session_root / "session.json", session)
            write_json(self.latest_session_path, session)
            return session
        finally:
            self._release_pending_hashes(reserved_hashes)

    def get_status(self) -> dict[str, Any]:
        return {
            "generated_at": iso_now(),
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "project_review": self._build_project_review_status(),
            "latest_upload": self._build_session_status(self.get_latest_session()),
            "recent_uploads": [self._build_session_status(session) for session in self.list_recent_sessions(limit=6)],
            "vector_index": self.index_manager.get_status(),
            "openai": self.get_openai_status(),
            "embedding_comparison": self.get_embedding_comparison_status(),
            "chunking_comparison": self.get_chunking_comparison_status(),
        }

    def get_document_list(self) -> list[dict[str, Any]]:
        documents_by_source: dict[str, dict[str, Any]] = {}
        for path in self._iter_structured_document_paths():
            payload = self._read_json(path)
            if not payload:
                continue
            payload = ensure_basic_summary(payload)
            document_id = str(payload.get("document_id", ""))
            if not document_id:
                continue
            try:
                modified_at = path.stat().st_mtime
            except OSError:
                modified_at = 0.0
            source_name = str(payload.get("source_name", ""))
            source_key = self._normalize_source_key(source_name) or document_id
            entry = {
                "document_id": document_id,
                "source_name": source_name,
                "extension": payload.get("extension", ""),
                "document_type": (payload.get("classification") or {}).get("document_type", ""),
                "basic_summary": payload.get("basic_summary", {}),
                "llm_summary": payload.get("llm_summary") or None,
                "ui_summary": payload.get("ui_summary") or None,
                "origin": "upload" if "/outputs/ui_runs/" in path.as_posix().replace("\\", "/") else "project",
                "document_path": path.as_posix(),
                "modified_at": modified_at,
                "pipeline_timing": None,
                "duplicate_document_ids": [],
                "variant_count": 1,
            }
            existing = documents_by_source.get(source_key)
            if existing is None:
                documents_by_source[source_key] = entry
                continue
            existing["duplicate_document_ids"] = [
                *existing.get("duplicate_document_ids", []),
                document_id,
            ]
            existing["variant_count"] = int(existing.get("variant_count") or 1) + 1
        documents = list(documents_by_source.values())
        documents.sort(
            key=lambda item: (float(item.get("modified_at") or 0.0), str(item.get("document_id") or "")),
            reverse=True,
        )
        return documents

    def _find_session_timing_for_document(self, *, document_id: str, source_name: str) -> dict[str, Any] | None:
        normalized_source = self._normalize_source_key(source_name)
        for session in self.list_recent_sessions(limit=100):
            timings = session.get("timings") or {}
            if not timings:
                continue
            vector_documents = (session.get("vector_index") or {}).get("documents") or []
            for item in vector_documents:
                if document_id and str(item.get("document_id") or "") == document_id:
                    return dict(timings)
                item_source = self._normalize_source_key(str(item.get("source_name") or ""))
                if normalized_source and item_source == normalized_source:
                    return dict(timings)
        return None

    def get_document_payload(self, *, document_id: str = "", source_name: str = "") -> dict[str, Any] | None:
        normalized_source = self._normalize_source_key(source_name)
        for path in self._iter_structured_document_paths():
            payload = self._read_json(path)
            if not payload:
                continue
            payload_document_id = str(payload.get("document_id", "")).strip()
            payload_source_name = self._normalize_source_key(str(payload.get("source_name", "")))
            if document_id and payload_document_id == document_id:
                return payload
            if not document_id and normalized_source and payload_source_name == normalized_source:
                return payload
        return None

    def get_document_source_path(self, *, document_id: str = "", source_name: str = "") -> Path | None:
        payload = self.get_document_payload(document_id=document_id, source_name=source_name)
        if not payload:
            return None
        raw_source_path = str(payload.get("source_path") or "").strip()
        if not raw_source_path:
            return None
        candidate = Path(raw_source_path)
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(self.project_root.resolve())
        except ValueError:
            return None
        return candidate if candidate.exists() else None

    _LIBREOFFICE_CONVERTIBLE = LIBREOFFICE_CONVERTIBLE_EXTENSIONS

    def _find_libreoffice(self) -> Path | None:
        return find_libreoffice()

    def _iter_libreoffice_candidates(self, preferred: Path | None) -> list[Path]:
        return iter_libreoffice_candidates(preferred)

    def _get_libreoffice_user_profile(self) -> Path | None:
        candidates = [
            Path.home() / "AppData" / "Roaming" / "LibreOffice" / "4" / "user",
            Path("C:/Users/yongseop.im/AppData/Roaming/LibreOffice/4/user"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _libreoffice_has_h2orestart(self) -> bool:
        return libreoffice_has_h2orestart()

    def _build_libreoffice_env(
        self,
        *,
        source_path: Path,
        profile_root: Path,
    ) -> dict[str, str]:
        env = os.environ.copy()
        if source_path.suffix.lower() not in {".hwp", ".hwpx"} or not self._libreoffice_has_h2orestart():
            return env
        home_dir = profile_root / "home"
        temp_dir = profile_root / "tmp"
        ensure_directory(home_dir)
        ensure_directory(temp_dir)
        java_tool_options = str(env.get("JAVA_TOOL_OPTIONS") or "").strip()
        user_home_option = f"-Duser.home={home_dir}"
        env["JAVA_TOOL_OPTIONS"] = f"{java_tool_options} {user_home_option}".strip() if java_tool_options else user_home_option
        env["USERPROFILE"] = str(home_dir)
        env["HOME"] = str(home_dir)
        env["TMP"] = str(temp_dir)
        env["TEMP"] = str(temp_dir)
        return env

    def _seed_libreoffice_profile(self, *, source_path: Path, profile_root: Path) -> None:
        if source_path.suffix.lower() not in {".hwp", ".hwpx"} or not self._libreoffice_has_h2orestart():
            return
        user_profile = self._get_libreoffice_user_profile()
        if user_profile is None:
            return
        src_uno_packages = user_profile / "uno_packages"
        if not src_uno_packages.exists():
            return
        dest_uno_packages = profile_root / "user" / "uno_packages"
        ensure_directory(dest_uno_packages.parent)
        shutil.copytree(src_uno_packages, dest_uno_packages, dirs_exist_ok=True)

    def convert_source_to_pdf(self, source_path: Path, dest_path: Path) -> bool:
        """LibreOffice로 doc/docx/hwp/hwpx → PDF 변환. 성공 시 True 반환."""
        return convert_office_source_to_pdf(source_path, dest_path)

    def _detect_image_format(self, content: bytes) -> str | None:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if content.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if content.startswith((b"GIF87a", b"GIF89a")):
            return "gif"
        if content.startswith(b"BM"):
            return "bmp"
        if content.startswith(b"II*\x00") or content.startswith(b"MM\x00*"):
            return "tiff"
        return None

    def _extract_hwp_preview_image(self, source_path: Path) -> tuple[bytes, str] | None:
        if source_path.suffix.lower() != ".hwp" or not source_path.exists():
            return None
        if not olefile.isOleFile(source_path):
            return None
        try:
            with olefile.OleFileIO(source_path) as ole:
                if not ole.exists("PrvImage"):
                    return None
                preview_bytes = ole.openstream("PrvImage").read()
        except Exception:
            return None

        image_format = self._detect_image_format(preview_bytes)
        if not image_format:
            return None
        return preview_bytes, image_format

    def get_document_preview_image_path(self, *, document_id: str = "", source_name: str = "") -> Path | None:
        payload = self.get_document_payload(document_id=document_id, source_name=source_name)
        if not payload:
            return None
        source_path = self.get_document_source_path(document_id=document_id, source_name=source_name)
        if source_path is None or source_path.suffix.lower() != ".hwp":
            return None
        preview_image = self._extract_hwp_preview_image(source_path)
        if preview_image is None:
            return None

        resolved_source_name = str(payload.get("source_name") or source_name or "").strip()
        preview_cache_key = self._build_preview_cache_key(
            payload=payload,
            source_name=source_name,
            document_id=document_id,
        )
        if not preview_cache_key:
            return None

        preview_bytes, image_format = preview_image
        preview_path = self.preview_root / f"{preview_cache_key}.{image_format}"
        if self._is_preview_cache_valid(
            preview_path=preview_path,
            source_path=source_path,
            source_name=resolved_source_name,
        ):
            return preview_path

        self._safe_remove_preview_artifacts(preview_path)
        try:
            ensure_directory(preview_path.parent)
            write_bytes(preview_path, preview_bytes)
        except OSError:
            return None
        if not preview_path.exists():
            return None
        self._write_preview_metadata(
            preview_path=preview_path,
            source_path=source_path,
            source_name=resolved_source_name,
            strategy="hwp-preview-image-raw",
        )
        return preview_path

    def _build_preview_pdf_from_image(self, image_bytes: bytes, dest_path: Path) -> bool:
        try:
            ensure_directory(dest_path.parent)
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass

            image_document = fitz.open(stream=image_bytes)
            try:
                image_rect = image_document[0].rect
                image_width = max(float(image_rect.width), 1.0)
                image_height = max(float(image_rect.height), 1.0)
            finally:
                image_document.close()

            page_width = 595.0
            page_height = max(842.0, page_width * (image_height / image_width))
            margin = 24.0
            available_width = page_width - (margin * 2)
            available_height = page_height - (margin * 2)
            scale = min(available_width / image_width, available_height / image_height)
            render_width = image_width * scale
            render_height = image_height * scale
            x0 = (page_width - render_width) / 2
            y0 = (page_height - render_height) / 2

            document = fitz.open()
            try:
                page = document.new_page(width=page_width, height=page_height)
                page.insert_image(fitz.Rect(x0, y0, x0 + render_width, y0 + render_height), stream=image_bytes)
                document.save(dest_path)
            finally:
                document.close()
            return dest_path.exists()
        except Exception:
            return False

    def _get_preview_meta_path(self, preview_path: Path) -> Path:
        return preview_path.with_suffix(".meta.json")

    def _build_preview_cache_key(
        self,
        *,
        payload: dict[str, Any] | None = None,
        source_name: str = "",
        document_id: str = "",
    ) -> str:
        resolved_source_name = str((payload or {}).get("source_name") or source_name or "").strip()
        normalized_source = self._normalize_source_key(resolved_source_name)
        if normalized_source:
            stem = self._sanitize_filename(Path(resolved_source_name).stem) or "document"
            digest = hashlib.sha1(normalized_source.encode("utf-8")).hexdigest()[:10]
            return f"{stem}--src-{digest}"
        fallback_id = str((payload or {}).get("document_id") or document_id or "").strip()
        return fallback_id or "document-preview"

    def _write_preview_metadata(
        self,
        *,
        preview_path: Path,
        source_path: Path | None,
        source_name: str,
        strategy: str,
    ) -> None:
        meta_path = self._get_preview_meta_path(preview_path)
        payload = {
            "preview_path": preview_path.as_posix(),
            "source_path": source_path.as_posix() if source_path else None,
            "source_name": source_name,
            "normalized_source_name": self._normalize_source_key(source_name),
            "source_mtime": round(source_path.stat().st_mtime, 6) if source_path and source_path.exists() else None,
            "source_size": int(source_path.stat().st_size) if source_path and source_path.exists() else None,
            "preview_mtime": round(preview_path.stat().st_mtime, 6) if preview_path.exists() else None,
            "strategy": strategy,
            "generated_at": iso_now(),
        }
        write_json(meta_path, payload)

    def _is_preview_cache_valid(self, *, preview_path: Path, source_path: Path | None, source_name: str) -> bool:
        if not preview_path.exists():
            return False
        meta_path = self._get_preview_meta_path(preview_path)
        if not meta_path.exists():
            return False
        metadata = self._read_json(meta_path)
        if not metadata:
            return False
        if str(metadata.get("preview_path") or "").strip() != preview_path.as_posix():
            return False
        if self._normalize_source_key(str(metadata.get("source_name") or "")) != self._normalize_source_key(source_name):
            return False
        if source_path is None:
            return True
        cached_strategy = str(metadata.get("strategy") or "").strip().lower()
        if (
            source_path.suffix.lower() == ".hwp"
            and cached_strategy in {"markdown-fallback", "hwp-preview-image"}
            and (
                self._libreoffice_has_h2orestart()
                or self._extract_hwp_preview_image(source_path) is not None
            )
        ):
            return False
        try:
            cached_mtime = float(metadata.get("source_mtime"))
        except (TypeError, ValueError):
            return False
        try:
            current_mtime = round(source_path.stat().st_mtime, 6)
        except OSError:
            return False
        try:
            cached_size = int(metadata.get("source_size"))
        except (TypeError, ValueError):
            return False
        try:
            current_size = int(source_path.stat().st_size)
        except OSError:
            return False
        return abs(cached_mtime - current_mtime) < 0.000001 and cached_size == current_size

    def _safe_remove_preview_artifacts(self, preview_path: Path) -> None:
        for candidate in (preview_path, self._get_preview_meta_path(preview_path)):
            if not candidate.exists():
                continue
            try:
                candidate.unlink()
            except OSError:
                pass

    def get_document_preview_pdf_path(self, *, document_id: str = "", source_name: str = "") -> Path | None:
        payload = self.get_document_payload(document_id=document_id, source_name=source_name)
        if not payload:
            return None
        preview_cache_key = self._build_preview_cache_key(
            payload=payload,
            source_name=source_name,
            document_id=document_id,
        )
        if not preview_cache_key:
            return None
        preview_path = self.preview_root / f"{preview_cache_key}.pdf"
        source_path = self.get_document_source_path(document_id=document_id, source_name=source_name)
        resolved_source_name = str(payload.get("source_name") or source_name or "").strip()
        if self._is_preview_cache_valid(
            preview_path=preview_path,
            source_path=source_path,
            source_name=resolved_source_name,
        ):
            return preview_path

        # doc/docx/hwp/hwpx → LibreOffice로 직접 변환
        if source_path and source_path.suffix.lower() in self._LIBREOFFICE_CONVERTIBLE:
            self._safe_remove_preview_artifacts(preview_path)
            if self.convert_source_to_pdf(source_path, preview_path):
                self._write_preview_metadata(
                    preview_path=preview_path,
                    source_path=source_path,
                    source_name=resolved_source_name,
                    strategy="libreoffice",
                )
                return preview_path

        if source_path and source_path.suffix.lower() == ".hwp":
            preview_image = self._extract_hwp_preview_image(source_path)
            if preview_image is not None:
                self._safe_remove_preview_artifacts(preview_path)
                preview_bytes, _image_format = preview_image
                if self._build_preview_pdf_from_image(preview_bytes, preview_path):
                    self._write_preview_metadata(
                        preview_path=preview_path,
                        source_path=source_path,
                        source_name=resolved_source_name,
                        strategy="hwp-preview-image",
                    )
                    return preview_path

        markdown = str(payload.get("markdown") or "").strip()
        if not markdown:
            return None

        font_path: Path | None = None
        for candidate in (
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/NanumGothic.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ):
            if candidate.exists():
                font_path = candidate
                break

        document = fitz.open()
        try:
            page_width = 595
            page_height = 842
            margin = 40
            cursor_y = margin
            page = document.new_page(width=page_width, height=page_height)
            font_name = "helv"
            if font_path is not None:
                try:
                    font_name = "fallback-ui-font"
                    page.insert_font(fontname=font_name, fontfile=str(font_path))
                except Exception:
                    font_name = "helv"

            normalized_lines: list[str] = []
            for block in markdown.splitlines():
                line = block.rstrip()
                if len(line) <= 92:
                    normalized_lines.append(line)
                    continue
                while len(line) > 92:
                    normalized_lines.append(line[:92])
                    line = line[92:]
                normalized_lines.append(line)

            for raw_line in normalized_lines:
                line = raw_line or " "
                box = fitz.Rect(margin, cursor_y, page_width - margin, cursor_y + 20)
                overflow = page.insert_textbox(
                    box,
                    line,
                    fontsize=10,
                    fontname=font_name,
                    color=(0.12, 0.16, 0.23),
                    lineheight=1.35,
                )
                if overflow < 0:
                    page = document.new_page(width=page_width, height=page_height)
                    cursor_y = margin
                    if font_path is not None and font_name != "helv":
                        try:
                            page.insert_font(fontname=font_name, fontfile=str(font_path))
                        except Exception:
                            font_name = "helv"
                    box = fitz.Rect(margin, cursor_y, page_width - margin, cursor_y + 20)
                    page.insert_textbox(
                        box,
                        line,
                        fontsize=10,
                        fontname=font_name,
                        color=(0.12, 0.16, 0.23),
                        lineheight=1.35,
                    )
                cursor_y += 16
                if cursor_y >= page_height - margin - 20:
                    page = document.new_page(width=page_width, height=page_height)
                    cursor_y = margin
                    if font_path is not None and font_name != "helv":
                        try:
                            page.insert_font(fontname=font_name, fontfile=str(font_path))
                        except Exception:
                            font_name = "helv"

            document.save(preview_path)
        finally:
            document.close()

        if preview_path.exists():
            self._write_preview_metadata(
                preview_path=preview_path,
                source_path=source_path,
                source_name=resolved_source_name,
                strategy="markdown-fallback",
            )

        return preview_path if preview_path.exists() else None

    def get_related_document_ids(self, *, document_id: str = "", source_name: str = "") -> list[str]:
        normalized_source = self._normalize_source_key(source_name)
        if not normalized_source and document_id:
            payload = self.get_document_payload(document_id=document_id)
            if payload:
                normalized_source = self._normalize_source_key(str(payload.get("source_name") or ""))

        related_ids: list[str] = []
        seen_ids: set[str] = set()
        for path in self._iter_structured_document_paths():
            payload = self._read_json(path)
            if not payload:
                continue
            payload_document_id = str(payload.get("document_id", "")).strip()
            payload_source_key = self._normalize_source_key(str(payload.get("source_name") or ""))
            if not payload_document_id:
                continue
            matches = bool(document_id and payload_document_id == document_id)
            if normalized_source and payload_source_key == normalized_source:
                matches = True
            if not matches or payload_document_id in seen_ids:
                continue
            seen_ids.add(payload_document_id)
            related_ids.append(payload_document_id)
        return related_ids

    def delete_document_files(self, document_ids: list[str] | str) -> int:
        """Delete document payload and preview files for the given document ids."""
        target_ids = [document_ids] if isinstance(document_ids, str) else list(document_ids)
        target_set = {str(item).strip() for item in target_ids if str(item).strip()}
        if not target_set:
            return 0
        removed = 0
        preview_cache_keys: set[str] = set()
        for path in self._iter_structured_document_paths():
            payload = self._read_json(path)
            if not payload:
                continue
            if str(payload.get("document_id", "")) in target_set:
                preview_cache_keys.add(self._build_preview_cache_key(payload=payload))
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
        for preview_cache_key in preview_cache_keys:
            preview_path = self.preview_root / f"{preview_cache_key}.pdf"
            meta_path = self._get_preview_meta_path(preview_path)
            for candidate in (preview_path, meta_path):
                if not candidate.exists():
                    continue
                try:
                    candidate.unlink()
                    removed += 1
                except Exception:
                    pass
        return removed

    def get_openai_status(self) -> dict[str, Any]:
        settings = load_openai_settings()
        return {
            "enabled": settings.enabled,
            "model": settings.model,
        }

    def get_embedding_comparison_status(self) -> dict[str, Any]:
        settings = load_openai_settings()
        report_path = self.project_root / "outputs" / "reports" / "embedding_model_comparison.json"
        report = self._read_json(report_path)
        return {
            "enabled": settings.enabled,
            "models": TARGET_MODELS,
            "report_path": report_path.as_posix() if report_path.exists() else None,
            "latest_report": report,
        }

    def get_chunking_comparison_status(self) -> dict[str, Any]:
        latest_dir = self._find_latest_chunking_comparison_dir()
        if not latest_dir:
            return {
                "exists": False,
                "report_url": None,
                "generated_at": None,
                "summary": None,
                "output_dir": None,
            }
        summary = self._read_json(latest_dir / "summary.json")
        return {
            "exists": bool(summary),
            "report_url": "/chunking-compare",
            "generated_at": (summary or {}).get("finished_at") or (summary or {}).get("started_at"),
            "summary": summary,
            "output_dir": latest_dir.as_posix(),
        }

    def get_chunking_comparison_document(self, document_id: str) -> dict[str, Any] | None:
        latest_dir = self._find_latest_chunking_comparison_dir()
        if not latest_dir:
            return None
        summary = self._read_json(latest_dir / "summary.json")
        if not summary:
            return None
        document = next((item for item in summary.get("documents", []) if item.get("document_id") == document_id), None)
        if not document:
            return None
        doc_dir = latest_dir / "documents" / document_id
        strategies: dict[str, Any] = {}
        for json_path in sorted(doc_dir.glob("*.json")):
            payload = self._read_json(json_path)
            if payload:
                strategies[json_path.stem] = payload
        return {
            "document": document,
            "strategies": strategies,
            "output_dir": doc_dir.as_posix(),
        }

    def start_embedding_comparison(self) -> dict[str, Any]:
        job_id = uuid4().hex[:12]
        with self.job_lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "job_type": "embedding_comparison",
                "status": "queued",
                "message": "임베딩 비교 작업을 준비 중입니다.",
                "created_at": iso_now(),
                "progress": {"current": 0, "total": 3},
                "result": None,
                "error": None,
            }
        threading.Thread(target=self._run_embedding_comparison_job, args=(job_id,), daemon=True).start()
        return self.get_job(job_id) or {}

    def get_latest_session(self) -> dict[str, Any] | None:
        latest = self._read_json(self.latest_session_path)
        if latest:
            return latest
        sessions = self.list_recent_sessions(limit=1)
        return sessions[0] if sessions else None

    def list_recent_sessions(self, limit: int = 6) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.runs_root.glob("*/session.json"):
            payload = self._read_json(path)
            if payload:
                sessions.append(payload)
        sessions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return sessions[:limit]

    def resolve_static_path(self, request_path: str) -> Path | None:
        raw_parts = [part for part in PurePosixPath(unquote(request_path)).parts if part not in {"", "/", "."}]
        if not raw_parts:
            return None
        if raw_parts[0] == "static":
            return self._resolve_relative_path(self.project_root / "src" / "ui" / "static", list(raw_parts[1:]))
        if raw_parts[0] == "project-review":
            return self._resolve_relative_path(self.project_review_root, list(raw_parts[1:]) or ["index.html"])
        if raw_parts[0] == "project-parsing":
            return self._resolve_relative_path(self.project_parsing_root, list(raw_parts[1:]) or ["index.html"])
        if raw_parts[0] == "project-compare":
            return self._resolve_relative_path(self.project_compare_root, list(raw_parts[1:]) or ["index.html"])
        if raw_parts[0] == "project-process-compare":
            return self._resolve_relative_path(self.project_process_compare_root, list(raw_parts[1:]) or ["index.html"])
        if raw_parts[0] == "latest-upload":
            latest = self.get_latest_session()
            if not latest:
                return None
            return self._resolve_relative_path(self.runs_root / latest["run_id"], list(raw_parts[1:]) or ["reports", "pdf_overlay_review", "index.html"])
        if raw_parts[0] == "latest-upload-parsing":
            latest = self.get_latest_session()
            if not latest:
                return None
            return self._resolve_relative_path(self.runs_root / latest["run_id"], list(raw_parts[1:]) or ["reports", "parsing_review", "index.html"])
        if raw_parts[0] == "runs" and len(raw_parts) >= 2:
            return self._resolve_relative_path(self.runs_root / raw_parts[1], list(raw_parts[2:]) or ["reports", "pdf_overlay_review", "index.html"])
        return None

    def _run_job(
        self,
        job_id: str,
        uploads: list[UploadedDocument],
        duplicate_uploads: list[dict[str, Any]],
        reserved_hashes: list[str],
    ) -> None:
        self._update_job(
            job_id,
            status="running",
            message="작업을 시작했습니다.",
            current=0,
            total=max(len(uploads) + len(duplicate_uploads), 1),
        )

        def progress(message: str, current: int, total: int) -> None:
            self._update_job(job_id, status="running", message=message, current=current, total=max(total, 1))

        try:
            session = self.create_run(
                uploads,
                progress_callback=progress,
                duplicate_uploads=duplicate_uploads,
                reserved_hashes=reserved_hashes,
            )
        except Exception as error:
            self._update_job(job_id, status="failed", message=str(error), error=str(error))
            return

        message = "동일 문서가 확인되어 기존 결과를 안내합니다." if session["summary"]["status"] == "duplicate_only" else "파싱과 벡터 인덱싱이 완료되었습니다."
        final_total = max(int(session.get("upload_count", 0) or 0), 1)
        self._update_job(job_id, status="completed", message=message, current=final_total, total=final_total, result=session)

    def _run_embedding_comparison_job(self, job_id: str) -> None:
        settings = load_openai_settings()
        if not settings.enabled:
            self._update_job(job_id, status="failed", message="OPENAI_API_KEY가 없어 임베딩 비교를 실행할 수 없습니다.", error="missing_openai_api_key")
            return

        self._update_job(job_id, status="running", message="임베딩 모델 비교를 시작했습니다.", current=0, total=len(TARGET_MODELS))
        try:
            runner = EmbeddingModelComparisonRunner(
                structured_documents_root=self.project_root / "data" / "structured" / "documents",
                reports_root=self.project_root / "outputs" / "reports",
                api_key=settings.api_key or "",
            )
            summary = runner.run()
        except Exception as error:
            self._update_job(job_id, status="failed", message=str(error), error=str(error))
            return
        self._update_job(job_id, status="completed", message="임베딩 모델 비교가 완료되었습니다.", current=len(TARGET_MODELS), total=len(TARGET_MODELS), result=summary)

    def subscribe_sse(self) -> "queue.Queue[str]":
        q: queue.Queue[str] = queue.Queue(maxsize=32)
        with self._sse_lock:
            self._sse_clients.append(q)
        return q

    def unsubscribe_sse(self, q: "queue.Queue[str]") -> None:
        with self._sse_lock:
            try:
                self._sse_clients.remove(q)
            except ValueError:
                pass

    def broadcast_sse(self, event: str, data: dict[str, Any]) -> None:
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        with self._sse_lock:
            for q in list(self._sse_clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def _update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.job_lock:
            job = self.jobs[job_id]
            if status is not None:
                job["status"] = status
            if message is not None:
                job["message"] = message
            if current is not None:
                job["progress"]["current"] = current
            if total is not None:
                job["progress"]["total"] = total
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            if status in {"completed", "failed"}:
                job["finished_at"] = iso_now()
        if status == "completed":
            self.broadcast_sse("documents_updated", {"job_id": job_id})

    def _build_project_review_status(self) -> dict[str, Any]:
        manifest = self._read_json(self.project_review_manifest_path)
        parsing_manifest = self._read_json(self.project_parsing_manifest_path)
        latest_run = self._read_json(self.project_latest_run_path)
        return {
            "exists": (
                self.project_review_root.joinpath("index.html").exists()
                or self.project_parsing_root.joinpath("index.html").exists()
                or self.project_compare_root.joinpath("index.html").exists()
                or self.project_process_compare_root.joinpath("index.html").exists()
            ),
            "review_url": "/project-parsing/index.html" if self.project_parsing_root.joinpath("index.html").exists() else "/project-review/index.html",
            "compare_url": "/project-compare/index.html" if self.project_compare_root.joinpath("index.html").exists() else None,
            "process_compare_url": "/project-process-compare/index.html" if self.project_process_compare_root.joinpath("index.html").exists() else None,
            "generated_at": (parsing_manifest or manifest or {}).get("generated_at"),
            "document_count": (parsing_manifest or manifest or {}).get("document_count"),
            "version": (latest_run or {}).get("version"),
            "source_root": (latest_run or {}).get("source_root"),
        }

    def _build_session_status(self, session: dict[str, Any] | None) -> dict[str, Any] | None:
        if not session:
            return None
        summary = session.get("summary", {})
        vector_index = session.get("vector_index", {})
        return {
            "exists": True,
            "run_id": session.get("run_id"),
            "created_at": session.get("created_at"),
            "review_url": session.get("review_url"),
            "parsing_review_url": session.get("parsing_review_url"),
            "upload_count": session.get("upload_count", 0),
            "duplicate_upload_count": len(session.get("duplicate_uploads", [])) + len(session.get("content_duplicate_uploads", [])),
            "total_documents": summary.get("total_documents", 0),
            "parsed_documents": summary.get("parsed_documents", 0),
            "failed_documents": summary.get("failed_documents", 0),
            "vector_indexed_documents": vector_index.get("indexed_documents", vector_index.get("document_count", 0)),
            "qa_ready_seconds": (session.get("timings") or {}).get("qa_ready_seconds"),
        }

    def _write_uploads(self, source_root: Path, uploads: list[UploadedDocument]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ensure_directory(source_root)
        used_names: set[str] = set()
        saved_uploads: list[dict[str, Any]] = []
        skipped_uploads: list[dict[str, Any]] = []
        for upload in uploads:
            safe_name = self._sanitize_filename(upload.filename)
            if not safe_name:
                skipped_uploads.append({"filename": upload.filename, "reason": "invalid_filename"})
                continue
            if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped_uploads.append({"filename": upload.filename, "reason": "unsupported_extension"})
                continue
            stored_name = self._dedupe_name(safe_name, used_names)
            output_path = source_root / stored_name
            write_bytes(output_path, upload.content)
            saved_uploads.append({
                "original_name": upload.filename,
                "stored_name": stored_name,
                "stored_path": output_path.as_posix(),
                "size_bytes": len(upload.content),
                "source_hash": self._hash_bytes(upload.content),
            })
        return saved_uploads, skipped_uploads

    def _partition_uploads(self, uploads: list[UploadedDocument]) -> tuple[list[UploadedDocument], list[dict[str, Any]], list[str]]:
        uploads_to_process: list[UploadedDocument] = []
        duplicate_uploads: list[dict[str, Any]] = []
        seen_batch_hashes: set[str] = set()
        with self.job_lock:
            pending_hashes = set(self._pending_source_hashes)
        for upload in uploads:
            source_hash = self._hash_bytes(upload.content)
            if source_hash in seen_batch_hashes:
                duplicate_uploads.append({
                    "original_name": upload.filename,
                    "stored_name": self._sanitize_filename(upload.filename) or upload.filename,
                    "stored_path": "",
                    "size_bytes": len(upload.content),
                    "source_hash": source_hash,
                    "reason": "same_batch_source_hash",
                    "existing_document_id": None,
                    "existing_source_name": "same batch upload",
                })
                continue
            if source_hash in pending_hashes:
                duplicate_uploads.append({
                    "original_name": upload.filename,
                    "stored_name": self._sanitize_filename(upload.filename) or upload.filename,
                    "stored_path": "",
                    "size_bytes": len(upload.content),
                    "source_hash": source_hash,
                    "reason": "already_processing",
                    "existing_document_id": None,
                    "existing_source_name": "processing",
                })
                continue
            existing = self.index_manager.find_duplicate_source_by_hash(source_hash)
            if existing:
                duplicate_uploads.append({
                    "original_name": upload.filename,
                    "stored_name": self._sanitize_filename(upload.filename) or upload.filename,
                    "stored_path": "",
                    "size_bytes": len(upload.content),
                    "source_hash": source_hash,
                    "reason": "same_source_hash",
                    "existing_document_id": existing.get("document_id"),
                    "existing_source_name": existing.get("source_name"),
                })
                continue
            seen_batch_hashes.add(source_hash)
            uploads_to_process.append(upload)
        return uploads_to_process, duplicate_uploads, list(seen_batch_hashes)

    def _reserve_uploads(self, uploads: list[UploadedDocument]) -> tuple[list[UploadedDocument], list[dict[str, Any]], list[str]]:
        uploads_to_process: list[UploadedDocument] = []
        duplicate_uploads: list[dict[str, Any]] = []
        reserved_hashes: list[str] = []
        seen_batch_hashes: set[str] = set()

        for upload in uploads:
            source_hash = self._hash_bytes(upload.content)
            stored_name = self._sanitize_filename(upload.filename) or upload.filename
            if source_hash in seen_batch_hashes:
                duplicate_uploads.append({
                    "original_name": upload.filename,
                    "stored_name": stored_name,
                    "stored_path": "",
                    "size_bytes": len(upload.content),
                    "source_hash": source_hash,
                    "reason": "same_batch_source_hash",
                    "existing_document_id": None,
                    "existing_source_name": "same batch upload",
                })
                continue

            existing = self.index_manager.find_duplicate_source_by_hash(source_hash)
            if existing:
                duplicate_uploads.append({
                    "original_name": upload.filename,
                    "stored_name": stored_name,
                    "stored_path": "",
                    "size_bytes": len(upload.content),
                    "source_hash": source_hash,
                    "reason": "same_source_hash",
                    "existing_document_id": existing.get("document_id"),
                    "existing_source_name": existing.get("source_name"),
                })
                continue

            with self.job_lock:
                if source_hash in self._pending_source_hashes:
                    duplicate_uploads.append({
                        "original_name": upload.filename,
                        "stored_name": stored_name,
                        "stored_path": "",
                        "size_bytes": len(upload.content),
                        "source_hash": source_hash,
                        "reason": "already_processing",
                        "existing_document_id": None,
                        "existing_source_name": "processing",
                    })
                    continue
                self._pending_source_hashes.add(source_hash)

            seen_batch_hashes.add(source_hash)
            reserved_hashes.append(source_hash)
            uploads_to_process.append(upload)

        return uploads_to_process, duplicate_uploads, reserved_hashes

    def _register_pending_hashes(self, hashes: list[str]) -> None:
        valid_hashes = [item for item in hashes if item]
        if not valid_hashes:
            return
        with self.job_lock:
            self._pending_source_hashes.update(valid_hashes)

    def _release_pending_hashes(self, hashes: list[str]) -> None:
        valid_hashes = [item for item in hashes if item]
        if not valid_hashes:
            return
        with self.job_lock:
            for item in valid_hashes:
                self._pending_source_hashes.discard(item)

    def _hash_bytes(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _make_run_id(self) -> str:
        base = "review-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = base
        counter = 2
        while (self.runs_root / candidate).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _sanitize_filename(self, filename: str) -> str:
        cleaned = filename.replace("\\", "/").split("/")[-1].strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" .")
        return re.sub(r'[<>:"/\\\\|?*]+', "_", cleaned)

    def _normalize_source_key(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    def _find_existing_document_by_name(self, filename: str) -> dict[str, Any] | None:
        target = self._normalize_source_key(filename)
        if not target:
            return None
        for document in self.get_document_list():
            source_name = self._normalize_source_key(str(document.get("source_name") or ""))
            if source_name and source_name == target:
                return document
        return None

    def _dedupe_name(self, filename: str, used_names: set[str]) -> str:
        path = Path(filename)
        candidate = filename
        counter = 2
        while candidate.lower() in used_names:
            candidate = f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        used_names.add(candidate.lower())
        return candidate

    def _resolve_relative_path(self, base_root: Path, parts: list[str]) -> Path | None:
        base_resolved = base_root.resolve()
        candidate = (base_root / Path(*parts)).resolve()
        try:
            candidate.relative_to(base_resolved)
        except ValueError:
            return None
        if candidate.is_dir():
            candidate = candidate / "index.html"
        return candidate if candidate.exists() else None

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(read_text_with_fallback(path)[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _build_office_pdf_strategy_compare(
        self,
        *,
        session_root: Path,
        saved_uploads: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        office_exts = {".doc", ".docx", ".hwp", ".hwpx"}
        compare_targets = [
            item for item in saved_uploads
            if Path(str(item.get("stored_path") or "")).suffix.lower() in office_exts
        ]
        if not compare_targets:
            return None

        compare_root = session_root / "reports" / "office_pdf_strategy_compare"
        # LibreOffice conversion on Windows becomes unreliable with long nested paths.
        # Keep the staging area short and use hashed artifact stems for filenames.
        converted_root = self.project_root / "outputs" / "tmp_office_pdf_compare" / session_root.name
        artifacts_root = compare_root / "artifacts"
        ensure_directory(compare_root)
        ensure_directory(converted_root)
        ensure_directory(artifacts_root)

        extractor = PdfMarkdownExtractor(enable_omitted_picture_ocr=False)
        documents: list[dict[str, Any]] = []

        for item in compare_targets:
            source_path = Path(str(item.get("stored_path") or ""))
            if not source_path.exists():
                continue
            artifact_stem = make_artifact_stem(source_path)

            converted_pdf_path = converted_root / f"{artifact_stem}.pdf"
            conversion_ok = self.convert_source_to_pdf(source_path, converted_pdf_path)
            document_entry: dict[str, Any] = {
                "source_name": source_path.name,
                "source_path": source_path.as_posix(),
                "source_extension": source_path.suffix.lower(),
                "converted_pdf_path": converted_pdf_path.as_posix(),
                "conversion_succeeded": conversion_ok and converted_pdf_path.exists(),
                "strategies": [],
                "document_path": None,
                "error": None,
            }
            if not conversion_ok or not converted_pdf_path.exists():
                document_entry["error"] = "pdf_conversion_failed"
                documents.append(document_entry)
                continue

            try:
                classification = classify_document(converted_pdf_path)
                with fitz.open(converted_pdf_path) as document:
                    producer = str((document.metadata or {}).get("producer") or "")
                    page_count = document.page_count
                    document_entry["pdf_metadata"] = {
                        "producer": producer,
                        "page_count": page_count,
                        "selected_strategy": classification.pdf_parser_strategy,
                    }
                    for strategy_name in ("structtree-actualtext", "pymupdf4llm"):
                        result = extractor.extract(
                            converted_pdf_path,
                            document,
                            classification,
                            strategy_name=strategy_name,
                        )
                        strategy_stem = f"{artifact_stem}__{strategy_name}"
                        markdown_path = artifacts_root / f"{strategy_stem}.md"
                        metadata_path = artifacts_root / f"{strategy_stem}.json"
                        markdown_path.write_text(result.markdown.rstrip() + "\n", encoding="utf-8")
                        write_json(
                            metadata_path,
                            {
                                "strategy_name": result.strategy_name,
                                "applied_strategy": result.applied_strategy,
                                "metadata": result.metadata,
                                "elapsed_ms": result.elapsed_ms,
                                "issue_count": len(result.issues),
                                "issues": [issue.__dict__ for issue in result.issues],
                                "char_count": len(result.markdown),
                                "line_count": len(result.markdown.splitlines()),
                            },
                        )
                        document_entry["strategies"].append(
                            {
                                "strategy_name": strategy_name,
                                "applied_strategy": result.applied_strategy,
                                "markdown_path": markdown_path.relative_to(compare_root).as_posix(),
                                "metadata_path": metadata_path.relative_to(compare_root).as_posix(),
                                "elapsed_ms": result.elapsed_ms,
                                "issue_count": len(result.issues),
                                "char_count": len(result.markdown),
                                "line_count": len(result.markdown.splitlines()),
                                "markdown": result.markdown,
                                "issues": [issue.__dict__ for issue in result.issues],
                            }
                        )
            except Exception as error:
                document_entry["error"] = str(error)

            document_html_path = compare_root / f"{artifact_stem}.html"
            write_json(
                compare_root / f"{artifact_stem}.summary.json",
                {
                    key: value
                    for key, value in document_entry.items()
                    if key != "strategies"
                }
                | {
                    "strategies": [
                        {
                            key: value
                            for key, value in strategy.items()
                            if key not in {"markdown", "issues"}
                        }
                        | {"issues": strategy.get("issues", [])}
                        for strategy in document_entry.get("strategies", [])
                    ]
                },
            )
            document_entry["document_path"] = document_html_path.as_posix()
            write_text(document_html_path, self._render_office_pdf_compare_document_html(document_entry, compare_root))
            documents.append(document_entry)

        manifest = {
            "generated_at": iso_now(),
            "index_path": (compare_root / "index.html").as_posix(),
            "document_count": len(documents),
            "documents": [
                {
                    **{
                        key: value
                        for key, value in item.items()
                        if key not in {"strategies"}
                    },
                    "strategies": [
                        {
                            key: value
                            for key, value in strategy.items()
                            if key not in {"markdown", "issues"}
                        }
                        | {"issues": strategy.get("issues", [])}
                        for strategy in item.get("strategies", [])
                    ],
                }
                for item in documents
            ],
        }
        write_json(compare_root / "manifest.json", manifest)
        write_text(compare_root / "index.html", self._render_office_pdf_compare_index_html(manifest, compare_root))
        return manifest

    def _render_office_pdf_compare_index_html(self, manifest: dict[str, Any], compare_root: Path) -> str:
        rows: list[str] = []
        for document in manifest.get("documents", []):
            source_name = html.escape(str(document.get("source_name") or ""))
            source_ext = html.escape(str(document.get("source_extension") or ""))
            selected_strategy = html.escape(
                str((document.get("pdf_metadata") or {}).get("selected_strategy") or "n/a")
            )
            producer = html.escape(str((document.get("pdf_metadata") or {}).get("producer") or "n/a"))
            page_count = html.escape(str((document.get("pdf_metadata") or {}).get("page_count") or ""))
            error = html.escape(str(document.get("error") or ""))
            raw_doc_path = str(document.get("document_path") or "").strip()
            doc_href = html.escape(Path(raw_doc_path).relative_to(compare_root).as_posix()) if raw_doc_path else ""
            action = f'<a href="{doc_href}">Open Compare</a>' if doc_href else "-"
            status = "ok" if document.get("conversion_succeeded") and not error else error or "conversion_failed"
            rows.append(
                "<tr><td>{name}</td><td>{ext}</td><td>{pages}</td><td>{producer}</td><td>{selected}</td><td>{status}</td><td>{action}</td></tr>".format(
                    name=source_name,
                    ext=source_ext,
                    pages=page_count,
                    producer=producer,
                    selected=selected_strategy,
                    status=html.escape(status),
                    action=action,
                )
            )
        if not rows:
            rows.append("<tr><td colspan='7'>No office documents were converted in this run.</td></tr>")
        return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Office to PDF Strategy Compare</title>
<style>
body {{ font-family: "Segoe UI", "Malgun Gothic", sans-serif; margin: 0; background: #f6f8fb; color: #172033; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }}
.card {{ background: #fff; border: 1px solid #dce4f1; border-radius: 18px; padding: 20px; box-shadow: 0 10px 30px rgba(15,23,42,0.05); }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 12px 10px; border-bottom: 1px solid #e7edf6; text-align: left; vertical-align: top; font-size: 14px; }}
th {{ color: #4a5d79; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
a {{ color: #2457a7; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ background: #f2f5fb; padding: 2px 6px; border-radius: 6px; }}
</style></head><body><main>
<div class="card">
  <h1>Office Upload PDF Strategy Compare</h1>
  <p>DOC/HWP uploads are converted to PDF and compared side by side using <code>structtree-actualtext</code> and <code>pymupdf4llm</code>.</p>
  <table>
    <thead><tr><th>Document</th><th>Ext</th><th>Pages</th><th>Producer</th><th>Metadata Selected</th><th>Status</th><th>Compare</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
</main></body></html>""".format(rows="".join(rows))

    def _render_office_pdf_compare_document_html(self, document: dict[str, Any], compare_root: Path) -> str:
        strategy_sections: list[str] = []
        for strategy in document.get("strategies", []):
            metadata_href = html.escape(str(strategy.get("metadata_path") or ""))
            markdown_href = html.escape(str(strategy.get("markdown_path") or ""))
            issues = strategy.get("issues") or []
            issues_html = (
                "<ul>{items}</ul>".format(
                    items="".join(
                        "<li><strong>{code}</strong>: {message}</li>".format(
                            code=html.escape(str(issue.get("code") or "")),
                            message=html.escape(str(issue.get("message") or "")),
                        )
                        for issue in issues
                    )
                )
                if issues
                else "<p>No issues recorded.</p>"
            )
            strategy_sections.append(
                """
<section class="strategy-card">
  <div class="strategy-head">
    <h2>{name}</h2>
    <div class="meta">Applied: <code>{applied}</code> | {chars} chars | {lines} lines | {elapsed} ms</div>
    <div class="links"><a href="{md}">Markdown</a> · <a href="{meta}">Metadata JSON</a></div>
  </div>
  <div class="issues">{issues}</div>
  <pre>{markdown}</pre>
</section>
""".format(
                    name=html.escape(str(strategy.get("strategy_name") or "")),
                    applied=html.escape(str(strategy.get("applied_strategy") or "")),
                    chars=html.escape(str(strategy.get("char_count") or 0)),
                    lines=html.escape(str(strategy.get("line_count") or 0)),
                    elapsed=html.escape(str(strategy.get("elapsed_ms") or 0)),
                    md=markdown_href,
                    meta=metadata_href,
                    issues=issues_html,
                    markdown=html.escape(str(strategy.get("markdown") or "")),
                )
            )
        if not strategy_sections:
            strategy_sections.append("<p>No strategy output available.</p>")

        producer = html.escape(str((document.get("pdf_metadata") or {}).get("producer") or "n/a"))
        page_count = html.escape(str((document.get("pdf_metadata") or {}).get("page_count") or ""))
        selected = html.escape(str((document.get("pdf_metadata") or {}).get("selected_strategy") or "n/a"))
        return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font-family: "Segoe UI", "Malgun Gothic", sans-serif; margin: 0; background: #f6f8fb; color: #172033; }}
main {{ max-width: 1600px; margin: 0 auto; padding: 32px 24px 48px; }}
.hero, .strategy-grid {{ display: grid; gap: 18px; }}
.hero-card, .strategy-card {{ background: #fff; border: 1px solid #dce4f1; border-radius: 18px; padding: 20px; box-shadow: 0 10px 30px rgba(15,23,42,0.05); }}
.strategy-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }}
.meta {{ color: #4a5d79; font-size: 14px; line-height: 1.7; }}
.links a {{ color: #2457a7; text-decoration: none; }}
.links a:hover {{ text-decoration: underline; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafe; border: 1px solid #e5ebf5; border-radius: 12px; padding: 14px; font-size: 12px; line-height: 1.6; overflow: auto; max-height: 70vh; }}
code {{ background: #f2f5fb; padding: 2px 6px; border-radius: 6px; }}
ul {{ margin: 8px 0 0; padding-left: 18px; }}
@media (max-width: 1100px) {{ .strategy-grid {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
<div class="hero">
  <div class="hero-card">
    <p><a href="index.html">Back to index</a></p>
    <h1>{title}</h1>
    <div class="meta">
      <div>Source: <code>{source_path}</code></div>
      <div>Converted PDF: <code>{pdf_path}</code></div>
      <div>PDF Pages: <code>{page_count}</code></div>
      <div>Producer: <code>{producer}</code></div>
      <div>Metadata-selected strategy: <code>{selected}</code></div>
    </div>
  </div>
</div>
<div class="strategy-grid">{sections}</div>
</main></body></html>""".format(
            title=html.escape(str(document.get("source_name") or "")),
            source_path=html.escape(str(document.get("source_path") or "")),
            pdf_path=html.escape(str(document.get("converted_pdf_path") or "")),
            page_count=page_count,
            producer=producer,
            selected=selected,
            sections="".join(strategy_sections),
        )

    def _find_latest_chunking_comparison_dir(self) -> Path | None:
        root = self.project_root / "outputs" / "chunking_strategy_compare"
        if not root.exists():
            return None
        candidates = [
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name.startswith("chunking_compare_")
            and (path / "summary.json").exists()
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.name, reverse=True)
        return candidates[0]

    def _iter_structured_document_paths(self) -> list[Path]:
        roots = [self.project_root / "data" / "structured" / "documents"]
        if self.runs_root.exists():
            for path in sorted(self.runs_root.glob("*/structured/documents"), reverse=True):
                roots.append(path)
            for path in sorted(self.runs_root.glob("*/parsing/json"), reverse=True):
                roots.append(path)
        paths: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json")):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
        return paths


def render_launcher_html(status: dict[str, Any]) -> str:
    project_review = status.get("project_review") or {}
    latest_upload = status.get("latest_upload") or {}
    vector_index = status.get("vector_index") or {}
    openai_status = status.get("openai") or {}
    embedding_comparison = status.get("embedding_comparison") or {}
    chunking_comparison = status.get("chunking_comparison") or {}

    recent_rows = []
    for session in status.get("recent_uploads", []):
        if not session:
            continue
        recent_rows.append(
            "<tr><td>{run_id}</td><td>{created_at}</td><td>{upload_count}</td><td>{parsed_documents}</td><td>{duplicate_upload_count}</td></tr>".format(
                run_id=html.escape(session.get("run_id") or ""),
                created_at=html.escape(session.get("created_at") or ""),
                upload_count=session.get("upload_count", 0),
                parsed_documents=session.get("parsed_documents", 0),
                duplicate_upload_count=session.get("duplicate_upload_count", 0),
            )
        )
    if not recent_rows:
        recent_rows.append("<tr><td colspan='5'>아직 업로드 실행 이력이 없습니다.</td></tr>")

    qa_mode = (
        f"OpenAI {html.escape(openai_status.get('model') or 'gpt-5.2')} 모델로 답변을 생성하고 출처를 함께 보여줍니다."
        if openai_status.get("enabled")
        else "현재 OPENAI_API_KEY가 없어 로컬 근거 기반 응답만 사용합니다."
    )

    comparison_rows = []
    for item in (embedding_comparison.get("latest_report") or {}).get("models", []):
        comparison_rows.append(
            "<tr><td>{model}</td><td>{hit1}</td><td>{hit3}</td><td>{mrr}</td></tr>".format(
                model=html.escape(item.get("model", "")),
                hit1=item.get("hit_at_1", ""),
                hit3=item.get("hit_at_3", ""),
                mrr=item.get("mrr", ""),
            )
        )
    if not comparison_rows:
        comparison_rows.append("<tr><td colspan='4'>비교 결과가 아직 없습니다.</td></tr>")

    secondary_action = ""
    if project_review.get("exists"):
        compare_href = html.escape(project_review.get("compare_url") or "/project-parsing/index.html")
        secondary_action = f'<a class="button secondary" href="{compare_href}">프로젝트 비교 열기</a>'
    tertiary_action = ""
    if project_review.get("process_compare_url"):
        tertiary_action = f'<a class="button secondary" href="{html.escape(project_review.get("process_compare_url") or "")}">새 프로세스 비교 열기</a>'

    comparison_mode = (
        "OpenAI API가 연결되어 있으면 임베딩 모델 비교를 실행하고 아래 표를 갱신합니다."
        if embedding_comparison.get("enabled")
        else "현재 OPENAI_API_KEY가 없어 임베딩 비교를 실행할 수 없습니다."
    )
    chunking_mode = (
        "문서를 선택해서 최신 5가지 청킹 결과를 비교할 수 있습니다."
        if chunking_comparison.get("exists")
        else "청킹 비교 결과가 아직 없습니다."
    )
    chunking_action = (
        '<a class="button secondary" href="/chunking-compare">5가지 청킹 결과 보기</a>'
        if chunking_comparison.get("exists")
        else ""
    )

    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Document Review Launcher</title>
<style>
body {{ font-family: Georgia, "Malgun Gothic", serif; margin: 0; background: #f7f1e8; color: #231c14; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
.grid {{ display: grid; gap: 16px; grid-template-columns: 1.2fr 0.8fr; }}
.panel {{ background: rgba(255,255,255,0.82); border: 1px solid #dacbb8; border-radius: 20px; padding: 18px; }}
.stack {{ display: grid; gap: 16px; }}
.guide {{ margin: 16px 0; padding: 16px 18px; border-radius: 18px; background: rgba(32,79,122,0.08); border: 1px solid #c8d6e4; }}
.guide strong {{ display:block; margin-bottom: 8px; }}
.guide ol {{ margin: 0; padding-left: 20px; line-height: 1.7; }}
.button {{ display: inline-block; background: #ab4e2d; color: #fff; padding: 10px 16px; border-radius: 999px; text-decoration: none; border: 0; cursor: pointer; }}
.button.secondary {{ background: #204f7a; }}
.status {{ white-space: pre-wrap; padding: 12px; min-height: 86px; border-radius: 14px; background: #fff; border: 1px solid #e7dccc; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ padding: 10px 8px; border-bottom: 1px solid #eadfce; text-align: left; }}
code {{ overflow-wrap: anywhere; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><main>
<p>Local Review Launcher</p>
<h1>기본 문서 비교와 신규 문서 파이프라인 실행</h1>
<p>업로드한 문서는 기존 파싱 결과와 비교한 뒤 처리하고, 프로젝트 기본 문서는 비교 대시보드에서 바로 확인할 수 있습니다.</p>
<section class="guide">
<strong>처음 화면에서 이렇게 보시면 됩니다.</strong>
<ol>
<li>프로젝트에 이미 있는 문서를 바로 보려면 `프로젝트 비교 열기` 버튼을 누르세요. 기본 리뷰 목록 화면으로 이동합니다.</li>
<li>목록 화면에서는 보고 싶은 문서 제목을 클릭하세요. 그러면 문서 상세 리뷰 페이지가 열립니다.</li>
<li>문서 상세 페이지에서 `Postprocess Diff` 섹션으로 내려가면 후처리 전후 변경 부분이 하이라이트되어 보입니다.</li>
<li>새 문서를 직접 검사하려면 파일을 선택한 뒤 `파이프라인 실행`을 누르세요. 완료되면 해당 문서의 상세 리뷰 페이지로 자동 이동합니다.</li>
</ol>
</section>
<div class="grid">
<section class="panel">
<h2>Upload And Run</h2>
<p>지원 확장자: <code>{supported}</code></p>
<form id="upload-form">
<input id="documents" name="documents" type="file" multiple accept="{accept}">
<p><button class="button" id="submit-button" type="submit">파이프라인 실행</button> {secondary_action} {tertiary_action}</p>
<div class="status" id="status-box">문서를 선택하고 실행하면 진행 상황을 여기서 보여줍니다. 기존 문서만 보고 싶다면 위 안내대로 `프로젝트 비교 열기`를 사용하세요.</div>
</form>
</section>
<aside class="stack">
<section class="panel"><h3>Project Review</h3><p>문서 수: <strong>{project_count}</strong></p><p>버전: <code>{version}</code></p><p>기존 문서 리뷰를 보려면 왼쪽의 <strong>프로젝트 비교 열기</strong> 버튼을 누르세요.</p></section>
<section class="panel"><h3>Latest Upload</h3><p>최근 parsed: <strong>{latest_parsed}</strong></p><p>중복 문서: <strong>{latest_duplicates}</strong></p><p>새 업로드가 성공하면 상세 리뷰 페이지로 자동 이동합니다.</p></section>
<section class="panel"><h3>Vector Index</h3><p>인덱싱 문서 수: <strong>{vector_count}</strong></p><p>업데이트: <code>{vector_updated}</code></p></section>
</aside></div>
<section class="panel"><h2>Recent Upload Runs</h2><table><thead><tr><th>Run</th><th>Created</th><th>Uploads</th><th>Parsed</th><th>Duplicates</th></tr></thead><tbody>{recent_rows}</tbody></table></section>
<section class="panel">
<h2>Ask Questions</h2>
<p>{qa_mode}</p>
<p>기존 custom QA와 LangChain RetrievalQA 결과를 나란히 보여줍니다.</p>
<form id="query-form">
<textarea id="query-input" rows="4" style="width:100%;padding:12px;border-radius:14px;border:1px solid #dbcdbb;">질문을 입력하세요.</textarea>
<p>
<select id="query-strategy" style="padding:8px;border-radius:10px;border:1px solid #dbcdbb;">
<option value="rule_based">rule_based</option>
<option value="semantic">semantic</option>
</select>
<button class="button secondary" id="query-button" type="submit">질의응답 실행</button>
</p>
<div class="status" id="query-status">문서를 인덱싱한 뒤 질문을 실행할 수 있습니다.</div>
</form>
</section>
<section class="panel">
<h2>문서 요약</h2>
<p id="summarize-mode-desc">인덱싱된 문서를 선택해 LLM 기반 요약을 생성합니다.</p>
<select id="summarize-doc-select" style="padding:8px;border-radius:10px;border:1px solid #dbcdbb;width:100%;margin-bottom:8px;"><option value="">-- 문서 선택 --</option></select>
<button class="button secondary" id="summarize-button" type="button">요약 생성</button>
<div class="status" id="summarize-status">문서를 선택하고 요약 생성을 클릭하세요.</div>
</section>
<section class="panel">
<h2>Embedding Comparison</h2>
<p>{comparison_mode}</p>
<p><button class="button secondary" id="embedding-compare-button" type="button">임베딩 비교 실행</button></p>
<div class="status" id="embedding-status">현재 상태를 불러오는 중입니다.</div>
<table><thead><tr><th>Model</th><th>Hit@1</th><th>Hit@3</th><th>MRR</th></tr></thead><tbody>{comparison_rows}</tbody></table>
</section>
<section class="panel">
<h2>Chunking Comparison</h2>
<p>{chunking_mode}</p>
<p>{chunking_action}</p>
</section>
<script>
const form = document.getElementById('upload-form');
const input = document.getElementById('documents');
const button = document.getElementById('submit-button');
const statusBox = document.getElementById('status-box');
const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const queryButton = document.getElementById('query-button');
const queryStatus = document.getElementById('query-status');
const embeddingCompareButton = document.getElementById('embedding-compare-button');
const embeddingStatus = document.getElementById('embedding-status');
async function pollJob(jobId) {{
  while (true) {{
    const response = await fetch('/api/jobs/' + jobId);
    const payload = await response.json();
    const progress = payload.progress || {{current:0,total:1}};
    statusBox.textContent = 'Status: ' + payload.status + '\\nProgress: ' + progress.current + ' / ' + progress.total + '\\n' + (payload.message || '');
    if (payload.status === 'completed') {{
      button.disabled = false;
      const result = payload.result || {{}};
      const summary = result.summary || {{}};
      const duplicates = [...(result.duplicate_uploads || []), ...(result.content_duplicate_uploads || [])];
      const vectorIndex = result.vector_index || {{}};
      statusBox.textContent = [
        'Run ID: ' + (result.run_id || ''),
        'Parsed: ' + (summary.parsed_documents ?? 0),
        'Duplicates: ' + duplicates.length,
        'Vector Indexed: ' + (vectorIndex.indexed_documents ?? vectorIndex.document_count ?? 0),
        payload.message || ''
      ].join('\\n');
      if (duplicates.length) {{
        statusBox.textContent += '\\n기존 문서: ' + duplicates.map((item) => item.original_name + ' -> ' + (item.existing_source_name || item.existing_document_id || 'existing document')).join(', ');
      }}
      if (result.review_url && (summary.parsed_documents ?? 0) > 0) {{
        window.location.href = result.review_url;
      }}
      return;
    }}
    if (payload.status === 'failed') {{
      button.disabled = false;
      statusBox.textContent = payload.error || payload.message || '작업이 실패했습니다.';
      return;
    }}
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }}
}}
form.addEventListener('submit', async (event) => {{
  event.preventDefault();
  if (!input.files.length) {{
    statusBox.textContent = '실행할 문서를 먼저 선택해주세요.';
    return;
  }}
  const formData = new FormData();
  for (const file of input.files) formData.append('documents', file, file.name);
  button.disabled = true;
  statusBox.textContent = '업로드를 시작했습니다. 기존 문서와 비교 중입니다.';
  try {{
    const response = await fetch('/api/run', {{ method: 'POST', body: formData }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '파이프라인 실행에 실패했습니다.');
    await pollJob(payload.job_id);
  }} catch (error) {{
    button.disabled = false;
    statusBox.textContent = String(error);
  }}
}});
queryForm.addEventListener('submit', async (event) => {{
  event.preventDefault();
  const query = queryInput.value.trim();
  const strategy = document.getElementById('query-strategy').value;
  if (!query) {{
    queryStatus.textContent = '질문을 입력해주세요.';
    return;
  }}
  queryButton.disabled = true;
  queryStatus.textContent = '질문에 맞는 청크를 검색 중입니다.';
  try {{
    const response = await fetch('/api/query-compare', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ query, strategy }})
    }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '질의응답에 실패했습니다.');
    const renderBlock = (title, result) => {{
      const matches = result.matches || result.source_documents || [];
      const citations = result.citations || [];
      const documentSummaries = result.document_summaries || [];
      const lines = [
        '[' + title + ']',
        'Answer: ' + (result.answer || ''),
        'Model: ' + (result.used_model || 'local evidence synthesis'),
        ''
      ];
      if (result.warning) {{
        lines.push('Warning: ' + result.warning);
        lines.push('');
      }}
      lines.push('Citations:');
      if (citations.length) {{
        citations.forEach((citation, index) => {{
          lines.push((index + 1) + '. ' + (citation.source_name || 'document') + ' / ' + (citation.section_hint || 'section'));
          lines.push('   quote: ' + (citation.quote || ''));
        }});
      }} else {{
        lines.push('No citations available');
      }}
      lines.push('');
      lines.push('Document summaries:');
      if (documentSummaries.length) {{
        documentSummaries.forEach((item, index) => {{
          lines.push((index + 1) + '. ' + (item.source_name || item.document_id || 'document'));
          lines.push('   summary: ' + (item.summary_text || ''));
        }});
      }} else {{
        lines.push('No summaries available');
      }}
      lines.push('');
      lines.push('Top matches:');
      matches.slice(0, 3).forEach((match, index) => {{
        const metadata = match.metadata || {{}};
        lines.push((index + 1) + '. ' + (metadata.source_name || metadata.document_id || 'document') + ' / ' + (metadata.section_hint || 'section'));
        lines.push(((match.document || match.page_content || '')).slice(0, 240));
        lines.push('');
      }});
      return lines.join('\\n');
    }};
    queryStatus.textContent = [
      renderBlock('Custom QA', payload.custom || {{}}),
      '',
      renderBlock('LangChain RetrievalQA', payload.langchain || {{}})
    ].join('\\n');
  }} catch (error) {{
    queryStatus.textContent = String(error);
  }} finally {{
    queryButton.disabled = false;
  }}
}});
const summarizeButton = document.getElementById('summarize-button');
const summarizeStatus = document.getElementById('summarize-status');
const summarizeSelect = document.getElementById('summarize-doc-select');
fetch('/api/documents').then(r => r.json()).then(data => {{
  const docs = data.documents || [];
  summarizeSelect.innerHTML = '<option value="">-- 문서 선택 --</option>' +
    docs.map(d => '<option value="' + esc(d.document_id) + '">' + esc(d.source_name || d.document_id) + '</option>').join('');
}}).catch(() => {{}});
summarizeButton.addEventListener('click', async () => {{
  const selectedId = summarizeSelect.value;
  if (!selectedId) {{
    summarizeStatus.textContent = '요약할 문서를 선택해주세요.';
    return;
  }}
  summarizeButton.disabled = true;
  summarizeStatus.textContent = '요약을 생성하는 중입니다...';
  try {{
    const response = await fetch('/api/summarize', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ document_id: selectedId }})
    }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '요약 생성에 실패했습니다.');
    const lines = [
      'Document: ' + (payload.source_name || payload.document_id || ''),
      'Type: ' + (payload.document_type || ''),
      '',
      'Summary:',
      payload.summary_text || '',
      '',
      'Key Points:',
    ];
    (payload.key_points || []).forEach((pt, i) => lines.push((i + 1) + '. ' + pt));
    lines.push('');
    lines.push('Model: ' + (payload.used_model || ''));
    summarizeStatus.textContent = lines.join('\\n');
  }} catch (error) {{
    summarizeStatus.textContent = String(error);
  }} finally {{
    summarizeButton.disabled = false;
  }}
}});
embeddingCompareButton.addEventListener('click', async () => {{
  embeddingCompareButton.disabled = true;
  embeddingStatus.textContent = '임베딩 비교 작업을 시작합니다.';
  try {{
    const response = await fetch('/api/embedding-comparison', {{ method: 'POST' }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '임베딩 비교 실행에 실패했습니다.');
    while (true) {{
      const jobResponse = await fetch('/api/jobs/' + payload.job_id);
      const jobPayload = await jobResponse.json();
      const progress = jobPayload.progress || {{ current: 0, total: 1 }};
      embeddingStatus.textContent = 'Status: ' + jobPayload.status + '\\nProgress: ' + progress.current + ' / ' + progress.total + '\\n' + (jobPayload.message || '');
      if (jobPayload.status === 'completed') {{
        window.location.reload();
        return;
      }}
      if (jobPayload.status === 'failed') {{
        embeddingStatus.textContent = jobPayload.error || jobPayload.message || '임베딩 비교가 실패했습니다.';
        break;
      }}
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }}
  }} catch (error) {{
    embeddingStatus.textContent = String(error);
  }} finally {{
    embeddingCompareButton.disabled = false;
  }}
}});
</script></main></body></html>
""".format(
        supported=html.escape(", ".join(status.get("supported_extensions", []))),
        accept=html.escape(",".join(sorted(SUPPORTED_EXTENSIONS))),
        secondary_action=secondary_action,
        project_count=project_review.get("document_count", 0),
        version=html.escape(project_review.get("version") or "n/a"),
        latest_parsed=latest_upload.get("parsed_documents", 0),
        latest_duplicates=latest_upload.get("duplicate_upload_count", 0),
        vector_count=vector_index.get("document_count", 0),
        vector_updated=html.escape(vector_index.get("updated_at") or "n/a"),
        qa_mode=qa_mode,
        comparison_mode=comparison_mode,
        chunking_mode=chunking_mode,
        chunking_action=chunking_action,
        comparison_rows="\n".join(comparison_rows),
        recent_rows="\n".join(recent_rows),
    )


def render_document_studio_html(status: dict[str, Any]) -> str:
    project_review = status.get("project_review") or {}
    latest_upload = status.get("latest_upload") or {}
    vector_index = status.get("vector_index") or {}
    openai_status = status.get("openai") or {}
    chunking_comparison = status.get("chunking_comparison") or {}

    qa_mode = (
        f"선택한 문서를 우선 기준으로 답변합니다. OpenAI 모델은 {html.escape(openai_status.get('model') or 'configured model')} 입니다."
        if openai_status.get("enabled")
        else "OpenAI 키가 없으면 로컬 근거 기반 응답만 표시합니다."
    )
    review_action = ""
    if project_review.get("exists"):
        review_href = html.escape(project_review.get("review_url") or "/project-parsing/index.html")
        review_action = f'<a class="ghost-link" href="{review_href}">기존 문서 전체 보기</a>'
    chunking_action = ""
    if chunking_comparison.get("exists"):
        chunking_action = '<a class="ghost-link" href="/chunking-compare">청킹 비교 보기</a>'

    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document Studio</title>
<style>
:root {{
  --bg:#f4efe7;
  --ink:#1f2937;
  --muted:#5f6b7a;
  --line:rgba(31,41,55,0.10);
  --surface:rgba(255,255,255,0.86);
  --surface-strong:#ffffff;
  --accent:#0f766e;
  --accent-soft:rgba(15,118,110,0.10);
  --shadow:0 24px 80px rgba(31,41,55,0.10);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  color:var(--ink);
  font-family:"Segoe UI","Malgun Gothic",sans-serif;
  background:
    radial-gradient(circle at top left, rgba(217,119,6,0.12), transparent 28%),
    radial-gradient(circle at top right, rgba(15,118,110,0.12), transparent 30%),
    linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
}}
main {{ max-width:1360px; margin:0 auto; padding:32px 24px 48px; }}
.shell {{ display:grid; grid-template-columns:280px minmax(0, 1fr); gap:20px; align-items:start; }}
.sidebar, .panel {{
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:28px;
  box-shadow:var(--shadow);
  backdrop-filter: blur(18px);
}}
.sidebar {{ padding:22px; position:sticky; top:24px; }}
.brand {{ font-size:28px; font-weight:800; letter-spacing:-0.04em; margin:0 0 10px; }}
.sidebar p {{ color:var(--muted); line-height:1.6; margin:0; }}
.quickstats {{ display:grid; gap:12px; margin-top:20px; }}
.stat {{ padding:14px 16px; border-radius:18px; background:var(--surface-strong); border:1px solid var(--line); }}
.stat-label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; }}
.stat-value {{ font-size:24px; font-weight:700; margin-top:6px; }}
.sidebar-links {{ display:grid; gap:10px; margin-top:18px; }}
.ghost-link {{ display:inline-flex; align-items:center; gap:8px; width:fit-content; color:var(--accent); text-decoration:none; font-weight:600; }}
.content {{ display:grid; gap:20px; }}
.hero {{ padding:28px; overflow:hidden; position:relative; }}
.hero:before {{ content:""; position:absolute; inset:auto -10% -40% auto; width:320px; height:320px; background:radial-gradient(circle, rgba(15,118,110,0.15), transparent 70%); pointer-events:none; }}
.eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:0.08em; text-transform:uppercase; font-size:12px; }}
.hero-grid {{ display:grid; grid-template-columns:minmax(0, 1.2fr) minmax(280px, 0.8fr); gap:20px; align-items:stretch; }}
.hero-copy h1 {{ margin:10px 0 12px; font-size:clamp(34px, 5vw, 58px); line-height:1.02; letter-spacing:-0.05em; }}
.hero-copy p {{ margin:0; color:var(--muted); font-size:17px; line-height:1.7; max-width:760px; }}
.upload-box {{ margin-top:24px; border:1.5px dashed rgba(15,118,110,0.30); border-radius:28px; background:linear-gradient(180deg, rgba(255,255,255,0.88), rgba(240,249,247,0.92)); padding:22px; }}
.upload-box.dragover {{ border-color:var(--accent); background:linear-gradient(180deg, rgba(222,247,244,0.95), rgba(255,255,255,0.96)); }}
.upload-title {{ font-size:24px; font-weight:700; margin:0; }}
.upload-subtitle {{ margin:6px 0 0; color:var(--muted); line-height:1.6; }}
.upload-actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }}
.button {{ appearance:none; border:0; border-radius:999px; padding:14px 20px; font:inherit; font-weight:700; cursor:pointer; }}
.button.primary {{ background:#111827; color:#fff; }}
.button.secondary {{ background:var(--accent-soft); color:var(--accent); }}
.upload-filelist {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:10px; }}
.file-chip {{ padding:10px 14px; border-radius:999px; background:var(--surface-strong); border:1px solid var(--line); font-size:13px; }}
.hero-notes {{ display:grid; gap:12px; align-content:start; }}
.note-card {{ padding:18px; border-radius:22px; background:linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,245,239,0.88)); border:1px solid var(--line); }}
.note-card h3 {{ margin:0 0 8px; font-size:16px; }}
.note-card p {{ margin:0; color:var(--muted); line-height:1.6; }}
.workspace {{ display:grid; grid-template-columns:minmax(320px, 0.9fr) minmax(0, 1.1fr); gap:20px; }}
.stack {{ display:grid; gap:20px; }}
.panel {{ padding:22px; }}
.panel h2 {{ margin:0 0 10px; font-size:22px; letter-spacing:-0.03em; }}
.panel p.helper {{ margin:0 0 16px; color:var(--muted); line-height:1.6; }}
.status-box {{ white-space:pre-wrap; min-height:120px; padding:16px; border-radius:18px; background:linear-gradient(180deg, rgba(248,250,252,0.98), rgba(255,255,255,0.98)); border:1px solid var(--line); line-height:1.6; }}
.document-list {{ display:grid; gap:12px; max-height:560px; overflow:auto; padding-right:2px; }}
.document-card {{ width:100%; text-align:left; border:1px solid var(--line); background:var(--surface-strong); border-radius:20px; padding:16px; cursor:pointer; }}
.document-card.active {{ border-color:rgba(15,118,110,0.45); box-shadow:0 0 0 3px rgba(15,118,110,0.10); }}
.document-card small {{ display:block; color:var(--muted); margin-bottom:6px; }}
.document-card strong {{ display:block; font-size:16px; line-height:1.5; }}
.document-card p {{ margin:8px 0 0; color:var(--muted); line-height:1.5; font-size:14px; }}
.tag-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
.tag {{ display:inline-flex; padding:6px 10px; border-radius:999px; background:rgba(15,118,110,0.08); color:var(--accent); font-size:12px; font-weight:700; }}
.summary-card {{ padding:20px; border-radius:24px; background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,250,251,0.96)); border:1px solid var(--line); }}
.summary-meta {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; color:var(--muted); font-size:13px; }}
.summary-text {{ font-size:16px; line-height:1.8; margin:0; }}
.keypoints {{ display:grid; gap:10px; margin-top:18px; }}
.keypoint {{ padding:14px 16px; border-radius:18px; background:rgba(15,118,110,0.06); border:1px solid rgba(15,118,110,0.10); }}
.question-box {{ width:100%; min-height:120px; resize:vertical; border-radius:20px; border:1px solid var(--line); padding:16px; font:inherit; background:var(--surface-strong); }}
.toolbar {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-top:14px; }}
select {{ padding:12px 14px; border-radius:999px; border:1px solid var(--line); background:var(--surface-strong); font:inherit; }}
.answer-stack {{ display:grid; gap:14px; }}
.answer-card {{ padding:18px; border-radius:22px; background:var(--surface-strong); border:1px solid var(--line); }}
.answer-card h3 {{ margin:0 0 10px; font-size:17px; }}
.answer-card p {{ margin:0; line-height:1.7; }}
.evidence-list {{ display:grid; gap:10px; margin-top:14px; }}
.evidence-item {{ padding:14px; border-radius:16px; background:#f8fafc; border:1px solid var(--line); }}
.muted {{ color:var(--muted); }}
input[type=file] {{ display:none; }}
.doc-toolbar {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
.select-all-label {{ display:flex; align-items:center; gap:6px; cursor:pointer; font-weight:600; font-size:14px; user-select:none; }}
.button.danger {{ background:#dc2626; color:#fff; padding:8px 18px; font-size:14px; }}
.button.danger:disabled {{ opacity:0.45; cursor:not-allowed; }}
.doc-card-row {{ display:flex; align-items:flex-start; gap:10px; }}
.doc-card-row .doc-checkbox {{ margin-top:18px; width:16px; height:16px; flex-shrink:0; cursor:pointer; accent-color:var(--accent); }}
.doc-card-row .document-card {{ flex:1; min-width:0; }}
@media (max-width: 1100px) {{ .shell, .hero-grid, .workspace {{ grid-template-columns:1fr; }} .sidebar {{ position:static; }} }}
</style></head>
<body><main>
<div class="shell">
  <aside class="sidebar">
    <p class="brand">Document Studio</p>
    <p>첫 화면에서 문서를 올리고, 처리 상태를 보면서 바로 요약과 질의를 이어가는 흐름에 맞춘 UI입니다.</p>
    <div class="quickstats">
      <div class="stat"><div class="stat-label">Indexed Documents</div><div class="stat-value">{vector_count}</div></div>
      <div class="stat"><div class="stat-label">Latest Parsed</div><div class="stat-value">{latest_parsed}</div></div>
      <div class="stat"><div class="stat-label">Project Docs</div><div class="stat-value">{project_count}</div></div>
    </div>
    <div class="sidebar-links">
      {review_action}
      {chunking_action}
    </div>
  </aside>
  <section class="content">
    <section class="panel hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow">NotebookLM Style Flow</div>
          <h1>문서를 던지면, 요약과 질문이 바로 이어지는 첫 화면</h1>
          <p>업로드 이후의 파싱, 인덱싱, 후처리는 내부에서 처리하고, 화면에는 현재 상태와 선택한 문서의 요약과 질의응답만 선명하게 보여줍니다.</p>
          <form id="upload-form" class="upload-box">
            <p class="upload-title">드래그 앤 드롭 또는 파일 선택</p>
            <p class="upload-subtitle">지원 형식: <code>{supported}</code><br>문서가 완료되면 자동으로 목록에 반영되고, 바로 요약과 질문을 이어갈 수 있습니다.</p>
            <input id="documents" name="documents" type="file" multiple accept="{accept}">
            <div class="upload-actions">
              <button class="button primary" id="pick-button" type="button">문서 선택</button>
              <button class="button secondary" id="submit-button" type="submit">업로드 후 처리 시작</button>
            </div>
            <div class="upload-filelist" id="file-list">
              <span class="file-chip">아직 선택된 파일이 없습니다.</span>
            </div>
          </form>
        </div>
        <div class="hero-notes">
          <div class="note-card"><h3>처리 방식</h3><p>문서 업로드 후 파싱과 벡터 인덱싱을 순서대로 수행합니다. 중복 문서는 기존 결과를 재사용합니다.</p></div>
          <div class="note-card"><h3>질의응답</h3><p>{qa_mode}</p></div>
          <div class="note-card"><h3>최근 상태</h3><p>벡터 인덱스 마지막 업데이트: <code>{vector_updated}</code><br>최근 업로드 중복 문서: <strong>{latest_duplicates}</strong></p></div>
        </div>
      </div>
    </section>
    <section class="workspace">
      <div class="stack">
        <section class="panel">
          <h2>Processing</h2>
          <p class="helper">문서 업로드 이후 상태를 실시간으로 보여줍니다.</p>
          <div class="status-box" id="status-box">문서를 올리면 여기에서 진행 상황을 확인할 수 있습니다.</div>
        </section>
        <section class="panel">
          <h2>Documents</h2>
          <p class="helper">프로젝트 문서와 최근 업로드 문서를 함께 보여줍니다. 선택한 문서 기준으로 요약과 질문이 동작합니다.</p>
          <div class="doc-toolbar">
            <label class="select-all-label"><input type="checkbox" id="select-all-checkbox"> 전체 선택</label>
            <button class="button danger" id="delete-selected-button" type="button" disabled>삭제</button>
          </div>
          <div class="document-list" id="document-list">
            <div class="status-box">문서 목록을 불러오는 중입니다.</div>
          </div>
        </section>
      </div>
      <div class="stack">
        <section class="panel">
          <h2>Summary</h2>
          <p class="helper">선택한 문서를 한 번에 훑을 수 있는 요약 결과입니다.</p>
          <div id="summary-panel" class="summary-card">
            <p class="summary-text muted">왼쪽에서 문서를 선택한 뒤 요약을 실행하세요.</p>
          </div>
          <div class="toolbar">
            <button class="button secondary" id="summarize-button" type="button">선택 문서 요약</button>
          </div>
        </section>
        <section class="panel">
          <h2>Ask</h2>
          <p class="helper">질문은 선택한 문서를 우선 탐색합니다. 필요하면 인덱스 전체에서 보강합니다.</p>
          <form id="query-form">
            <textarea id="query-input" class="question-box" placeholder="예: 이 문서의 핵심 조건을 3줄로 정리해줘"></textarea>
            <div class="toolbar">
              <select id="query-strategy">
                <option value="rule_based">rule_based</option>
                <option value="semantic">semantic</option>
              </select>
              <button class="button primary" id="query-button" type="submit">질문하기</button>
            </div>
          </form>
          <div id="query-results" class="answer-stack" style="margin-top:16px;">
            <div class="status-box">질문 결과가 여기에 표시됩니다.</div>
          </div>
        </section>
      </div>
    </section>
  </section>
</div>
<script>
const state = {{ documents: [], selectedDocumentId: '', latestRun: null, checkedIds: new Set() }};
const uploadForm = document.getElementById('upload-form');
const uploadBox = uploadForm;
const fileInput = document.getElementById('documents');
const pickButton = document.getElementById('pick-button');
const submitButton = document.getElementById('submit-button');
const fileList = document.getElementById('file-list');
const statusBox = document.getElementById('status-box');
const documentList = document.getElementById('document-list');
const summarizeButton = document.getElementById('summarize-button');
const summaryPanel = document.getElementById('summary-panel');
const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const queryButton = document.getElementById('query-button');
const queryResults = document.getElementById('query-results');
const selectAllCheckbox = document.getElementById('select-all-checkbox');
const deleteSelectedButton = document.getElementById('delete-selected-button');
selectAllCheckbox.addEventListener('change', () => {{
  if (selectAllCheckbox.checked) {{
    state.documents.forEach((doc) => state.checkedIds.add(doc.document_id));
  }} else {{
    state.checkedIds.clear();
  }}
  renderDocumentList();
}});
deleteSelectedButton.addEventListener('click', async () => {{
  if (state.checkedIds.size === 0) return;
  const ids = Array.from(state.checkedIds);
  if (!confirm(`선택한 문서 ${{ids.length}}개를 벡터 DB에서 삭제합니다. 계속하시겠습니까?`)) return;
  deleteSelectedButton.disabled = true;
  deleteSelectedButton.textContent = '삭제 중...';
  try {{
    const response = await fetch('/api/delete-documents', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ document_ids: ids }})
    }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '삭제에 실패했습니다.');
    state.checkedIds.clear();
    await loadDocuments();
  }} catch (error) {{
    alert(String(error));
  }} finally {{
    deleteSelectedButton.textContent = '삭제';
    updateDeleteButton();
  }}
}});
function esc(value) {{
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}}
function setStatus(lines) {{
  statusBox.textContent = Array.isArray(lines) ? lines.join('\\n') : String(lines ?? '');
}}
function renderFileList(files) {{
  if (!files.length) {{
    fileList.innerHTML = '<span class="file-chip">아직 선택된 파일이 없습니다.</span>';
    return;
  }}
  fileList.innerHTML = files.map((file) => `<span class="file-chip">${{esc(file.name)}} · ${{Math.max(1, Math.round(file.size / 1024))}}KB</span>`).join('');
}}
function getSelectedDocument() {{
  return state.documents.find((item) => item.document_id === state.selectedDocumentId) || null;
}}
function chooseDocument(documentId) {{
  state.selectedDocumentId = documentId;
  renderDocumentList();
  const selected = getSelectedDocument();
  if (!selected) {{
    summaryPanel.innerHTML = '<p class="summary-text muted">선택된 문서가 없습니다.</p>';
    queryResults.innerHTML = '<div class="status-box">질문 결과가 여기에 표시됩니다.</div>';
    return;
  }}
  summaryPanel.innerHTML = `<div class="summary-meta"><span>${{esc(selected.source_name || selected.document_id)}}</span><span>${{esc(selected.document_type || selected.extension || 'unknown')}}</span><span>${{esc(selected.origin || 'project')}}</span></div><p class="summary-text">${{esc((selected.basic_summary || {{}}).summary_text || '요약을 실행하면 결과가 여기에 표시됩니다.')}}</p>`;
}}
function updateDeleteButton() {{
  const deleteBtn = document.getElementById('delete-selected-button');
  const selectAllCb = document.getElementById('select-all-checkbox');
  if (deleteBtn) deleteBtn.disabled = state.checkedIds.size === 0;
  if (selectAllCb) {{
    selectAllCb.indeterminate = state.checkedIds.size > 0 && state.checkedIds.size < state.documents.length;
    selectAllCb.checked = state.documents.length > 0 && state.checkedIds.size === state.documents.length;
  }}
}}
function renderDocumentList() {{
  if (!state.documents.length) {{
    documentList.innerHTML = '<div class="status-box">표시할 문서가 없습니다.</div>';
    updateDeleteButton();
    return;
  }}
  documentList.innerHTML = state.documents.map((doc) => {{
    const summary = (doc.basic_summary || {{}}).summary_text || '문서 요약이 아직 없습니다.';
    const activeClass = doc.document_id === state.selectedDocumentId ? ' active' : '';
    const checked = state.checkedIds.has(doc.document_id) ? 'checked' : '';
    return `<div class="doc-card-row"><input type="checkbox" class="doc-checkbox" data-id="${{esc(doc.document_id)}}" ${{checked}}><button class="document-card${{activeClass}}" type="button" data-document-id="${{esc(doc.document_id)}}"><small>${{esc(doc.origin || 'project')}} · ${{esc(doc.document_type || doc.extension || 'unknown')}}</small><strong>${{esc(doc.source_name || doc.document_id)}}</strong><p>${{esc(summary)}}</p><div class="tag-row"><span class="tag">${{esc(doc.document_id)}}</span></div></button></div>`;
  }}).join('');
  documentList.querySelectorAll('[data-document-id]').forEach((button) => {{
    button.addEventListener('click', () => chooseDocument(button.getAttribute('data-document-id') || ''));
  }});
  documentList.querySelectorAll('.doc-checkbox').forEach((cb) => {{
    cb.addEventListener('change', () => {{
      const id = cb.getAttribute('data-id') || '';
      if (cb.checked) state.checkedIds.add(id); else state.checkedIds.delete(id);
      updateDeleteButton();
    }});
  }});
  updateDeleteButton();
}}
function renderSummary(payload) {{
  const keyPoints = payload.key_points || [];
  summaryPanel.innerHTML = `<div class="summary-meta"><span>${{esc(payload.source_name || payload.document_id || '')}}</span><span>${{esc(payload.document_type || 'document')}}</span><span>${{esc(payload.used_model || '')}}</span></div><p class="summary-text">${{esc(payload.summary_text || '')}}</p><div class="keypoints">${{keyPoints.length ? keyPoints.map((item) => `<div class="keypoint">${{esc(item)}}</div>`).join('') : '<div class="keypoint">핵심 포인트가 따로 반환되지 않았습니다.</div>'}}</div>`;
}}
function renderAnswerCard(title, result) {{
  const citations = result.citations || [];
  const matches = result.matches || result.source_documents || [];
  return `<section class="answer-card"><h3>${{esc(title)}}</h3><p>${{esc(result.answer || '답변이 없습니다.')}}</p><div class="evidence-list">${{citations.length ? citations.map((citation) => `<div class="evidence-item"><strong>${{esc(citation.source_name || 'document')}}</strong><div class="muted">${{esc(citation.section_hint || 'section')}}</div><div style="margin-top:8px; white-space:pre-wrap;">${{esc(String(citation.quote || '').replace(/<br\\s*\\/?>/gi, '\\n'))}}</div></div>`).join('') : matches.slice(0, 3).map((match) => {{ const metadata = match.metadata || {{}}; const excerpt = match.document || match.page_content || ''; return `<div class="evidence-item"><strong>${{esc(metadata.source_name || metadata.document_id || 'document')}}</strong><div class="muted">${{esc(metadata.section_hint || 'section')}}</div><div style="margin-top:8px; white-space:pre-wrap;">${{esc(String(excerpt).replace(/<br\\s*\\/?>/gi, '\\n'))}}</div></div>`; }}).join('') || '<div class="evidence-item">근거가 없습니다.</div>'}}</div></section>`;
}}
async function loadDocuments(preferredDocumentId = '') {{
  const response = await fetch('/api/documents');
  const payload = await response.json();
  state.documents = payload.documents || [];
  if (!state.documents.length) {{
    state.selectedDocumentId = '';
    renderDocumentList();
    return;
  }}
  if (preferredDocumentId && state.documents.some((item) => item.document_id === preferredDocumentId)) {{
    state.selectedDocumentId = preferredDocumentId;
  }} else if (!state.selectedDocumentId || !state.documents.some((item) => item.document_id === state.selectedDocumentId)) {{
    state.selectedDocumentId = state.documents[0].document_id;
  }}
  renderDocumentList();
  chooseDocument(state.selectedDocumentId);
}}
async function pollJob(jobId) {{
  while (true) {{
    const response = await fetch('/api/jobs/' + jobId);
    const payload = await response.json();
    const progress = payload.progress || {{ current: 0, total: 1 }};
    setStatus(['상태: ' + (payload.status || ''), '진행: ' + progress.current + ' / ' + progress.total, payload.message || '']);
    if (payload.status === 'completed') {{
      submitButton.disabled = false;
      const result = payload.result || {{}};
      const summary = result.summary || {{}};
      const vectorSummary = result.vector_index || {{}};
      const indexedDocs = (vectorSummary.documents || []).filter((item) => item.status === 'indexed');
      const firstIndexed = indexedDocs[0] || null;
      setStatus(['Run ID: ' + (result.run_id || ''), 'Parsed: ' + (summary.parsed_documents ?? 0), 'Indexed: ' + (vectorSummary.indexed_documents ?? vectorSummary.document_count ?? 0), payload.message || '']);
      state.latestRun = result;
      await loadDocuments(firstIndexed ? firstIndexed.document_id : '');
      if (firstIndexed) {{
        await summarizeSelectedDocument();
      }}
      return;
    }}
    if (payload.status === 'failed') {{
      submitButton.disabled = false;
      setStatus(payload.error || payload.message || '작업이 실패했습니다.');
      return;
    }}
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }}
}}
async function summarizeSelectedDocument() {{
  const selected = getSelectedDocument();
  if (!selected) {{
    summaryPanel.innerHTML = '<p class="summary-text muted">먼저 문서를 선택하세요.</p>';
    return;
  }}
  summarizeButton.disabled = true;
  summaryPanel.innerHTML = '<p class="summary-text muted">요약을 생성하고 있습니다...</p>';
  try {{
    const response = await fetch('/api/summarize', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ document_id: selected.document_id, source_name: selected.source_name }}) }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '요약 생성에 실패했습니다.');
    renderSummary(payload);
  }} catch (error) {{
    summaryPanel.innerHTML = `<p class="summary-text muted">${{esc(String(error))}}</p>`;
  }} finally {{
    summarizeButton.disabled = false;
  }}
}}
pickButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => renderFileList(Array.from(fileInput.files || [])));
['dragenter', 'dragover'].forEach((eventName) => {{
  uploadBox.addEventListener(eventName, (event) => {{ event.preventDefault(); uploadBox.classList.add('dragover'); }});
}});
['dragleave', 'drop'].forEach((eventName) => {{
  uploadBox.addEventListener(eventName, (event) => {{
    event.preventDefault();
    if (eventName === 'drop') {{
      const files = Array.from(event.dataTransfer?.files || []);
      if (files.length) {{
        const dt = new DataTransfer();
        files.forEach((file) => dt.items.add(file));
        fileInput.files = dt.files;
        renderFileList(files);
      }}
    }}
    uploadBox.classList.remove('dragover');
  }});
}});
uploadForm.addEventListener('submit', async (event) => {{
  event.preventDefault();
  const files = Array.from(fileInput.files || []);
  if (!files.length) {{
    setStatus('업로드할 문서를 먼저 선택하세요.');
    return;
  }}
  submitButton.disabled = true;
  setStatus('업로드를 시작했습니다. 문서 파이프라인을 준비하고 있습니다.');
  const formData = new FormData();
  files.forEach((file) => formData.append('documents', file, file.name));
  try {{
    const response = await fetch('/api/run', {{ method: 'POST', body: formData }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '파이프라인 실행에 실패했습니다.');
    await pollJob(payload.job_id);
  }} catch (error) {{
    submitButton.disabled = false;
    setStatus(String(error));
  }}
}});
summarizeButton.addEventListener('click', summarizeSelectedDocument);
queryForm.addEventListener('submit', async (event) => {{
  event.preventDefault();
  const selected = getSelectedDocument();
  const query = queryInput.value.trim();
  const strategy = document.getElementById('query-strategy').value;
  if (!selected) {{
    queryResults.innerHTML = '<div class="status-box">먼저 문서를 선택하세요.</div>';
    return;
  }}
  if (!query) {{
    queryResults.innerHTML = '<div class="status-box">질문을 입력하세요.</div>';
    return;
  }}
  queryButton.disabled = true;
  queryResults.innerHTML = '<div class="status-box">선택한 문서를 기준으로 답변을 찾고 있습니다...</div>';
  try {{
    const response = await fetch('/api/query-compare', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ query, strategy, document_id: selected.document_id, source_name: selected.source_name }})
    }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '질의응답에 실패했습니다.');
    queryResults.innerHTML = [renderAnswerCard('Custom QA', payload.custom || {{}}), renderAnswerCard('LangChain RetrievalQA (No Reranker)', payload.langchain || {{}}), renderAnswerCard('LangChain RetrievalQA (+ Cross-Encoder Reranker)', payload.langchain_reranked || {{}})].join('');
  }} catch (error) {{
    queryResults.innerHTML = `<div class="status-box">${{esc(String(error))}}</div>`;
  }} finally {{
    queryButton.disabled = false;
  }}
}});
loadDocuments().catch((error) => {{
  documentList.innerHTML = `<div class="status-box">${{esc(String(error))}}</div>`;
}});
</script></main></body></html>
""".format(
        supported=html.escape(", ".join(status.get("supported_extensions", []))),
        accept=html.escape(",".join(sorted(SUPPORTED_EXTENSIONS))),
        vector_count=vector_index.get("document_count", 0),
        latest_parsed=latest_upload.get("parsed_documents", 0),
        project_count=project_review.get("document_count", 0),
        review_action=review_action,
        chunking_action=chunking_action,
        qa_mode=qa_mode,
        vector_updated=html.escape(vector_index.get("updated_at") or "n/a"),
        latest_duplicates=latest_upload.get("duplicate_upload_count", 0),
    )


render_launcher_html = render_document_studio_html


def render_chunking_comparison_html(status: dict[str, Any]) -> str:
    comparison = status.get("chunking_comparison") or {}
    summary = comparison.get("summary") or {}
    documents = summary.get("documents") or []
    overall = summary.get("overall_ranking") or []
    document_options = "\n".join(
        "<option value=\"{value}\">{label}</option>".format(
            value=html.escape(doc.get("document_id", "")),
            label=html.escape(doc.get("source_name", doc.get("document_id", ""))),
        )
        for doc in documents
    ) or "<option value=\"\">No documents</option>"

    html_template = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chunking Strategy Compare</title>
<style>
body {{ margin:0; font-family: Georgia, "Malgun Gothic", serif; color:#1f1a16; background:linear-gradient(135deg, #f4ecdf 0%, #e7efe7 100%); }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
.topbar {{ display:flex; justify-content:space-between; align-items:end; gap:16px; flex-wrap:wrap; margin-bottom:18px; }}
.panel {{ background:rgba(255,255,255,0.84); border:1px solid #d7cfbf; border-radius:22px; padding:18px; box-shadow:0 12px 30px rgba(59,45,30,0.08); }}
.toolbar {{ display:grid; grid-template-columns:minmax(18rem, 28rem) 1fr; gap:16px; margin-bottom:18px; }}
.summary {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; }}
.metric {{ background:#fff; border:1px solid #d7cfbf; border-radius:16px; padding:12px; }}
.results {{ display:grid; grid-template-columns:repeat(5, minmax(18rem, 1fr)); gap:14px; align-items:start; }}
.card {{ background:#fff; border:1px solid #d7cfbf; border-radius:18px; overflow:hidden; }}
.card header {{ padding:14px; border-bottom:1px solid #d7cfbf; background:linear-gradient(180deg, rgba(30,95,105,0.10), rgba(30,95,105,0)); }}
.card h2 {{ margin:0; font-size:20px; }}
.metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:12px 14px; border-bottom:1px solid #d7cfbf; }}
.metric-chip {{ padding:8px 10px; border-radius:12px; background:#f7f4ee; font-size:12px; }}
.chunks {{ padding:14px; display:grid; gap:12px; max-height:70vh; overflow:auto; }}
.chunk {{ border:1px solid #d7cfbf; border-radius:14px; padding:12px; background:#fcfbf8; }}
.chunk h4 {{ margin:0 0 8px; font-size:14px; }}
.chunk pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-family:"Consolas","Malgun Gothic",monospace; font-size:12px; line-height:1.5; }}
select {{ width:100%; padding:10px 12px; border-radius:12px; border:1px solid #d7cfbf; font:inherit; }}
a {{ color:#1d5f69; }}
@media (max-width: 1300px) {{ .results {{ grid-template-columns:1fr 1fr; }} .summary {{ grid-template-columns:1fr 1fr; }} .toolbar {{ grid-template-columns:1fr; }} }}
@media (max-width: 820px) {{ .results, .summary {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<div class="topbar">
  <div>
    <p style="margin:0 0 6px;">Chunking Strategy Compare</p>
    <h1 style="margin:0;">문서를 선택해서 5가지 청킹 결과 보기</h1>
  </div>
  <a href="/">런처로 돌아가기</a>
</div>
<section class="panel toolbar">
  <div>
    <label for="document-select">문서 선택</label>
    <select id="document-select" style="margin-top:8px;">__DOCUMENT_OPTIONS__</select>
  </div>
  <div class="summary">
    <div class="metric"><strong>Generated</strong><div>__GENERATED_AT__</div></div>
    <div class="metric"><strong>Documents</strong><div>__DOCUMENT_COUNT__</div></div>
    <div class="metric"><strong>Top Strategy</strong><div>__TOP_STRATEGY__</div></div>
    <div class="metric"><strong>Output</strong><div style="word-break:break-all;">__OUTPUT_DIR__</div></div>
  </div>
</section>
<section id="document-meta" class="panel" style="margin-bottom:18px;">문서를 선택하면 평가 결과와 실제 청크를 불러옵니다.</section>
<section class="panel" style="margin-bottom:18px;">
  <strong>Overall Ranking</strong>
  <div style="margin-top:8px;">__OVERALL_TEXT__</div>
</section>
<section id="results" class="results"></section>
<script>
const selectEl = document.getElementById('document-select');
const resultsEl = document.getElementById('results');
const metaEl = document.getElementById('document-meta');
function esc(value) {{
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}}
function metric(label, value) {{
  return `<div class="metric-chip"><strong>${{esc(label)}}</strong><div>${{esc(value)}}</div></div>`;
}}
function chunkView(chunk) {{
  return `<article class="chunk">
    <h4>Chunk ${{esc(chunk.chunk_index)}} | ${{esc(chunk.char_count)}} chars</h4>
    <div style="font-size:12px; margin-bottom:8px;">paragraph_ids: [${{esc((chunk.paragraph_ids || []).join(', '))}}]</div>
    <div style="font-size:12px; margin-bottom:8px;">section_hint: ${{esc(chunk.section_hint || '')}}</div>
    <pre>${{esc(chunk.text || '')}}</pre>
  </article>`;
}}
function strategyCard(name, evaluation, chunks) {{
  return `<section class="card">
    <header><div style="font-size:12px; opacity:0.7;">strategy</div><h2>${{esc(name)}}</h2></header>
    <div class="metrics">
      ${metric('score', evaluation?.composite_score ?? '')}
      ${metric('chunks', evaluation?.chunk_count ?? chunks.length)}
      ${metric('coverage', evaluation?.coverage_ratio ?? '')}
      ${metric('retrieval', evaluation?.retrieval_hit_rate ?? '')}
      ${metric('redundancy', evaluation?.adjacent_redundancy ?? '')}
      ${metric('avg chars', evaluation?.avg_characters ?? '')}
    </div>
    <div class="chunks">${{chunks.map(chunkView).join('')}}</div>
  </section>`;
}}
async function loadDocument(documentId) {{
  if (!documentId) {{
    resultsEl.innerHTML = '';
    return;
  }}
  metaEl.textContent = '문서 결과를 불러오는 중입니다...';
  const response = await fetch('/api/chunking-comparison/document?document_id=' + encodeURIComponent(documentId));
  const payload = await response.json();
  if (!response.ok) {{
    metaEl.textContent = payload.error || '불러오기에 실패했습니다.';
    resultsEl.innerHTML = '';
    return;
  }}
  const document = payload.document || {{}};
  const evaluations = new Map((document.evaluations || []).map((item) => [item.strategy, item]));
  metaEl.innerHTML = `<strong>${{esc(document.source_name || document.document_id || '')}}</strong><br>
    path: ${{esc(document.path || '')}}<br>
    paragraph_count: ${{esc(document.paragraph_count || 0)}}<br>
    ranking: ${{esc((document.ranking || []).map((item) => `${{item.strategy}} (${{item.composite_score}})`).join(' | '))}}`;
  const order = ['rule_based', 'semantic', 'agentic_section', 'agentic_topic_shift', 'agentic_qa'];
  resultsEl.innerHTML = order
    .filter((name) => payload.strategies && payload.strategies[name])
    .map((name) => strategyCard(name, evaluations.get(name), (payload.strategies[name] || {{}}).chunks || []))
    .join('');
}}
selectEl.addEventListener('change', () => loadDocument(selectEl.value));
if (selectEl.value) loadDocument(selectEl.value);
</script>
</main></body></html>
"""
    return (
        html_template
        .replace("__DOCUMENT_OPTIONS__", document_options)
        .replace("__GENERATED_AT__", html.escape(comparison.get("generated_at") or "n/a"))
        .replace("__DOCUMENT_COUNT__", html.escape(str(len(documents))))
        .replace("__TOP_STRATEGY__", html.escape((overall[0] or {}).get("strategy", "n/a") if overall else "n/a"))
        .replace("__OUTPUT_DIR__", html.escape(comparison.get("output_dir") or "n/a"))
        .replace("__OVERALL_TEXT__", html.escape(" | ".join(f"{item.get('strategy')} ({item.get('avg_composite_score')})" for item in overall) or "n/a"))
        .replace("{{", "{")
        .replace("}}", "}")
    )
