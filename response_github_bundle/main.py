# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import cgi
import html
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import sys
import threading
import traceback
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import fitz


ROOT = Path(__file__).resolve().parent
UI_ROOT = ROOT / "src" / "ui"
DOCUMENT_STUDIO_TEMPLATE_PATH = UI_ROOT / "templates" / "document_studio.html"
SERVER_ERROR_LOG = ROOT / "outputs" / "document_studio_server_errors.log"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.chroma_retriever import ChromaRetriever  # noqa: E402
from src.retrieval.document_summary import build_page_summaries  # noqa: E402
from src.parsers.pdf.markdown_extractor import PdfMarkdownExtractor  # noqa: E402
from src.shared.constants import SUPPORTED_EXTENSIONS  # noqa: E402
from src.ui.review_server import ReviewSessionManager, UploadedDocument  # noqa: E402


DEFAULT_QA_STRATEGY = "semantic"

LLM_SUMMARY_SOURCES = {
    "on_demand_llm",
    "cached_ui_summary",
    "pipeline_llm_summary",
}

LLM_PAGE_SUMMARY_SOURCE = "on_demand_page_llm"


def is_semantic_strategy_available() -> bool:
    try:
        import sentence_transformers  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def clear_process_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def log_server_exception(context: str, error: Exception) -> None:
    SERVER_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SERVER_ERROR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{context}] {error}\n")
        handle.write(traceback.format_exc())
        handle.write("\n")


def make_json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return str(value)


def _is_llm_summary_payload(summary: dict[str, object] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    source = str(summary.get("summary_source") or "")
    if source in LLM_SUMMARY_SOURCES:
        return True
    used_model = str(summary.get("used_model") or "")
    return bool(used_model and used_model not in {"cached_basic_summary", "cached_llm_summary"})


def _normalize_page_summary_items(raw_page_summaries: object) -> list[dict[str, object]]:
    if not isinstance(raw_page_summaries, list):
        return []

    normalized: list[dict[str, object]] = []
    for item in raw_page_summaries:
        if not isinstance(item, dict):
            continue
        try:
            page_number = int(item.get("page_number") or 0)
        except (TypeError, ValueError):
            page_number = 0
        summary_text = str(item.get("summary_text") or "").strip()
        if page_number <= 0 or not summary_text:
            continue
        normalized.append(
            {
                "page_number": page_number,
                "summary_text": summary_text,
                "key_points": [str(point).strip() for point in (item.get("key_points") or []) if str(point).strip()][:4],
                "summary_source": str(item.get("summary_source") or ""),
                "char_count": int(item.get("char_count") or 0),
            }
        )
    normalized.sort(key=lambda item: int(item.get("page_number") or 0))
    return normalized


def _has_llm_page_summaries(raw_page_summaries: object) -> bool:
    page_summaries = _normalize_page_summary_items(raw_page_summaries)
    return bool(page_summaries) and all(
        str(item.get("summary_source") or "") == LLM_PAGE_SUMMARY_SOURCE
        for item in page_summaries
    )


def clean_viewer_markdown(markdown: str) -> str:
    cleaned_lines: list[str] = []
    skip_unit_line = False
    for raw_line in str(markdown or "").splitlines():
        line = raw_line
        # Remove omitted-picture markers inline without dropping neighboring content.
        line = re.sub(
            r"\s*==>\s*picture\s*\[[^\]]+\]\s*intentionally omitted\s*<==\s*",
            " ",
            line,
            flags=re.IGNORECASE,
        )
        stripped = line.strip()
        lowered = stripped.lower()
        if re.fullmatch(r"[| ]{5,}", stripped):
            continue
        if lowered == "end of document":
            continue
        if "[financial fact table]" in lowered or "[row_path]" in lowered:
            skip_unit_line = True
            continue
        if skip_unit_line and re.match(r"^\(unit:\s*.*\)$", stripped, flags=re.IGNORECASE):
            skip_unit_line = False
            continue
        skip_unit_line = False
        if stripped:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_match_text(text: object) -> str:
    normalized = re.sub(r"<br\s*/?>", "\n", str(text or ""), flags=re.IGNORECASE)
    normalized = re.sub(r"[*_`#>\-\[\]\(\)\|]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _score_text_overlap(candidate: object, target: object) -> float:
    normalized_candidate = _normalize_match_text(candidate)
    normalized_target = _normalize_match_text(target)
    if not normalized_candidate or not normalized_target:
        return -1.0
    if normalized_candidate == normalized_target:
        return 20000.0
    if normalized_candidate in normalized_target:
        return 12000.0 - abs(len(normalized_target) - len(normalized_candidate))
    if normalized_target in normalized_candidate:
        return 10000.0 - abs(len(normalized_candidate) - len(normalized_target))

    candidate_tokens = {token for token in normalized_candidate.split(" ") if token}
    target_tokens = [token for token in normalized_target.split(" ") if token]
    if not candidate_tokens or not target_tokens:
        return -1.0

    overlap = 0.0
    for token in target_tokens:
        if token in candidate_tokens:
            overlap += 12.0 if len(token) > 2 else 4.0
    overlap -= abs(len(normalized_candidate) - len(normalized_target)) * 0.05
    return overlap


def _infer_semantic_chunk_page_numbers(
    payload: dict[str, object],
    normalized_chunks: list[dict[str, object]],
) -> list[dict[str, object]]:
    structured_chunks = payload.get("chunks") if isinstance(payload.get("chunks"), list) else []
    candidates: list[dict[str, object]] = []
    for item in structured_chunks:
        if not isinstance(item, dict):
            continue
        try:
            page_number = int(item.get("page") or item.get("page_number") or 0)
        except (TypeError, ValueError):
            page_number = 0
        serialized_text = clean_viewer_markdown(str(item.get("serialized_text") or ""))
        if page_number <= 0 or not serialized_text:
            continue
        candidates.append(
            {
                "page_number": page_number,
                "text": serialized_text,
                "section": str(item.get("section") or ""),
            }
        )

    if not candidates:
        return normalized_chunks

    enriched_chunks: list[dict[str, object]] = []
    for chunk in normalized_chunks:
        chunk_text = clean_viewer_markdown(str(chunk.get("text") or ""))
        chunk_section = str(chunk.get("section_hint") or "")
        best_page_number: int | None = None
        best_score = -1.0
        for candidate in candidates:
            score = _score_text_overlap(candidate.get("text"), chunk_text)
            if chunk_section and candidate.get("section") == chunk_section:
                score += 80.0
            if score > best_score:
                best_score = score
                best_page_number = int(candidate["page_number"])
        enriched = dict(chunk)
        if best_page_number and best_score > 0:
            enriched["page_number"] = best_page_number
        enriched_chunks.append(enriched)
    return enriched_chunks


def enrich_pdf_markdown_for_viewer(payload: dict[str, object], markdown: str) -> str:
    source_path_raw = str(payload.get("source_path") or "").strip()
    if not source_path_raw or "intentionally omitted <==" not in str(markdown or ""):
        return markdown

    source_path = Path(source_path_raw)
    if not source_path.exists() or source_path.suffix.lower() != ".pdf":
        return markdown

    extractor = PdfMarkdownExtractor(enable_omitted_picture_ocr=True)
    try:
        with fitz.open(source_path) as document:
            enriched_markdown, _, _ = extractor._apply_omitted_picture_ocr(source_path, markdown, document)
            return enriched_markdown or markdown
    except Exception as error:
        log_server_exception("enrich_pdf_markdown_for_viewer", error)
        return markdown


def _load_ui_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render_ui_template(path: Path, replacements: dict[str, str]) -> str:
    content = _load_ui_template(path)
    for key, value in replacements.items():
        content = content.replace(f"{{{{{key}}}}}", value)
    return content


def build_openapi_spec(host: str, port: int) -> dict[str, object]:
    server_url = f"http://{host}:{port}"
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Document Studio API",
            "version": "1.0.0",
            "description": "ThreadingHTTPServer-based API for document upload, summary, QA, and review job tracking.",
        },
        "servers": [{"url": server_url}],
        "tags": [
            {"name": "system", "description": "Server and status endpoints"},
            {"name": "documents", "description": "Document list and summary endpoints"},
            {"name": "jobs", "description": "Upload and job tracking endpoints"},
            {"name": "qa", "description": "Question answering endpoints"},
        ],
        "paths": {
            "/api/status": {
                "get": {
                    "tags": ["system"],
                    "summary": "Get server status",
                    "responses": {
                        "200": {
                            "description": "Current Document Studio status",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "additionalProperties": True}
                                }
                            },
                        }
                    },
                }
            },
            "/api/documents": {
                "get": {
                    "tags": ["documents"],
                    "summary": "List indexed documents",
                    "responses": {
                        "200": {
                            "description": "Document list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "documents": {
                                                "type": "array",
                                                "items": {"type": "object", "additionalProperties": True},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/download-summary": {
                "get": {
                    "tags": ["documents"],
                    "summary": "Download a document summary",
                    "parameters": [
                        {
                            "name": "document_id",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Target document ID",
                        },
                        {
                            "name": "source_name",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Target document source name",
                        },
                        {
                            "name": "format",
                            "in": "query",
                            "schema": {"type": "string", "enum": ["md"], "default": "md"},
                            "description": "Download format",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Summary file",
                            "content": {
                                "text/markdown": {"schema": {"type": "string", "format": "binary"}},
                                "text/plain": {"schema": {"type": "string", "format": "binary"}},
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                        "404": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/events": {
                "get": {
                    "tags": ["system"],
                    "summary": "Subscribe to server-sent events",
                    "responses": {
                        "200": {
                            "description": "SSE stream",
                            "content": {
                                "text/event-stream": {"schema": {"type": "string"}}
                            },
                        }
                    },
                }
            },
            "/api/jobs/{job_id}": {
                "get": {
                    "tags": ["jobs"],
                    "summary": "Get upload job status",
                    "parameters": [
                        {
                            "name": "job_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Job payload",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "additionalProperties": True}
                                }
                            },
                        },
                        "404": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/run": {
                "post": {
                    "tags": ["jobs"],
                    "summary": "Upload one or more documents and start a processing job",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["documents"],
                                    "properties": {
                                        "documents": {
                                            "type": "array",
                                            "items": {"type": "string", "format": "binary"},
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "202": {
                            "description": "Accepted job",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "additionalProperties": True}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                        "500": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/summarize": {
                "post": {
                    "tags": ["documents"],
                    "summary": "Generate or fetch a summary for a document",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DocumentSelector"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Summary payload",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SummaryResponse"}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                        "503": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/query": {
                "post": {
                    "tags": ["qa"],
                    "summary": "Ask a question across indexed documents",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueryRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "QA response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "additionalProperties": True}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                        "500": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/query-compare": {
                "post": {
                    "tags": ["qa"],
                    "summary": "Compare custom retriever and LangChain QA answers",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueryRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Comparison response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "custom": {"type": "object", "additionalProperties": True},
                                            "langchain": {"type": "object", "additionalProperties": True},
                                            "langchain_reranked": {"type": "object", "additionalProperties": True},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                        "500": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/delete-documents": {
                "post": {
                    "tags": ["documents"],
                    "summary": "Delete indexed documents and related files",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["document_ids"],
                                    "properties": {
                                        "document_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Deletion results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "deleted": {
                                                "type": "array",
                                                "items": {"type": "object", "additionalProperties": True},
                                            }
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                        "500": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "DocumentSelector": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "Internal document ID"},
                        "source_name": {"type": "string", "description": "Original file/source name"},
                    },
                    "description": "At least one of document_id or source_name should be provided.",
                },
                "SummaryResponse": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "source_name": {"type": "string"},
                        "summary_text": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "document_type": {"type": "string"},
                        "used_model": {"type": "string"},
                        "framework": {"type": "string"},
                        "summary_source": {"type": "string"},
                    },
                },
                "QueryRequest": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "strategy": {
                            "type": "string",
                            "default": DEFAULT_QA_STRATEGY,
                            "description": "Retriever strategy such as semantic",
                        },
                        "document_id": {"type": "string"},
                        "source_name": {"type": "string"},
                        "selected_document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"}
                    },
                    "required": ["error"],
                },
            },
            "responses": {
                "ErrorResponse": {
                    "description": "Error payload",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                }
            },
        },
    }


def render_swagger_ui_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Studio API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    html, body { margin: 0; padding: 0; background: #f7f7f9; }
    body { font-family: "Segoe UI", Arial, sans-serif; }
    .topbar { display: none; }
    .swagger-ui .info { margin: 24px 0; }
    .offline-note {
      margin: 16px 24px 0;
      padding: 12px 14px;
      border-radius: 10px;
      background: #fff7ed;
      border: 1px solid #fdba74;
      color: #9a3412;
    }
  </style>
</head>
<body>
  <div class="offline-note">
    Swagger UI assets are loaded from a CDN. If the page looks unstyled, check internet access and open <code>/openapi.json</code> directly.
  </div>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
      layout: 'BaseLayout',
      tryItOutEnabled: true,
      displayRequestDuration: true,
      defaultModelsExpandDepth: 1
    });
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Document Studio UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8233)
    parser.add_argument("--markdown-mode", choices=["original", "llm_ready", "both"], default="both")
    parser.add_argument("--qa-mode", choices=["original", "llm_ready", "hybrid"], default="hybrid")
    return parser.parse_args()


class LazyLangChainService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._service = None
        self._lock = threading.Lock()
        self._init_error: Exception | None = None
        self._warmup_lock = threading.Lock()
        self._warmup_started = False
        self._warmup_status: dict[str, object] = {
            "started": False,
            "completed": False,
            "error": None,
            "details": None,
        }

    def _get_service(self):
        if self._service is not None:
            return self._service
        if self._init_error is not None:
            raise self._init_error
        with self._lock:
            if self._service is not None:
                return self._service
            if self._init_error is not None:
                raise self._init_error
            try:
                from src.retreival_lanchain.retrieval_qa import LangChainRetrievalQAService  # noqa: WPS433

                self._service = LangChainRetrievalQAService(project_root=self.project_root)
            except Exception as error:
                self._init_error = error
                raise
        return self._service

    def summarize_document(self, *args: object, **kwargs: object):
        return self._get_service().summarize_document(*args, **kwargs)

    def answer_question(self, *args: object, **kwargs: object):
        return self._get_service().answer_question(*args, **kwargs)

    def get_warmup_status(self) -> dict[str, object]:
        with self._warmup_lock:
            return dict(self._warmup_status)

    def start_background_warmup(self, *, preload_summaries: bool = True, max_documents: int | None = None) -> None:
        with self._warmup_lock:
            if self._warmup_started:
                return
            self._warmup_started = True
            self._warmup_status = {
                "started": True,
                "completed": False,
                "error": None,
                "details": None,
            }
        threading.Thread(
            target=self._run_warmup,
            kwargs={
                "preload_summaries": preload_summaries,
                "max_documents": max_documents,
            },
            daemon=True,
        ).start()

    def _run_warmup(self, *, preload_summaries: bool, max_documents: int | None) -> None:
        try:
            details = self._get_service().warmup(
                preload_summaries=preload_summaries,
                max_documents=max_documents,
            )
            with self._warmup_lock:
                self._warmup_status = {
                    "started": True,
                    "completed": True,
                    "error": None,
                    "details": details,
                }
        except Exception as error:
            with self._warmup_lock:
                self._warmup_status = {
                    "started": True,
                    "completed": False,
                    "error": str(error),
                    "details": None,
                }


def _render_document_studio_html_legacy(status: dict[str, object]) -> str:
    project_review = status.get("project_review") or {}
    latest_upload = status.get("latest_upload") or {}
    openai_status = status.get("openai") or {}

    qa_caption = (
        (
            "?낅줈????semantic 泥?궧, ?꾨쿋???몃뜳?? 臾몄꽌 ?붿빟??癒쇱? 以鍮꾪븳 ??"
            f"LangChain QA + Cross-Encoder reranker濡?諛붾줈 吏덈Ц?????덉뒿?덈떎. ?꾩옱 紐⑤뜽: {openai_status.get('model') or 'local fallback'}"
        )
        if openai_status.get("enabled")
        else "OpenAI ?곌껐???놁뼱 臾몄꽌 ?붿빟怨?LangChain QA瑜?以鍮꾪븯吏 紐삵뻽?듬땲?? ?꾩옱??濡쒖뺄 寃??湲곕컲 ?묐떟??????ъ슜?⑸땲??"
    )

    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Studio</title>
  <style>
    :root {{
      --bg: #ffffff;
      --surface: rgba(255,255,255,0.94);
      --surface-strong: #ffffff;
      --line: rgba(15, 23, 42, 0.10);
      --text: #111827;
      --muted: #667085;
      --navy: #113b68;
      --navy-soft: rgba(17, 59, 104, 0.08);
      --shadow: 0 12px 32px rgba(15, 23, 42, 0.05);
      --radius-xl: 22px;
    }}

    * {{ box-sizing: border-box; }}

    html, body {{
      height: 100%;
      overflow: hidden;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
      color: var(--text);
      background: var(--bg);
      padding: 10px;
    }}

    .app {{
      width: 100%;
      max-width: none;
      height: calc(100vh - 20px);
      min-height: calc(100vh - 20px);
      margin: 0;
      display: grid;
      grid-template-columns: 278px minmax(0, 1fr);
      overflow: hidden;
      background: #fff;
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 22px;
      box-shadow: 0 16px 44px rgba(15, 23, 42, 0.06);
    }}

    .sidebar {{
      height: 100%;
      padding: 14px 12px;
      border-right: 1px solid rgba(15, 23, 42, 0.06);
      background: #fff;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
      overflow: hidden;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 18px;
      font-weight: 800;
      color: var(--navy);
      letter-spacing: -0.03em;
    }}

    .brand-badge {{
      width: 30px;
      height: 30px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--navy), #2d5d8f);
      color: #fff;
      display: grid;
      place-items: center;
      font-size: 14px;
      box-shadow: 0 8px 18px rgba(17, 59, 104, 0.18);
    }}

    .sidebar-workspace {{
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(0, 1.2fr) minmax(260px, 0.95fr);
      gap: 12px;
      overflow: hidden;
    }}

    .sidebar-workspace > .sidebar-section:first-child {{
      overflow: hidden;
    }}

    .sidebar-section {{
      padding: 12px;
      border-radius: 18px;
      background: var(--surface-strong);
      border: 1px solid var(--line);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
      min-height: 0;
      overflow: hidden;
    }}

    .sidebar-section h3 {{
      margin: 0;
      font-size: 17px;
      letter-spacing: -0.03em;
    }}

    /* 臾몄꽌 ?좏깮 ?뱀뀡: ?곗뒪?ы깙?먯꽌 ?⑥? 怨듦컙 梨꾩슦湲?*/
    .sidebar-section-docs {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      overflow: hidden;
      min-height: 0;
    }}
    .sidebar-section-docs .doc-list {{
      flex: 1 1 0;
      min-height: 0;
      overflow-y: scroll;
      max-height: 232px;
    }}

    .sidebar-upload {{
      display: grid;
      grid-template-rows: minmax(96px, 0.72fr) auto auto minmax(124px, 1fr);
      gap: 8px;
      height: 100%;
    }}

    .sidebar-upload-actions {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}

    .sidebar-upload .circle-button {{
      width: 40px;
      height: 40px;
      font-size: 24px;
    }}

    .sidebar-upload .send-button {{
      flex: 1;
      padding: 12px 16px;
    }}

    .sidebar-selected-files {{
      display: grid;
      gap: 6px;
      max-height: 56px;
      overflow: auto;
    }}

    #status-box {{
      min-height: 0;
      height: 100%;
      overflow-y: auto;
      scrollbar-width: auto;
      padding: 10px 12px !important;
      border-radius: 14px;
      background: rgba(248,250,252,0.96);
      border: 1px solid var(--line);
      line-height: 1.55 !important;
      font-size: 13px !important;
      word-break: keep-all;
      overflow-wrap: anywhere;
    }}

    .sidebar-list-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}

    .sidebar-list-count {{
      flex: 0 0 auto;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(17, 59, 104, 0.08);
      color: var(--navy);
      font-size: 12px;
      font-weight: 700;
    }}

    .selection-status {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(248,250,252,0.96);
      border: 1px solid var(--line);
      font-size: 13px;
      min-height: 46px;
      flex: 0 0 auto;
    }}

    .selection-status-names {{
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}

    .selection-status-name-item {{
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 100%;
    }}

    .selection-status strong {{
      color: var(--navy);
    }}

    .main {{
      height: 100%;
      min-height: 100%;
      padding: 12px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 10px;
    }}

    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      min-height: 28px;
    }}

    .page-title {{
      min-width: 0;
    }}

    .page-title h1 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.05em;
      line-height: 1.1;
    }}

    .topbar-right {{
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .top-pill {{
      padding: 10px 16px;
      border-radius: 999px;
      background: rgba(255,255,255,0.98);
      border: 1px solid var(--line);
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.03);
      font-weight: 600;
      white-space: nowrap;
    }}

    .panel {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
    }}

    .status-box {{
      min-height: 56px;
      white-space: pre-wrap;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(248,250,252,0.96);
      border: 1px solid var(--line);
      line-height: 1.65;
      font-size: 14px;
    }}

    .doc-list {{
      display: grid;
      gap: 8px;
      align-content: start;
      overflow-y: scroll;
      padding-right: 4px;
      min-height: 140px;
      scrollbar-gutter: stable;
    }}

    .sidebar-doc-list {{
      max-height: none;
      padding-bottom: 4px;
      min-height: 140px;
    }}

    .doc-card {{
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr) 28px;
      gap: 10px;
      align-items: center;
      border: 1px solid rgba(17, 59, 104, 0.08);
      background: rgba(255,255,255,0.96);
      border-radius: 14px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      box-shadow: none;
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }}

    .doc-card:hover {{
      transform: translateY(-1px);
      border-color: rgba(17, 59, 104, 0.22);
      box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
    }}

    .doc-card.active {{
      border-color: rgba(17, 59, 104, 0.24);
      background: rgba(227, 236, 247, 0.92);
      box-shadow: 0 0 0 2px rgba(17, 59, 104, 0.08);
    }}

    .doc-icon {{
      width: 22px;
      height: 22px;
      border-radius: 7px;
      background: rgba(17, 59, 104, 0.08);
      color: var(--navy);
      display: grid;
      place-items: center;
      font-size: 11px;
      font-weight: 800;
    }}

    .doc-copy {{
      min-width: 0;
    }}

    .doc-card strong {{
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
      line-height: 1.4;
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      white-space: normal;
    }}

    .doc-card p {{
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }}

    .sidebar-selected-files,
    #status-box,
    .doc-list {{
      scrollbar-color: rgba(17, 59, 104, 0.35) transparent;
    }}

    .sidebar-selected-files::-webkit-scrollbar,
    #status-box::-webkit-scrollbar,
    .doc-list::-webkit-scrollbar {{
      width: 10px;
    }}

    .sidebar-selected-files::-webkit-scrollbar-thumb,
    #status-box::-webkit-scrollbar-thumb,
    .doc-list::-webkit-scrollbar-thumb {{
      background: rgba(17, 59, 104, 0.28);
      border-radius: 999px;
      border: 2px solid transparent;
      background-clip: padding-box;
    }}

    .doc-check {{
      width: 20px;
      height: 20px;
      border-radius: 5px;
      border: 1.5px solid rgba(17, 59, 104, 0.25);
      background: #fff;
      color: transparent;
      display: grid;
      place-items: center;
      font-size: 13px;
      font-weight: 900;
      align-self: center;
      flex-shrink: 0;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
    }}

    .doc-check.checked {{
      background: var(--navy);
      border-color: var(--navy);
      color: #fff;
    }}

    .doc-card.active .doc-check {{
      border-color: var(--navy);
    }}

    .workspace {{
      min-height: 0;
      overflow: hidden;
    }}

    .content-pane {{
      height: 100%;
      padding: 0;
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.35fr);
      min-height: 0;
      overflow: hidden;
    }}

    .content-pane h2 {{
      margin: 0;
      letter-spacing: -0.04em;
    }}

    .summary-box,
    .qa-box {{
      padding: 14px;
      min-height: 0;
      overflow: hidden;
    }}

    .summary-box {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 14px;
    }}

    .summary-box.guide-collapsed {{
      grid-template-rows: auto 0 auto;
      min-height: auto;
    }}

    .summary-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }}

    .source-guide-label {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 700;
      color: var(--navy);
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}

    .source-guide-label::before {{
      content: "??;
      font-size: 11px;
    }}

    .summary-toggle {{
      border: 0;
      background: transparent;
      color: var(--muted);
      font-size: 16px;
      cursor: pointer;
      padding: 0 4px;
      line-height: 1;
    }}

    .summary-toggle:hover {{ color: var(--navy); }}

    .source-guide-label::before {{
      content: "??;
      font-size: 10px;
    }}

    #summary-content.guide-collapsed {{
      display: none;
    }}

    .summary-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
    }}

    .summary-meta span {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 100%;
    }}

    .summary-text {{
      margin: 0;
      font-size: 15px;
      line-height: 1.75;
      padding: 14px 16px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid rgba(15, 23, 42, 0.08);
    }}

    #summary-content {{
      min-height: 0;
      overflow: auto;
      padding-right: 4px;
    }}

    .points {{
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }}

    .point {{
      padding: 10px 12px;
      border-radius: 14px;
      background: #fbfcfe;
      border: 1px solid rgba(17,59,104,0.08);
      box-shadow: inset 3px 0 0 rgba(17, 59, 104, 0.12);
      line-height: 1.6;
      font-size: 13px;
    }}

    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 0;
    }}

    .circle-button {{
      width: 46px;
      height: 46px;
      border-radius: 999px;
      border: 1px solid rgba(15,23,42,0.10);
      background: #fff;
      display: grid;
      place-items: center;
      font-size: 28px;
      color: var(--navy);
      cursor: pointer;
    }}

    .send-button {{
      border: 0;
      border-radius: 999px;
      background: var(--navy);
      color: #fff;
      padding: 14px 22px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}

    .pill-button {{
      border: 0;
      border-radius: 999px;
      background: var(--navy);
      color: #fff;
      padding: 9px 13px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}

    .ghost-button {{
      border: 0;
      border-radius: 999px;
      background: rgba(17,59,104,0.08);
      color: var(--navy);
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}

    .file-chip {{
      padding: 9px 12px;
      border-radius: 999px;
      background: #fff;
      border: 1px solid var(--line);
      font-size: 13px;
      color: var(--text);
    }}

    .upload-dropzone {{
      border: 2px dashed rgba(17, 59, 104, 0.22);
      border-radius: 14px;
      padding: 14px 12px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      cursor: pointer;
      transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
      user-select: none;
      line-height: 1.7;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }}

    .upload-dropzone-icon {{
      font-size: 22px;
      line-height: 1;
      color: rgba(17, 59, 104, 0.35);
      transition: color 0.2s ease;
    }}

    .upload-dropzone.drag-over {{
      border-color: var(--navy);
      background: rgba(17, 59, 104, 0.06);
      color: var(--navy);
      font-weight: 600;
    }}

    .upload-dropzone.drag-over .upload-dropzone-icon {{
      color: var(--navy);
    }}

    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}

    .upload-spinner {{
      display: none;
      width: 20px;
      height: 20px;
      border: 2.5px solid rgba(17, 59, 104, 0.15);
      border-top-color: var(--navy);
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
      flex-shrink: 0;
    }}

    .upload-spinner.active {{
      display: block;
    }}

    .select-all-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid rgba(17, 59, 104, 0.12);
      background: rgba(17, 59, 104, 0.06);
      border-radius: 14px;
      padding: 10px 12px;
      cursor: pointer;
      user-select: none;
      flex: 0 0 auto;
    }}

    .select-all-text {{
      flex: 1;
      min-width: 0;
      font-size: 14px;
      font-weight: 700;
      color: var(--navy);
      padding-left: 32px;
    }}

    .question-area {{
      width: 100%;
      min-height: 42px;
      max-height: 42px;
      resize: none;
      border-radius: 16px;
      border: 1px solid rgba(15, 23, 42, 0.10);
      background: #fff;
      padding: 9px 14px;
      font: inherit;
      line-height: 1.45;
    }}

    .evidence-list {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}

    .evidence-item {{
      padding: 13px 14px;
      border-radius: 16px;
      background: rgba(248,250,252,0.95);
      border: 1px solid var(--line);
      line-height: 1.7;
    }}

    .qa-box {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 10px;
    }}

    .chat-shell {{
      min-height: 0;
      border-radius: 20px;
      background: #fff;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 14px;
      overflow-y: auto;
    }}

    .chat-history {{
      display: flex;
      flex-direction: column;
      gap: 18px;
      min-height: 100%;
      padding: 8px 6px 12px;
      scroll-behavior: smooth;
    }}

    .chat-empty {{
      min-height: 100%;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px 18px;
      border-radius: 18px;
      border: 1px dashed rgba(17, 59, 104, 0.18);
      background: #fff;
      color: var(--muted);
      line-height: 1.65;
    }}

    .chat-turn {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .chat-bubble-user {{
      align-self: flex-end;
      background: var(--navy);
      color: #fff;
      border-radius: 22px 22px 6px 22px;
      padding: 14px 18px;
      max-width: min(86%, 560px);
      line-height: 1.7;
      font-size: 15px;
      white-space: pre-wrap;
      box-shadow: 0 16px 30px rgba(17, 59, 104, 0.18);
    }}

    .chat-bubble-ai {{
      align-self: stretch;
      width: 100%;
      display: grid;
      gap: 10px;
    }}

    .chat-answer-card {{
      padding: 20px 20px 18px;
      border-radius: 24px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: rgba(255,255,255,0.98);
      margin-bottom: 0;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
    }}

    .chat-answer-head {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 10px;
    }}

    .chat-answer-head h3 {{
      margin: 0;
      font-size: 14px;
      color: var(--navy);
      font-weight: 700;
    }}

    .answer-badge {{
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(17, 59, 104, 0.08);
      color: var(--navy);
      font-size: 12px;
      font-weight: 700;
    }}

    .chat-answer-text {{
      margin: 0;
      color: var(--text);
      line-height: 1.75;
      font-size: 15px;
      white-space: pre-wrap;
    }}

    .chat-loading {{
      padding: 16px 18px;
      border-radius: 22px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: rgba(255,255,255,0.96);
      color: var(--muted);
      font-size: 14px;
    }}
    .chat-composer {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 10px;
        align-items: end;
        padding: 10px;
        border-radius: 18px;
        background: #fff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03);
        min-height: 0;
    }}
    

    .chat-toolbar {{
      display: flex;
      align-items: stretch;
      justify-content: flex-end;
    }}

    input[type=file] {{
      display: none;
    }}

    /* ?몃줈 怨듦컙??吏㏃쓣 ???쒕∼議?異뺤냼 */
    @media (max-height: 700px) {{
      .upload-dropzone {{
        padding: 10px 12px;
        font-size: 12px;
      }}
      .upload-dropzone-icon {{
        font-size: 16px;
      }}
      .upload-dropzone span:last-child {{
        display: none;
      }}
    }}

    @media (max-width: 1100px) {{
      html, body {{
        height: auto;
        overflow-y: auto;
      }}

      body {{
        padding: 0;
      }}

      .app {{
        grid-template-columns: 1fr;
        height: auto;
        min-height: 100vh;
        width: 100%;
        max-width: none;
        margin: 0;
        overflow: visible;
        border-radius: 0;
        border: 0;
        box-shadow: none;
      }}

      .sidebar {{
        height: auto;
        overflow: visible;
        border-right: 0;
        border-bottom: 1px solid rgba(15, 23, 42, 0.06);
      }}

      .sidebar-workspace {{
        overflow: visible;
        grid-template-rows: auto auto;
      }}

      .sidebar-section {{
        overflow: visible;
      }}

      /* 紐⑤컮?? flex ?좎??섎릺 overflow visible濡??꾩껜 ?쒖떆 */
      .sidebar-section-docs {{
        overflow: visible;
      }}
      .sidebar-section-docs .doc-list {{
        flex: none;
        max-height: none;
      }}

      /* 紐⑤컮?? ?대? ?ㅽ겕濡??쒓굅 ??紐⑸줉 ?꾩껜 ?쒖떆, ?섏씠吏 ?ㅽ겕濡ㅻ줈 ?먯깋 */
      #document-list {{
        max-height: none;
        overflow-y: visible;
      }}

      .main {{
        height: auto;
        min-height: 0;
        overflow: visible;
      }}

      .workspace {{
        height: auto;
        overflow: visible;
      }}

      .content-pane {{
        height: auto;
        overflow: visible;
        grid-template-columns: 1fr;
        grid-template-rows: auto auto;
      }}

      .summary-box {{
        min-height: 160px;
        overflow: visible;
      }}

      .qa-box {{
        overflow: visible;
      }}

      .chat-shell {{
        min-height: 300px;
      }}
    }}

    @media (max-width: 720px) {{
      .sidebar {{
        padding: 16px 14px;
      }}

      .upload-dropzone {{
        padding: 10px 12px;
        font-size: 12px;
      }}

      .upload-dropzone-icon {{
        font-size: 18px;
      }}

      .main {{
        padding: 14px;
      }}

      .topbar {{
        align-items: flex-start;
        flex-direction: column;
      }}

      .topbar-right {{
        width: 100%;
        justify-content: flex-start;
        flex-wrap: wrap;
      }}

      .doc-card {{
        grid-template-columns: 20px minmax(0, 1fr) 24px;
        padding: 10px;
        border-radius: 12px;
      }}

      .doc-icon {{
        width: 20px;
        height: 20px;
        border-radius: 6px;
        font-size: 10px;
      }}

      .doc-check {{
        width: 22px;
        height: 22px;
        font-size: 12px;
      }}

      .qa-box {{
        padding: 14px;
      }}

      .chat-shell {{
        min-height: 260px;
        padding: 12px;
        border-radius: 22px;
      }}

      .chat-history {{
        min-height: 220px;
      }}

      .chat-composer {{
        padding: 10px;
        border-radius: 20px;
      }}

      .chat-toolbar {{
        align-items: stretch;
      }}

      .summary-header {{
        flex-direction: column;
        align-items: stretch;
      }}

      .page-title h1 {{
        font-size: 22px;
      }}

      .pill-button {{
        width: 100%;
        justify-content: center;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-badge">D</div>
        <span>Document Studio</span>
      </div>

      <div class="sidebar-workspace" id="sidebar-workspace">
        <section class="sidebar-section">
          <div class="sidebar-list-head">
            <h3>臾몄꽌 ?낅줈??/h3>
            <span class="sidebar-list-count">Quick Add</span>
          </div>
          <form class="sidebar-upload" id="workspace-upload-form">
            <input id="workspace-documents" name="documents" type="file" multiple accept="{accept}">
            <div id="upload-dropzone" class="upload-dropzone">
              <span class="upload-dropzone-icon">&#8659;</span>
              <span>?뚯씪???ш린???쒕옒洹명븯???낅줈??/span>
              <span style="font-size:11px; opacity:0.7;">?먮뒗 ?꾨옒 踰꾪듉?쇰줈 ?좏깮</span>
            </div>
            <div class="sidebar-upload-actions">
              <button class="circle-button" id="workspace-pick-button" type="button">+</button>
              <button class="send-button" id="workspace-upload-button" type="submit">?낅줈??/button>
              <div class="upload-spinner" id="upload-spinner"></div>
            </div>
            <div class="sidebar-selected-files" id="workspace-selected-files">
              <span class="file-chip">?좏깮??臾몄꽌媛 ?놁뒿?덈떎.</span>
            </div>
            <div id="status-box" style="display:none; font-size:12px; color:var(--muted); line-height:1.6; padding:6px 4px; white-space:pre-wrap;"></div>
          </form>
        </section>

        <section class="sidebar-section sidebar-section-docs">
          <div class="sidebar-list-head">
            <h3>臾몄꽌 ?좏깮</h3>
            <span class="sidebar-list-count" id="document-count-chip">0 docs</span>
          </div>

          <div class="selection-status" id="checked-summary">
            <span>QA</span>
            <strong>0 selected</strong>
          </div>

          <div class="select-all-row" id="select-all-row" style="display:none;">
            <span class="select-all-text">?꾩껜 ?좏깮</span>
            <span class="doc-check" id="select-all-check"></span>
            <button id="delete-selected-button" type="button" style="margin-left:auto;padding:5px 14px;border:none;border-radius:999px;background:#dc2626;color:#fff;font:inherit;font-size:13px;font-weight:700;cursor:pointer;opacity:0.4;pointer-events:none;">??젣</button>
          </div>

          <div class="doc-list sidebar-doc-list" id="document-list">
            <div class="status-box">臾몄꽌 紐⑸줉??遺덈윭?ㅻ뒗 以묒엯?덈떎.</div>
          </div>
        </section>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div class="page-title">
          <h1>臾몄꽌 蹂닿린</h1>
        </div>
      </div>

      

      <section class="workspace" id="workspace-screen">
        <section class="content-pane">
          <section class="panel summary-box">
            <div class="summary-header">
              <div>
                <h2>?붿빟</h2>
              </div>
              <div class="toolbar">
                <button class="ghost-button" id="download-md-button" type="button">?ㅼ슫濡쒕뱶 .md</button>
                <button class="summary-toggle" id="summary-toggle" type="button" title="?묎린 / ?쇱튂湲?>??/button>
              </div>
            </div>
            <div id="summary-content" style="margin-top:16px;">
              <p class="summary-text" style="color: var(--muted);">臾몄꽌瑜??좏깮?섎㈃ ?먮룞?쇰줈 ?붿빟??遺덈윭?듬땲??</p>
            </div>
          </section>

          <section class="panel qa-box">
            <h2>吏덈Ц</h2>
            <div class="chat-shell">
              <div class="chat-history" id="chat-history">
                <div class="chat-empty">臾몄꽌瑜??좏깮????吏덈Ц???낅젰?섏꽭??</div>
              </div>
            </div>
            <form class="chat-composer" id="query-form">
              <textarea class="question-area" id="query-input" placeholder="?? ??臾몄꽌???듭떖 議곌굔????以꾨줈 ?뺣━?댁쨾"></textarea>
              <div class="chat-toolbar">
                <button class="pill-button" id="query-button" type="submit">吏덈Ц?섍린</button>
              </div>
            </form>
          </section>
        </section>
      </section>
    </main>
  </div>

  <script>
    const state = {{
      documents: [],
      selectedDocumentId: '',
      checkedDocumentIds: [],
      latestRun: null,
      chatHistory: {{}},
      summaryCache: {{}},
      summaryInflight: {{}},
      guideCollapsed: false,
    }};

    const sidebarWorkspace = document.getElementById('sidebar-workspace');
    const workspaceScreen = document.getElementById('workspace-screen');

    const workspaceUploadForm = document.getElementById('workspace-upload-form');
    const workspaceFileInput = document.getElementById('workspace-documents');
    const workspacePickButton = document.getElementById('workspace-pick-button');
    const workspaceUploadButton = document.getElementById('workspace-upload-button');
    const workspaceSelectedFiles = document.getElementById('workspace-selected-files');

    const statusBox = document.getElementById('status-box');
    const documentList = document.getElementById('document-list');
    const checkedSummary = document.getElementById('checked-summary');
    const documentCountChip = document.getElementById('document-count-chip');
    const uploadSpinner = document.getElementById('upload-spinner');
    const selectAllRow = document.getElementById('select-all-row');
    const selectAllCheck = document.getElementById('select-all-check');
    const summaryContent = document.getElementById('summary-content');
    const downloadMdButton = document.getElementById('download-md-button');
    const summaryToggle = document.getElementById('summary-toggle');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const queryButton = document.getElementById('query-button');
    const chatHistoryEl = document.getElementById('chat-history');

    sidebarWorkspace.classList.add('active');
    workspaceScreen.classList.add('active');

    // ?뚯뒪 媛?대뱶 ?묎린/?쇱튂湲??좉?
    summaryToggle.addEventListener('click', () => {{
      state.guideCollapsed = !state.guideCollapsed;
      summaryContent.classList.toggle('guide-collapsed', state.guideCollapsed);
      summaryContent.closest('.summary-box').classList.toggle('guide-collapsed', state.guideCollapsed);
      summaryToggle.textContent = state.guideCollapsed ? '?? : '??;
      // ?묒쑝硫??꾩옱 ?좏깮 臾몄꽌??泥댄겕 ?댁젣 (?먯쑀 ?좏깮 蹂듦?)
      if (state.guideCollapsed && state.selectedDocumentId) {{
        state.checkedDocumentIds = state.checkedDocumentIds.filter(id => id !== state.selectedDocumentId);
        renderDocumentList();
        renderCheckedSummary();
      }}
    }});

    function esc(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function setStatus(message) {{
      if (!statusBox) return;
      const text = Array.isArray(message) ? message.join('\\n') : String(message ?? '');
      statusBox.textContent = text;
      statusBox.style.display = text ? 'block' : 'none';
    }}

    function renderSelectedFiles(files, targetEl) {{
      if (!targetEl) return;
      if (!files.length) {{
        targetEl.innerHTML = '<span class="file-chip">?좏깮??臾몄꽌媛 ?놁뒿?덈떎.</span>';
        return;
      }}
      targetEl.innerHTML = files
        .map((file) => `<span class="file-chip">${{esc(file.name)}} 쨌 ${{Math.max(1, Math.round(file.size / 1024))}}KB</span>`)
        .join('');
    }}

    function resetUploadUiState(buttonEl = workspaceUploadButton) {{
      if (buttonEl) buttonEl.disabled = false;
      uploadSpinner.classList.remove('active');
      workspaceFileInput.value = '';
      renderSelectedFiles([], workspaceSelectedFiles);
    }}

    function getSelectedDocument() {{
      return state.documents.find((item) => item.document_id === state.selectedDocumentId) || null;
    }}

    function dedupeDocumentIds(ids) {{
      return [...new Set((ids || []).filter(Boolean))];
    }}

    function isCheckedDocument(documentId) {{
      return state.checkedDocumentIds.includes(documentId);
    }}

    function getScopedDocumentIds() {{
      if (state.checkedDocumentIds.length) return dedupeDocumentIds(state.checkedDocumentIds);
      return state.selectedDocumentId ? [state.selectedDocumentId] : [];
    }}

    function renderCheckedSummary() {{
      const scoped = getScopedDocumentIds();
      documentCountChip.textContent = `${{state.documents.length}} docs`;
      if (!scoped.length) {{
        checkedSummary.innerHTML = '<span>QA</span><strong>0 selected</strong>';
        return;
      }}
      const selectedDocs = scoped
        .map((id) => state.documents.find((d) => d.document_id === id))
        .filter(Boolean);
      const MAX_SHOW = 2;
      const shownDocs = selectedDocs.slice(0, MAX_SHOW);
      const restCount = selectedDocs.length - shownDocs.length;
      const nameItemsHtml = shownDocs
        .map((d) => `<span class="selection-status-name-item" title="${{esc(d.source_name || d.document_id)}}">${{esc(d.source_name || d.document_id)}}</span>`)
        .join('');
      const restHtml = restCount > 0
        ? `<span class="selection-status-name-item" style="font-style:italic;">? ${{restCount}}?</span>`
        : '';
      checkedSummary.innerHTML = `
        <div class="selection-status-names">
          <strong style="font-size:13px;">${{scoped.length}} selected</strong>
          ${{nameItemsHtml}}${{restHtml}}
        </div>`;
    }}

    function toggleCheckedDocument(documentId) {{
      if (!documentId) return;
      if (isCheckedDocument(documentId)) {{
        state.checkedDocumentIds = state.checkedDocumentIds.filter((item) => item !== documentId);
      }} else {{
        state.checkedDocumentIds = dedupeDocumentIds([...state.checkedDocumentIds, documentId]);
      }}
      if (!state.checkedDocumentIds.length && state.selectedDocumentId) {{
        state.checkedDocumentIds = [state.selectedDocumentId];
      }}
      renderDocumentList();
      renderCheckedSummary();
    }}

    const deleteSelectedButton = document.getElementById('delete-selected-button');

    function updateSelectAllCheckbox() {{
      if (!state.documents.length) {{
        selectAllRow.style.display = 'none';
        return;
      }}
      selectAllRow.style.display = 'flex';
      const allChecked = state.documents.every((doc) => state.checkedDocumentIds.includes(doc.document_id));
      if (allChecked) {{
        selectAllCheck.classList.add('checked');
        selectAllCheck.textContent = '??;
      }} else {{
        selectAllCheck.classList.remove('checked');
        selectAllCheck.textContent = '';
      }}
      const hasChecked = state.checkedDocumentIds.length > 0;
      deleteSelectedButton.style.opacity = hasChecked ? '1' : '0.4';
      deleteSelectedButton.style.pointerEvents = hasChecked ? 'auto' : 'none';
    }}

    deleteSelectedButton.addEventListener('click', async (event) => {{
      event.stopPropagation();
      const ids = [...state.checkedDocumentIds];
      if (!ids.length) return;
      if (!confirm(`?좏깮??臾몄꽌 ${{ids.length}}媛쒕? 踰≫꽣 DB?먯꽌 ??젣?⑸땲?? 怨꾩냽?섏떆寃좎뒿?덇퉴?`)) return;
      deleteSelectedButton.textContent = '??젣 以?..';
      deleteSelectedButton.style.pointerEvents = 'none';
      try {{
        const response = await fetch('/api/delete-documents', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ document_ids: ids }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '??젣???ㅽ뙣?덉뒿?덈떎.');
        state.checkedDocumentIds = [];
        state.selectedDocumentId = '';
        await loadDocuments();
      }} catch (error) {{
        alert(String(error));
      }} finally {{
        deleteSelectedButton.textContent = '??젣';
        updateSelectAllCheckbox();
      }}
    }});

    function toggleSelectAll() {{
      const allChecked = state.documents.every((doc) => state.checkedDocumentIds.includes(doc.document_id));
      if (allChecked) {{
        state.checkedDocumentIds = state.selectedDocumentId ? [state.selectedDocumentId] : [];
      }} else {{
        state.checkedDocumentIds = state.documents.map((doc) => doc.document_id);
      }}
      renderDocumentList();
      renderCheckedSummary();
    }}

    selectAllRow.addEventListener('click', toggleSelectAll);

    function renderDocumentList() {{
      if (!state.documents.length) {{
        documentList.innerHTML = '<div class="status-box">?쒖떆??臾몄꽌媛 ?놁뒿?덈떎.</div>';
        selectAllRow.style.display = 'none';
        renderCheckedSummary();
        return;
      }}

      documentList.innerHTML = state.documents.map((doc) => {{
        const active = doc.document_id === state.selectedDocumentId ? ' active' : '';
        const checked = isCheckedDocument(doc.document_id);

        return `<button class="doc-card${{active}}" type="button" data-id="${{esc(doc.document_id)}}" title="${{esc(doc.source_name || doc.document_id)}}">
          <span class="doc-icon">${{esc((doc.document_type || doc.extension || 'D').slice(0, 1).toUpperCase())}}</span>
          <span class="doc-copy">
            <strong>${{esc(doc.source_name || doc.document_id)}}</strong>
          </span>
          <span class="doc-check${{checked ? ' checked' : ''}}" data-check-id="${{esc(doc.document_id)}}">${{checked ? '?? : ''}}</span>
        </button>`;
      }}).join('');

      documentList.querySelectorAll('[data-id]').forEach((button) => {{
        button.addEventListener('click', () => {{
          const docId = button.getAttribute('data-id') || '';
          state.selectedDocumentId = docId;
          // ?뚯뒪 ?좏깮 ????긽 泥댄겕諛뺤뒪 ?좏깮
          if (docId && !isCheckedDocument(docId)) {{
            state.checkedDocumentIds = dedupeDocumentIds([...state.checkedDocumentIds, docId]);
          }}
          // ?뚯뒪 媛?대뱶 ?쇱튂湲?
          if (state.guideCollapsed) {{
            state.guideCollapsed = false;
            summaryContent.classList.remove('guide-collapsed');
            summaryToggle.textContent = '??;
          }}
          renderDocumentList();
          renderDocumentPlaceholder();
          queryInput.value = '';
          renderChat();
          renderCheckedSummary();
        }});
      }});

      documentList.querySelectorAll('[data-check-id]').forEach((checkEl) => {{
        checkEl.addEventListener('click', (event) => {{
          event.stopPropagation();
          toggleCheckedDocument(checkEl.getAttribute('data-check-id') || '');
        }});
      }});

      updateSelectAllCheckbox();
      renderCheckedSummary();
    }}

    function renderDocumentPlaceholder() {{
      const selected = getSelectedDocument();
      if (!selected) {{
        summaryContent.innerHTML = '<p class="summary-text" style="color: var(--muted);">臾몄꽌瑜??좏깮?섎㈃ ?붿빟???뺤씤?????덉뒿?덈떎.</p>';
        return;
      }}

      const cached = state.summaryCache[selected.document_id];
      if (cached) {{
        renderSummaryHtml(cached);
        return;
      }}

      summaryContent.innerHTML =
        `<div class="summary-meta">
          <span>${{esc(selected.source_name || selected.document_id)}}</span>
          <span>${{esc(selected.document_type || selected.extension || 'document')}}</span>
          <span>${{esc(selected.origin || 'project')}}</span>
        </div>
        <p class="summary-text" style="color:var(--muted);">?붿빟??遺덈윭?ㅻ뒗 以묒엯?덈떎...</p>`;

      autoSummarizeDocument(selected.document_id);
    }}

    function renderSummaryHtml(payload) {{
      const keyPoints = payload.key_points || [];
      summaryContent.innerHTML =
        `<div class="summary-meta">
          <span>${{esc(payload.source_name || payload.document_id || '')}}</span>
          <span>${{esc(payload.document_type || 'document')}}</span>
          <span>${{esc(payload.used_model || '')}}</span>
        </div>
        <p class="summary-text">${{esc(payload.summary_text || '')}}</p>
        <div class="points">
          ${{
            keyPoints.length
              ? keyPoints.map((item) => `<div class="point">${{esc(item)}}</div>`).join('')
              : '<div class="point">蹂꾨룄 ?듭떖 ?ъ씤?멸? ?놁뒿?덈떎.</div>'
          }}
        </div>`;
    }}

    function renderSummary(payload, forDocumentId) {{
      state.summaryCache[forDocumentId] = {{
        ...payload,
        summary_kind: 'ui',
      }};
      delete state.summaryInflight[forDocumentId];
      if (state.selectedDocumentId === forDocumentId) {{
        renderSummaryHtml(state.summaryCache[forDocumentId]);
      }}
    }}

    async function autoSummarizeDocument(documentId, {{ force = false }} = {{}}) {{
      if (state.summaryInflight[documentId]) return;
      if (!force && state.summaryCache[documentId]) return;
      const doc = state.documents.find((d) => d.document_id === documentId);
      if (!doc) return;

      state.summaryInflight[documentId] = true;
      try {{
        const response = await fetch('/api/summarize', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            document_id: doc.document_id,
            source_name: doc.source_name,
          }}),
        }});
        const payload = await response.json();
        if (response.ok) {{
          renderSummary(payload, documentId);
        }}
      }} catch (_) {{
        // silent fail
      }} finally {{
        delete state.summaryInflight[documentId];
      }}
    }}

    function renderAnswerBubble(result) {{
      var answerHtml = result.error
        ? ('<span style="color:var(--muted);">오류: ' + esc(String(result.error || 'unknown_error')) + '</span>')
        : esc(result.answer || '응답을 생성할 수 없습니다.');

      return '<div class="chat-answer-card">'
        + '<div class="chat-answer-head">'
        + '<h3>QA 결과</h3>'
        + (result.used_model ? ('<span class="answer-badge">' + esc(result.used_model) + '</span>') : '')
        + '</div>'
        + '<p class="chat-answer-text">' + answerHtml + '</p>'
        + '</div>';
    }}

    function renderChat(scrollToBottom = false) {{
      const history = state.chatHistory[state.selectedDocumentId] || [];
      if (!history.length) {{
        chatHistoryEl.innerHTML = '<div class="chat-empty">臾몄꽌瑜??좏깮????吏덈Ц???낅젰?섏꽭??</div>';
        return;
      }}

      const turns = history.map((entry) => {{
        const aiHtml = entry.loading
          ? '<div class="chat-loading">?듬???李얜뒗 以묒엯?덈떎...</div>'
          : renderAnswerBubble(entry.result || {{}});

        return `<div class="chat-turn">
          <div class="chat-bubble-user">${{esc(entry.question)}}</div>
          <div class="chat-bubble-ai">${{aiHtml}}</div>
        </div>`;
      }}).join('');

      chatHistoryEl.innerHTML = turns;
      if (scrollToBottom) {{
        requestAnimationFrame(() => {{
          const shell = chatHistoryEl.parentElement;
          shell.scrollTop = shell.scrollHeight;
        }});
      }}
    }}

    async function loadDocuments(preferredDocumentId = '', {{ skipSummary = false }} = {{}}) {{
      const response = await fetch('/api/documents', {{ cache: 'no-store' }});
      const payload = await response.json();
      state.documents = payload.documents || [];
      state.documents.forEach((doc) => {{
        const cached = doc.ui_summary || doc.llm_summary || doc.basic_summary;
        if (!cached || !doc.document_id) return;
        state.summaryCache[doc.document_id] = {{
          document_id: doc.document_id,
          source_name: doc.source_name,
          document_type: doc.document_type || doc.extension || 'document',
          summary_text: cached.summary_text || '',
          key_points: cached.key_points || cached.highlights || [],
          used_model: cached.used_model || (doc.ui_summary ? 'cached_ui_summary' : (doc.llm_summary ? 'cached_llm_summary' : 'cached_basic_summary')),
          summary_kind: doc.ui_summary ? 'ui' : (doc.llm_summary ? 'llm' : 'basic'),
        }};
      }});

      if (!state.documents.length) {{
        state.selectedDocumentId = '';
        state.checkedDocumentIds = [];
        renderDocumentList();
        return;
      }}

      if (preferredDocumentId && state.documents.some((item) => item.document_id === preferredDocumentId)) {{
        state.selectedDocumentId = preferredDocumentId;
      }} else if (!state.selectedDocumentId || !state.documents.some((item) => item.document_id === state.selectedDocumentId)) {{
        state.selectedDocumentId = state.documents[0].document_id;
      }}

      const availableIds = new Set(state.documents.map((item) => item.document_id));
      state.checkedDocumentIds = dedupeDocumentIds(
        state.checkedDocumentIds.filter((item) => availableIds.has(item))
      );

      if (!state.checkedDocumentIds.length && state.selectedDocumentId) {{
        state.checkedDocumentIds = [state.selectedDocumentId];
      }}

      renderDocumentList();
      // skipSummary=true???뚮뒗 ?대쭅 以?誘몄셿???몃뜳?ㅻ줈 ?붿빟?섏? ?딆쓬
      if (!skipSummary) renderDocumentPlaceholder();
      renderCheckedSummary();
    }}

    async function pollJob(jobId) {{
      let lastCompletedCount = 0;
      let pollTick = 0;
      while (true) {{
        pollTick += 1;
        const response = await fetch('/api/jobs/' + jobId, {{ cache: 'no-store' }});
        const payload = await response.json();
        const progress = payload.progress || {{ current: 0, total: 1 }};
        const progressCurrent = Number(progress.current) || 0;
        const progressTotal = Number(progress.total) || 0;
        const dots = '.'.repeat((pollTick % 3) + 1);

        const progressText = progressTotal > 1
          ? Math.min(progressCurrent, progressTotal) + ' / ' + progressTotal + '媛??꾨즺'
          : (payload.status === 'running' ? ('泥섎━ 以? + dots) : '');
        setStatus([
          payload.message || '',
          progressText
        ].filter(Boolean));

        // 吏꾪뻾 ?곹솴??諛붾뚭굅?? ?ㅽ뻾 以묒씪 ??3?깅쭏??二쇨린?곸쑝濡?紐⑸줉 媛깆떊
        // skipSummary:true ???몃뜳???꾨즺 ?꾩뿉 鍮??붿빟??罹먯떆????λ릺??寃껋쓣 諛⑹?
        const shouldRefresh = progressCurrent > lastCompletedCount || (payload.status === 'running' && pollTick % 3 === 0);
        if (shouldRefresh) {{
          lastCompletedCount = Math.max(lastCompletedCount, progressCurrent);
          try {{
            const prevIds = new Set(state.documents.map((d) => d.document_id));
            await loadDocuments(state.selectedDocumentId || '', {{ skipSummary: true }});
            const newDocs = state.documents.filter((d) => !prevIds.has(d.document_id));
            if (newDocs.length > 0) {{
              state.checkedDocumentIds = dedupeDocumentIds([
                ...state.checkedDocumentIds,
                ...newDocs.map((d) => d.document_id),
              ]);
              renderDocumentList();
              renderCheckedSummary();
            }}
          }} catch (_) {{
            // 紐⑸줉 媛깆떊 ?ㅽ뙣 ??臾댁떆?섍퀬 ?대쭅 怨꾩냽
          }}
        }}

        if (payload.status === 'completed') {{
          resetUploadUiState();
          workspaceFileInput.value = '';
          renderSelectedFiles([], workspaceSelectedFiles);

          const result = payload.result || {{}};
          const summary = result.summary || {{}};
          const vectorSummary = result.vector_index || {{}};
          const timings = result.timings || {{}};

          // ?뚯씪 ?덈꺼 以묐났 (duplicate_uploads): existing_document_id ?ы븿
          const duplicateUploads = result.duplicate_uploads || [];
          // 而⑦뀗痢??덈꺼 以묐났 (content_duplicate_uploads): document_id ?ы븿
          const contentDuplicates = result.content_duplicate_uploads || [];
          // ?덈줈 indexed??臾몄꽌??
          const indexedDocs = (vectorSummary.documents || []).filter((item) => item.status === 'indexed');

          const newCount = indexedDocs.length;
          const dupCount = duplicateUploads.length + contentDuplicates.length;

          // 以묐났 臾몄꽌??湲곗〈 document_id ?섏쭛
          const dupDocIds = [
            ...duplicateUploads.map((d) => d.existing_document_id).filter(Boolean),
            ...contentDuplicates.map((d) => d.document_id).filter(Boolean),
          ];
          // ?좉퇋 臾몄꽌 ID ?섏쭛
          const newDocIds = indexedDocs.map((d) => d.document_id).filter(Boolean);
          // ?꾩껜 諛곗튂 ID (以묐났 ?곗꽑, ?좉퇋 異붽?, 以묐났 ?쒓굅)
          const batchDocIds = [...new Set([...dupDocIds, ...newDocIds])];
          // ?먮룞 ?좏깮: 以묐났???덉쑝硫?泥?以묐났, ?놁쑝硫?泥??좉퇋
          const preferredDocId = batchDocIds[0] || '';

          const statusLines = [
            'Run ID: ' + (result.run_id || ''),
            '?좉퇋 臾몄꽌: ' + newCount + '媛? /  以묐났 臾몄꽌: ' + dupCount + '媛?,
          ];
          if (typeof timings.qa_ready_seconds === 'number') statusLines.push('QA ready: ' + timings.qa_ready_seconds.toFixed(3) + 's');
          if (typeof timings.parse_seconds === 'number') statusLines.push('Parse: ' + timings.parse_seconds.toFixed(3) + 's');
          if (typeof timings.vector_index_seconds === 'number') statusLines.push('Index: ' + timings.vector_index_seconds.toFixed(3) + 's');
          if (typeof timings.llm_summary_seconds === 'number') statusLines.push('Summary: ' + timings.llm_summary_seconds.toFixed(3) + 's');
          if (payload.message) statusLines.push(payload.message);
          if (dupCount > 0) {{
            const filedupNames = duplicateUploads
              .map((d) => d.existing_source_name || d.existing_document_id || '')
              .filter(Boolean);
            const contentdupNames = contentDuplicates
              .map((d) => d.source_name || d.document_id || '')
              .filter(Boolean);
            if (filedupNames.length) {{
              const MAX_SHOW = 3;
              const shown = filedupNames.slice(0, MAX_SHOW).join(', ');
              const rest = filedupNames.length > MAX_SHOW ? ` ??${{filedupNames.length - MAX_SHOW}}媛? : '';
              statusLines.push('?뚯씪 以묐났 (' + filedupNames.length + '媛?: ' + shown + rest);
            }}
            if (contentdupNames.length) {{
              const MAX_SHOW = 3;
              const shown = contentdupNames.slice(0, MAX_SHOW).join(', ');
              const rest = contentdupNames.length > MAX_SHOW ? ` ??${{contentdupNames.length - MAX_SHOW}}媛? : '';
              statusLines.push('?댁슜 以묐났 (' + contentdupNames.length + '媛?: ' + shown + rest);
            }}
          }}
          setStatus(statusLines);

          state.latestRun = result;

          // 泥?踰덉㎏ 臾몄꽌(以묐났 ?곗꽑) ?먮룞 ?좏깮
          await loadDocuments(preferredDocId);

          // ?대쾲 諛곗튂???ы븿??紐⑤뱺 臾몄꽌(?좉퇋+以묐났) 泥댄겕諛뺤뒪 ?좏깮
          if (batchDocIds.length) {{
            const availableIds = new Set(state.documents.map((d) => d.document_id));
            const validBatchIds = batchDocIds.filter((id) => availableIds.has(id));
            if (validBatchIds.length) {{
              state.checkedDocumentIds = dedupeDocumentIds(validBatchIds);
              renderDocumentList();
              renderCheckedSummary();
            }}
          }}

          // ?좉퇋 臾몄꽌 ?꾩껜 ?붿빟: ?대쭅 以?鍮?罹먯떆媛 ??λ릱?????덉쑝誘濡?
          // newDocIds 罹먯떆瑜?珥덇린?뷀븳 ???ъ슂??
          newDocIds.forEach((id) => {{ delete state.summaryCache[id]; }});
          if (newDocIds.length > 0) {{
            // ?좏깮??臾몄꽌媛 ?좉퇋 紐⑸줉???덉쑝硫?癒쇱? 泥섎━
            const selectedFirst = newDocIds.includes(state.selectedDocumentId)
              ? [state.selectedDocumentId, ...newDocIds.filter((id) => id !== state.selectedDocumentId)]
              : newDocIds;
            // UI??"?붿빟 以?.." ?쒖떆 ??利됱떆 ?붿빟 ?쒖옉
            renderDocumentPlaceholder();
            await autoSummarizeDocument(selectedFirst[0]);
            for (const docId of selectedFirst.slice(1)) {{
              autoSummarizeDocument(docId); // fire-and-forget
            }}
          }} else {{
            // ?좉퇋 臾몄꽌 ?놁쓬(以묐났留??덈뒗 寃쎌슦) ???좏깮 臾몄꽌 ?붿빟? renderDocumentPlaceholder媛 泥섎━
            renderDocumentPlaceholder();
          }}

          renderChat();
          return;
        }}

        if (payload.status === 'failed') {{
          resetUploadUiState();
          setStatus(payload.error || payload.message || '?묒뾽???ㅽ뙣?덉뒿?덈떎.');
          return;
        }}

        await new Promise((resolve) => setTimeout(resolve, 1200));
      }}
    }}

    async function summarizeSelectedDocument() {{
      const selected = getSelectedDocument();
      if (!selected) {{
        summaryContent.innerHTML = '<p class="summary-text" style="color: var(--muted);">癒쇱? 臾몄꽌瑜??좏깮?섏꽭??</p>';
        return;
      }}

      const targetDocumentId = selected.document_id;
      downloadMdButton.disabled = true;

      if (state.selectedDocumentId === targetDocumentId) {{
        summaryContent.innerHTML = '<p class="summary-text" style="color: var(--muted);">?붿빟???앹꽦?섎뒗 以묒엯?덈떎...</p>';
      }}

      try {{
        const response = await fetch('/api/summarize', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            document_id: selected.document_id,
            source_name: selected.source_name,
          }}),
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '?붿빟 ?앹꽦???ㅽ뙣?덉뒿?덈떎.');
        renderSummary(payload, targetDocumentId);
      }} catch (error) {{
        if (state.selectedDocumentId === targetDocumentId) {{
          summaryContent.innerHTML = `<p class="summary-text" style="color: var(--muted);">${{esc(String(error))}}</p>`;
        }}
      }} finally {{
        downloadMdButton.disabled = false;
      }}
    }}

    function downloadSummary(format) {{
      const selected = getSelectedDocument();
      if (!selected) {{
        setStatus('癒쇱? ?ㅼ슫濡쒕뱶??臾몄꽌瑜??좏깮??二쇱꽭??');
        return;
      }}
      window.location.href =
        `/api/download-summary?document_id=${{encodeURIComponent(selected.document_id)}}&source_name=${{encodeURIComponent(selected.source_name || '')}}&format=${{encodeURIComponent(format)}}`;
    }}

    async function submitUploads(files, buttonEl) {{
      if (!files.length) {{
        setStatus('?낅줈?쒗븷 臾몄꽌瑜?癒쇱? ?좏깮?섏꽭??');
        return;
      }}

      buttonEl.disabled = true;
      uploadSpinner.classList.add('active');
      setStatus('?낅줈?쒕? ?쒖옉?덉뒿?덈떎. 臾몄꽌 泥섎━ ?뚯씠?꾨씪?몄쓣 以鍮꾪븯怨??덉뒿?덈떎.');

      const formData = new FormData();
      files.forEach((file) => formData.append('documents', file, file.name));

      try {{
        const response = await fetch('/api/run', {{ method: 'POST', body: formData }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '?뚯씠?꾨씪???ㅽ뻾???ㅽ뙣?덉뒿?덈떎.');
        await pollJob(payload.job_id);
      }} catch (error) {{
        resetUploadUiState(buttonEl);
        setStatus(String(error));
      }}
    }}

    workspacePickButton.addEventListener('click', () => workspaceFileInput.click());

    workspaceFileInput.addEventListener('change', () => {{
      renderSelectedFiles(Array.from(workspaceFileInput.files || []), workspaceSelectedFiles);
    }});

    workspaceUploadForm.addEventListener('submit', async (event) => {{
      event.preventDefault();
      await submitUploads(Array.from(workspaceFileInput.files || []), workspaceUploadButton);
    }});

    // ?쒕옒洹몄븻?쒕∼ ?낅줈??
    const uploadDropzone = document.getElementById('upload-dropzone');

    uploadDropzone.addEventListener('click', () => workspaceFileInput.click());

    uploadDropzone.addEventListener('dragenter', (event) => {{
      event.preventDefault();
      uploadDropzone.classList.add('drag-over');
      uploadDropzone.querySelector('span:nth-child(2)').textContent = '?ш린???볦쑝?몄슂!';
    }});

    uploadDropzone.addEventListener('dragover', (event) => {{
      event.preventDefault();
    }});

    uploadDropzone.addEventListener('dragleave', (event) => {{
      if (!uploadDropzone.contains(event.relatedTarget)) {{
        uploadDropzone.classList.remove('drag-over');
        uploadDropzone.querySelector('span:nth-child(2)').textContent = '?뚯씪???ш린???쒕옒洹명븯???낅줈??;
      }}
    }});

    uploadDropzone.addEventListener('drop', (event) => {{
      event.preventDefault();
      uploadDropzone.classList.remove('drag-over');
      uploadDropzone.querySelector('span:nth-child(2)').textContent = '?뚯씪???ш린???쒕옒洹명븯???낅줈??;
      const files = Array.from(event.dataTransfer.files || []);
      if (files.length) {{
        renderSelectedFiles(files, workspaceSelectedFiles);
        submitUploads(files, workspaceUploadButton);
      }}
    }});

    // ?섏씠吏 ?꾩껜?먯꽌 ?ㅼ닔濡??뚯씪???쒕∼?섎뒗 寃?諛⑹?
    document.addEventListener('dragover', (event) => event.preventDefault());
    document.addEventListener('drop', (event) => event.preventDefault());

    downloadMdButton.addEventListener('click', () => downloadSummary('md'));

    queryInput.addEventListener('keydown', (event) => {{
      if (event.key === 'Enter' && !event.shiftKey) {{
        event.preventDefault();
        queryForm.requestSubmit();
      }}
    }});

    queryForm.addEventListener('submit', async (event) => {{
      event.preventDefault();

      const selected = getSelectedDocument();
      const query = queryInput.value.trim();
      const scopedDocumentIds = getScopedDocumentIds();

      if (!selected) {{
        chatHistoryEl.innerHTML = '<div class="chat-empty">癒쇱? 臾몄꽌瑜??좏깮?섏꽭??</div>';
        return;
      }}

      if (!scopedDocumentIds.length) {{
        chatHistoryEl.innerHTML = '<div class="chat-empty">寃?됲븷 臾몄꽌瑜?泥댄겕?섏꽭??</div>';
        return;
      }}

      if (!query) {{
        queryInput.focus();
        return;
      }}

      queryInput.value = '';
      queryButton.disabled = true;

      if (!state.chatHistory[state.selectedDocumentId]) {{
        state.chatHistory[state.selectedDocumentId] = [];
      }}

      const entry = {{ question: query, result: null, loading: true }};
      state.chatHistory[state.selectedDocumentId].push(entry);
      renderChat(true);

      try {{
        // ?щ윭 臾몄꽌媛 ?좏깮??寃쎌슦 document_id瑜?鍮꾩썙??諛깆뿏??source_boost媛
        // ?쒖꽦 臾몄꽌 ?섎굹?먮쭔 吏묒쨷?섏? ?딅룄濡???
        const response = await fetch('/api/query', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            query,
            strategy: '{default_qa_strategy}',
            document_id: selected.document_id,
            source_name: selected.source_name,
            selected_document_ids: scopedDocumentIds,
          }}),
        }});

        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '吏덉쓽?묐떟???ㅽ뙣?덉뒿?덈떎.');
        entry.result = payload || {{}};
        entry.loading = false;
      }} catch (error) {{
        entry.result = {{ answer: String(error), citations: [], matches: [], error: String(error) }};
        entry.loading = false;
      }} finally {{
        queryButton.disabled = false;
        renderChat();
      }}
    }});

    renderSelectedFiles([], workspaceSelectedFiles);

    loadDocuments()
      .then(() => renderChat())
      .catch(() => {{
        documentList.innerHTML = '<div class="status-box">臾몄꽌 紐⑸줉??遺덈윭?ㅼ? 紐삵뻽?듬땲??</div>';
      }});

    // 踰≫꽣 DB ?몃뜳???꾨즺 ???쒕쾭 ?대깽?몃줈 臾몄꽌 紐⑸줉 ?먮룞 媛깆떊
    const evtSource = new EventSource('/api/events');
    evtSource.addEventListener('documents_updated', async () => {{
      const prevIds = new Set(state.documents.map((d) => d.document_id));
      await loadDocuments(state.selectedDocumentId || '');
      const newDocs = state.documents.filter((d) => !prevIds.has(d.document_id));
      if (newDocs.length > 0) {{
        state.checkedDocumentIds = dedupeDocumentIds([
          ...state.checkedDocumentIds,
          ...newDocs.map((d) => d.document_id),
        ]);
        renderDocumentList();
        renderCheckedSummary();
        if (workspaceUploadButton.disabled) {{
          resetUploadUiState();
          if (!newDocs.some((d) => d.document_id === state.selectedDocumentId)) {{
            state.selectedDocumentId = newDocs[0].document_id;
          }}
          renderDocumentList();
          renderDocumentPlaceholder();
          renderCheckedSummary();
        }}
      }}
    }});
  </script>
</body>
</html>
""".format(
        accept=",".join(sorted(SUPPORTED_EXTENSIONS)),
        default_qa_strategy=DEFAULT_QA_STRATEGY,
        latest_duplicates=latest_upload.get("duplicate_upload_count", 0),
        project_count=project_review.get("document_count", 0),
        qa_caption=qa_caption,
    )


def render_document_studio_html(status: dict[str, object]) -> str:
    project_review = status.get("project_review") or {}
    latest_upload = status.get("latest_upload") or {}
    openai_status = status.get("openai") or {}
    semantic_ready = is_semantic_strategy_available()

    qa_caption = ""

    initial_status = make_json_safe(
        {
            "project_review": project_review,
            "latest_upload": latest_upload,
            "openai": openai_status,
            "qa_caption": qa_caption,
            "semantic_ready": semantic_ready,
        }
    )
    initial_status_json = json.dumps(initial_status, ensure_ascii=False).replace("</", "<\\/")

    return _render_ui_template(
        DOCUMENT_STUDIO_TEMPLATE_PATH,
        {
            "INITIAL_STATUS_JSON": initial_status_json,
            "SUPPORTED_ACCEPT": html.escape(",".join(sorted(SUPPORTED_EXTENSIONS)), quote=True),
            "DEFAULT_QA_STRATEGY": html.escape(DEFAULT_QA_STRATEGY, quote=True),
        },
    )


class DocumentStudioRequestHandler(BaseHTTPRequestHandler):
    server_version = "DocumentStudio/1.0.0"

    def __init__(
        self,
        *args: object,
        manager: ReviewSessionManager,
        retriever: ChromaRetriever,
        langchain_service: LangChainRetrievalQAService | None,
        openapi_spec: dict[str, object],
        **kwargs: object,
    ) -> None:
        self.manager = manager
        self.retriever = retriever
        self.langchain_service = langchain_service
        self.openapi_spec = openapi_spec
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send_html(render_document_studio_html(self.manager.get_status()))
                return
            if path == "/docs":
                self._send_html(render_swagger_ui_html())
                return
            if path == "/openapi.json":
                self._send_json(self.openapi_spec)
                return
            if path == "/api/status":
                self._send_json(self.manager.get_status())
                return
            if path == "/api/documents":
                self._send_json({"documents": self.manager.get_document_list()})
                return
            if path == "/api/document-content":
                self._handle_document_content()
                return
            if path == "/api/document-element-preview":
                self._handle_document_element_preview()
                return
            if path == "/api/document-original":
                self._handle_document_original()
                return
            if path == "/api/download-summary":
                self._handle_download_summary()
                return
            if path == "/api/events":
                self._handle_sse()
                return
            if path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = self.manager.get_job(job_id)
                if not job:
                    self._send_json({"error": "job_not_found"}, status=404)
                    return
                self._send_json(job)
                return

            static_path = self.manager.resolve_static_path(path)
            if static_path:
                self._serve_file(static_path)
                return
            self.send_error(404, "Not found")
        except Exception as error:
            log_server_exception(f"GET {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/run":
                self._handle_run()
                return
            if path == "/api/summarize":
                self._handle_summarize()
                return
            if path == "/api/query-compare":
                self._handle_query_compare()
                return
            if path == "/api/query":
                self._handle_query()
                return
            if path == "/api/delete-documents":
                self._handle_delete_documents()
                return
            self.send_error(404, "Not found")
        except Exception as error:
            log_server_exception(f"POST {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def _handle_delete_documents(self) -> None:
        try:
            payload = self._read_json_payload()
            document_ids = [str(item).strip() for item in (payload.get("document_ids") or []) if str(item).strip()]
            if not document_ids:
                self._send_json({"error": "document_ids_required"}, status=400)
                return
            results = []
            for doc_id in document_ids:
                related_ids = self.manager.get_related_document_ids(document_id=doc_id)
                if not related_ids:
                    related_ids = [doc_id]
                vector_deleted_rule = 0
                vector_deleted_semantic = 0
                registry_removed = 0
                for target_doc_id in related_ids:
                    vector_result = self.manager.index_manager.delete_document(target_doc_id)
                    vector_deleted_rule += int(vector_result.get("deleted_rule_chunks") or 0)
                    vector_deleted_semantic += int(vector_result.get("deleted_semantic_chunks") or 0)
                    registry_removed += int(vector_result.get("removed_from_registry") or 0)
                files_removed = self.manager.delete_document_files(related_ids)
                results.append(
                    {
                        "document_id": doc_id,
                        "deleted_document_ids": related_ids,
                        "deleted_rule_chunks": vector_deleted_rule,
                        "deleted_semantic_chunks": vector_deleted_semantic,
                        "removed_from_registry": registry_removed,
                        "files_removed": files_removed,
                    }
                )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            import traceback
            self._send_json({"error": str(error), "traceback": traceback.format_exc()}, status=500)
            return
        self._send_json({"deleted": results}, status=200)

    def _handle_sse(self) -> None:
        import queue as _queue
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self.manager.subscribe_sse()
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except _queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.manager.unsubscribe_sse(q)

    def _handle_run(self) -> None:
        try:
            uploads = self._parse_uploads()
            job = self.manager.start_run(uploads)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            import traceback
            self._send_json({"error": str(error), "traceback": traceback.format_exc()}, status=500)
            return
        self._send_json(job, status=202)

    def _handle_summarize(self) -> None:
        try:
            payload = self._read_json_payload()
            document_id = str(payload.get("document_id", "")).strip()
            source_name = str(payload.get("source_name", "")).strip()
            if not document_id and not source_name:
                self._send_json({"error": "document_id_or_source_name_required"}, status=400)
                return
            document_payload = self.manager.get_document_payload(document_id=document_id, source_name=source_name)
            if document_payload:
                result = self._build_cached_summary_payload(
                    document_payload,
                    default_document_id=document_id,
                    default_source_name=source_name,
                )
                if result:
                    self._send_json(result, status=200)
                    return
            if self.langchain_service is None:
                self._send_json(
                    {
                        "error": "langchain_service_unavailable",
                        "warning": "LLM summary service is not available.",
                        "document_id": document_id,
                        "source_name": source_name,
                    },
                    status=503,
                )
                return
            result = self.langchain_service.summarize_document(document_id=document_id, source_name=source_name)
            if _is_llm_summary_payload(result):
                self._send_json(result, status=200)
                return
            warning = str(result.get("warning") or result.get("error") or "llm_summary_unavailable")
            self._send_json(
                {
                    "error": "llm_summary_unavailable",
                    "warning": warning,
                    "document_id": result.get("document_id", document_id),
                    "source_name": result.get("source_name", source_name),
                    "summary_source": result.get("summary_source") or "",
                },
                status=503,
            )
            return
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)
            return

    def _handle_download_summary(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            document_id = str((query.get("document_id") or [""])[0]).strip()
            source_name = str((query.get("source_name") or [""])[0]).strip()
            output_format = str((query.get("format") or ["md"])[0]).strip().lower() or "md"
            if output_format != "md":
                self._send_json({"error": "invalid_format"}, status=400)
                return
            if not document_id and not source_name:
                self._send_json({"error": "document_id_or_source_name_required"}, status=400)
                return
            payload = self.manager.get_document_payload(document_id=document_id, source_name=source_name)
            if not payload:
                self._send_json({"error": "document_not_found"}, status=404)
                return
            result = self._build_download_summary_payload(payload)
            source_label = str(result.get("source_name") or result.get("document_id") or "summary")
            summary_text = str(result.get("summary_text") or "").strip()
            key_points = [str(item).strip() for item in (result.get("key_points") or []) if str(item).strip()]
            content = self._build_summary_markdown(source_label, result, summary_text, key_points)
            content_type = "text/markdown; charset=utf-8"
            filename = self._make_download_filename(source_label, output_format)
            self._send_download(content=content, filename=filename, content_type=content_type)
        except Exception as error:
            log_server_exception(f"GET {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def _handle_document_content(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            document_id = str((query.get("document_id") or [""])[0]).strip()
            source_name = str((query.get("source_name") or [""])[0]).strip()
            if not document_id and not source_name:
                self._send_json({"error": "document_id_or_source_name_required"}, status=400)
                return
            payload = self.manager.get_document_payload(document_id=document_id, source_name=source_name)
            if not payload:
                self._send_json({"error": "document_not_found"}, status=404)
                return
            self._send_json(self._build_document_content_payload(payload), status=200)
        except Exception as error:
            log_server_exception(f"GET {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def _handle_document_original(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            document_id = str((query.get("document_id") or [""])[0]).strip()
            source_name = str((query.get("source_name") or [""])[0]).strip()
            if not document_id and not source_name:
                self._send_json({"error": "document_id_or_source_name_required"}, status=400)
                return
            source_path = self.manager.get_document_source_path(document_id=document_id, source_name=source_name)
            if source_path and source_path.suffix.lower() == ".pdf":
                self._serve_file(source_path)
                return
            # NOTE: Keep PDF preview as the primary path for HWP as well. The
            # intended UX is "show the LibreOffice/H2Orestart-converted PDF";
            # the embedded HWP preview image is only a last-resort fallback for
            # cases where conversion truly produced no usable PDF.
            preview_path = self.manager.get_document_preview_pdf_path(document_id=document_id, source_name=source_name)
            if preview_path:
                self._serve_file(preview_path)
                return
            preview_image_path = self.manager.get_document_preview_image_path(
                document_id=document_id,
                source_name=source_name,
            )
            if preview_image_path:
                self._serve_file(preview_image_path)
                return
            if source_path and source_path.suffix.lower() == ".pdf":
                self._serve_file(source_path)
                return
            self._send_json(
                {
                    "error": "document_preview_not_available",
                    "document_id": document_id,
                    "source_name": source_name,
                },
                status=404,
            )
        except Exception as error:
            log_server_exception(f"GET {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def _handle_document_element_preview(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            document_id = str((query.get("document_id") or [""])[0]).strip()
            source_name = str((query.get("source_name") or [""])[0]).strip()
            element_id = str((query.get("element_id") or [""])[0]).strip()
            raw_page_number = str((query.get("page_number") or [""])[0]).strip()
            raw_bbox = str((query.get("bbox") or [""])[0]).strip()
            if (not document_id and not source_name) or (not element_id and (not raw_page_number or not raw_bbox)):
                self._send_json({"error": "document_selector_and_preview_target_required"}, status=400)
                return

            payload = self.manager.get_document_payload(document_id=document_id, source_name=source_name)
            if not payload:
                self._send_json({"error": "document_not_found"}, status=404)
                return

            if element_id:
                page_number, bbox = self._find_payload_element_bbox(payload, element_id=element_id)
            else:
                try:
                    page_number = int(raw_page_number or 0)
                except (TypeError, ValueError):
                    page_number = 0
                bbox = self._parse_preview_bbox(raw_bbox)
            if page_number <= 0 or bbox is None:
                self._send_json({"error": "document_element_not_found"}, status=404)
                return

            preview_pdf_path = self._resolve_document_element_preview_pdf_path(
                payload,
                document_id=document_id,
                source_name=source_name,
            )
            if not preview_pdf_path:
                self._send_json({"error": "pdf_source_required_for_preview"}, status=404)
                return

            body = self._render_pdf_element_preview(preview_pdf_path, page_number=page_number, bbox=bbox)
            self._send_binary(body, content_type="image/png")
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:
            log_server_exception(f"GET {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def _resolve_document_element_preview_pdf_path(
        self,
        payload: dict[str, object],
        *,
        document_id: str,
        source_name: str,
    ) -> Path | None:
        source_path = self.manager.get_document_source_path(document_id=document_id, source_name=source_name)
        if source_path and source_path.suffix.lower() == ".pdf":
            return source_path

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        supporting_pdf = metadata.get("supporting_pdf") if isinstance(metadata.get("supporting_pdf"), dict) else {}
        conversion = supporting_pdf.get("conversion") if isinstance(supporting_pdf.get("conversion"), dict) else {}
        converted_pdf_raw = str(conversion.get("converted_pdf_path") or "").strip()
        if converted_pdf_raw:
            converted_pdf_path = Path(converted_pdf_raw)
            if not converted_pdf_path.is_absolute():
                converted_pdf_path = (ROOT / converted_pdf_path).resolve()
            else:
                converted_pdf_path = converted_pdf_path.resolve()
            if converted_pdf_path.exists() and converted_pdf_path.suffix.lower() == ".pdf":
                return converted_pdf_path

        preview_pdf_path = self.manager.get_document_preview_pdf_path(
            document_id=document_id,
            source_name=source_name,
        )
        if preview_pdf_path and preview_pdf_path.exists() and preview_pdf_path.suffix.lower() == ".pdf":
            return preview_pdf_path
        return None

    def _find_payload_element_bbox(
        self,
        payload: dict[str, object],
        *,
        element_id: str,
    ) -> tuple[int, list[float] | None]:
        for page_collection_name in ("pages", "asset_pages"):
            pages = payload.get(page_collection_name) if isinstance(payload.get(page_collection_name), list) else []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                try:
                    page_number = int(page.get("page_number") or 0)
                except (TypeError, ValueError):
                    page_number = 0
                elements = page.get("elements") if isinstance(page.get("elements"), list) else []
                for element in elements:
                    if not isinstance(element, dict):
                        continue
                    if str(element.get("element_id") or "").strip() != element_id:
                        continue
                    bbox = element.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        return page_number, None
                    try:
                        return page_number, [float(value) for value in bbox]
                    except (TypeError, ValueError):
                        return page_number, None
        return 0, None

    def _render_pdf_element_preview(
        self,
        source_path: Path,
        *,
        page_number: int,
        bbox: list[float],
    ) -> bytes:
        if page_number <= 0:
            raise ValueError("invalid_page_number")
        if len(bbox) != 4:
            raise ValueError("invalid_bbox")

        with fitz.open(source_path) as document:
            if page_number > document.page_count:
                raise ValueError("page_out_of_range")
            page = document[page_number - 1]
            rect = fitz.Rect(*(float(value) for value in bbox))
            if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                raise ValueError("invalid_bbox")

            padding = max(10.0, min(rect.width, rect.height) * 0.08)
            clip = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding) & page.rect
            if clip.is_empty or clip.width <= 0 or clip.height <= 0:
                raise ValueError("invalid_bbox")

            longest_edge = max(float(clip.width), float(clip.height), 1.0)
            scale = min(3.0, max(1.5, 720.0 / longest_edge))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
            if pixmap.colorspace is not None and pixmap.colorspace.n > 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            return pixmap.tobytes("png")

    def _parse_preview_bbox(self, raw_bbox: str) -> list[float] | None:
        parts = [part.strip() for part in str(raw_bbox or "").split(",") if part.strip()]
        if len(parts) != 4:
            return None
        try:
            return [float(part) for part in parts]
        except (TypeError, ValueError):
            return None

    def _decorate_query_payload(self, payload: dict[str, object]) -> dict[str, object]:
        citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
        if not citations:
            return payload

        enriched = dict(payload)
        updated_citations: list[dict[str, object]] = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            updated_citation = dict(citation)
            document_id = str(updated_citation.get("document_id") or "")
            source_name = str(updated_citation.get("source_name") or "")
            supporting_assets = updated_citation.get("supporting_assets") if isinstance(updated_citation.get("supporting_assets"), list) else []
            updated_assets: list[dict[str, object]] = []
            for asset in supporting_assets:
                if not isinstance(asset, dict):
                    continue
                updated_asset = dict(asset)
                element_id = str(updated_asset.get("element_id") or "")
                page_number = 0
                try:
                    page_number = int(updated_asset.get("page_number") or 0)
                except (TypeError, ValueError):
                    page_number = 0
                bbox = updated_asset.get("preview_bbox") if isinstance(updated_asset.get("preview_bbox"), list) else (
                    updated_asset.get("bbox") if isinstance(updated_asset.get("bbox"), list) else None
                )
                if document_id and element_id:
                    updated_asset["preview_url"] = (
                        "/api/document-element-preview?document_id={document_id}&source_name={source_name}&element_id={element_id}".format(
                            document_id=quote(document_id),
                            source_name=quote(source_name),
                            element_id=quote(element_id),
                        )
                    )
                elif document_id and page_number > 0 and isinstance(bbox, list) and len(bbox) == 4:
                    bbox_value = ",".join(str(float(value)) for value in bbox)
                    updated_asset["preview_url"] = (
                        "/api/document-element-preview?document_id={document_id}&source_name={source_name}&page_number={page_number}&bbox={bbox}".format(
                            document_id=quote(document_id),
                            source_name=quote(source_name),
                            page_number=page_number,
                            bbox=quote(bbox_value),
                        )
                    )
                updated_assets.append(updated_asset)
            updated_citation["supporting_assets"] = updated_assets
            updated_citations.append(updated_citation)
        enriched["citations"] = updated_citations
        return enriched

    def _build_cached_summary_payload(
        self,
        payload: dict[str, object],
        *,
        default_document_id: str = "",
        default_source_name: str = "",
    ) -> dict[str, object] | None:
        ui_summary = payload.get("ui_summary")
        if _is_llm_summary_payload(ui_summary):
            return {
                "document_id": payload.get("document_id", default_document_id),
                "source_name": payload.get("source_name", default_source_name),
                "summary_text": ui_summary.get("summary_text", ""),
                "key_points": ui_summary.get("key_points") or ui_summary.get("highlights") or [],
                "page_summaries": ui_summary.get("page_summaries") or [],
                "document_type": (
                    ui_summary.get("document_type")
                    or (payload.get("classification") or {}).get("document_type", "")
                    or payload.get("extension", "")
                ),
                "used_model": ui_summary.get("used_model") or "cached_ui_summary",
                "framework": ui_summary.get("framework") or "langchain",
                "summary_source": ui_summary.get("summary_source") or "cached_ui_summary",
            }
        llm_summary = payload.get("llm_summary")
        if _is_llm_summary_payload(llm_summary):
            return {
                "document_id": payload.get("document_id", default_document_id),
                "source_name": payload.get("source_name", default_source_name),
                "summary_text": llm_summary.get("summary_text", ""),
                "key_points": llm_summary.get("key_points") or llm_summary.get("highlights") or [],
                "page_summaries": ui_summary.get("page_summaries") if isinstance(ui_summary, dict) else [],
                "document_type": (
                    llm_summary.get("document_type")
                    or (payload.get("classification") or {}).get("document_type", "")
                    or payload.get("extension", "")
                ),
                "used_model": llm_summary.get("used_model") or "cached_llm_summary",
                "framework": llm_summary.get("framework") or "pipeline_llm_summary",
                "summary_source": llm_summary.get("summary_source") or "pipeline_llm_summary",
            }
        return None

    def _build_basic_summary_payload(
        self,
        payload: dict[str, object] | None,
        *,
        document_id: str = "",
        source_name: str = "",
    ) -> dict[str, object] | None:
        if not payload:
            return None
        page_summaries = payload.get("page_summaries") if isinstance(payload.get("page_summaries"), list) else []
        if not page_summaries:
            page_summaries = build_page_summaries(payload)
        basic_summary = payload.get("basic_summary")
        if not isinstance(basic_summary, dict) or not basic_summary:
            return None
        return {
            "document_id": payload.get("document_id", document_id),
            "source_name": payload.get("source_name", source_name),
            "summary_text": basic_summary.get("summary_text", ""),
            "key_points": basic_summary.get("key_points") or basic_summary.get("highlights") or [],
            "page_summaries": page_summaries,
            "document_type": (
                basic_summary.get("document_type")
                or (payload.get("classification") or {}).get("document_type", "")
                or payload.get("extension", "")
            ),
            "used_model": basic_summary.get("used_model") or "cached_basic_summary",
            "framework": basic_summary.get("framework") or "basic_summary_fallback",
            "summary_source": "basic_summary_fallback",
        }

    def _build_download_summary_payload(self, payload: dict[str, object]) -> dict[str, object]:
        ui_summary = payload.get("ui_summary") if isinstance(payload.get("ui_summary"), dict) else None
        llm_summary = payload.get("llm_summary") if isinstance(payload.get("llm_summary"), dict) else None
        basic_summary = payload.get("basic_summary") if isinstance(payload.get("basic_summary"), dict) else None
        summary = ui_summary or llm_summary or basic_summary or {}
        key_points = summary.get("key_points") if isinstance(summary.get("key_points"), list) else None
        if key_points is None:
            key_points = summary.get("highlights") if isinstance(summary.get("highlights"), list) else []
        return {
            "document_id": payload.get("document_id", ""),
            "source_name": payload.get("source_name", ""),
            "summary_text": summary.get("summary_text", ""),
            "key_points": key_points,
            "document_type": (
                summary.get("document_type")
                or (payload.get("classification") or {}).get("document_type", "")
                or payload.get("extension", "")
            ),
            "used_model": summary.get("used_model") or ("cached_llm_summary" if llm_summary else "cached_basic_summary"),
        }

    def _build_document_content_payload(self, payload: dict[str, object]) -> dict[str, object]:
        basic_summary = payload.get("basic_summary") if isinstance(payload.get("basic_summary"), dict) else {}
        llm_summary = payload.get("llm_summary") if isinstance(payload.get("llm_summary"), dict) else {}
        ui_summary = payload.get("ui_summary") if isinstance(payload.get("ui_summary"), dict) else {}
        preferred_summary = ui_summary if _is_llm_summary_payload(ui_summary) else llm_summary if _is_llm_summary_payload(llm_summary) else basic_summary
        document_id = str(payload.get("document_id", "")).strip()
        source_name = str(payload.get("source_name", "")).strip()
        original_preview_kind = "pdf"
        preview_pdf_path = self.manager.get_document_preview_pdf_path(
            document_id=document_id,
            source_name=source_name,
        )
        if preview_pdf_path is None and self.manager.get_document_preview_image_path(
            document_id=document_id,
            source_name=source_name,
        ) is not None:
            original_preview_kind = "image"
        viewer_markdown = enrich_pdf_markdown_for_viewer(payload, str(payload.get("markdown") or ""))
        sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
        normalized_sections = []
        for item in sections[:8]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if title:
                normalized_sections.append(title)
        semantic_chunks = payload.get("semantic_chunks") if isinstance(payload.get("semantic_chunks"), list) else []
        normalized_chunks: list[dict[str, object]] = []
        for index, chunk in enumerate(semantic_chunks):
            if not isinstance(chunk, dict):
                continue
            chunk_text = clean_viewer_markdown(str(chunk.get("text") or ""))
            if not chunk_text.strip():
                continue
            raw_chunk_index = chunk.get("chunk_index")
            try:
                chunk_index = int(raw_chunk_index if raw_chunk_index is not None else index)
            except (TypeError, ValueError):
                chunk_index = index
            normalized_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "section_hint": str(chunk.get("section_hint") or ""),
                    "strategy": str(chunk.get("strategy") or "semantic"),
                    "char_count": int(chunk.get("char_count", len(chunk_text)) or len(chunk_text)),
                }
            )
        normalized_chunks = _infer_semantic_chunk_page_numbers(payload, normalized_chunks)
        key_points = preferred_summary.get("key_points") or preferred_summary.get("highlights") or []
        raw_page_summaries = ui_summary.get("page_summaries") if isinstance(ui_summary.get("page_summaries"), list) else []
        if not _has_llm_page_summaries(raw_page_summaries):
            raw_page_summaries = llm_summary.get("page_summaries") if isinstance(llm_summary.get("page_summaries"), list) else []
        if not _has_llm_page_summaries(raw_page_summaries):
            raw_page_summaries = payload.get("page_summaries") if isinstance(payload.get("page_summaries"), list) else []
        page_summaries = _normalize_page_summary_items(raw_page_summaries)
        return {
            "document_id": document_id,
            "source_name": source_name,
            "document_type": (payload.get("classification") or {}).get("document_type", "") or payload.get("extension", ""),
            "origin": payload.get("origin", ""),
            "summary_text": preferred_summary.get("summary_text", ""),
            "key_points": [str(item).strip() for item in key_points if str(item).strip()][:6],
            "page_summaries": page_summaries,
            "section_titles": normalized_sections,
            "markdown": clean_viewer_markdown(viewer_markdown),
            "semantic_chunks": normalized_chunks,
            "original_preview_kind": original_preview_kind,
            "original_url": "/api/document-original?document_id={document_id}&source_name={source_name}".format(
                document_id=quote(document_id),
                source_name=quote(source_name),
            ),
            "extension": str(payload.get("extension") or ""),
        }

    def _build_summary_markdown(
        self,
        source_label: str,
        result: dict[str, object],
        summary_text: str,
        key_points: list[str],
    ) -> str:
        lines = [
            f"# {source_label} ?붿빟",
            "",
            f"- Document ID: {result.get('document_id', '')}",
            f"- Document Type: {result.get('document_type', '')}",
            "",
            "## 요약",
            "",
            summary_text or "(요약 없음)",
            "",
            "## 핵심 포인트",
            "",
        ]
        if key_points:
            lines.extend([f"- {item}" for item in key_points])
        else:
            lines.append("- 핵심 포인트 없음")
        return "\n".join(lines) + "\n"

    def _make_download_filename(self, source_label: str, output_format: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        safe_stem = "".join("_" if ch in invalid_chars else ch for ch in Path(source_label).stem).strip().rstrip(".")
        safe_stem = safe_stem or "summary"
        return f"{safe_stem}_요약.md"

    def _handle_query(self) -> None:
        try:
            payload = self._read_json_payload()
            query = str(payload.get("query", "")).strip()
            strategy = str(payload.get("strategy", DEFAULT_QA_STRATEGY)).strip() or DEFAULT_QA_STRATEGY
            if strategy == "semantic" and not is_semantic_strategy_available():
                strategy = "rule_based"
            document_id = str(payload.get("document_id", "")).strip()
            source_name = str(payload.get("source_name", "")).strip()
            selected_document_ids = [str(item).strip() for item in (payload.get("selected_document_ids") or []) if str(item).strip()]
            if not query:
                self._send_json({"error": "query_is_required"}, status=400)
                return
            answer = self.retriever.answer_question(
                query=query,
                strategy=strategy,
                document_id=document_id,
                source_name=source_name,
                document_ids=selected_document_ids,
            )
            answer = self._decorate_query_payload(answer)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            import traceback
            self._send_json({"error": str(error), "traceback": traceback.format_exc()}, status=500)
            return
        self._send_json(answer, status=200)

    def _handle_query_compare(self) -> None:
        try:
            payload = self._read_json_payload()
            query = str(payload.get("query", "")).strip()
            strategy = str(payload.get("strategy", DEFAULT_QA_STRATEGY)).strip() or DEFAULT_QA_STRATEGY
            document_id = str(payload.get("document_id", "")).strip()
            source_name = str(payload.get("source_name", "")).strip()
            selected_document_ids = [str(item).strip() for item in (payload.get("selected_document_ids") or []) if str(item).strip()]
            if not query:
                self._send_json({"error": "query_is_required"}, status=400)
                return
            try:
                custom_answer = self.retriever.answer_question(
                    query=query,
                    strategy=strategy,
                    document_id=document_id,
                    source_name=source_name,
                    document_ids=selected_document_ids,
                )
            except Exception as error:
                custom_answer = {
                    "answer": "Custom QA ?듬? ?앹꽦???ㅽ뙣?덉뒿?덈떎.",
                    "citations": [],
                    "matches": [],
                    "used_model": None,
                    "warning": str(error),
                }
            if self.langchain_service is None:
                langchain_answer = {
                    "answer": "OPENAI_API_KEY媛 ?놁뼱 LangChain ?쒕퉬?ㅻ? ?ъ슜?????놁뒿?덈떎.",
                    "citations": [],
                    "source_documents": [],
                    "used_model": None,
                    "warning": "langchain_service_unavailable",
                }
                langchain_reranked_answer = {
                    "answer": "OPENAI_API_KEY媛 ?놁뼱 LangChain + reranker ?쒕퉬?ㅻ? ?ъ슜?????놁뒿?덈떎.",
                    "citations": [],
                    "source_documents": [],
                    "used_model": None,
                    "warning": "langchain_service_unavailable",
                }
            else:
                try:
                    langchain_answer = self.langchain_service.answer_question(
                        query=query,
                        strategy=strategy,
                        document_id=document_id,
                        source_name=source_name,
                        document_ids=selected_document_ids,
                        use_reranker=False,
                    )
                    langchain_reranked_answer = self.langchain_service.answer_question(
                        query=query,
                        strategy=strategy,
                        document_id=document_id,
                        source_name=source_name,
                        document_ids=selected_document_ids,
                        use_reranker=True,
                    )
                except Exception as error:
                    langchain_answer = {
                        "answer": "LangChain RetrievalQA ?듬? ?앹꽦???ㅽ뙣?덉뒿?덈떎.",
                        "citations": [],
                        "source_documents": [],
                        "used_model": None,
                        "warning": str(error),
                    }
                    langchain_reranked_answer = {
                        "answer": "LangChain RetrievalQA + reranker ?듬? ?앹꽦???ㅽ뙣?덉뒿?덈떎.",
                        "citations": [],
                        "source_documents": [],
                        "used_model": None,
                        "warning": str(error),
                    }
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)
            return
        self._send_json(
            {
                "custom": custom_answer,
                "langchain": langchain_answer,
                "langchain_reranked": langchain_reranked_answer,
            },
            status=200,
        )

    def _prime_query_summaries(
        self,
        *,
        document_id: str,
        source_name: str,
        selected_document_ids: list[str],
    ) -> None:
        if self.langchain_service is None:
            return

        candidate_ids: list[str] = []
        if document_id:
            candidate_ids.append(document_id)
        for selected_id in selected_document_ids:
            if selected_id and selected_id not in candidate_ids:
                candidate_ids.append(selected_id)

        for target_document_id in candidate_ids[:3]:
            try:
                self.langchain_service.summarize_document(document_id=target_document_id)
            except Exception:
                continue

        if not candidate_ids and source_name:
            try:
                self.langchain_service.summarize_document(source_name=source_name)
            except Exception:
                pass

    def _parse_uploads(self) -> list[UploadedDocument]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "documents" not in form:
            return []

        fields = form["documents"]
        if not isinstance(fields, list):
            fields = [fields]

        uploads: list[UploadedDocument] = []
        for field in fields:
            filename = getattr(field, "filename", "") or ""
            if not filename:
                continue
            uploads.append(UploadedDocument(filename=filename, content=field.file.read()))
        return uploads

    def _send_html(self, content: str, status: int = 200) -> None:
        payload = content.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.error):
            return

    def _read_json_payload(self) -> dict[str, object]:
        content_length_header = (self.headers.get("Content-Length") or "").strip()
        if not content_length_header:
            raise ValueError("request_body_required")
        try:
            content_length = int(content_length_header)
        except ValueError as error:
            raise ValueError("invalid_content_length") from error
        raw_body = self.rfile.read(content_length)
        body_text = raw_body.decode("utf-8-sig").strip()
        if not body_text:
            raise ValueError("empty_request_body")
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid_json: {error.msg}") from error
        if not isinstance(payload, dict):
            raise ValueError("json_object_required")
        return payload

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        safe_payload = make_json_safe(payload)
        try:
            body = json.dumps(safe_payload, ensure_ascii=False, indent=2).encode("utf-8")
        except Exception as error:
            log_server_exception("json_encode", error)
            fallback = json.dumps({"error": f"json_encode_failed: {error}"}, ensure_ascii=False).encode("utf-8")
            body = fallback
            status = 500
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.error):
            return

    def _send_download(self, *, content: str, filename: str, content_type: str) -> None:
        body = content.encode("utf-8")
        ascii_filename = "".join(ch if ord(ch) < 128 and ch not in {'"', "\\"} else "_" for ch in filename) or "download.txt"
        encoded_filename = quote(filename, safe="")
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Disposition",
                f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}",
            )
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.error):
            return

    def _send_binary(self, payload: bytes, *, content_type: str, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.error):
            return

    def _serve_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.error):
            return

    def log_message(self, format: str, *args: object) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    clear_process_proxy_env()
    args = parse_args()
    manager = ReviewSessionManager(project_root=ROOT, markdown_mode=args.markdown_mode, qa_mode=args.qa_mode)
    retriever = ChromaRetriever(project_root=ROOT)
    langchain_service = LazyLangChainService(project_root=ROOT)

    handler = partial(
        DocumentStudioRequestHandler,
        manager=manager,
        retriever=retriever,
        langchain_service=langchain_service,
        openapi_spec=build_openapi_spec(args.host, args.port),
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving Document Studio from: {ROOT}")
    print(f"Open: {url}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopped Document Studio.")
    finally:
        retriever.close()
        manager.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

