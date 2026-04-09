# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import cgi
import json
import math
import mimetypes
import shutil
import socket
import sys
import threading
import traceback
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
SERVER_ERROR_LOG = ROOT / "outputs" / "document_studio_server_errors.log"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.chroma_retriever import ChromaRetriever  # noqa: E402
from src.shared.constants import SUPPORTED_EXTENSIONS  # noqa: E402
from src.ui.review_server import ReviewSessionManager, UploadedDocument  # noqa: E402


DEFAULT_QA_STRATEGY = "semantic"


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
                            "schema": {"type": "string", "enum": ["md", "txt"], "default": "md"},
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


def render_document_studio_html(status: dict[str, object]) -> str:
    project_review = status.get("project_review") or {}
    latest_upload = status.get("latest_upload") or {}
    openai_status = status.get("openai") or {}

    qa_caption = (
        (
            "업로드 시 semantic 청킹, 임베딩 인덱싱, 문서 요약을 먼저 준비한 뒤 "
            f"LangChain QA + Cross-Encoder reranker로 바로 질문할 수 있습니다. 현재 모델: {openai_status.get('model') or 'local fallback'}"
        )
        if openai_status.get("enabled")
        else "OpenAI 연결이 없어 문서 요약과 LangChain QA를 준비하지 못했습니다. 현재는 로컬 검색 기반 응답을 대신 사용합니다."
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

    /* 문서 선택 섹션: 데스크탑에서 남은 공간 채우기 */
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
      content: "✦";
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
      content: "•";
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

    /* 세로 공간이 짧을 때 드롭존 축소 */
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

      /* 모바일: flex 유지하되 overflow visible로 전체 표시 */
      .sidebar-section-docs {{
        overflow: visible;
      }}
      .sidebar-section-docs .doc-list {{
        flex: none;
        max-height: none;
      }}

      /* 모바일: 내부 스크롤 제거 → 목록 전체 표시, 페이지 스크롤로 탐색 */
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
            <h3>문서 업로드</h3>
            <span class="sidebar-list-count">Quick Add</span>
          </div>
          <form class="sidebar-upload" id="workspace-upload-form">
            <input id="workspace-documents" name="documents" type="file" multiple accept="{accept}">
            <div id="upload-dropzone" class="upload-dropzone">
              <span class="upload-dropzone-icon">&#8659;</span>
              <span>파일을 여기에 드래그하여 업로드</span>
              <span style="font-size:11px; opacity:0.7;">또는 아래 버튼으로 선택</span>
            </div>
            <div class="sidebar-upload-actions">
              <button class="circle-button" id="workspace-pick-button" type="button">+</button>
              <button class="send-button" id="workspace-upload-button" type="submit">업로드</button>
              <div class="upload-spinner" id="upload-spinner"></div>
            </div>
            <div class="sidebar-selected-files" id="workspace-selected-files">
              <span class="file-chip">선택된 문서가 없습니다.</span>
            </div>
            <div id="status-box" style="display:none; font-size:12px; color:var(--muted); line-height:1.6; padding:6px 4px; white-space:pre-wrap;"></div>
          </form>
        </section>

        <section class="sidebar-section sidebar-section-docs">
          <div class="sidebar-list-head">
            <h3>문서 선택</h3>
            <span class="sidebar-list-count" id="document-count-chip">0 docs</span>
          </div>

          <div class="selection-status" id="checked-summary">
            <span>QA</span>
            <strong>0 selected</strong>
          </div>

          <div class="select-all-row" id="select-all-row" style="display:none;">
            <span class="select-all-text">전체 선택</span>
            <span class="doc-check" id="select-all-check"></span>
            <button id="delete-selected-button" type="button" style="margin-left:auto;padding:5px 14px;border:none;border-radius:999px;background:#dc2626;color:#fff;font:inherit;font-size:13px;font-weight:700;cursor:pointer;opacity:0.4;pointer-events:none;">삭제</button>
          </div>

          <div class="doc-list sidebar-doc-list" id="document-list">
            <div class="status-box">문서 목록을 불러오는 중입니다.</div>
          </div>
        </section>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div class="page-title">
          <h1>문서 보기</h1>
        </div>
      </div>

      

      <section class="workspace" id="workspace-screen">
        <section class="content-pane">
          <section class="panel summary-box">
            <div class="summary-header">
              <div>
                <h2>요약</h2>
              </div>
              <div class="toolbar">
                <button class="ghost-button" id="download-md-button" type="button">다운로드 .md</button>
                <button class="ghost-button" id="download-txt-button" type="button">다운로드 .txt</button>
                <button class="summary-toggle" id="summary-toggle" type="button" title="접기 / 펼치기">▲</button>
              </div>
            </div>
            <div id="summary-content" style="margin-top:16px;">
              <p class="summary-text" style="color: var(--muted);">문서를 선택하면 자동으로 요약을 불러옵니다.</p>
            </div>
          </section>

          <section class="panel qa-box">
            <h2>질문</h2>
            <div class="chat-shell">
              <div class="chat-history" id="chat-history">
                <div class="chat-empty">문서를 선택한 뒤 질문을 입력하세요.</div>
              </div>
            </div>
            <form class="chat-composer" id="query-form">
              <textarea class="question-area" id="query-input" placeholder="예: 이 문서의 핵심 조건을 세 줄로 정리해줘"></textarea>
              <div class="chat-toolbar">
                <button class="pill-button" id="query-button" type="submit">질문하기</button>
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
    const downloadTxtButton = document.getElementById('download-txt-button');
    const summaryToggle = document.getElementById('summary-toggle');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const queryButton = document.getElementById('query-button');
    const chatHistoryEl = document.getElementById('chat-history');

    sidebarWorkspace.classList.add('active');
    workspaceScreen.classList.add('active');

    // 소스 가이드 접기/펼치기 토글
    summaryToggle.addEventListener('click', () => {{
      state.guideCollapsed = !state.guideCollapsed;
      summaryContent.classList.toggle('guide-collapsed', state.guideCollapsed);
      summaryContent.closest('.summary-box').classList.toggle('guide-collapsed', state.guideCollapsed);
      summaryToggle.textContent = state.guideCollapsed ? '▼' : '▲';
      // 접으면 현재 선택 문서의 체크 해제 (자유 선택 복귀)
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
        targetEl.innerHTML = '<span class="file-chip">선택된 문서가 없습니다.</span>';
        return;
      }}
      targetEl.innerHTML = files
        .map((file) => `<span class="file-chip">${{esc(file.name)}} · ${{Math.max(1, Math.round(file.size / 1024))}}KB</span>`)
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
        selectAllCheck.textContent = '✓';
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
      if (!confirm(`선택한 문서 ${{ids.length}}개를 벡터 DB에서 삭제합니다. 계속하시겠습니까?`)) return;
      deleteSelectedButton.textContent = '삭제 중...';
      deleteSelectedButton.style.pointerEvents = 'none';
      try {{
        const response = await fetch('/api/delete-documents', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ document_ids: ids }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '삭제에 실패했습니다.');
        state.checkedDocumentIds = [];
        state.selectedDocumentId = '';
        await loadDocuments();
      }} catch (error) {{
        alert(String(error));
      }} finally {{
        deleteSelectedButton.textContent = '삭제';
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
        documentList.innerHTML = '<div class="status-box">표시할 문서가 없습니다.</div>';
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
          <span class="doc-check${{checked ? ' checked' : ''}}" data-check-id="${{esc(doc.document_id)}}">${{checked ? '✓' : ''}}</span>
        </button>`;
      }}).join('');

      documentList.querySelectorAll('[data-id]').forEach((button) => {{
        button.addEventListener('click', () => {{
          const docId = button.getAttribute('data-id') || '';
          state.selectedDocumentId = docId;
          // 소스 선택 시 항상 체크박스 선택
          if (docId && !isCheckedDocument(docId)) {{
            state.checkedDocumentIds = dedupeDocumentIds([...state.checkedDocumentIds, docId]);
          }}
          // 소스 가이드 펼치기
          if (state.guideCollapsed) {{
            state.guideCollapsed = false;
            summaryContent.classList.remove('guide-collapsed');
            summaryToggle.textContent = '▲';
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
        summaryContent.innerHTML = '<p class="summary-text" style="color: var(--muted);">문서를 선택하면 요약을 확인할 수 있습니다.</p>';
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
        <p class="summary-text" style="color:var(--muted);">요약을 불러오는 중입니다...</p>`;

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
              : '<div class="point">별도 핵심 포인트가 없습니다.</div>'
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
      const isError = Boolean(result.error);
      return `<div class="chat-answer-card">
        <div class="chat-answer-head">
          <h3>QA 寃곌낵</h3>
          ${{result.used_model ? `<span class="answer-badge">${{esc(result.used_model)}}</span>` : ''}}
        </div>
        <p class="chat-answer-text">${{esc(result.answer || '?듬????놁뒿?덈떎.')}}</p>
        ${{
          isError
            ? `<div class="evidence-list"><div class="evidence-item">QA ?붿껌 泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎. ${{
                esc(String(result.error || 'unknown_error'))
              }}</div></div>`
            : ''
        }}
      </div>`;

      const evidenceHtml = isError
        ? `<div class="evidence-item">QA 요청 처리 중 오류가 발생했습니다. ${{
            esc(String(result.error || 'unknown_error'))
          }}</div>`
        : citations.length
        ? citations.map((c) =>
            `<div class="evidence-item">
              <strong>${{esc(c.source_name || 'document')}}</strong><br>
              <span style="color:var(--muted);">${{esc(c.section_hint || 'section')}}</span>
              <div style="margin-top:8px;">${{esc(c.quote || '')}}</div>
            </div>`
          ).join('')
        : matches.slice(0, 3).map((m) => {{
            const md = m.metadata || {{}};
            const ex = m.document || m.page_content || '';
            return `<div class="evidence-item">
              <strong>${{esc(md.source_name || md.document_id || 'document')}}</strong><br>
              <span style="color:var(--muted);">${{esc(md.section_hint || 'section')}}</span>
              <div style="margin-top:8px;">${{esc(String(ex).slice(0, 220))}}</div>
            </div>`;
          }}).join('') || '<div class="evidence-item">근거가 없습니다.</div>';

      return `<div class="chat-answer-card">
        <div class="chat-answer-head">
          <h3>QA 결과</h3>
          ${{result.used_model ? `<span class="answer-badge">${{esc(result.used_model)}}</span>` : ''}}
        </div>
        <p class="chat-answer-text">${{esc(result.answer || '답변이 없습니다.')}}</p>
        <div class="evidence-list">${{evidenceHtml}}</div>
      </div>`;
    }}

    function renderAnswerBubble(result) {{
      const isError = Boolean(result.error);
      return `<div class="chat-answer-card">
        <div class="chat-answer-head">
          <h3>QA 결과</h3>
          ${{result.used_model ? `<span class="answer-badge">${{esc(result.used_model)}}</span>` : ''}}
        </div>
        <p class="chat-answer-text">${{esc(result.answer || '답변이 없습니다.')}}</p>
        ${{
          isError
            ? `<div class="evidence-list"><div class="evidence-item">QA 요청 처리 중 오류가 발생했습니다. ${{
                esc(String(result.error || 'unknown_error'))
              }}</div></div>`
            : ''
        }}
      </div>`;
    }}

    function renderChat(scrollToBottom = false) {{
      const history = state.chatHistory[state.selectedDocumentId] || [];
      if (!history.length) {{
        chatHistoryEl.innerHTML = '<div class="chat-empty">문서를 선택한 뒤 질문을 입력하세요.</div>';
        return;
      }}

      const turns = history.map((entry) => {{
        const aiHtml = entry.loading
          ? '<div class="chat-loading">답변을 찾는 중입니다...</div>'
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
      // skipSummary=true일 때는 폴링 중 미완성 인덱스로 요약하지 않음
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
          ? Math.min(progressCurrent, progressTotal) + ' / ' + progressTotal + '개 완료'
          : (payload.status === 'running' ? ('처리 중' + dots) : '');
        setStatus([
          payload.message || '',
          progressText
        ].filter(Boolean));

        // 진행 상황이 바뀌거나, 실행 중일 때 3틱마다 주기적으로 목록 갱신
        // skipSummary:true — 인덱싱 완료 전에 빈 요약이 캐시에 저장되는 것을 방지
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
            // 목록 갱신 실패 시 무시하고 폴링 계속
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

          // 파일 레벨 중복 (duplicate_uploads): existing_document_id 포함
          const duplicateUploads = result.duplicate_uploads || [];
          // 컨텐츠 레벨 중복 (content_duplicate_uploads): document_id 포함
          const contentDuplicates = result.content_duplicate_uploads || [];
          // 새로 indexed된 문서들
          const indexedDocs = (vectorSummary.documents || []).filter((item) => item.status === 'indexed');

          const newCount = indexedDocs.length;
          const dupCount = duplicateUploads.length + contentDuplicates.length;

          // 중복 문서의 기존 document_id 수집
          const dupDocIds = [
            ...duplicateUploads.map((d) => d.existing_document_id).filter(Boolean),
            ...contentDuplicates.map((d) => d.document_id).filter(Boolean),
          ];
          // 신규 문서 ID 수집
          const newDocIds = indexedDocs.map((d) => d.document_id).filter(Boolean);
          // 전체 배치 ID (중복 우선, 신규 추가, 중복 제거)
          const batchDocIds = [...new Set([...dupDocIds, ...newDocIds])];
          // 자동 선택: 중복이 있으면 첫 중복, 없으면 첫 신규
          const preferredDocId = batchDocIds[0] || '';

          const statusLines = [
            'Run ID: ' + (result.run_id || ''),
            '신규 문서: ' + newCount + '개  /  중복 문서: ' + dupCount + '개',
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
              const rest = filedupNames.length > MAX_SHOW ? ` 외 ${{filedupNames.length - MAX_SHOW}}개` : '';
              statusLines.push('파일 중복 (' + filedupNames.length + '개): ' + shown + rest);
            }}
            if (contentdupNames.length) {{
              const MAX_SHOW = 3;
              const shown = contentdupNames.slice(0, MAX_SHOW).join(', ');
              const rest = contentdupNames.length > MAX_SHOW ? ` 외 ${{contentdupNames.length - MAX_SHOW}}개` : '';
              statusLines.push('내용 중복 (' + contentdupNames.length + '개): ' + shown + rest);
            }}
          }}
          setStatus(statusLines);

          state.latestRun = result;

          // 첫 번째 문서(중복 우선) 자동 선택
          await loadDocuments(preferredDocId);

          // 이번 배치에 포함된 모든 문서(신규+중복) 체크박스 선택
          if (batchDocIds.length) {{
            const availableIds = new Set(state.documents.map((d) => d.document_id));
            const validBatchIds = batchDocIds.filter((id) => availableIds.has(id));
            if (validBatchIds.length) {{
              state.checkedDocumentIds = dedupeDocumentIds(validBatchIds);
              renderDocumentList();
              renderCheckedSummary();
            }}
          }}

          // 신규 문서 전체 요약: 폴링 중 빈 캐시가 저장됐을 수 있으므로
          // newDocIds 캐시를 초기화한 뒤 재요약
          newDocIds.forEach((id) => {{ delete state.summaryCache[id]; }});
          if (newDocIds.length > 0) {{
            // 선택된 문서가 신규 목록에 있으면 먼저 처리
            const selectedFirst = newDocIds.includes(state.selectedDocumentId)
              ? [state.selectedDocumentId, ...newDocIds.filter((id) => id !== state.selectedDocumentId)]
              : newDocIds;
            // UI에 "요약 중..." 표시 후 즉시 요약 시작
            renderDocumentPlaceholder();
            await autoSummarizeDocument(selectedFirst[0]);
            for (const docId of selectedFirst.slice(1)) {{
              autoSummarizeDocument(docId); // fire-and-forget
            }}
          }} else {{
            // 신규 문서 없음(중복만 있는 경우) — 선택 문서 요약은 renderDocumentPlaceholder가 처리
            renderDocumentPlaceholder();
          }}

          renderChat();
          return;
        }}

        if (payload.status === 'failed') {{
          resetUploadUiState();
          setStatus(payload.error || payload.message || '작업이 실패했습니다.');
          return;
        }}

        await new Promise((resolve) => setTimeout(resolve, 1200));
      }}
    }}

    async function summarizeSelectedDocument() {{
      const selected = getSelectedDocument();
      if (!selected) {{
        summaryContent.innerHTML = '<p class="summary-text" style="color: var(--muted);">먼저 문서를 선택하세요.</p>';
        return;
      }}

      const targetDocumentId = selected.document_id;
      downloadMdButton.disabled = true;
      downloadTxtButton.disabled = true;

      if (state.selectedDocumentId === targetDocumentId) {{
        summaryContent.innerHTML = '<p class="summary-text" style="color: var(--muted);">요약을 생성하는 중입니다...</p>';
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
        if (!response.ok) throw new Error(payload.error || '요약 생성에 실패했습니다.');
        renderSummary(payload, targetDocumentId);
      }} catch (error) {{
        if (state.selectedDocumentId === targetDocumentId) {{
          summaryContent.innerHTML = `<p class="summary-text" style="color: var(--muted);">${{esc(String(error))}}</p>`;
        }}
      }} finally {{
        downloadMdButton.disabled = false;
        downloadTxtButton.disabled = false;
      }}
    }}

    function downloadSummary(format) {{
      const selected = getSelectedDocument();
      if (!selected) {{
        setStatus('먼저 다운로드할 문서를 선택해 주세요.');
        return;
      }}
      window.location.href =
        `/api/download-summary?document_id=${{encodeURIComponent(selected.document_id)}}&source_name=${{encodeURIComponent(selected.source_name || '')}}&format=${{encodeURIComponent(format)}}`;
    }}

    async function submitUploads(files, buttonEl) {{
      if (!files.length) {{
        setStatus('업로드할 문서를 먼저 선택하세요.');
        return;
      }}

      buttonEl.disabled = true;
      uploadSpinner.classList.add('active');
      setStatus('업로드를 시작했습니다. 문서 처리 파이프라인을 준비하고 있습니다.');

      const formData = new FormData();
      files.forEach((file) => formData.append('documents', file, file.name));

      try {{
        const response = await fetch('/api/run', {{ method: 'POST', body: formData }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || '파이프라인 실행에 실패했습니다.');
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

    // 드래그앤드롭 업로드
    const uploadDropzone = document.getElementById('upload-dropzone');

    uploadDropzone.addEventListener('click', () => workspaceFileInput.click());

    uploadDropzone.addEventListener('dragenter', (event) => {{
      event.preventDefault();
      uploadDropzone.classList.add('drag-over');
      uploadDropzone.querySelector('span:nth-child(2)').textContent = '여기에 놓으세요!';
    }});

    uploadDropzone.addEventListener('dragover', (event) => {{
      event.preventDefault();
    }});

    uploadDropzone.addEventListener('dragleave', (event) => {{
      if (!uploadDropzone.contains(event.relatedTarget)) {{
        uploadDropzone.classList.remove('drag-over');
        uploadDropzone.querySelector('span:nth-child(2)').textContent = '파일을 여기에 드래그하여 업로드';
      }}
    }});

    uploadDropzone.addEventListener('drop', (event) => {{
      event.preventDefault();
      uploadDropzone.classList.remove('drag-over');
      uploadDropzone.querySelector('span:nth-child(2)').textContent = '파일을 여기에 드래그하여 업로드';
      const files = Array.from(event.dataTransfer.files || []);
      if (files.length) {{
        renderSelectedFiles(files, workspaceSelectedFiles);
        submitUploads(files, workspaceUploadButton);
      }}
    }});

    // 페이지 전체에서 실수로 파일을 드롭하는 것 방지
    document.addEventListener('dragover', (event) => event.preventDefault());
    document.addEventListener('drop', (event) => event.preventDefault());

    downloadMdButton.addEventListener('click', () => downloadSummary('md'));
    downloadTxtButton.addEventListener('click', () => downloadSummary('txt'));

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
        chatHistoryEl.innerHTML = '<div class="chat-empty">먼저 문서를 선택하세요.</div>';
        return;
      }}

      if (!scopedDocumentIds.length) {{
        chatHistoryEl.innerHTML = '<div class="chat-empty">검색할 문서를 체크하세요.</div>';
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
        // 여러 문서가 선택된 경우 document_id를 비워서 백엔드 source_boost가
        // 활성 문서 하나에만 집중되지 않도록 함
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
        if (!response.ok) throw new Error(payload.error || '질의응답에 실패했습니다.');
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
        documentList.innerHTML = '<div class="status-box">문서 목록을 불러오지 못했습니다.</div>';
      }});

    // 벡터 DB 인덱싱 완료 시 서버 이벤트로 문서 목록 자동 갱신
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
                vector_result = self.manager.index_manager.delete_document(doc_id)
                files_removed = self.manager.delete_document_files(doc_id)
                results.append({**vector_result, "files_removed": files_removed})
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)
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
            self._send_json({"error": str(error)}, status=500)
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
                basic_result = self._build_basic_summary_payload(
                    document_payload,
                    document_id=document_id,
                    source_name=source_name,
                )
                if basic_result:
                    self._send_json(basic_result, status=200)
                    return
                self._send_json({"error": "langchain_service_unavailable"}, status=503)
                return
            result = self.langchain_service.summarize_document(document_id=document_id, source_name=source_name)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)
            return
        self._send_json(result, status=200)

    def _handle_download_summary(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            document_id = str((query.get("document_id") or [""])[0]).strip()
            source_name = str((query.get("source_name") or [""])[0]).strip()
            output_format = str((query.get("format") or ["md"])[0]).strip().lower() or "md"
            if output_format not in {"md", "txt"}:
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
            if output_format == "md":
                content = self._build_summary_markdown(source_label, result, summary_text, key_points)
                content_type = "text/markdown; charset=utf-8"
            else:
                content = self._build_summary_text(source_label, result, summary_text, key_points)
                content_type = "text/plain; charset=utf-8"
            filename = self._make_download_filename(source_label, output_format)
            self._send_download(content=content, filename=filename, content_type=content_type)
        except Exception as error:
            log_server_exception(f"GET {self.path}", error)
            self._send_json({"error": str(error)}, status=500)

    def _build_cached_summary_payload(
        self,
        payload: dict[str, object],
        *,
        default_document_id: str = "",
        default_source_name: str = "",
    ) -> dict[str, object] | None:
        ui_summary = payload.get("ui_summary")
        if isinstance(ui_summary, dict) and ui_summary:
            return {
                "document_id": payload.get("document_id", default_document_id),
                "source_name": payload.get("source_name", default_source_name),
                "summary_text": ui_summary.get("summary_text", ""),
                "key_points": ui_summary.get("key_points") or ui_summary.get("highlights") or [],
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
        if isinstance(llm_summary, dict) and llm_summary:
            return {
                "document_id": payload.get("document_id", default_document_id),
                "source_name": payload.get("source_name", default_source_name),
                "summary_text": llm_summary.get("summary_text", ""),
                "key_points": llm_summary.get("key_points") or llm_summary.get("highlights") or [],
                "document_type": (
                    llm_summary.get("document_type")
                    or (payload.get("classification") or {}).get("document_type", "")
                    or payload.get("extension", "")
                ),
                "used_model": llm_summary.get("used_model") or "cached_llm_summary",
                "framework": llm_summary.get("framework") or "pipeline_llm_summary",
                "summary_source": "pipeline_llm_summary",
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
        basic_summary = payload.get("basic_summary")
        if not isinstance(basic_summary, dict) or not basic_summary:
            return None
        return {
            "document_id": payload.get("document_id", document_id),
            "source_name": payload.get("source_name", source_name),
            "summary_text": basic_summary.get("summary_text", ""),
            "key_points": basic_summary.get("key_points") or basic_summary.get("highlights") or [],
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

    def _build_summary_markdown(
        self,
        source_label: str,
        result: dict[str, object],
        summary_text: str,
        key_points: list[str],
    ) -> str:
        lines = [
            f"# {source_label} 요약",
            "",
            f"- Document ID: {result.get('document_id', '')}",
            f"- Document Type: {result.get('document_type', '')}",
            f"- Model: {result.get('used_model', '')}",
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

    def _build_summary_text(
        self,
        source_label: str,
        result: dict[str, object],
        summary_text: str,
        key_points: list[str],
    ) -> str:
        lines = [
            f"{source_label} 요약",
            "=" * max(8, len(source_label) + 3),
            f"Document ID: {result.get('document_id', '')}",
            f"Document Type: {result.get('document_type', '')}",
            f"Model: {result.get('used_model', '')}",
            "",
            "[요약]",
            summary_text or "(요약 없음)",
            "",
            "[핵심 포인트]",
        ]
        if key_points:
            lines.extend([f"- {item}" for item in key_points])
        else:
            lines.append("- 핵심 포인트 없음")
        return "\n".join(lines) + "\n"

    def _make_download_filename(self, source_label: str, output_format: str) -> str:
        safe_stem = "".join(
            ch if (ch.isascii() and (ch.isalnum() or ch in {"-", "_", " "})) else "_"
            for ch in Path(source_label).stem
        ).strip()
        safe_stem = "_".join(safe_stem.split()) or "summary"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_stem}_summary_{stamp}.{output_format}"

    def _handle_query(self) -> None:
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
            if self.langchain_service is not None:
                self._prime_query_summaries(
                    document_id=document_id,
                    source_name=source_name,
                    selected_document_ids=selected_document_ids,
                )
                answer = self.langchain_service.answer_question(
                    query=query,
                    strategy=strategy,
                    document_id=document_id,
                    source_name=source_name,
                    document_ids=selected_document_ids,
                    use_reranker=True,
                )
            else:
                answer = self.retriever.answer_question(
                    query=query,
                    strategy=strategy,
                    document_id=document_id,
                    source_name=source_name,
                    document_ids=selected_document_ids,
                )
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
            return
        except Exception as error:
            self._send_json({"error": str(error)}, status=500)
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
                    "answer": "Custom QA 답변 생성에 실패했습니다.",
                    "citations": [],
                    "matches": [],
                    "used_model": None,
                    "warning": str(error),
                }
            if self.langchain_service is None:
                langchain_answer = {
                    "answer": "OPENAI_API_KEY가 없어 LangChain 서비스를 사용할 수 없습니다.",
                    "citations": [],
                    "source_documents": [],
                    "used_model": None,
                    "warning": "langchain_service_unavailable",
                }
                langchain_reranked_answer = {
                    "answer": "OPENAI_API_KEY가 없어 LangChain + reranker 서비스를 사용할 수 없습니다.",
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
                        "answer": "LangChain RetrievalQA 답변 생성에 실패했습니다.",
                        "citations": [],
                        "source_documents": [],
                        "used_model": None,
                        "warning": str(error),
                    }
                    langchain_reranked_answer = {
                        "answer": "LangChain RetrievalQA + reranker 답변 생성에 실패했습니다.",
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
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)
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
    args = parse_args()
    manager = ReviewSessionManager(project_root=ROOT)
    retriever = ChromaRetriever(project_root=ROOT)
    langchain_service = LazyLangChainService(project_root=ROOT)
    langchain_service.start_background_warmup(preload_summaries=True)

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
    print("LangChain warmup started in background.")

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
