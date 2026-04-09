from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentIssue:
    code: str
    message: str
    severity: str = "info"


@dataclass
class DocumentClassification:
    document_type: str
    parser_route: str
    confidence: float
    pdf_kind: str | None = None
    origin_hint: str | None = None
    pdf_parser_strategy: str | None = None
    text_layer_present: bool | None = None
    recommended_action: str | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentElement:
    element_id: str
    element_type: str
    page_number: int
    order: int
    bbox: list[float] | None = None
    text: str | None = None
    markdown: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPage:
    page_number: int
    width: float
    height: float
    text_length: int
    elements: list[DocumentElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    document_id: str
    source_name: str
    source_path: str
    extension: str
    status: str
    parser_name: str
    classification: DocumentClassification
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    pages: list[ParsedPage] = field(default_factory=list)
    markdown: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    issues: list[DocumentIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
