from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from src.shared.runtime_deps import ensure_local_dependency_path


ensure_local_dependency_path()

from langchain_core.embeddings import Embeddings  # type: ignore  # noqa: E402
from langchain_experimental.text_splitter import SemanticChunker  # type: ignore  # noqa: E402


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+", re.UNICODE)
TABLE_ROW_RE = re.compile(r"^\s*\|")
TABLE_META_RE = re.compile(r"^\[표: \d+행 × \d+열\]$")
_TABLE_NEWLINE_PLACEHOLDER = "\x00TBNL\x00"


@dataclass
class ChunkComparison:
    strategy: str
    chunk_count: int
    total_characters: int
    average_characters: float
    min_characters: int
    max_characters: int


class DeterministicTextEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def clamp_chunks_for_embedding(
    chunks: list[dict[str, Any]],
    max_characters: int = 6000,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    normalized_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        if len(text) <= max_characters:
            normalized_chunks.append(
                {
                    **chunk,
                    "chunk_index": len(normalized_chunks),
                    "char_count": len(text),
                }
            )
            continue

        for split_text in _split_large_paragraph(text, max_characters, overlap):
            normalized_chunks.append(
                {
                    **chunk,
                    "chunk_index": len(normalized_chunks),
                    "text": split_text,
                    "char_count": len(split_text),
                }
            )
    return normalized_chunks


def build_rule_based_chunks(markdown: str, chunk_size: int = 900, overlap: int = 150) -> list[dict[str, Any]]:
    paragraphs = _split_markdown_paragraphs(markdown)
    if not paragraphs:
        return []

    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_heading: str | None = None
    current_sections: list[str] = []

    for paragraph in paragraphs:
        heading_match = HEADING_RE.match(paragraph)
        if heading_match:
            current_heading = heading_match.group(2)
            if current_parts:
                chunks.append(_make_rule_chunk(chunks, current_parts, current_sections))
                current_parts = []
                current_sections = []
            current_parts.append(paragraph)
            current_sections = [current_heading]
            continue

        candidate_parts = [*current_parts, paragraph] if current_parts else [paragraph]
        candidate_text = "\n\n".join(candidate_parts)
        if len(candidate_text) <= chunk_size:
            current_parts = candidate_parts
            if current_heading and current_heading not in current_sections:
                current_sections.append(current_heading)
            continue

        if current_parts:
            chunks.append(_make_rule_chunk(chunks, current_parts, current_sections))
            if len(paragraph) <= chunk_size:
                overlap_text = chunks[-1]["text"][-overlap:].strip() if overlap > 0 else ""
                current_parts = [part for part in [overlap_text, paragraph] if part]
                current_sections = [current_heading] if current_heading else []
            else:
                current_parts = []
                current_sections = []
                for split_part in _split_large_paragraph(paragraph, chunk_size, overlap):
                    chunks.append(
                        {
                            "strategy": "rule_based",
                            "chunk_index": len(chunks),
                            "text": split_part,
                            "char_count": len(split_part),
                            "section_hint": current_heading,
                        }
                    )
            continue

        for split_part in _split_large_paragraph(paragraph, chunk_size, overlap):
            chunks.append(
                {
                    "strategy": "rule_based",
                    "chunk_index": len(chunks),
                    "text": split_part,
                    "char_count": len(split_part),
                    "section_hint": current_heading,
                }
            )

    if current_parts:
        chunks.append(_make_rule_chunk(chunks, current_parts, current_sections))

    return clamp_chunks_for_embedding(chunks, max_characters=chunk_size * 2, overlap=overlap)


def build_semantic_chunks(markdown: str, embeddings: Embeddings | None = None) -> list[dict[str, Any]]:
    clean_text = markdown.strip()
    if not clean_text:
        return []

    active_embeddings = embeddings or DeterministicTextEmbeddings()
    if len(clean_text) < 280:
        return [
            {
                "strategy": "semantic",
                "chunk_index": 0,
                "text": clean_text,
                "char_count": len(clean_text),
                "section_hint": _find_first_heading(clean_text),
            }
        ]

    protected_text = _protect_table_newlines(clean_text)
    splitter = SemanticChunker(
        active_embeddings,
        sentence_split_regex=r"(?<=[.!?])\s+|\n+",
    )
    try:
        raw_chunks = [_restore_table_newlines(chunk).strip() for chunk in splitter.split_text(protected_text) if chunk.strip()]
    except Exception:
        raw_chunks = _fallback_semantic_chunks(clean_text)
    if not raw_chunks:
        raw_chunks = _fallback_semantic_chunks(clean_text)

    semantic_chunks = [
        {
            "strategy": "semantic",
            "chunk_index": index,
            "text": chunk,
            "char_count": len(chunk),
            "section_hint": _find_first_heading(chunk),
        }
        for index, chunk in enumerate(raw_chunks)
    ]
    return clamp_chunks_for_embedding(semantic_chunks, max_characters=6000, overlap=180)


def compare_chunk_strategies(document_id: str, rule_chunks: list[dict[str, Any]], semantic_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    rule_metrics = _build_metrics("rule_based", rule_chunks)
    semantic_metrics = _build_metrics("semantic", semantic_chunks)
    return {
        "document_id": document_id,
        "rule_based": rule_metrics.__dict__,
        "semantic": semantic_metrics.__dict__,
        "difference": {
            "chunk_count_delta": semantic_metrics.chunk_count - rule_metrics.chunk_count,
            "average_characters_delta": round(semantic_metrics.average_characters - rule_metrics.average_characters, 2),
        },
    }


def _build_metrics(strategy: str, chunks: list[dict[str, Any]]) -> ChunkComparison:
    lengths = [chunk["char_count"] for chunk in chunks] or [0]
    total = sum(lengths)
    return ChunkComparison(
        strategy=strategy,
        chunk_count=len(chunks),
        total_characters=total,
        average_characters=round(total / len(chunks), 2) if chunks else 0.0,
        min_characters=min(lengths),
        max_characters=max(lengths),
    )


def _split_markdown_paragraphs(markdown: str) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append("\n".join(buffer).strip())
    return [paragraph for paragraph in paragraphs if paragraph]


def _make_rule_chunk(chunks: list[dict[str, Any]], parts: list[str], sections: list[str]) -> dict[str, Any]:
    text = "\n\n".join(part for part in parts if part).strip()
    return {
        "strategy": "rule_based",
        "chunk_index": len(chunks),
        "text": text,
        "char_count": len(text),
        "section_hint": sections[-1] if sections else None,
    }


def _is_table_line(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line) or TABLE_META_RE.match(line.strip()))


def _is_table_block(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return len(lines) >= 2 and all(_is_table_line(line) for line in lines)


def _protect_table_newlines(text: str) -> str:
    """표 블록 내부의 \n을 placeholder로 치환해 청커가 표 행을 분리하지 못하게 함."""
    result: list[str] = []
    in_table = False
    for line in text.splitlines():
        if _is_table_line(line):
            if in_table:
                result.append(_TABLE_NEWLINE_PLACEHOLDER + line)
            else:
                in_table = True
                result.append(line)
        else:
            in_table = False
            result.append(line)
    return "\n".join(result)


def _restore_table_newlines(text: str) -> str:
    return text.replace(_TABLE_NEWLINE_PLACEHOLDER, "\n")


def _split_large_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    if _is_table_block(paragraph):
        return [paragraph]

    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY_RE.split(paragraph) if sentence.strip()]
    if not sentences:
        return [paragraph[:chunk_size]]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            carry = current[-overlap:].strip() if overlap > 0 else ""
            current = f"{carry} {sentence}".strip() if carry else sentence
            continue

        chunks.extend(_hard_split_text(sentence, chunk_size, overlap))
        current = ""

    if current:
        chunks.append(current.strip())
    return chunks


def _hard_split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    slices: list[str] = []
    if chunk_size <= 0:
        return [text]
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(text):
        part = text[start : start + chunk_size].strip()
        if part:
            slices.append(part)
        start += step
    return slices or [text[:chunk_size]]


def _find_first_heading(text: str) -> str | None:
    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            return match.group(2)
    return None


def _fallback_semantic_chunks(text: str) -> list[str]:
    paragraphs = _split_markdown_paragraphs(text)
    if not paragraphs:
        return _hard_split_text(text, 3000, 180)

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= 3000:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= 3000:
            current = paragraph
            continue
        chunks.extend(_split_large_paragraph(paragraph, 3000, 180))
    if current:
        chunks.append(current)
    return chunks or _hard_split_text(text, 3000, 180)
