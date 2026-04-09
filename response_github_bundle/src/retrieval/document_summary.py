from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.indexing.chunking import build_semantic_chunks
from src.retrieval.openai_answerer import load_openai_settings
from src.shared.io import read_text_with_fallback
from src.shared.retry import call_with_retry

from urllib import error, request as urllib_request


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_CHUNK_SUMMARIZE_PROMPT = """\
당신은 한국어 문서 분석 전문가입니다.
아래 문서 일부의 핵심 내용을 추출하세요.

JSON 형식으로만 답변하세요:
{{"summary_text": "2-3문장 핵심 요약", "key_points": ["포인트 1", "포인트 2"]}}

규칙:
- 문서에 실제로 있는 내용만 반영하세요.
- 수치, 날짜, 기관명은 문서에 있을 때만 적으세요.
- JSON 외 다른 설명은 쓰지 마세요.

문서 일부:
{content}
"""

_REDUCE_PROMPT = """\
당신은 한국어 문서 분석 전문가입니다.
아래는 한 문서의 각 부분별 요약입니다. 전체 문서를 종합하여 최종 요약을 작성하세요.

JSON 형식으로만 답변하세요:
{{"summary_text": "3-5문장 전체 요약", "key_points": ["핵심 포인트 1", "핵심 포인트 2", "핵심 포인트 3"], "document_type": "문서 유형"}}

규칙:
- 각 부분의 내용을 종합하되 중복은 제거하세요.
- JSON 외 다른 설명은 쓰지 마세요.

각 부분 요약:
{summaries}
"""

# 이 글자 수 이하면 청킹 없이 전체를 한 번에 LLM에 전달
_STUFF_THRESHOLD = 12_000

def ensure_basic_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("basic_summary"):
        return payload
    payload["basic_summary"] = build_basic_summary(payload)
    return payload


def ensure_llm_summary(payload: dict[str, Any], *, source_path: Path | None = None) -> dict[str, Any]:
    """LLM 기반 요약을 payload에 추가한다.

    source_path(structured JSON 경로)를 캐시로 사용한다.
    해당 파일에 이미 llm_summary가 있으면 LLM 호출 없이 로드만 한다.
    """
    # 디스크 캐시 확인 — 이미 요약된 문서는 스킵
    if source_path and source_path.exists():
        try:
            cached = json.loads(source_path.read_text(encoding="utf-8"))
            if cached.get("llm_summary"):
                payload["llm_summary"] = cached["llm_summary"]
                return payload
        except Exception:
            pass

    if payload.get("llm_summary"):
        return payload

    settings = load_openai_settings()
    if not settings.enabled:
        return payload

    markdown = str(payload.get("markdown") or "")
    if not markdown.strip():
        return payload

    semantic_chunks = extract_semantic_summary_chunks(payload)
    if semantic_chunks and not payload.get("semantic_chunks"):
        payload["semantic_chunks"] = semantic_chunks

    payload["llm_summary"] = _build_llm_summary(
        markdown,
        settings,
        semantic_chunks=semantic_chunks,
    )

    # structured JSON 파일에 llm_summary 추가 저장 (캐시)
    if source_path and source_path.exists():
        try:
            cached = json.loads(source_path.read_text(encoding="utf-8"))
            cached["llm_summary"] = payload["llm_summary"]
            source_path.write_text(
                json.dumps(cached, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    return payload


def build_basic_summary(payload: dict[str, Any]) -> dict[str, Any]:
    source_name = str(payload.get("source_name") or payload.get("document_id") or "document")
    markdown = str(payload.get("markdown") or "")
    sections = payload.get("sections") or []
    section_titles = [str(section.get("title")) for section in sections if section.get("title")][:5]
    sentences = _extract_candidate_sentences(markdown)
    highlights = sentences[:3]

    summary_parts = [source_name]
    if section_titles:
        summary_parts.append("주요 섹션: " + ", ".join(section_titles[:3]))
    if highlights:
        summary_parts.append("핵심 내용: " + " ".join(highlights[:2]))

    return {
        "source_name": source_name,
        "section_titles": section_titles,
        "highlights": highlights,
        "summary_text": " | ".join(summary_parts),
    }


def load_summary_map(structured_documents_root: Path | Iterable[Path]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in _iter_structured_document_paths(structured_documents_root):
        try:
            payload = json.loads(read_text_with_fallback(path)[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        payload = ensure_basic_summary(payload)
        document_id = str(payload.get("document_id") or "")
        if document_id:
            summaries[document_id] = payload.get("llm_summary") or payload["basic_summary"]
    return summaries


def backfill_basic_summaries(structured_documents_root: Path) -> int:
    updated = 0
    if not structured_documents_root.exists():
        return updated
    for path in sorted(structured_documents_root.glob("*.json")):
        try:
            payload = json.loads(read_text_with_fallback(path)[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("basic_summary"):
            continue
        ensure_basic_summary(payload)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated += 1
    return updated


def backfill_llm_summaries(structured_documents_root: Path) -> int:
    """이미 파싱된 구조화 문서들에 llm_summary를 일괄 추가한다."""
    updated = 0
    if not structured_documents_root.exists():
        return updated
    for path in sorted(structured_documents_root.glob("*.json")):
        try:
            payload = json.loads(read_text_with_fallback(path)[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("llm_summary"):
            continue
        before = payload.get("llm_summary")
        ensure_llm_summary(payload, source_path=path)
        if payload.get("llm_summary") and payload.get("llm_summary") != before:
            updated += 1
    return updated


# ── 내부 구현 ──────────────────────────────────────────────────────────────────

def extract_semantic_summary_chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    precomputed = _normalize_semantic_chunks(payload.get("semantic_chunks"))
    if precomputed:
        return precomputed

    markdown = str(payload.get("markdown") or "")
    if not markdown.strip():
        return []
    return build_semantic_chunks(markdown)


def _build_llm_summary(markdown: str, settings: Any, *, semantic_chunks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    normalized_chunks = _normalize_semantic_chunks(semantic_chunks)
    if normalized_chunks:
        return _map_reduce_summarize(markdown, settings, semantic_chunks=normalized_chunks)
    if len(markdown) <= _STUFF_THRESHOLD:
        return _stuff_summarize(markdown, settings)
    return _map_reduce_summarize(markdown, settings)


def _stuff_summarize(content: str, settings: Any) -> dict[str, Any]:
    prompt = _CHUNK_SUMMARIZE_PROMPT.format(content=content)
    raw = _call_openai(prompt, settings)
    return _parse_json_response(raw)


def _map_reduce_summarize(
    markdown: str,
    settings: Any,
    *,
    semantic_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_chunks = _normalize_semantic_chunks(semantic_chunks)
    if not raw_chunks:
        raw_chunks = build_semantic_chunks(markdown)
    if len(raw_chunks) <= 1:
        single_chunk = str((raw_chunks[0].get("text") if raw_chunks else markdown) or "").strip()
        return _stuff_summarize(single_chunk or markdown, settings)

    chunk_summaries: list[str] = []
    for chunk in raw_chunks:
        text = chunk.get("text", "")
        if not text.strip():
            continue
        result = _stuff_summarize(text, settings)
        summary_text = result.get("summary_text", "")
        points = result.get("key_points", [])
        parts = [summary_text] + [f"- {p}" for p in points]
        chunk_summaries.append("\n".join(p for p in parts if p))

    combined = "\n\n---\n\n".join(chunk_summaries)
    prompt = _REDUCE_PROMPT.format(summaries=combined)
    raw = _call_openai(prompt, settings)
    return _parse_json_response(raw)


def _normalize_semantic_chunks(raw_chunks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_chunks, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        strategy = str(chunk.get("strategy") or "semantic").strip() or "semantic"
        normalized.append(
            {
                "strategy": strategy,
                "chunk_index": int(chunk.get("chunk_index", index) or index),
                "text": text,
                "char_count": int(chunk.get("char_count", len(text)) or len(text)),
                "section_hint": chunk.get("section_hint"),
            }
        )
    return normalized


def _call_openai(prompt: str, settings: Any) -> str:
    body = json.dumps({
        "model": settings.model,
        "input": prompt,
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    def _do() -> str:
        req = urllib_request.Request(
            "https://api.openai.com/v1/responses",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            exc.reason = f"OpenAI API error: {exc.code} {detail}"
            raise
        for item in raw.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return content["text"].strip()
        if raw.get("output_text"):
            return str(raw["output_text"]).strip()
        raise RuntimeError("OpenAI response did not include output text")

    return call_with_retry(_do, context="document_summary._call_openai")


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"summary_text": text}


def _extract_candidate_sentences(markdown: str) -> list[str]:
    candidates: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(markdown):
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue
        if normalized.startswith("#"):
            continue
        if len(normalized) < 15:
            continue
        if normalized in candidates:
            continue
        candidates.append(normalized)
    return candidates


def _iter_structured_document_paths(structured_documents_root: Path | Iterable[Path]) -> list[Path]:
    if isinstance(structured_documents_root, Path):
        roots = [structured_documents_root]
    else:
        roots = list(structured_documents_root)

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
