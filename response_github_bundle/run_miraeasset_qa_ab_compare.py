from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from src.indexing.chunking import build_semantic_chunks
from src.retrieval.openai_answerer import OpenAIAnswerSynthesizer


PROJECT_ROOT = Path(__file__).resolve().parent
PAYLOAD_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manual_compare"
    / "miraeasset_main_pipeline"
    / "structured"
    / "documents"
    / "미래에셋증권_3분기_실적보고서--59a0fab3.json"
)
CURATED_CHUNKS_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "miraeasset_q3_semantic_chunks_curated.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "miraeasset_q3_qa_ab_compare"
REPORT_PATH = OUTPUT_DIR / "miraeasset_q3_qa_ab_compare.md"
JSON_PATH = OUTPUT_DIR / "miraeasset_q3_qa_ab_compare.json"

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣%]+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current markdown semantic QA vs proposed LLM-ready chunk QA.")
    parser.add_argument("--query", action="append", default=[], help="Question to compare. Can be repeated.")
    parser.add_argument("--question-set", choices=["sample", "pagewise"], default="sample")
    parser.add_argument("--max-questions", type=int, default=0, help="Limit the number of generated questions for a quick run.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answer-mode", choices=["llm", "extractive"], default="llm")
    args = parser.parse_args()

    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    baseline_chunks = build_baseline_chunks(payload)
    proposed_chunks = build_proposed_chunks()
    hybrid_chunks = build_hybrid_chunks(baseline_chunks, proposed_chunks)
    backend = LocalDeterministicEmbeddings()

    indexes = {
        "current_main_markdown_semantic": build_index(baseline_chunks, backend),
        "proposed_llm_ready_curated": build_index(proposed_chunks, backend),
        "hybrid_main_plus_llm_ready": build_index(hybrid_chunks, backend),
    }

    queries = args.query or questions_for_set(args.question_set)
    if args.max_questions > 0:
        queries = queries[: args.max_questions]
    results = []
    for query in queries:
        current = answer_query(
            query=query,
            index=indexes["current_main_markdown_semantic"],
            backend=backend,
            top_k=args.top_k,
            answer_mode=args.answer_mode,
            strategy_label="current_main_markdown_semantic",
        )
        proposed = answer_query(
            query=query,
            index=indexes["proposed_llm_ready_curated"],
            backend=backend,
            top_k=args.top_k,
            answer_mode=args.answer_mode,
            strategy_label="proposed_llm_ready_curated",
        )
        hybrid = answer_query(
            query=query,
            index=indexes["hybrid_main_plus_llm_ready"],
            backend=backend,
            top_k=args.top_k,
            answer_mode=args.answer_mode,
            strategy_label="hybrid_main_plus_llm_ready",
        )
        results.append({"query": query, "current": current, "proposed": proposed, "hybrid": hybrid})

    summary = {
        "payload": PAYLOAD_PATH.as_posix(),
        "curated_chunks": CURATED_CHUNKS_PATH.as_posix(),
        "answer_mode": args.answer_mode,
        "question_set": "custom" if args.query else args.question_set,
        "question_count": len(queries),
        "top_k": args.top_k,
        "chunk_counts": {
            "current_main_markdown_semantic": len(baseline_chunks),
            "proposed_llm_ready_curated": len(proposed_chunks),
            "hybrid_main_plus_llm_ready": len(hybrid_chunks),
        },
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"markdown": REPORT_PATH.as_posix(), "json": JSON_PATH.as_posix()}, ensure_ascii=False, indent=2))


def build_baseline_chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    markdown = str(payload.get("markdown") or "").strip()
    chunks = build_semantic_chunks(markdown)
    page_texts = split_markdown_pages(markdown)
    source_name = str(payload.get("source_name") or payload.get("file_name") or "미래에셋증권_3분기_실적보고서.pdf")
    document_id = str(payload.get("document_id") or "miraeasset_q3_main_pipeline")
    normalized = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        page_number = chunk.get("page_number") or infer_page_number(text) or infer_chunk_page_by_overlap(text, page_texts)
        normalized.append(
            {
                "chunk_id": f"main_semantic_{index:03d}",
                "chunk_type": "main_markdown_semantic",
                "semantic_type": "markdown_semantic",
                "strategy": "semantic",
                "chunk_index": index,
                "page_number": page_number,
                "section_hint": chunk.get("section_hint") or "",
                "text": text,
                "metadata": {
                    "source_name": source_name,
                    "document_id": document_id,
                },
            }
        )
    return normalized


def build_proposed_chunks() -> list[dict[str, Any]]:
    chunks = json.loads(CURATED_CHUNKS_PATH.read_text(encoding="utf-8"))
    normalized = []
    for index, chunk in enumerate(chunks):
        normalized.append(
            {
                **chunk,
                "chunk_id": chunk.get("chunk_id") or f"curated_{index:03d}",
                "chunk_index": chunk.get("chunk_index", index),
                "metadata": {
                    "source_name": "미래에셋증권_3분기_실적보고서.pdf",
                    "document_id": "miraeasset_q3_curated",
                    **(chunk.get("metadata") or {}),
                },
            }
        )
    return normalized


def build_hybrid_chunks(
    baseline_chunks: list[dict[str, Any]],
    proposed_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hybrid: list[dict[str, Any]] = []
    for chunk in baseline_chunks:
        hybrid.append(
            {
                **chunk,
                "chunk_id": f"hybrid_legacy_{chunk.get('chunk_id')}",
                "retrieval_lane": "legacy_context",
            }
        )
    for chunk in proposed_chunks:
        lane = "relation" if chunk.get("chunk_type") == "structured_relation" else "clean_context"
        hybrid.append(
            {
                **chunk,
                "chunk_id": f"hybrid_curated_{chunk.get('chunk_id')}",
                "retrieval_lane": lane,
            }
        )
    return hybrid


def build_index(chunks: list[dict[str, Any]], backend: "LocalDeterministicEmbeddings") -> dict[str, Any]:
    texts = [embedding_text(chunk) for chunk in chunks]
    return {
        "chunks": chunks,
        "embedding_texts": texts,
        "embeddings": backend.embed_documents(texts),
    }


def answer_query(
    *,
    query: str,
    index: dict[str, Any],
    backend: "LocalDeterministicEmbeddings",
    top_k: int,
    answer_mode: str,
    strategy_label: str,
) -> dict[str, Any]:
    query_embedding = backend.embed_query(query)
    query_tokens = tokens(query)
    query_key_terms = key_terms(query)
    requested_pages = extract_requested_pages(query)
    scored = []
    for chunk, text, embedding in zip(index["chunks"], index["embedding_texts"], index["embeddings"]):
        vector_score = cosine_similarity(query_embedding, embedding)
        chunk_tokens = tokens(text)
        overlap = len(query_tokens & chunk_tokens)
        numeric_overlap = len({token for token in query_tokens if has_digit(token)} & {token for token in chunk_tokens if has_digit(token)})
        exact_key_score = sum(term_weight(term) for term in query_key_terms if term in normalize_for_match(text))
        type_bonus = 0.16 if chunk.get("chunk_type") == "structured_relation" else 0.0
        if strategy_label == "hybrid_main_plus_llm_ready" and chunk.get("chunk_type") == "clean_text":
            type_bonus += 0.04
        page_score = page_relevance_score(chunk, text, requested_pages)
        score = vector_score + overlap * 0.035 + numeric_overlap * 0.08 + exact_key_score + type_bonus + page_score
        scored.append(
            {
                "score": round(score, 6),
                "vector_score": round(vector_score, 6),
                "token_overlap": overlap,
                "numeric_overlap": numeric_overlap,
                "exact_key_score": round(exact_key_score, 6),
                "type_bonus": type_bonus,
                "page_score": page_score,
                "chunk": chunk,
                "highlight": best_highlight(query_tokens, chunk.get("text") or ""),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    matches = select_hybrid_matches(scored, top_k) if strategy_label == "hybrid_main_plus_llm_ready" else scored[:top_k]
    evidence = [evidence_record(match, index + 1, strategy_label) for index, match in enumerate(matches)]
    answer = synthesize_llm_answer(query, evidence) if answer_mode == "llm" else synthesize_extractive_answer(matches)
    return {
        "strategy": strategy_label,
        "answer": answer,
        "matches": matches,
        "evidence": evidence,
    }


def select_hybrid_matches(scored: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    lane_targets = [
        ("structured_relation", 2),
        ("clean_text", 1),
        ("main_markdown_semantic", 1),
    ]
    for chunk_type, target_count in lane_targets:
        for match in scored:
            chunk = match.get("chunk") or {}
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id in selected_ids or chunk.get("chunk_type") != chunk_type:
                continue
            selected.append(match)
            selected_ids.add(chunk_id)
            if len(selected) >= top_k or sum(1 for item in selected if (item.get("chunk") or {}).get("chunk_type") == chunk_type) >= target_count:
                break
        if len(selected) >= top_k:
            break

    for match in scored:
        chunk = match.get("chunk") or {}
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id in selected_ids:
            continue
        selected.append(match)
        selected_ids.add(chunk_id)
        if len(selected) >= top_k:
            break

    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected[:top_k]


def embedding_text(chunk: dict[str, Any]) -> str:
    metadata_parts = [
        f"chunk_type: {chunk.get('chunk_type', '')}",
        f"page: {chunk.get('page_number', '')}",
        f"region: {chunk.get('region_id', '')}",
        f"semantic_type: {chunk.get('semantic_type', '')}",
        f"section_hint: {chunk.get('section_hint', '')}",
    ]
    return "\n".join([*metadata_parts, "", str(chunk.get("text") or "")]).strip()


def evidence_record(match: dict[str, Any], source_number: int, strategy_label: str) -> dict[str, Any]:
    chunk = match["chunk"]
    metadata = chunk.get("metadata") or {}
    return {
        "source_number": source_number,
        "source_name": metadata.get("source_name", "미래에셋증권_3분기_실적보고서.pdf"),
        "document_id": metadata.get("document_id", strategy_label),
        "page_number": chunk.get("page_number", ""),
        "section_hint": chunk.get("semantic_type") or chunk.get("section_hint") or chunk.get("chunk_type"),
        "chunk_index": chunk.get("chunk_id") or chunk.get("chunk_index", ""),
        "highlight_text": match.get("highlight", ""),
        "excerpt": truncate(chunk.get("text") or "", 1100),
    }


def infer_page_number(text: str) -> int | str:
    match = re.search(r"(?:^|\n)#\s*Page\s+(\d+)\b", str(text or ""))
    if match:
        return int(match.group(1))
    return ""


def split_markdown_pages(markdown: str) -> dict[int, str]:
    pages: dict[int, str] = {}
    parts = re.split(r"(?=^#\s*Page\s+\d+\b)", str(markdown or ""), flags=re.MULTILINE)
    for part in parts:
        match = re.match(r"#\s*Page\s+(\d+)\b", part.strip())
        if match:
            pages[int(match.group(1))] = part
    return pages


def infer_chunk_page_by_overlap(text: str, page_texts: dict[int, str]) -> int | str:
    chunk_tokens = {token for token in tokens(text) if len(token) >= 3}
    if not chunk_tokens:
        return ""
    best_page = ""
    best_score = 0
    for page_number, page_text in page_texts.items():
        page_tokens = {token for token in tokens(page_text) if len(token) >= 3}
        overlap = len(chunk_tokens & page_tokens)
        if overlap > best_score:
            best_score = overlap
            best_page = page_number
    return best_page if best_score >= 4 else ""


def extract_requested_pages(query: str) -> set[int]:
    pages = set()
    for match in re.finditer(r"(?:페이지|page|p)\s*\.?\s*(\d{1,2})(?!\d)", str(query or ""), flags=re.IGNORECASE):
        pages.add(int(match.group(1)))
    return pages


def page_relevance_score(chunk: dict[str, Any], text: str, requested_pages: set[int]) -> float:
    if not requested_pages:
        return 0.0

    chunk_page = normalize_page_number(chunk.get("page_number"))
    text_pages = {int(item) for item in re.findall(r"#\s*Page\s+(\d+)\b", str(text or ""), flags=re.IGNORECASE)}
    candidate_pages = set(text_pages)
    if chunk_page:
        candidate_pages.add(chunk_page)

    if candidate_pages & requested_pages:
        return 1.2
    if candidate_pages:
        return -1.0
    return -0.35


def normalize_page_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"\d+", str(value or ""))
    if match:
        return int(match.group(0))
    return None


def synthesize_llm_answer(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    synthesizer = OpenAIAnswerSynthesizer()
    if not synthesizer.is_enabled():
        return {
            "answer": "OPENAI_API_KEY가 설정되어 있지 않아 LLM 답변을 만들지 못했습니다. --answer-mode extractive로 검색 결과만 비교할 수 있습니다.",
            "citations": [],
            "warning": "missing_openai_api_key",
        }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return synthesizer.answer(query=query, evidence=evidence)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.6 + attempt * 0.8)
    return {
        "answer": f"LLM 답변 생성 중 오류가 발생했습니다: {last_error}",
        "citations": [],
        "warning": "llm_answer_failed",
    }


def synthesize_extractive_answer(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {"answer": "관련 근거를 찾지 못했습니다.", "citations": []}
    best = matches[0]
    chunk = best["chunk"]
    highlight = best.get("highlight") or first_content_line(chunk.get("text") or "")
    return {
        "answer": f"가장 관련성이 높은 근거는 {chunk.get('chunk_id')}입니다. {highlight}",
        "citations": [{"source_number": 1, "quote": truncate(highlight, 240)}],
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Mirae Asset Q3 QA A/B Comparison",
        "",
        f"- Current source: `{summary['payload']}`",
        f"- Proposed source: `{summary['curated_chunks']}`",
        f"- Answer mode: `{summary['answer_mode']}`",
        f"- Question set: `{summary['question_set']}`",
        f"- Question count: `{summary['question_count']}`",
        f"- Top K: `{summary['top_k']}`",
        f"- Current chunks: `{summary['chunk_counts']['current_main_markdown_semantic']}`",
        f"- Proposed chunks: `{summary['chunk_counts']['proposed_llm_ready_curated']}`",
        f"- Hybrid chunks: `{summary['chunk_counts']['hybrid_main_plus_llm_ready']}`",
        "",
    ]
    for item in summary["results"]:
        lines.extend([f"## Q. {item['query']}", ""])
        lines.extend(render_strategy_block("Current main.py style markdown semantic chunks", item["current"]))
        lines.extend(render_strategy_block("Proposed LLM-ready curated chunks", item["proposed"]))
        lines.extend(render_strategy_block("Hybrid main.py + LLM-ready chunks", item["hybrid"]))
    return "\n".join(lines).rstrip() + "\n"


def render_strategy_block(title: str, result: dict[str, Any]) -> list[str]:
    answer = result.get("answer") or {}
    lines = [
        f"### {title}",
        "",
        "**Answer**",
        "",
        str(answer.get("answer") if isinstance(answer, dict) else answer),
        "",
        "**Top evidence**",
        "",
    ]
    for index, match in enumerate(result.get("matches") or [], start=1):
        chunk = match["chunk"]
        lines.extend(
            [
                f"{index}. `{chunk.get('chunk_id')}` score `{match.get('score')}`",
                f"   - page: `{chunk.get('page_number', '')}` / type: `{chunk.get('chunk_type', '')}` / semantic: `{chunk.get('semantic_type', '')}`",
                f"   - lane: `{chunk.get('retrieval_lane', '')}`",
                f"   - highlight: {match.get('highlight', '')}",
                "",
                "```md",
                truncate(chunk.get("text") or "", 1300),
                "```",
                "",
            ]
        )
    return lines


def questions_for_set(question_set: str) -> list[str]:
    if question_set == "pagewise":
        return pagewise_queries()
    return default_queries()


def default_queries() -> list[str]:
    return [
        "3Q25 연결 세전이익과 순이익은 얼마인가?",
        "3Q25 별도 영업수익은 얼마이고 3Q24와 비교하면 어떤가?",
        "p4 r2 mixed chart에서 3Q25와 3Q24 값은 각각 얼마인가?",
        "주주환원율의 2024년 값은 얼마인가?",
        "해외법인 세전손익은 전년 대비 어떻게 변했나?",
    ]


def pagewise_queries() -> list[str]:
    return [
        "페이지 1에서 이 자료의 투자 권유 및 법적 책임 관련 유의사항은 무엇인가?",
        "페이지 1에서 보고서 작성 주체와 작성 시점은 무엇인가?",
        "페이지 2의 목차에는 어떤 주요 섹션들이 포함되어 있는가?",
        "페이지 2에서 Key Highlights와 사업별 주요 실적은 각각 몇 페이지로 안내되는가?",
        "페이지 3의 2025년 중점사업 추진전략은 무엇인가?",
        "페이지 3에서 글로벌 비즈니스와 WM/연금 관련 우선순위는 어떻게 표현되어 있는가?",
        "페이지 4에서 3Q25 연결 세전이익과 순이익은 얼마인가?",
        "페이지 4의 3Q25 별도 순영업수익은 3Q24와 비교해 얼마인가?",
        "페이지 4에서 Brokerage와 WM 관련 Key Highlights는 무엇인가?",
        "페이지 5에서 3Q25 순영업수익, 영업이익, 연결 당기순이익은 각각 얼마인가?",
        "페이지 5에서 3Q25 Brokerage 수수료와 WM 수수료는 각각 얼마인가?",
        "페이지 6에서 3Q25 Brokerage 수수료 수익과 QoQ 변화는 무엇인가?",
        "페이지 6에서 국내주식과 해외주식 수수료 수익은 각각 얼마인가?",
        "페이지 7에서 3Q25 WM 수수료 수익은 얼마이며 어떤 항목들이 포함되는가?",
        "페이지 7에서 연금 잔고와 고객자산 관련 3Q25 수치는 무엇인가?",
        "페이지 8에서 3Q25 Trading 운용손익과 분배 및 배당금 수익은 각각 얼마인가?",
        "페이지 8에서 연결 투자목적자산 공정가치평가 손익은 어떻게 설명되어 있는가?",
        "페이지 9에서 3Q25 IB 수수료 수익과 QoQ 변화는 무엇인가?",
        "페이지 9에서 주요 IB Deals에는 어떤 사례들이 언급되는가?",
        "페이지 10에서 3Q25 해외법인 세전이익과 QoQ 변화는 무엇인가?",
        "페이지 10에서 선진지역과 이머징지역의 세전이익은 각각 얼마인가?",
        "페이지 11에서 3Q25 연결 ROE와 지배주주 자기자본은 얼마인가?",
        "페이지 11에서 3Q25 BPS와 EPS는 각각 얼마인가?",
        "페이지 12의 Appendix에는 어떤 항목들이 포함되어 있는가?",
        "페이지 12에서 회사 개요와 주주환원 현황은 Appendix에서 각각 어떤 위치로 안내되는가?",
        "페이지 13에서 발행주식 총 수와 시가총액은 얼마인가?",
        "페이지 13에서 주요 주주와 지분율은 어떻게 제시되어 있는가?",
        "페이지 14에서 미래에셋캐피탈과 미래에셋증권의 지배구조상 지분율은 어떻게 제시되는가?",
        "페이지 14에서 주요 계열사 지배구조에 포함된 계열사는 무엇인가?",
        "페이지 15에서 글로벌 지역, 글로벌 거점, 글로벌 임직원, 글로벌 자기자본은 각각 얼마인가?",
        "페이지 15에서 인도, 브라질, 홍콩의 주요 비즈니스는 어떻게 설명되는가?",
        "페이지 16에서 2024년 총 주주환원율과 주주환원 총액은 얼마인가?",
        "페이지 16에서 보통주와 2우선주의 자기주식 소각 목표 대비 이행률은 어떻게 제시되는가?",
        "페이지 16에서 2024년 자기주식소각 총액과 배당 총액은 얼마인가?",
        "페이지 17에서 ESG 등급 또는 리더십 등급은 어떻게 제시되어 있는가?",
        "페이지 17에서 ESG 주요 성과로 언급된 이니셔티브나 등급 정보는 무엇인가?",
        "페이지 18에서 3Q25 별도 레버리지비율과 연결 순자본비율은 얼마인가?",
        "페이지 18에서 순자본비율과 레버리지비율은 어떻게 정의되어 있는가?",
        "페이지 19에서 ESG&IR Team의 연락처 정보는 무엇인가?",
        "페이지 19는 문서의 어떤 종료 정보를 제공하는가?",
    ]


def best_highlight(query_tokens: set[str], text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("|---")]
    best_line = ""
    best_score = -1
    for line in lines:
        line_tokens = tokens(line)
        overlap = len(query_tokens & line_tokens)
        numeric_overlap = len({token for token in query_tokens if has_digit(token)} & {token for token in line_tokens if has_digit(token)})
        score = overlap * 2 + numeric_overlap * 3 + (1 if "|" in line else 0)
        if score > best_score:
            best_score = score
            best_line = line
    return best_line


def first_content_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("Reference:"):
            return stripped
    return truncate(text, 240)


def tokens(text: str) -> set[str]:
    output: set[str] = set()
    for token in TOKEN_RE.findall(str(text or "").lower()):
        output.add(token)
        stripped = strip_korean_suffix(token)
        if stripped:
            output.add(stripped)
    return output


def key_terms(text: str) -> list[str]:
    ignored = {"얼마", "각각", "비교", "어떤가", "어떻게", "변했나", "값은"}
    terms = []
    for token in tokens(text):
        if len(token) < 2 or token in ignored:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        terms.append(normalize_for_match(token))
    return sorted(set(terms), key=len, reverse=True)


def term_weight(term: str) -> float:
    if re.fullmatch(r"(?:20\d{2}|[1-4]?q\d{2}|ytd|qoq)", term, flags=re.IGNORECASE):
        return 0.05
    if len(term) >= 4:
        return 0.42
    return 0.18


def strip_korean_suffix(token: str) -> str:
    suffixes = [
        "으로는",
        "에서는",
        "에게는",
        "하고",
        "이며",
        "인가",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "와",
        "과",
        "로",
        "으로",
    ]
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def has_digit(text: str) -> bool:
    return any(char.isdigit() for char in str(text or ""))


def truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class LocalDeterministicEmbeddings:
    model_name = "local-deterministic-v1"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        found_tokens = TOKEN_RE.findall(str(text or "").lower())
        if not found_tokens:
            return vector
        for token in found_tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + digest[5] / 255.0
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


if __name__ == "__main__":
    main()
