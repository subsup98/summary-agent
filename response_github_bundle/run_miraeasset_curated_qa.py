from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from src.retrieval.openai_answerer import OpenAIAnswerSynthesizer


PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "miraeasset_q3_semantic_chunks_curated.json"
PAYLOAD_DIR = PROJECT_ROOT / "outputs" / "manual_compare" / "miraeasset_main_pipeline" / "structured" / "documents"
INDEX_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "miraeasset_q3_curated_embedding_index.json"
HYBRID_INDEX_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "miraeasset_q3_hybrid_embedding_index.json"
QA_REPORT_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "miraeasset_q3_curated_qa_last.md"
QA_JSON_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "miraeasset_q3_curated_qa_last.json"


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣.%]+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed curated Mirae Asset Q3 semantic chunks and run test QA.")
    parser.add_argument("--query", action="append", default=[], help="Question to ask. Can be repeated.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild embedding index even if it already exists.")
    parser.add_argument("--embedding-backend", choices=["local", "openai"], default="local")
    parser.add_argument("--strategy", choices=["curated", "hybrid"], default="curated", help="Chunk set to search.")
    parser.add_argument("--payload-path", default="", help="Structured JSON payload to use for hybrid legacy markdown chunks.")
    parser.add_argument(
        "--answer-mode",
        choices=["llm", "extractive"],
        default="llm",
        help="Answer with the OpenAI evidence synthesizer by default, or show extractive retrieval debug output.",
    )
    parser.add_argument("--llm-answer", action="store_true", help="Deprecated alias for --answer-mode llm.")
    parser.add_argument("--interactive", action="store_true", help="Ask questions repeatedly in the terminal.")
    args = parser.parse_args()

    index_path = index_path_for_strategy(args.strategy)
    if args.rebuild or not index_path.exists():
        index = build_index(args.embedding_backend, args.strategy, payload_path=Path(args.payload_path) if args.payload_path else None)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        index = json.loads(index_path.read_text(encoding="utf-8"))

    backend = embedding_backend_for_index(index)
    if args.interactive:
        run_interactive(index, backend, top_k=args.top_k, answer_mode=resolve_answer_mode(args))
        return

    queries = args.query or default_queries()
    results = [answer_query(query, index, backend, top_k=args.top_k, answer_mode=resolve_answer_mode(args)) for query in queries]

    QA_JSON_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    QA_REPORT_PATH.write_text(render_report(results), encoding="utf-8")
    print(json.dumps({"index": index_path.as_posix(), "qa_markdown": QA_REPORT_PATH.as_posix(), "qa_json": QA_JSON_PATH.as_posix()}, ensure_ascii=False, indent=2))


def resolve_answer_mode(args: argparse.Namespace) -> str:
    if args.llm_answer:
        return "llm"
    return str(args.answer_mode or "llm")


def index_path_for_strategy(strategy: str) -> Path:
    if strategy == "hybrid":
        return HYBRID_INDEX_PATH
    return INDEX_PATH


def run_interactive(index: dict[str, Any], backend: Any, *, top_k: int, answer_mode: str) -> None:
    print("Mirae Asset Q3 QA")
    print("질문을 입력하세요. 종료하려면 exit 또는 quit.")
    print(
        f"strategy={index.get('strategy')}, chunks={len(index.get('chunks') or [])}, "
        f"embedding_model={index.get('embedding_model')}, top_k={top_k}, answer_mode={answer_mode}"
    )
    if answer_mode == "llm":
        print("LLM 답변 모드입니다. 검색된 evidence가 OpenAI 답변 생성기로 전달됩니다.")
    print()
    history: list[dict[str, Any]] = []
    while True:
        try:
            query = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        result = answer_query(query, index, backend, top_k=top_k, answer_mode=answer_mode)
        history.append(result)
        answer = result.get("answer") or {}
        print()
        print("A>", answer.get("answer") if isinstance(answer, dict) else answer)
        print()
        print("Top evidence:")
        for rank, match in enumerate(result.get("matches") or [], start=1):
            chunk = match["chunk"]
            print(
                f"{rank}. {chunk.get('chunk_id')} "
                f"(score={match.get('score')}, page={chunk.get('page_number')}, type={chunk.get('chunk_type')}/{chunk.get('semantic_type')})"
            )
            print(f"   {match.get('highlight', '')}")
        print()

    if history:
        QA_JSON_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        QA_REPORT_PATH.write_text(render_report(history), encoding="utf-8")
        print(f"Saved session: {QA_REPORT_PATH.as_posix()}")


def build_index(backend_name: str, strategy: str, payload_path: Path | None = None) -> dict[str, Any]:
    chunks = build_chunks_for_strategy(strategy, payload_path=payload_path)
    if backend_name == "openai":
        raise RuntimeError("OpenAI embeddings are not enabled in this dependency-light QA script. Use --embedding-backend local.")
    backend = LocalDeterministicEmbeddings()
    texts = [embedding_text(chunk) for chunk in chunks]
    embeddings = backend.embed_documents(texts)
    return {
        "strategy": strategy,
        "source_chunks": CHUNKS_PATH.as_posix(),
        "source_payload": payload_path.as_posix() if payload_path else "",
        "embedding_model": backend.model_name,
        "chunks": chunks,
        "embedding_texts": texts,
        "embeddings": embeddings,
    }


def build_chunks_for_strategy(strategy: str, payload_path: Path | None = None) -> list[dict[str, Any]]:
    curated_chunks = build_curated_chunks()
    if strategy == "hybrid":
        return build_hybrid_chunks(build_baseline_chunks(payload_path=payload_path), curated_chunks)
    return curated_chunks


def build_curated_chunks() -> list[dict[str, Any]]:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
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


def build_baseline_chunks(payload_path: Path | None = None) -> list[dict[str, Any]]:
    from src.indexing.chunking import build_semantic_chunks

    active_payload_path = payload_path or resolve_payload_path()
    payload = json.loads(active_payload_path.read_text(encoding="utf-8"))
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


def build_hybrid_chunks(baseline_chunks: list[dict[str, Any]], curated_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hybrid: list[dict[str, Any]] = []
    for chunk in baseline_chunks:
        hybrid.append({**chunk, "chunk_id": f"hybrid_legacy_{chunk.get('chunk_id')}", "retrieval_lane": "legacy_context"})
    for chunk in curated_chunks:
        lane = "relation" if chunk.get("chunk_type") == "structured_relation" else "clean_context"
        hybrid.append({**chunk, "chunk_id": f"hybrid_curated_{chunk.get('chunk_id')}", "retrieval_lane": lane})
    return hybrid


def resolve_payload_path() -> Path:
    candidates = sorted(PAYLOAD_DIR.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No structured document JSON found under {PAYLOAD_DIR}")
    return candidates[0]


def embedding_backend_for_index(index: dict[str, Any]) -> Any:
    model = str(index.get("embedding_model") or "")
    if model == "local-deterministic-v1":
        return LocalDeterministicEmbeddings()
    raise RuntimeError(f"Unsupported embedding model in index: {model}")


def answer_query(query: str, index: dict[str, Any], backend: Any, *, top_k: int, answer_mode: str) -> dict[str, Any]:
    query_embedding = backend.embed_query(query)
    query_tokens = tokens(query)
    query_key_terms = key_terms(query)
    requested_pages = extract_requested_pages(query)
    strategy = str(index.get("strategy") or "curated")
    scored = []
    for chunk, text, embedding in zip(index["chunks"], index["embedding_texts"], index["embeddings"]):
        vector_score = cosine_similarity(query_embedding, embedding)
        chunk_tokens = tokens(text)
        overlap = len(query_tokens & chunk_tokens)
        numeric_overlap = len({token for token in query_tokens if has_digit(token)} & {token for token in chunk_tokens if has_digit(token)})
        exact_key_score = sum(term_weight(term) for term in query_key_terms if term in normalize_for_match(text))
        type_bonus = 0.16 if chunk.get("chunk_type") == "structured_relation" else 0.0
        if strategy == "hybrid" and chunk.get("chunk_type") == "clean_text":
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
    matches = select_hybrid_matches(scored, top_k) if strategy == "hybrid" else scored[:top_k]
    evidence = [evidence_record(match, index + 1) for index, match in enumerate(matches)]
    answer = synthesize_llm_answer(query, evidence) if answer_mode == "llm" else synthesize_extractive_answer(query, matches)
    return {
        "query": query,
        "answer": answer,
        "matches": matches,
        "evidence": evidence,
    }


def embedding_text(chunk: dict[str, Any]) -> str:
    metadata_parts = [
        f"chunk_type: {chunk.get('chunk_type', '')}",
        f"page: {chunk.get('page_number', '')}",
        f"region: {chunk.get('region_id', '')}",
        f"semantic_type: {chunk.get('semantic_type', '')}",
        f"section_hint: {chunk.get('section_hint', '')}",
    ]
    return "\n".join([*metadata_parts, "", str(chunk.get("text") or "")]).strip()


def select_hybrid_matches(scored: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    lane_targets = [("structured_relation", 2), ("clean_text", 1), ("main_markdown_semantic", 1)]
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


def evidence_record(match: dict[str, Any], source_number: int) -> dict[str, Any]:
    chunk = match["chunk"]
    return {
        "source_number": source_number,
        "source_name": (chunk.get("metadata") or {}).get("source_name", "미래에셋증권 3분기 실적보고서.pdf"),
        "document_id": "miraeasset_q3_curated",
        "page_number": chunk.get("page_number", ""),
        "section_hint": chunk.get("semantic_type") or chunk.get("chunk_type"),
        "chunk_index": chunk.get("chunk_id", ""),
        "highlight_text": match.get("highlight", ""),
        "excerpt": truncate(chunk.get("text") or "", 1000),
    }


def synthesize_llm_answer(query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    synthesizer = OpenAIAnswerSynthesizer()
    if not synthesizer.is_enabled():
        return {
            "answer": "OpenAI API key is not configured. `--answer-mode extractive`로 검색 결과만 확인하거나, .env에 OPENAI_API_KEY를 설정해 주세요.",
            "citations": [],
            "warning": "missing_openai_api_key",
        }
    try:
        return synthesizer.answer(query=query, evidence=evidence)
    except Exception as error:
        return {
            "answer": f"LLM 답변 생성 중 오류가 발생했습니다: {error}",
            "citations": [],
            "warning": "llm_answer_failed",
        }


def synthesize_extractive_answer(query: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {"answer": "관련 근거를 찾지 못했습니다.", "citations": []}
    best = matches[0]
    chunk = best["chunk"]
    highlight = best.get("highlight") or first_content_line(chunk.get("text") or "")
    return {
        "answer": f"가장 관련성이 높은 근거는 page {chunk.get('page_number')}의 {chunk.get('semantic_type') or chunk.get('chunk_type')} chunk입니다: {highlight}",
        "citations": [
            {
                "source_number": 1,
                "quote": truncate(highlight, 240),
            }
        ],
    }


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
    best_page: int | str = ""
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


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Mirae Asset Q3 Curated Semantic QA Test",
        "",
        f"- Index: `{INDEX_PATH.as_posix()}`",
        f"- Source chunks: `{CHUNKS_PATH.as_posix()}`",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## Q. {result['query']}",
                "",
                "Answer:",
                "",
                str((result.get("answer") or {}).get("answer") or result.get("answer")),
                "",
                "Top evidence:",
                "",
            ]
        )
        for index, match in enumerate(result.get("matches") or [], start=1):
            chunk = match["chunk"]
            lines.extend(
                [
                    f"### {index}. {chunk.get('chunk_id')} (score {match.get('score')})",
                    "",
                    f"- Page: {chunk.get('page_number')}",
                    f"- Type: {chunk.get('chunk_type')} / {chunk.get('semantic_type')}",
                    f"- Highlight: {match.get('highlight', '')}",
                    "",
                    "```md",
                    truncate(chunk.get("text") or "", 1400),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def default_queries() -> list[str]:
    return [
        "3Q25 연결 세전이익과 순이익은 얼마인가?",
        "3Q25 별도 순영업수익은 얼마이고 3Q24와 비교하면 어떤가?",
        "주주환원율 2024년 값은 얼마인가?",
    ]


def tokens(text: str) -> set[str]:
    output: set[str] = set()
    for token in TOKEN_RE.findall(str(text or "").lower()):
        output.add(token)
        stripped = strip_korean_suffix(token)
        if stripped:
            output.add(stripped)
    return output


def key_terms(text: str) -> list[str]:
    terms = []
    for token in tokens(text):
        if len(token) < 3:
            continue
        if token in {"얼마인가", "얼마야", "비교하면", "어떤가", "알려줘"}:
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
        "과",
        "와",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "에",
        "로",
        "으로",
        "하고",
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
