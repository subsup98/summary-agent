from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src.indexing.embedding_backends import EmbeddingBackend, load_openai_embedding_settings, normalize_model_token, resolve_embedding_backend
from src.retrieval.document_summary import load_summary_map
from src.retrieval.openai_answerer import load_openai_settings
from src.shared.retry import call_with_retry
from src.shared.io import read_text_with_fallback
from src.shared.runtime_deps import ensure_local_dependency_path


ensure_local_dependency_path()

from langchain_classic.chains import RetrievalQA  # type: ignore  # noqa: E402
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker  # type: ignore  # noqa: E402
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # type: ignore  # noqa: E402
from langchain_community.vectorstores import Chroma  # type: ignore  # noqa: E402
from langchain_core.documents import Document  # type: ignore  # noqa: E402
from langchain_core.prompts import PromptTemplate  # type: ignore  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # type: ignore  # noqa: E402
from langchain_openai import ChatOpenAI  # type: ignore  # noqa: E402
from pydantic import Field  # type: ignore  # noqa: E402


def _format_quote_text(text: str) -> str:
    cleaned = re.sub(r"<br\s*/?>", "\n", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _tokenize_lookup_text(text: str) -> set[str]:
    return {token for token in re.split(r"[^\w\uAC00-\uD7A3]+", str(text or "").lower()) if token}


def _pick_highlight_text(query: str, text: str) -> str:
    formatted = _format_quote_text(text)
    sentences = [item.strip() for item in re.split(r"(?<=[.!?\u3002\uFF01\uFF1F])\s+|\n+", formatted) if item.strip()]
    if not sentences:
        return formatted
    query_tokens = _tokenize_lookup_text(query)
    if not query_tokens:
        return sentences[0]
    scored = []
    for sentence in sentences:
        sentence_tokens = _tokenize_lookup_text(sentence)
        overlap = len(query_tokens & sentence_tokens)
        scored.append((overlap, len(sentence), sentence))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


LANGCHAIN_SUMMARIZE_PROMPT = """당신은 한국어 문서를 빠르게 읽고 요약하는 AI입니다.

아래 문서 내용을 바탕으로 다음 JSON 형식으로만 답변하세요.
{{"summary_text": "2-3문장 요약", "key_points": ["핵심 포인트 1", "핵심 포인트 2"], "document_type": "문서 유형"}}

규칙:
- 요약은 문서에 실제로 있는 내용만 반영하세요.
- 수치, 날짜, 기관명은 문서에 있을 때만 적으세요.
- JSON 외 다른 설명은 쓰지 마세요.

문서 내용:
{content}
"""


LANGCHAIN_QA_PROMPT = """You are a document QA assistant for Korean financial, earnings, and business documents.

Follow these rules strictly:
- Use only the supplied context.
- Treat [Document Summary] as high-level background and use [Retrieved Chunk] for concrete facts, figures, dates, and wording.
- If the user names a specific document, company, quarter, report, or period, prioritize that target and do not generalize from unrelated documents.
- If the context contains multiple documents, prefer the one that best matches the named document or period.
- Do not guess missing numbers, dates, entities, or conclusions.
- If the question is ambiguous, first say what is unclear, then provide only the limited facts that are explicitly supported.
- If the context is insufficient, say exactly what is missing.
- Write naturally in Korean and avoid boilerplate AI phrasing.
- Do not add inline citation markers like [1] or [2] in the answer text.

Output format:
- Write one natural Korean answer block only.
- Do not output headings like "Short answer", "Key points", or "Evidence".
- Do not add a separate evidence section or chunk dump.
Context:
{context}

Question:
{question}

Answer:
"""


def _normalize_lookup_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", text).lower()


def _build_document_id_filter(document_ids: list[str]) -> dict[str, Any] | None:
    valid_ids = [item for item in document_ids if item]
    if not valid_ids:
        return None
    if len(valid_ids) == 1:
        return {"document_id": valid_ids[0]}
    return {"$or": [{"document_id": item} for item in valid_ids]}


def _extract_fields_from_text(text: str) -> dict[str, Any]:
    """Fallback: manually extract summary_text, key_points, document_type via regex."""
    result: dict[str, Any] = {}

    m = re.search(r'"summary_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        result["summary_text"] = m.group(1)

    m = re.search(r'"key_points"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
    if m:
        raw_items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        # Drop any item that is a bare key name like "document_type"
        clean = [pt for pt in raw_items if not re.match(r"^document_type$", pt, re.IGNORECASE)]
        result["key_points"] = clean

    m = re.search(r'"document_type"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        result["document_type"] = m.group(1)

    return result


def _parse_llm_summary_response(result_text: str) -> dict[str, Any]:
    """Parse LLM summary JSON, repairing common malformed patterns."""
    # 1. Direct parse
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        pass

    # 2. Extract JSON object substring and repair leaked document_type in key_points
    match = re.search(r"\{.*\}", result_text, re.DOTALL)
    candidate = match.group() if match else result_text

    repaired = re.sub(
        r'("key_points"\s*:\s*\[(?:[^\[\]]*?),\s*)"document_type"\s*:\s*"([^"]*)"(\s*\])',
        lambda m: re.sub(r",\s*$", "", m.group(1)) + m.group(3) + ', "document_type": "' + m.group(2) + '"',
        candidate,
        flags=re.DOTALL,
    )
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 3. Field-by-field regex extraction
    extracted = _extract_fields_from_text(result_text)
    if extracted:
        return extracted

    # 4. Last resort
    return {"summary_text": result_text}


class StaticDocumentRetriever(BaseRetriever):
    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return list(self.documents)


class LangChainRetrievalQAService:
    cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        project_root: Path,
        *,
        vector_root: Path | None = None,
        structured_document_roots: list[Path] | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.project_root = project_root
        self.vector_root = vector_root or (project_root / "outputs" / "vector_index")
        self.embedding_backend = embedding_backend or resolve_embedding_backend()
        self.embedding_settings = load_openai_embedding_settings()
        self.api_key = self.embedding_settings.api_key
        self.embedding_model = self.embedding_backend.model_name
        self.embedding_token = normalize_model_token(self.embedding_model)
        self.answer_model = load_openai_settings().model if self.api_key else None
        self.structured_document_roots = structured_document_roots or self._structured_document_roots()
        self.summary_map = load_summary_map(self.structured_document_roots)
        self.document_catalog = self._load_document_catalog()
        self._cross_encoder_reranker: CrossEncoderReranker | None = None
        self._chat_llm: ChatOpenAI | None = None
        self._vectorstores: dict[str, Chroma] = {}

    def refresh_catalog(self) -> None:
        self.structured_document_roots = self._structured_document_roots()
        self.summary_map = load_summary_map(self.structured_document_roots)
        self.document_catalog = self._load_document_catalog()

    def warmup(self, *, preload_summaries: bool = False, max_documents: int | None = None) -> dict[str, Any]:
        self.refresh_catalog()
        warmed = {
            "catalog_documents": len(self.document_catalog),
            "chat_model_ready": False,
            "semantic_vectorstore_ready": False,
            "reranker_ready": False,
            "preloaded_summary_count": 0,
            "warnings": [],
        }
        if self.api_key and self.answer_model:
            try:
                self._get_chat_llm()
                warmed["chat_model_ready"] = True
            except Exception as error:
                warmed["warnings"].append(f"chat_model: {error}")
        try:
            self._build_vectorstore("semantic")
            warmed["semantic_vectorstore_ready"] = True
        except Exception as error:
            warmed["warnings"].append(f"vectorstore: {error}")
        try:
            self._get_cross_encoder_reranker(top_n=4)
            warmed["reranker_ready"] = True
        except Exception as error:
            warmed["warnings"].append(f"reranker: {error}")
        if preload_summaries:
            try:
                warmed["preloaded_summary_count"] = self.preload_document_summaries(max_documents=max_documents)
            except Exception as error:
                warmed["warnings"].append(f"summary_preload: {error}")
        return warmed

    def preload_document_summaries(self, *, max_documents: int | None = None) -> int:
        count = 0
        records = self.document_catalog[: max_documents or None]
        for record in records:
            target_path = record.get("path")
            if not target_path:
                continue
            try:
                payload = json.loads(read_text_with_fallback(Path(target_path))[0])
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("ui_summary"):
                continue
            result = self.summarize_document(document_id=record["document_id"])
            if not result.get("error"):
                count += 1
        return count

    def answer_question(
        self,
        query: str,
        strategy: str = "rule_based",
        top_k: int = 4,
        *,
        document_id: str = "",
        source_name: str = "",
        document_ids: list[str] | None = None,
        use_reranker: bool = False,
    ) -> dict[str, Any]:
        self.refresh_catalog()
        vectorstore = self._build_vectorstore(strategy)
        scoped_ids = [item for item in (document_ids or []) if item]
        explicit_target = None if scoped_ids else self._resolve_explicit_target(document_id=document_id, source_name=source_name)
        source_candidate = explicit_target or self._select_source_candidate(query)
        # 여러 문서가 선택된 경우 문서당 최소 2청크씩 가져올 수 있도록 top_k 확대
        effective_top_k = max(top_k, len(document_ids) * 2) if document_ids and len(document_ids) > 1 else top_k
        source_documents = self._retrieve_documents(
            vectorstore,
            query,
            top_k=effective_top_k,
            source_candidate=source_candidate,
            document_ids=document_ids,
            use_reranker=use_reranker,
        )
        if self.api_key and self.answer_model:
            try:
                prompt = PromptTemplate(template=LANGCHAIN_QA_PROMPT)
                numbered_documents = [
                    Document(
                        page_content=f"[SOURCE {index + 1}]\n{document.page_content}",
                        metadata=dict(document.metadata),
                    )
                    for index, document in enumerate(source_documents)
                ]
                chain = RetrievalQA.from_chain_type(
                    llm=self._get_chat_llm(),
                    retriever=StaticDocumentRetriever(documents=numbered_documents),
                    return_source_documents=True,
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt},
                )
                response = call_with_retry(
                    lambda: chain.invoke({"query": query}),
                    context="LangChain.answer_question",
                )
                used_documents = response.get("source_documents", [])
                citations = [
                    {
                        "source_number": index + 1,
                        "source_name": document.metadata.get("source_name", ""),
                        "document_id": document.metadata.get("document_id", ""),
                        "section_hint": document.metadata.get("section_hint", ""),
                        "page_number": self._resolve_document_page_number(
                            document=document,
                            query=query,
                        ),
                        "quote": _format_quote_text(
                            document.metadata.get("original_page_content", "")
                            or re.sub(r"^\[SOURCE\s+\d+\]\s*", "", document.page_content)
                        ),
                        "highlight_text": _pick_highlight_text(
                            query,
                            document.metadata.get("original_page_content", "")
                            or re.sub(r"^\[SOURCE\s+\d+\]\s*", "", document.page_content),
                        ),
                    }
                    for index, document in enumerate(used_documents)
                ]
                return {
                    "query": query,
                    "strategy": strategy,
                    "answer": str(response.get("result", "") or "").strip(),
                    "citations": citations,
                    "document_summaries": self._build_document_summaries(used_documents),
                    "source_documents": [
                        {
                            "page_content": str(document.metadata.get("original_page_content", "") or re.sub(r"^\[SOURCE\s+\d+\]\s*", "", document.page_content)),
                            "metadata": dict(document.metadata),
                        }
                        for document in used_documents
                    ],
                    "used_model": self.answer_model,
                    "embedding_model": self.embedding_model,
                    "framework": "langchain",
                    "retrieval_mode": "cross_encoder_rerank" if use_reranker else "heuristic_only",
                    "reranker_model": self.cross_encoder_model_name if use_reranker else None,
                    "source_hint": source_candidate,
                    "document_id": document_id or None,
                    "source_name": source_name or None,
                    "document_ids": document_ids or [],
                }
            except Exception as error:
                fallback = self._fallback_answer(query=query, strategy=strategy, source_documents=source_documents)
                fallback["warning"] = str(error)
                fallback["source_hint"] = source_candidate
                fallback["document_id"] = document_id or None
                fallback["source_name"] = source_name or None
                fallback["document_ids"] = document_ids or []
                fallback["retrieval_mode"] = "cross_encoder_rerank" if use_reranker else "heuristic_only"
                fallback["reranker_model"] = self.cross_encoder_model_name if use_reranker else None
                return fallback

        fallback = self._fallback_answer(query=query, strategy=strategy, source_documents=source_documents)
        fallback["source_hint"] = source_candidate
        fallback["document_id"] = document_id or None
        fallback["source_name"] = source_name or None
        fallback["document_ids"] = document_ids or []
        fallback["retrieval_mode"] = "cross_encoder_rerank" if use_reranker else "heuristic_only"
        fallback["reranker_model"] = self.cross_encoder_model_name if use_reranker else None
        return fallback

    def _build_vectorstore(self, strategy: str) -> Chroma:
        if strategy in self._vectorstores:
            return self._vectorstores[strategy]
        collection_name = f"parsed_documents_{'rule' if strategy == 'rule_based' else 'semantic'}_{self.embedding_token}"
        vectorstore = Chroma(
            persist_directory=str(self.vector_root / "chroma_db"),
            collection_name=collection_name,
            embedding_function=self.embedding_backend,
        )
        self._vectorstores[strategy] = vectorstore
        return vectorstore

    def _retrieve_documents(
        self,
        vectorstore: Chroma,
        query: str,
        *,
        top_k: int,
        source_candidate: dict[str, Any] | None,
        document_ids: list[str] | None,
        use_reranker: bool,
    ) -> list[Document]:
        search_limit = max(top_k * 4, 8)
        docs_with_scores: list[tuple[Document, float]] = []
        scoped_ids = [item for item in (document_ids or []) if item]
        explicit_candidate_id = source_candidate.get("document_id") if source_candidate else None

        # 선택된 문서 전체를 한 번에 검색 — 활성 문서 하나만 우선 검색하면
        # 나머지 선택 문서(예: 1분기)의 청크가 결과에서 빠지는 문제가 발생함
        if scoped_ids:
            doc_filter = _build_document_id_filter(scoped_ids)
            docs_with_scores = vectorstore.similarity_search_with_score(
                query,
                k=search_limit,
                filter=doc_filter,
            )
        elif source_candidate:
            doc_filter = {"document_id": explicit_candidate_id} if explicit_candidate_id else {"source_name": source_candidate["source_name"]}
            docs_with_scores = vectorstore.similarity_search_with_score(
                query,
                k=search_limit,
                filter=doc_filter,
            )
        if not docs_with_scores and not (scoped_ids or source_candidate):
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=search_limit)

        reranked: list[tuple[Document, float]] = []
        query_norm = _normalize_lookup_text(query)
        for document, score in docs_with_scores:
            metadata = dict(document.metadata)
            metadata["original_page_content"] = document.page_content
            candidate_document_id = str(metadata.get("document_id", ""))
            candidate_source_name = str(metadata.get("source_name", ""))
            source_boost = 0.0
            if source_candidate:
                if candidate_document_id and candidate_document_id == explicit_candidate_id:
                    source_boost += 25.0
                elif candidate_source_name == source_candidate.get("source_name"):
                    source_boost += 15.0
            source_norm = _normalize_lookup_text(Path(candidate_source_name).stem)
            if source_norm and source_norm in query_norm:
                source_boost += 2.0
            summary = self.summary_map.get(candidate_document_id, {})
            enriched_text = self._enrich_document_text(document, summary)
            reranked.append(
                (
                    Document(page_content=enriched_text, metadata=metadata),
                    float(source_boost - score),
                )
            )
        reranked.sort(key=lambda item: item[1], reverse=True)
        ordered_documents = [document for document, _ in reranked]
        if not use_reranker:
            return ordered_documents[:top_k]

        compressor = self._get_cross_encoder_reranker(top_n=top_k)
        return list(compressor.compress_documents(documents=ordered_documents, query=query))

    def _get_cross_encoder_reranker(self, *, top_n: int) -> CrossEncoderReranker:
        if self._cross_encoder_reranker is None:
            model = HuggingFaceCrossEncoder(
                model_name=self.cross_encoder_model_name,
                model_kwargs={"device": "cpu"},
            )
            self._cross_encoder_reranker = CrossEncoderReranker(model=model, top_n=top_n)
        self._cross_encoder_reranker.top_n = top_n
        return self._cross_encoder_reranker

    def _enrich_document_text(self, document: Document, summary: dict[str, Any]) -> str:
        summary_text = str(summary.get("summary_text", "")).strip()
        highlights = [str(item).strip() for item in summary.get("highlights", []) if str(item).strip()]
        parts = []
        if summary_text:
            parts.append("[Document Summary]\n" + summary_text)
        if highlights:
            parts.append("[Highlights]\n- " + "\n- ".join(highlights[:3]))
        parts.append("[Retrieved Chunk]\n" + document.page_content)
        return "\n\n".join(parts)

    def _build_document_summaries(self, documents: list[Document]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for document in documents:
            document_id = str(document.metadata.get("document_id", ""))
            if not document_id or document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            summary = self.summary_map.get(document_id)
            if not summary:
                continue
            summaries.append(
                {
                    "document_id": document_id,
                    "source_name": document.metadata.get("source_name", ""),
                    "summary_text": summary.get("summary_text", ""),
                    "highlights": summary.get("highlights", []),
                }
            )
        return summaries[:3]

    def _load_document_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for root in self.structured_document_roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json")):
                try:
                    payload = json.loads(read_text_with_fallback(path)[0])
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                source_name = str(payload.get("source_name") or "")
                document_id = str(payload.get("document_id") or "")
                if not document_id:
                    continue
                source_stem = Path(source_name).stem if source_name else document_id
                catalog.append(
                    {
                        "document_id": document_id,
                        "source_name": source_name,
                        "source_stem": source_stem,
                        "normalized_source": _normalize_lookup_text(source_stem),
                        "path": path,
                    }
                )
        return catalog

    def _select_source_candidate(self, query: str) -> dict[str, Any] | None:
        query_norm = _normalize_lookup_text(query)
        ranked: list[tuple[int, dict[str, Any]]] = []
        for record in self.document_catalog:
            source_norm = record["normalized_source"]
            if not source_norm:
                continue
            score = 0
            if source_norm in query_norm:
                score += 100
            for token in self._tokenize_source_name(record["source_stem"]):
                if token and token in query_norm:
                    score += len(token)
            if score:
                ranked.append((score, record))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return None
        if len(ranked) == 1 or ranked[0][0] >= ranked[1][0] + 3:
            return ranked[0][1]
        return None

    def summarize_document(self, document_id: str = "", source_name: str = "") -> dict[str, Any]:
        self.refresh_catalog()
        record = self._resolve_explicit_target(document_id=document_id, source_name=source_name)
        if not record:
            return {"error": "document_not_found", "document_id": document_id, "source_name": source_name}

        target_path = record.get("path")
        if not target_path:
            return {"error": "document_file_not_found", "document_id": record["document_id"]}

        try:
            payload = json.loads(read_text_with_fallback(Path(target_path))[0])
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"error": "document_json_invalid", "document_id": record["document_id"]}

        # Return cached UI summary if already generated
        if payload.get("ui_summary"):
            cached_ui_summary = dict(payload["ui_summary"])
            cached_ui_summary.setdefault("summary_elapsed_seconds", 0.0)
            cached_ui_summary.setdefault("summary_cached", True)
            cached_ui_summary.setdefault("summary_source", "cached_ui_summary")
            return cached_ui_summary
        if payload.get("llm_summary"):
            llm_summary = payload["llm_summary"]
            return {
                "document_id": record["document_id"],
                "source_name": record["source_name"],
                "summary_text": llm_summary.get("summary_text", ""),
                "key_points": llm_summary.get("key_points") or llm_summary.get("highlights") or [],
                "document_type": llm_summary.get("document_type", ""),
                "used_model": llm_summary.get("used_model") or "cached_llm_summary",
                "framework": llm_summary.get("framework") or "pipeline_llm_summary",
                "summary_elapsed_seconds": 0.0,
                "summary_cached": True,
                "summary_source": "pipeline_llm_summary",
            }

        markdown = str(payload.get("markdown") or "")
        if not markdown.strip():
            return {"error": "document_has_no_content", "document_id": record["document_id"]}
        basic_summary = payload.get("basic_summary") or self.summary_map.get(record["document_id"], {})
        document_type = str(
            (payload.get("classification") or {}).get("document_type")
            or payload.get("extension")
            or ""
        )

        max_chars = 8000
        if len(markdown) > max_chars:
            markdown = markdown[:max_chars] + "\n...(이하 생략)"

        if not self.api_key or not self.answer_model:
            return {
                "document_id": record["document_id"],
                "source_name": record["source_name"],
                "summary_text": basic_summary.get("summary_text", ""),
                "key_points": basic_summary.get("highlights", [])[:3],
                "document_type": document_type,
                "used_model": None,
                "framework": "basic_summary_fallback",
                "warning": "OPENAI_API_KEY is not configured for LangChain summarization",
                "summary_elapsed_seconds": 0.0,
                "summary_cached": False,
                "summary_source": "basic_summary_fallback",
            }

        summarize_started = time.perf_counter()
        try:
            prompt_text = LANGCHAIN_SUMMARIZE_PROMPT.format(content=markdown)
            response = call_with_retry(
                lambda: self._get_chat_llm().invoke(prompt_text),
                context="LangChain.summarize_document",
            )
            result_text = response.content if hasattr(response, "content") else str(response)

            result = _parse_llm_summary_response(result_text)
        except Exception as error:
            return {
                "document_id": record["document_id"],
                "source_name": record["source_name"],
                "summary_text": basic_summary.get("summary_text", ""),
                "key_points": basic_summary.get("highlights", [])[:3],
                "document_type": document_type,
                "used_model": None,
                "framework": "basic_summary_fallback",
                "warning": str(error),
                "summary_elapsed_seconds": round(time.perf_counter() - summarize_started, 3),
                "summary_cached": False,
                "summary_source": "basic_summary_fallback",
            }

        # Strip any key_points element that is a stray "document_type: ..." string
        raw_key_points = result.get("key_points") or []
        clean_key_points: list[str] = []
        leaked_doc_type = ""
        for item in raw_key_points:
            if isinstance(item, str):
                m = re.match(r"^\s*document_type\s*[:\"]\s*(.+)", item, re.IGNORECASE)
                if m:
                    leaked_doc_type = m.group(1).strip().strip('"')
                else:
                    clean_key_points.append(item)
            else:
                clean_key_points.append(item)

        ui_summary = {
            "document_id": record["document_id"],
            "source_name": record["source_name"],
            "summary_text": result.get("summary_text", "") or basic_summary.get("summary_text", ""),
            "key_points": clean_key_points or basic_summary.get("highlights", [])[:3],
            "document_type": result.get("document_type", "") or leaked_doc_type or document_type,
            "used_model": self.answer_model,
            "framework": "langchain",
            "summary_elapsed_seconds": round(time.perf_counter() - summarize_started, 3),
            "summary_cached": False,
            "summary_source": "on_demand_llm",
        }

        # Persist to JSON so future calls skip LLM
        try:
            cached = json.loads(read_text_with_fallback(Path(target_path))[0])
            cached["ui_summary"] = ui_summary
            Path(target_path).write_text(
                json.dumps(cached, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

        return ui_summary

    def _resolve_explicit_target(self, *, document_id: str, source_name: str) -> dict[str, Any] | None:
        if document_id:
            for record in self.document_catalog:
                if record["document_id"] == document_id:
                    return record
        if source_name:
            norm = _normalize_lookup_text(source_name)
            for record in self.document_catalog:
                if _normalize_lookup_text(record["source_name"]) == norm or _normalize_lookup_text(record["source_stem"]) == norm:
                    return record
        return None

    def _tokenize_source_name(self, text: str) -> list[str]:
        return [token for token in re.split(r"[\s_\-()]+", text.lower()) if len(token) >= 2]

    def _get_chat_llm(self) -> ChatOpenAI:
        if self._chat_llm is None:
            if not self.api_key or not self.answer_model:
                raise RuntimeError("OpenAI chat model is unavailable")
            self._chat_llm = ChatOpenAI(model=self.answer_model, temperature=0, api_key=self.api_key)
        return self._chat_llm

    def _structured_document_roots(self) -> list[Path]:
        roots = [self.project_root / "data" / "structured" / "documents"]
        ui_runs_root = self.project_root / "outputs" / "ui_runs"
        if ui_runs_root.exists():
            for path in sorted(ui_runs_root.glob("*/structured/documents")):
                roots.append(path)
        return roots

    def _resolve_document_page_number(self, *, document: Document, query: str) -> int | None:
        metadata = document.metadata or {}
        existing_page = metadata.get("page_number")
        quote_text = _format_quote_text(
            metadata.get("original_page_content", "") or re.sub(r"^\[SOURCE\s+\d+\]\s*", "", document.page_content)
        )
        highlight_text = _pick_highlight_text(query, quote_text)
        looked_up_page = self._lookup_page_number(
            str(metadata.get("document_id", "")),
            highlight_text,
            quote_text,
        )
        return looked_up_page if looked_up_page is not None else existing_page

    def _lookup_page_number(self, document_id: str, *text_candidates: str) -> int | None:
        if not document_id:
            return None
        normalized_candidates = [
            _format_quote_text(text)
            for text in text_candidates
            if _format_quote_text(text)
        ]
        if not normalized_candidates:
            return None

        primary_text = normalized_candidates[0]
        primary_tokens = _tokenize_lookup_text(primary_text)
        if not primary_tokens:
            return None

        best_page: int | None = None
        best_score = -1.0
        for root in self._structured_document_roots():
            if not root.exists():
                continue
            for path in root.glob("*.json"):
                try:
                    payload = json.loads(read_text_with_fallback(path)[0])
                except Exception:
                    continue
                if str(payload.get("document_id", "")) != document_id:
                    continue
                markdown_pages = self._extract_markdown_pages(payload)
                if markdown_pages:
                    for page_number, page_text in markdown_pages:
                        chunk_tokens = _tokenize_lookup_text(page_text)
                        if not chunk_tokens:
                            continue

                        score = 0.0
                        if primary_text and primary_text in page_text:
                            score += 1500.0
                        elif primary_text and page_text in primary_text:
                            score += 900.0

                        for candidate in normalized_candidates[1:]:
                            if candidate and candidate in page_text:
                                score += 220.0

                        overlap = len(primary_tokens & chunk_tokens)
                        if overlap:
                            score += overlap * 10.0

                        if score > best_score:
                            best_score = score
                            best_page = page_number
                for chunk in payload.get("chunks", []):
                    page = chunk.get("page") or chunk.get("page_number")
                    if not page:
                        continue
                    chunk_text = _format_quote_text(str(chunk.get("serialized_text") or chunk.get("text") or ""))
                    if not chunk_text:
                        continue
                    chunk_tokens = _tokenize_lookup_text(chunk_text)
                    if not chunk_tokens:
                        continue

                    score = 0.0
                    if primary_text and primary_text in chunk_text:
                        score += 1000.0
                    elif primary_text and chunk_text in primary_text:
                        score += 700.0

                    for candidate in normalized_candidates[1:]:
                        if candidate and candidate in chunk_text:
                            score += 180.0

                    overlap = len(primary_tokens & chunk_tokens)
                    if overlap:
                        score += overlap * 12.0

                    if score > best_score:
                        best_score = score
                        try:
                            best_page = int(page)
                        except (TypeError, ValueError):
                            pass
                break
        return best_page if best_score > 0 else None

    def _extract_markdown_pages(self, payload: dict[str, Any]) -> list[tuple[int, str]]:
        markdown = _format_quote_text(
            str(payload.get("markdown_raw") or payload.get("markdown") or "")
        )
        if not markdown:
            return []
        matches = list(re.finditer(r"(?im)^#\s*Page\s+(\d+)\s*$", markdown))
        if not matches:
            return []

        pages: list[tuple[int, str]] = []
        for index, match in enumerate(matches):
            try:
                page_number = int(match.group(1))
            except ValueError:
                continue
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
            page_text = _format_quote_text(markdown[start:end])
            if page_text:
                pages.append((page_number, page_text))
        return pages

    def _fallback_answer(self, *, query: str, strategy: str, source_documents: list[Document]) -> dict[str, Any]:
        citations = [
            {
                "source_number": index + 1,
                "source_name": document.metadata.get("source_name", ""),
                "document_id": document.metadata.get("document_id", ""),
                "section_hint": document.metadata.get("section_hint", ""),
                "page_number": self._resolve_document_page_number(
                    document=document,
                    query=query,
                ),
                "quote": _format_quote_text(
                    document.metadata.get("original_page_content", "") or document.page_content
                ),
                "highlight_text": _pick_highlight_text(
                    query,
                    document.metadata.get("original_page_content", "") or document.page_content,
                ),
            }
            for index, document in enumerate(source_documents[:4])
        ]
        answer = str(source_documents[0].metadata.get("original_page_content", "") or source_documents[0].page_content)[:400] if source_documents else "관련 문서를 찾지 못했습니다."
        return {
            "query": query,
            "strategy": strategy,
            "answer": answer,
            "citations": citations,
            "document_summaries": self._build_document_summaries(source_documents),
            "source_documents": [
                {
                    "page_content": str(document.metadata.get("original_page_content", "") or document.page_content),
                    "metadata": dict(document.metadata),
                }
                for document in source_documents
            ],
            "used_model": None,
            "embedding_model": self.embedding_model,
            "framework": "retrieval_fallback",
        }
