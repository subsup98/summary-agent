from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.indexing.chroma_store import ChromaIndexManager
from src.indexing.embedding_backends import EmbeddingBackend
from src.retrieval.document_summary import load_summary_map
from src.retrieval.openai_answerer import OpenAIAnswerSynthesizer, load_openai_settings
from src.shared.io import read_text_with_fallback


def _build_document_id_filter(document_ids: list[str]) -> dict[str, Any] | None:
    valid_ids = [item for item in document_ids if item]
    if not valid_ids:
        return None
    if len(valid_ids) == 1:
        return {"document_id": valid_ids[0]}
    return {"$or": [{"document_id": item} for item in valid_ids]}


class ChromaRetriever:
    def __init__(self, project_root: Path, embedding_backend: EmbeddingBackend | None = None) -> None:
        self.project_root = project_root
        self.index_manager = ChromaIndexManager(project_root=project_root, embedding_backend=embedding_backend)
        self.embeddings = self.index_manager.embedding_backend
        self.answer_synthesizer = OpenAIAnswerSynthesizer()
        self.summary_map = load_summary_map(self._structured_document_roots())
        self._document_payload_cache: dict[str, dict[str, Any] | None] = {}
        self._document_payload_path_cache: dict[str, Path | None] = {}
        self._page_table_presence_cache: dict[tuple[str, int], bool] = {}

    def search(
        self,
        query: str,
        strategy: str = "semantic",
        top_k: int = 5,
        *,
        document_id: str = "",
        source_name: str = "",
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        collection = (
            self.index_manager.rule_collection if strategy == "rule_based" else self.index_manager.semantic_collection
        )
        scoped_ids = {item for item in (document_ids or []) if item}
        if scoped_ids:
            document_id = ""
            source_name = ""
        require_filter = bool(scoped_ids or document_id or source_name)
        if require_filter:
            return self._search_structured_documents(
                query=query,
                strategy=strategy,
                top_k=top_k,
                document_id=document_id,
                source_name=source_name,
                document_ids=document_ids,
            )
        n_results = top_k

        try:
            query_embedding = self.embeddings.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
        except Exception:
            return self._search_structured_documents(
                query=query,
                strategy=strategy,
                top_k=top_k,
                document_id=document_id,
                source_name=source_name,
                document_ids=document_ids,
            )

        matches: list[dict[str, Any]] = []
        for index, matched_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][index]
            match_document_id = str(metadata.get("document_id", ""))
            match_source_name = str(metadata.get("source_name", ""))
            if scoped_ids and match_document_id not in scoped_ids:
                continue
            if document_id and match_document_id != document_id:
                continue
            if source_name and match_source_name and match_source_name != source_name:
                continue
            matches.append(
                {
                    "id": matched_id,
                    "document": results.get("documents", [[]])[0][index],
                    "distance": results.get("distances", [[]])[0][index],
                    "metadata": metadata,
                }
            )
        matches.sort(key=lambda item: self._match_sort_key(query, item))
        if len(matches) >= top_k:
            return matches[:top_k]
        if matches:
            return matches
        return self._search_structured_documents(
            query=query,
            strategy=strategy,
            top_k=top_k,
            document_id=document_id,
            source_name=source_name,
            document_ids=document_ids,
        )

    def answer_question(
        self,
        query: str,
        strategy: str = "semantic",
        top_k: int = 5,
        *,
        document_id: str = "",
        source_name: str = "",
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            matches = self.search(
                query=query,
                strategy=strategy,
                top_k=top_k,
                document_id=document_id,
                source_name=source_name,
                document_ids=document_ids,
            )
        except Exception as error:
            return {
                "query": query,
                "strategy": strategy,
                "answer": "질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "matches": [],
                "document_summaries": [],
                "citations": [],
                "evidence": [],
                "used_model": None,
                "openai_enabled": self.answer_synthesizer.is_enabled(),
                "warning": str(error),
                "document_id": document_id or None,
                "source_name": source_name or None,
                "document_ids": document_ids or [],
            }
        if not matches:
            return {
                "query": query,
                "strategy": strategy,
                "answer": "질문과 연결되는 문서를 아직 찾지 못했습니다. 먼저 문서를 업로드하거나 다른 표현으로 질문해 주세요.",
                "matches": [],
                "citations": [],
                "used_model": None,
                "document_id": document_id or None,
                "source_name": source_name or None,
                "document_ids": document_ids or [],
            }

        ranked_sentences = self._rank_sentences(query, matches)
        evidence = self._build_evidence(matches, ranked_sentences)
        answer_payload: dict[str, Any]
        if self.answer_synthesizer.is_enabled():
            try:
                answer_payload = self.answer_synthesizer.answer(query=query, evidence=evidence)
            except Exception as error:
                answer_payload = self._fallback_answer(matches, ranked_sentences)
                answer_payload["warning"] = f"OpenAI fallback used: {error}"
        else:
            answer_payload = self._fallback_answer(matches, ranked_sentences)

        normalized_citations = self._normalize_citations(
            answer_payload.get("citations", []),
            query=query,
            evidence=evidence,
            matches=matches,
        )
        normalized_citations = self._attach_supporting_assets(query, normalized_citations)
        rewritten_answer = self._rewrite_answer_citation_markers(
            answer_payload.get("answer", ""),
            raw_citations=answer_payload.get("citations", []),
            normalized_citations=normalized_citations,
        )

        return {
            "query": query,
            "strategy": strategy,
            "answer": rewritten_answer,
            "matches": matches,
            "document_summaries": self._build_document_summaries(matches),
            "citations": normalized_citations,
            "evidence": ranked_sentences[:5],
            "used_model": load_openai_settings().model if self.answer_synthesizer.is_enabled() else None,
            "openai_enabled": self.answer_synthesizer.is_enabled(),
            "warning": answer_payload.get("warning"),
            "document_id": document_id or None,
            "source_name": source_name or None,
            "document_ids": document_ids or [],
        }

    def close(self) -> None:
        self.index_manager.close()

    def _rank_sentences(self, query: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        candidates: list[dict[str, Any]] = []
        for match in matches:
            distance = float(match.get("distance", 0.0) or 0.0)
            metadata = match.get("metadata", {}) or {}
            for sentence in self._split_sentences(match.get("document", "")):
                sentence_tokens = self._tokens(sentence)
                score = self._score_sentence_relevance(
                    query=query,
                    sentence=sentence,
                    match=match,
                    distance=distance,
                )
                candidates.append(
                    {
                        "score": score,
                        "sentence": sentence,
                        "document": match.get("document", "") or "",
                        "metadata": metadata,
                        "distance": distance,
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def _search_structured_documents(
        self,
        *,
        query: str,
        strategy: str,
        top_k: int,
        document_id: str = "",
        source_name: str = "",
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        scoped_ids = {item for item in (document_ids or []) if item}
        if scoped_ids:
            document_id = ""
            source_name = ""
        scored: list[dict[str, Any]] = []
        query_tokens = self._tokens(query)
        candidate_payloads: list[dict[str, Any]] = []
        if scoped_ids:
            for scoped_document_id in sorted(scoped_ids):
                payload = self._load_document_payload(scoped_document_id)
                if payload:
                    candidate_payloads.append(payload)
        elif document_id:
            payload = self._load_document_payload(document_id)
            if payload:
                candidate_payloads.append(payload)
        else:
            for root in self._structured_document_roots():
                if not root.exists():
                    continue
                for path in sorted(root.glob("*.json")):
                    try:
                        payload = read_text_with_fallback(path)[0]
                    except OSError:
                        continue
                    try:
                        candidate_payloads.append(json.loads(payload))
                    except Exception:
                        continue

        for data in candidate_payloads:
            current_document_id = str(data.get("document_id", ""))
            current_source_name = str(data.get("source_name", ""))
            if source_name and current_source_name and current_source_name != source_name:
                continue

            chunks = []
            if strategy == "semantic":
                chunks = data.get("semantic_chunks") or []
            if not chunks:
                chunks = data.get("chunks") or data.get("semantic_chunks") or []

            for index, chunk in enumerate(chunks):
                if "text" in chunk:
                    chunk_text = str(chunk.get("text", ""))
                    chunk_index = chunk.get("chunk_index", index)
                    section_hint = str(chunk.get("section_hint", "")) or None
                    page_number = (
                        chunk.get("asset_page_number")
                        if chunk.get("asset_page_number") is not None
                        else self._extract_page_number(chunk.get("section_hint"), chunk_text)
                    )
                else:
                    chunk_text = str(chunk.get("serialized_text", ""))
                    chunk_index = index
                    section_hint = str(chunk.get("section") or "") or None
                    page_number = chunk.get("page")
                if not chunk_text.strip():
                    continue
                overlap = len(query_tokens & self._tokens(chunk_text))
                scored.append(
                    {
                        "id": f"{current_document_id}:structured:{chunk_index}",
                        "document": chunk_text,
                        "distance": float(max(0, 100 - overlap)),
                        "metadata": {
                            "document_id": current_document_id,
                            "source_name": current_source_name,
                            "section_hint": section_hint or (f"Page {page_number}" if page_number else ""),
                            "page_number": page_number,
                            "asset_page_number": chunk.get("asset_page_number"),
                            "chunk_index": chunk_index,
                            "strategy": "structured_fallback",
                        },
                    }
                )

        scored.sort(key=lambda item: self._match_sort_key(query, item))
        return scored[:top_k]

    def _match_sort_key(self, query: str, match: dict[str, Any]) -> tuple[float, float]:
        metadata = match.get("metadata", {}) or {}
        query_tokens = self._tokens(query)
        intent_tokens = self._query_intent_tokens(query)
        source_tokens = self._tokens(str(metadata.get("source_name") or ""))
        section_tokens = self._tokens(str(metadata.get("section_hint") or ""))
        document_text = str(match.get("document") or "")
        document_tokens = self._tokens(document_text)
        overlap = (
            len(query_tokens & source_tokens) * 6
            + len(query_tokens & section_tokens) * 3
            + len(query_tokens & document_tokens)
        )
        intent_overlap = (
            self._count_intent_matches(intent_tokens, str(metadata.get("source_name") or "")) * 7
            + self._count_intent_matches(intent_tokens, str(metadata.get("section_hint") or "")) * 12
            + self._count_intent_matches(intent_tokens, document_text) * 5
        )
        overlap += intent_overlap
        numeric_overlap = len(self._extract_numeric_tokens(query) & self._extract_numeric_tokens(str(match.get("document") or "")))
        temporal_overlap = len(self._extract_temporal_tokens(query) & self._extract_temporal_tokens(str(match.get("document") or "")))
        overlap += numeric_overlap * 8 + temporal_overlap * 6
        if self._query_prefers_tabular_asset(query) and (not intent_tokens or intent_overlap > 0):
            overlap += self._table_query_match_bonus(query=query, match=match)
        elif intent_tokens and intent_overlap <= 0:
            overlap -= 12
        distance = float(match.get("distance", 9999.0) or 9999.0)
        return (-overlap, distance)

    def _score_sentence_relevance(
        self,
        *,
        query: str,
        sentence: str,
        match: dict[str, Any],
        distance: float,
    ) -> float:
        metadata = match.get("metadata", {}) or {}
        query_tokens = self._tokens(query)
        sentence_tokens = self._tokens(sentence)
        section_hint = str(metadata.get("section_hint") or "")
        section_tokens = self._tokens(section_hint)

        score = len(query_tokens & sentence_tokens) * 10.0
        score += len(query_tokens & section_tokens) * 6.0
        score += len(self._extract_numeric_tokens(query) & self._extract_numeric_tokens(sentence)) * 12.0
        score += len(self._extract_temporal_tokens(query) & self._extract_temporal_tokens(sentence)) * 10.0
        score -= distance

        if self._query_prefers_tabular_asset(query):
            score += self._table_query_sentence_bonus(
                query=query,
                sentence=sentence,
                match=match,
            )
        return score

    def _table_query_match_bonus(self, *, query: str, match: dict[str, Any]) -> float:
        metadata = match.get("metadata", {}) or {}
        document_text = str(match.get("document") or "")
        section_hint = str(metadata.get("section_hint") or "")
        score = 0.0

        if self._text_has_table_structure(document_text):
            score += 12.0
        if self._looks_like_table_section(section_hint):
            score += 8.0
        if self._page_contains_table(
            document_id=str(metadata.get("document_id") or ""),
            page_number=self._safe_int(self._metadata_page_number(metadata)),
        ):
            score += 10.0

        query_tokens = self._tokens(query)
        section_tokens = self._tokens(section_hint)
        score += len(query_tokens & section_tokens) * 3.0
        return score

    def _table_query_sentence_bonus(self, *, query: str, sentence: str, match: dict[str, Any]) -> float:
        metadata = match.get("metadata", {}) or {}
        section_hint = str(metadata.get("section_hint") or "")
        score = 0.0
        query_tokens = self._tokens(query)
        sentence_tokens = self._tokens(sentence)
        header_like_tokens = {"성장률", "비율", "증감", "달성률", "개최일자", "참석인원", "항목", "실적", "계획"}

        if self._text_has_table_structure(sentence):
            score += 16.0
        if self._looks_like_table_section(section_hint):
            score += 10.0
        if self._page_contains_table(
            document_id=str(metadata.get("document_id") or ""),
            page_number=self._safe_int(self._metadata_page_number(metadata)),
        ):
            score += 12.0
        score += len((query_tokens & header_like_tokens) & sentence_tokens) * 8.0
        score += len((query_tokens & header_like_tokens) & self._tokens(section_hint)) * 6.0
        return score

    def _build_evidence(self, matches: list[dict[str, Any]], ranked_sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        used_keys: set[tuple[str, str]] = set()

        for item in ranked_sentences[:5]:
            metadata = item.get("metadata", {})
            key = (str(metadata.get("document_id", "")), item["sentence"])
            if key in used_keys:
                continue
            used_keys.add(key)
            resolved_page_number = self._lookup_page_number(
                str(metadata.get("document_id", "")),
                item["sentence"],
                item.get("document", "") or "",
            )
            page_number = resolved_page_number or self._metadata_page_number(metadata)
            evidence.append(
                {
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "page_number": page_number,
                    "chunk_index": metadata.get("chunk_index"),
                    "excerpt": self._format_quote_text(item.get("document", "") or item["sentence"]),
                    "highlight_text": self._format_quote_text(item["sentence"]),
                }
            )

        if evidence:
            return evidence

        for match in matches[:3]:
            metadata = match.get("metadata", {})
            resolved_page_number = self._lookup_page_number(
                str(metadata.get("document_id", "")),
                match.get("document", "") or "",
            )
            page_number = resolved_page_number or self._metadata_page_number(metadata)
            evidence.append(
                {
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "page_number": page_number,
                    "chunk_index": metadata.get("chunk_index"),
                    "excerpt": self._format_quote_text(match.get("document", "") or ""),
                    "highlight_text": "",
                }
            )
        return evidence

    def _lookup_page_number(self, document_id: str, *text_candidates: str) -> int | None:
        """structured JSON의 페이지 요소/markdown/chunks에서 가장 유사한 실제 페이지 번호를 반환."""
        if not document_id:
            return None
        normalized_candidates = [
            self._format_quote_text(text)
            for text in text_candidates
            if self._format_quote_text(text)
        ]
        if not normalized_candidates:
            return None

        primary_text = normalized_candidates[0]
        primary_tokens = self._tokens(primary_text)
        if not primary_tokens:
            return None

        payload = self._load_document_payload(document_id)
        if not payload:
            return None

        best_page: int | None = None
        best_score = -1.0
        for page_number, page_text in self._extract_page_lookup_texts(payload):
            score = self._score_lookup_candidate(
                page_text,
                primary_text=primary_text,
                primary_tokens=primary_tokens,
                additional_candidates=normalized_candidates[1:],
                exact_boost=1600.0,
                partial_boost=1000.0,
                candidate_boost=240.0,
                overlap_weight=18.0,
            )
            if score > best_score:
                best_score = score
                best_page = page_number

        for page_number, page_text in self._extract_markdown_pages(payload):
            score = self._score_lookup_candidate(
                page_text,
                primary_text=primary_text,
                primary_tokens=primary_tokens,
                additional_candidates=normalized_candidates[1:],
                exact_boost=1500.0,
                partial_boost=900.0,
                candidate_boost=220.0,
                overlap_weight=10.0,
            )
            if score > best_score:
                best_score = score
                best_page = page_number

        if best_score < 800.0:
            for item in payload.get("chunks", []):
                page = item.get("asset_page_number") or item.get("page") or item.get("page_number")
                if not page:
                    continue
                text = self._format_quote_text(str(item.get("serialized_text") or item.get("text") or ""))
                if not text:
                    continue
                score = self._score_lookup_candidate(
                    text,
                    primary_text=primary_text,
                    primary_tokens=primary_tokens,
                    additional_candidates=normalized_candidates[1:],
                    exact_boost=1000.0,
                    partial_boost=700.0,
                    candidate_boost=180.0,
                    overlap_weight=12.0,
                )
                if score > best_score:
                    best_score = score
                    try:
                        best_page = int(page)
                    except (TypeError, ValueError):
                        pass
        return best_page if best_score > 0 else None

    def _extract_page_lookup_texts(self, payload: dict[str, Any]) -> list[tuple[int, str]]:
        pages = self._get_asset_pages(payload)
        collected: list[tuple[int, str]] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            try:
                page_number = int(page.get("page_number") or 0)
            except (TypeError, ValueError):
                page_number = 0
            if page_number <= 0:
                continue

            parts: list[str] = []
            page_text = self._format_quote_text(str(page.get("text") or ""))
            if page_text:
                parts.append(page_text)

            elements = page.get("elements") if isinstance(page.get("elements"), list) else []
            for element in elements:
                if not isinstance(element, dict):
                    continue
                metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
                for candidate in (
                    element.get("text"),
                    element.get("markdown"),
                    metadata.get("mcid_text"),
                ):
                    normalized = self._format_quote_text(str(candidate or ""))
                    if normalized:
                        parts.append(normalized)

            combined = self._format_quote_text("\n".join(parts))
            if combined:
                collected.append((page_number, combined))
        return collected

    def _score_lookup_candidate(
        self,
        candidate_text: str,
        *,
        primary_text: str,
        primary_tokens: set[str],
        additional_candidates: list[str],
        exact_boost: float,
        partial_boost: float,
        candidate_boost: float,
        overlap_weight: float,
    ) -> float:
        text_tokens = self._tokens(candidate_text)
        if not text_tokens:
            return 0.0

        score = 0.0
        if primary_text and primary_text in candidate_text:
            score += exact_boost
        elif primary_text and candidate_text in primary_text:
            score += partial_boost

        for candidate in additional_candidates:
            if candidate and candidate in candidate_text:
                score += candidate_boost

        overlap = len(primary_tokens & text_tokens)
        if overlap:
            score += overlap * overlap_weight
        return score

    def _extract_markdown_pages(self, payload: dict[str, Any]) -> list[tuple[int, str]]:
        markdown = self._format_quote_text(
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
            page_text = self._format_quote_text(markdown[start:end])
            if page_text:
                pages.append((page_number, page_text))
        return pages

    def _fallback_answer(self, matches: list[dict[str, Any]], ranked_sentences: list[dict[str, Any]]) -> dict[str, Any]:
        answer_sentences = [item["sentence"] for item in ranked_sentences[:3]]
        citations = []
        for item in ranked_sentences[:3]:
            metadata = item.get("metadata", {})
            resolved_page_number = self._lookup_page_number(
                str(metadata.get("document_id", "")),
                item["sentence"],
                item.get("document", "") or "",
            )
            page_number = resolved_page_number or metadata.get("page_number")
            if page_number is None:
                page_number = self._metadata_page_number(metadata)
            citations.append(
                {
                    "source_number": len(citations) + 1,
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "page_number": page_number,
                    "chunk_index": metadata.get("chunk_index"),
                    "quote": self._format_quote_text(item.get("document", "") or item["sentence"]),
                    "highlight_text": self._format_quote_text(item["sentence"]),
                }
            )
        if not answer_sentences:
            answer_sentences = [self._format_quote_text(matches[0]["document"])]
            metadata = matches[0].get("metadata", {})
            resolved_page_number = self._lookup_page_number(
                str(metadata.get("document_id", "")),
                answer_sentences[0],
                matches[0].get("document", "") or "",
            )
            citations = [
                {
                    "source_number": 1,
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "page_number": resolved_page_number or self._metadata_page_number(metadata),
                    "chunk_index": metadata.get("chunk_index"),
                    "quote": answer_sentences[0],
                    "highlight_text": answer_sentences[0],
                }
            ]
        return {
            "answer": " ".join(answer_sentences).strip(),
            "citations": citations,
        }

    def _resolve_citation_match(
        self,
        *,
        citation: dict[str, Any],
        evidence_item: dict[str, Any],
        full_quote: str,
        matches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        target_document_id = str(citation.get("document_id") or evidence_item.get("document_id") or "")
        target_source_name = str(citation.get("source_name") or evidence_item.get("source_name") or "")
        normalized_quote = self._format_quote_text(full_quote)

        best_match: dict[str, Any] | None = None
        best_score = -1.0

        for match in matches:
            metadata = match.get("metadata", {}) or {}
            match_document_id = str(metadata.get("document_id") or "")
            match_source_name = str(metadata.get("source_name") or "")
            if target_document_id and match_document_id != target_document_id:
                continue
            if target_source_name and match_source_name and match_source_name != target_source_name:
                continue

            match_text = self._format_quote_text(match.get("document", "") or "")
            if not match_text:
                continue

            score = 0.0
            if normalized_quote and self._contains_normalized(match_text, normalized_quote):
                score += 1000.0
            elif normalized_quote and self._contains_normalized(normalized_quote, match_text):
                score += 600.0

            citation_chunk_index = citation.get("chunk_index")
            try:
                normalized_chunk_index = int(citation_chunk_index) if citation_chunk_index is not None else None
            except (TypeError, ValueError):
                normalized_chunk_index = None
            if normalized_chunk_index is not None and metadata.get("chunk_index") == normalized_chunk_index:
                score += 400.0

            distance = float(match.get("distance", 0.0) or 0.0)
            score += max(0.0, 100.0 - distance)

            if score > best_score:
                best_score = score
                best_match = match

        return best_match or {}

    def _normalize_citations(
        self,
        citations: list[dict[str, Any]],
        *,
        query: str,
        evidence: list[dict[str, Any]],
        matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evidence_by_number = {index + 1: item for index, item in enumerate(evidence)}
        normalized: list[dict[str, Any]] = []
        for index, citation in enumerate(citations or [], start=1):
            source_number = int(citation.get("source_number") or index)
            evidence_item = evidence_by_number.get(source_number, {})
            if evidence_item:
                resolved_page_number = self._lookup_page_number(
                    str(evidence_item.get("document_id") or citation.get("document_id") or ""),
                    str(evidence_item.get("excerpt") or ""),
                    str(citation.get("quote") or ""),
                    str(evidence_item.get("highlight_text") or ""),
                )
                normalized.append(
                    {
                        "source_number": source_number,
                        "source_name": evidence_item.get("source_name") or "",
                        "document_id": evidence_item.get("document_id") or "",
                        "section_hint": evidence_item.get("section_hint") or "",
                        "quote": self._format_quote_text(
                            evidence_item.get("excerpt") or citation.get("quote") or ""
                        ),
                        "highlight_text": self._best_highlight_for_chunk(
                            query,
                            citation.get("quote") or "",
                            evidence_item.get("highlight_text") or "",
                            evidence_item.get("excerpt") or "",
                        ),
                        "chunk_text": self._format_quote_text(evidence_item.get("excerpt") or ""),
                        "chunk_index": evidence_item.get("chunk_index"),
                        "page_number": (
                            resolved_page_number
                            or evidence_item.get("page_number")
                            or evidence_item.get("asset_page_number")
                            or citation.get("page_number")
                        ),
                        "chunk_strategy": evidence_item.get("chunk_strategy") or "",
                    }
                )
                continue
            full_quote = self._format_quote_text(
                evidence_item.get("excerpt") or citation.get("quote") or ""
            )
            resolved_match = self._resolve_citation_match(
                citation=citation,
                evidence_item=evidence_item,
                full_quote=full_quote,
                matches=matches,
            )
            resolved_metadata = resolved_match.get("metadata", {}) or {}
            chunk_text = self._format_quote_text(
                resolved_match.get("document", "") or full_quote
            )
            highlight_text = self._best_highlight_for_chunk(
                query,
                full_quote,
                citation.get("highlight_text") or "",
                citation.get("quote") or "",
                evidence_item.get("highlight_text") or "",
            )
            resolved_page_number = self._lookup_page_number(
                str(citation.get("document_id") or evidence_item.get("document_id") or ""),
                chunk_text,
                full_quote,
                highlight_text,
            )
            normalized.append(
                {
                    "source_number": source_number,
                    "source_name": citation.get("source_name") or evidence_item.get("source_name") or "",
                    "document_id": citation.get("document_id") or evidence_item.get("document_id") or "",
                    "section_hint": citation.get("section_hint") or evidence_item.get("section_hint") or "",
                    "quote": full_quote,
                    "highlight_text": highlight_text,
                    "chunk_text": chunk_text,
                    "chunk_index": citation.get("chunk_index") if citation.get("chunk_index") is not None else resolved_metadata.get("chunk_index"),
                    "page_number": (
                        resolved_page_number
                        if resolved_page_number is not None
                        else evidence_item.get("page_number")
                        if evidence_item.get("page_number") is not None
                        else resolved_metadata.get("page_number")
                        if resolved_metadata.get("page_number") is not None
                        else resolved_metadata.get("asset_page_number")
                        if resolved_metadata.get("asset_page_number") is not None
                        else citation.get("page_number")
                    ),
                    "chunk_strategy": resolved_metadata.get("strategy") or "",
                }
            )
        if normalized:
            return self._dedupe_citations(normalized)

        fallback: list[dict[str, Any]] = []
        for index, match in enumerate(matches[:3], start=1):
            metadata = match.get("metadata", {})
            fallback.append(
                {
                    "source_number": index,
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "quote": self._format_quote_text(match.get("document", "") or ""),
                    "highlight_text": "",
                    "chunk_text": self._format_quote_text(match.get("document", "") or ""),
                    "chunk_index": metadata.get("chunk_index"),
                    "page_number": self._metadata_page_number(metadata),
                    "chunk_strategy": metadata.get("strategy") or "",
                }
            )
        return self._dedupe_citations(fallback)

    def _metadata_page_number(self, metadata: dict[str, Any]) -> int | str | None:
        asset_page_number = metadata.get("asset_page_number")
        if asset_page_number not in (None, ""):
            return asset_page_number
        return metadata.get("page_number")

    def _dedupe_citations(self, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str, str]] = set()
        for citation in citations:
            key = (
                str(citation.get("document_id") or ""),
                str(citation.get("source_name") or ""),
                str(citation.get("page_number") or ""),
                str(citation.get("chunk_index") or ""),
                self._format_quote_text(citation.get("chunk_text") or citation.get("quote") or citation.get("highlight_text") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(dict(citation))

        for index, citation in enumerate(deduped, start=1):
            citation["source_number"] = index
        return deduped

    def _attach_supporting_assets(self, query: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for citation in citations:
            updated = dict(citation)
            debug_payload = self._collect_supporting_asset_candidates(query, updated, include_debug=True)
            updated["supporting_assets"] = [dict(asset) for asset in debug_payload.get("selected_assets", [])]
            updated["supporting_asset_debug"] = debug_payload
            enriched.append(updated)
        return enriched

    def debug_supporting_asset_candidates(self, query: str, citation: dict[str, Any]) -> dict[str, Any]:
        return self._collect_supporting_asset_candidates(query, citation, include_debug=True)

    def _build_supporting_assets_for_citation(self, query: str, citation: dict[str, Any]) -> list[dict[str, Any]]:
        debug_payload = self._collect_supporting_asset_candidates(query, citation, include_debug=False)
        return [dict(asset) for asset in debug_payload.get("selected_assets", [])]

    def _collect_supporting_asset_candidates(
        self,
        query: str,
        citation: dict[str, Any],
        *,
        include_debug: bool,
    ) -> dict[str, Any]:
        document_id = str(citation.get("document_id") or "").strip()
        query_tokens = self._tokens(query)
        signals = self._extract_supporting_asset_signals(query=query, citation=citation)
        focus_tokens = signals["tokens"]
        prefers_visual = self._query_prefers_visual_asset(query)
        prefers_table = self._query_prefers_tabular_asset(query)
        synthetic_table_candidate = self._build_synthetic_table_asset(
            citation=citation,
            query_tokens=query_tokens,
            focus_tokens=focus_tokens,
            signals=signals,
        )
        if not document_id:
            selected_assets = [dict(synthetic_table_candidate[1])] if synthetic_table_candidate else []
            return {
                "document_id": document_id,
                "resolved_page_number": 0,
                "prefers_visual": prefers_visual,
                "prefers_table": prefers_table,
                "signals": self._summarize_supporting_asset_signals(signals) if include_debug else {},
                "table_candidates": (
                    [self._debug_candidate_entry(synthetic_table_candidate, "synthetic", include_debug=include_debug)]
                    if synthetic_table_candidate
                    else []
                ),
                "visual_candidates": [],
                "selected_assets": selected_assets,
            }
        try:
            page_number = int(citation.get("page_number") or 0)
        except (TypeError, ValueError):
            page_number = 0
        resolved_page_number = self._lookup_page_number(
            document_id,
            str(citation.get("chunk_text") or ""),
            str(citation.get("quote") or ""),
            str(citation.get("highlight_text") or ""),
        )
        if resolved_page_number:
            page_number = resolved_page_number
        if page_number <= 0:
            selected_assets = [dict(synthetic_table_candidate[1])] if synthetic_table_candidate else []
            return {
                "document_id": document_id,
                "resolved_page_number": page_number,
                "prefers_visual": prefers_visual,
                "prefers_table": prefers_table,
                "signals": self._summarize_supporting_asset_signals(signals) if include_debug else {},
                "table_candidates": (
                    [self._debug_candidate_entry(synthetic_table_candidate, "synthetic", include_debug=include_debug)]
                    if synthetic_table_candidate
                    else []
                ),
                "visual_candidates": [],
                "selected_assets": selected_assets,
            }

        page_elements = self._load_page_elements(document_id=document_id, page_number=page_number)

        table_candidates: list[tuple[float, dict[str, Any]]] = []
        visual_candidates: list[tuple[float, dict[str, Any]]] = []
        if synthetic_table_candidate:
            table_candidates.append(synthetic_table_candidate)
        if page_elements:
            page_region_candidate = self._build_page_region_asset(
                query=query,
                citation=citation,
                page_elements=page_elements,
                page_number=page_number,
            )
            if page_region_candidate and self._asset_supports_citation(
                asset_text="\n".join(
                    [
                        str(page_region_candidate[1].get("title") or ""),
                        str(page_region_candidate[1].get("text") or ""),
                    ]
                ),
                asset_type="page_region",
                signals=signals,
            ):
                visual_candidates.append(page_region_candidate)
        if prefers_visual:
            document_visual_candidate = self._find_document_level_visual_candidate(
                document_id=document_id,
                query=query,
                citation=citation,
            )
            if document_visual_candidate is not None and self._asset_supports_citation(
                asset_text="\n".join(
                    [
                        str(document_visual_candidate[1].get("title") or ""),
                        str(document_visual_candidate[1].get("text") or ""),
                    ]
                ),
                asset_type="page_region",
                signals=signals,
                ):
                visual_candidates.append(document_visual_candidate)
        if page_elements:
            for element in page_elements:
                element_type = str(element.get("element_type") or "")
                if element_type not in {"table", "image"}:
                    continue
                if element_type == "image" and not self._is_meaningful_image_element(element):
                    continue

                title = self._build_element_title(page_elements, element)
                candidate_text = self._build_element_search_text(element, title)
                if not self._asset_supports_citation(
                    asset_text=candidate_text,
                    asset_type=element_type,
                    signals=signals,
                ):
                    continue
                bbox = element.get("bbox") if isinstance(element.get("bbox"), list) else None
                score = self._score_supporting_asset(
                    element_type=element_type,
                    candidate_text=candidate_text,
                    bbox=bbox,
                    query_tokens=query_tokens,
                    focus_tokens=focus_tokens,
                    signals=signals,
                    citation_section_hint=str(citation.get("section_hint") or ""),
                    candidate_title=title,
                    citation_page_number=page_number,
                    candidate_page_number=int(element.get("page_number") or page_number),
                    document_level=False,
                )
                if element_type == "image" and score <= 1.0:
                    continue

                metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
                asset = {
                    "type": element_type,
                    "element_id": str(element.get("element_id") or ""),
                    "page_number": int(element.get("page_number") or page_number),
                    "bbox": bbox,
                    "title": title,
                    "text": self._truncate_text(candidate_text, max_characters=220),
                    "_debug": self._build_supporting_asset_score_debug(
                        element_type=element_type,
                        candidate_text=candidate_text,
                        bbox=bbox,
                        query_tokens=query_tokens,
                        focus_tokens=focus_tokens,
                        signals=signals,
                        citation_section_hint=str(citation.get("section_hint") or ""),
                        candidate_title=title,
                        citation_page_number=page_number,
                        candidate_page_number=int(element.get("page_number") or page_number),
                        document_level=False,
                    ),
                }
                if element_type == "table":
                    asset["markdown"] = self._truncate_table_markdown(str(element.get("markdown") or ""))
                    asset["row_count"] = int(metadata.get("row_count") or 0)
                    asset["column_count"] = int(metadata.get("column_count") or 0)
                    table_candidates.append((score, asset))
                else:
                    asset["caption_text"] = self._truncate_text(
                        self._format_quote_text(str(element.get("text") or metadata.get("mcid_text") or title)),
                        max_characters=160,
                    )
                    visual_candidates.append((score, asset))

        if not table_candidates:
            document_table_candidate = self._find_document_level_table_candidate(
                document_id=document_id,
                query=query,
                citation=citation,
            )
            if document_table_candidate is not None:
                table_candidates.append(document_table_candidate)

        table_candidates.sort(key=lambda item: item[0], reverse=True)
        visual_candidates.sort(key=lambda item: item[0], reverse=True)

        selected_assets = self._select_supporting_assets(
            table_candidates=table_candidates,
            visual_candidates=visual_candidates,
            prefers_visual=prefers_visual,
            prefers_table=prefers_table,
        )
        return {
            "document_id": document_id,
            "resolved_page_number": page_number,
            "prefers_visual": prefers_visual,
            "prefers_table": prefers_table,
            "signals": self._summarize_supporting_asset_signals(signals) if include_debug else {},
            "table_candidates": self._debug_candidates(
                table_candidates,
                include_debug=include_debug,
                default_source="page",
            ),
            "visual_candidates": self._debug_candidates(
                visual_candidates,
                include_debug=include_debug,
                default_source="page",
            ),
            "selected_assets": [dict(asset) for asset in selected_assets],
        }

    def _select_supporting_assets(
        self,
        *,
        table_candidates: list[tuple[float, dict[str, Any]]],
        visual_candidates: list[tuple[float, dict[str, Any]]],
        prefers_visual: bool,
        prefers_table: bool,
    ) -> list[dict[str, Any]]:
        best_table = table_candidates[0] if table_candidates else None
        best_visual = visual_candidates[0] if visual_candidates else None
        second_table = table_candidates[1] if len(table_candidates) > 1 else None
        second_visual = visual_candidates[1] if len(visual_candidates) > 1 else None

        if best_table:
            min_table_score = 18.0 if best_table[1].get("synthetic") else 12.0
            if best_table[0] < min_table_score:
                best_table = None
            elif second_table is not None:
                min_margin = 5.0 if best_table[1].get("synthetic") else 4.0
                if (best_table[0] - second_table[0]) < min_margin:
                    best_table = None
        if best_visual and best_visual[0] < 9.0:
            best_visual = None
        elif best_visual and second_visual is not None and (best_visual[0] - second_visual[0]) < 3.0:
            best_visual = None

        if prefers_visual and best_visual:
            return [best_visual[1]]

        if prefers_table and best_table:
            selected = [best_table[1]]
            if (
                best_visual
                and best_table[1].get("synthetic")
                and best_visual[0] >= max(8.0, best_table[0] - 6.0)
            ):
                selected.append(best_visual[1])
            return selected[:2]

        if best_table and best_visual:
            if best_visual[0] >= best_table[0] + 8.0:
                return [best_visual[1]]
            if best_table[0] >= best_visual[0] + 10.0:
                selected = [best_table[1]]
                if best_table[1].get("synthetic") and best_visual[0] >= 8.0:
                    selected.append(best_visual[1])
                return selected[:2]

            selected: list[dict[str, Any]] = []
            if best_table[1].get("synthetic"):
                selected.extend([best_visual[1], best_table[1]])
            else:
                selected.extend([best_table[1], best_visual[1]])
            return selected[:2]

        if best_visual:
            return [best_visual[1]]
        if best_table:
            return [best_table[1]]
        return []

    def _find_document_level_table_candidate(
        self,
        *,
        document_id: str,
        query: str,
        citation: dict[str, Any],
    ) -> tuple[float, dict[str, Any]] | None:
        payload = self._load_document_payload(document_id)
        if not payload:
            return None

        query_tokens = self._tokens(query)
        signals = self._extract_supporting_asset_signals(query=query, citation=citation)
        focus_tokens = signals["tokens"]
        evidence_anchors = self._citation_anchor_tokens(citation)
        try:
            citation_page_number = int(citation.get("page_number") or 0)
        except (TypeError, ValueError):
            citation_page_number = 0
        best_candidate: tuple[float, dict[str, Any]] | None = None

        for page in self._get_asset_pages(payload):
            if not isinstance(page, dict):
                continue
            try:
                page_number = int(page.get("page_number") or 0)
            except (TypeError, ValueError):
                page_number = 0
            if page_number <= 0:
                continue
            if not evidence_anchors and citation_page_number > 0 and abs(page_number - citation_page_number) > 1:
                continue

            page_elements = page.get("elements") if isinstance(page.get("elements"), list) else []
            normalized_elements = [item for item in page_elements if isinstance(item, dict)]
            normalized_elements.sort(key=lambda item: int(item.get("order") or 0))
            if not normalized_elements:
                continue

            for element in normalized_elements:
                if str(element.get("element_type") or "") != "table":
                    continue
                title = self._build_element_title(normalized_elements, element)
                candidate_text = self._build_element_search_text(element, title)
                if evidence_anchors and not self._asset_matches_citation_anchors(evidence_anchors, candidate_text):
                    continue
                if not self._asset_supports_citation(
                    asset_text=candidate_text,
                    asset_type="table",
                    signals=signals,
                ):
                    continue
                score = self._score_supporting_asset(
                    element_type="table",
                    candidate_text=candidate_text,
                    bbox=element.get("bbox") if isinstance(element.get("bbox"), list) else None,
                    query_tokens=query_tokens,
                    focus_tokens=focus_tokens,
                    signals=signals,
                    citation_section_hint=str(citation.get("section_hint") or ""),
                    candidate_title=title,
                    citation_page_number=citation_page_number,
                    candidate_page_number=int(element.get("page_number") or page_number),
                    document_level=True,
                )
                metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
                asset = {
                    "type": "table",
                    "element_id": str(element.get("element_id") or ""),
                    "page_number": int(element.get("page_number") or page_number),
                    "bbox": element.get("bbox") if isinstance(element.get("bbox"), list) else None,
                    "title": title,
                    "text": self._truncate_text(candidate_text, max_characters=220),
                    "markdown": self._truncate_table_markdown(str(element.get("markdown") or "")),
                    "row_count": int(metadata.get("row_count") or 0),
                    "column_count": int(metadata.get("column_count") or 0),
                    "document_level": True,
                    "_debug": self._build_supporting_asset_score_debug(
                        element_type="table",
                        candidate_text=candidate_text,
                        bbox=element.get("bbox") if isinstance(element.get("bbox"), list) else None,
                        query_tokens=query_tokens,
                        focus_tokens=focus_tokens,
                        signals=signals,
                        citation_section_hint=str(citation.get("section_hint") or ""),
                        candidate_title=title,
                        citation_page_number=citation_page_number,
                        candidate_page_number=int(element.get("page_number") or page_number),
                        document_level=True,
                    ),
                }
                adjusted_score = score
                if best_candidate is None or adjusted_score > best_candidate[0]:
                    best_candidate = (adjusted_score, asset)
        return best_candidate

    def _debug_candidates(
        self,
        candidates: list[tuple[float, dict[str, Any]]],
        *,
        include_debug: bool,
        default_source: str,
    ) -> list[dict[str, Any]]:
        return [
            self._debug_candidate_entry(candidate, default_source, include_debug=include_debug)
            for candidate in candidates
        ]

    def _debug_candidate_entry(
        self,
        candidate: tuple[float, dict[str, Any]],
        default_source: str,
        *,
        include_debug: bool,
    ) -> dict[str, Any]:
        score, asset = candidate
        entry = {
            "score": round(float(score), 4),
            "asset": dict(asset),
            "candidate_source": self._infer_candidate_source(asset, default_source),
        }
        if include_debug:
            debug_payload = asset.get("_debug") if isinstance(asset.get("_debug"), dict) else {}
            if debug_payload:
                entry["debug"] = debug_payload
        return entry

    def _infer_candidate_source(self, asset: dict[str, Any], default_source: str) -> str:
        if asset.get("synthetic"):
            return "synthetic"
        if asset.get("document_level"):
            return "document_level"
        return default_source

    def _find_document_level_visual_candidate(
        self,
        *,
        document_id: str,
        query: str,
        citation: dict[str, Any],
    ) -> tuple[float, dict[str, Any]] | None:
        payload = self._load_document_payload(document_id)
        if not payload:
            return None

        best_candidate: tuple[float, dict[str, Any]] | None = None
        pages = self._get_asset_pages(payload)
        for page in pages:
            if not isinstance(page, dict):
                continue
            try:
                page_number = int(page.get("page_number") or 0)
            except (TypeError, ValueError):
                page_number = 0
            if page_number <= 0:
                continue
            page_elements = page.get("elements") if isinstance(page.get("elements"), list) else []
            normalized_elements = [item for item in page_elements if isinstance(item, dict)]
            normalized_elements.sort(key=lambda item: int(item.get("order") or 0))
            if not normalized_elements:
                continue

            page_region_candidate = self._build_page_region_asset(
                query=query,
                citation=citation,
                page_elements=normalized_elements,
                page_number=page_number,
            )
            if page_region_candidate is None:
                continue
            if best_candidate is None or page_region_candidate[0] > best_candidate[0]:
                best_candidate = page_region_candidate
        return best_candidate

    def _build_page_region_asset(
        self,
        *,
        query: str,
        citation: dict[str, Any],
        page_elements: list[dict[str, Any]],
        page_number: int,
    ) -> tuple[float, dict[str, Any]] | None:
        query_tokens = self._tokens(query)
        focus_tokens = self._tokens(
            "\n".join(
                [
                    str(citation.get("highlight_text") or ""),
                    str(citation.get("quote") or ""),
                    str(citation.get("chunk_text") or ""),
                ]
            )
        )
        fragmented_images = [
            element
            for element in page_elements
            if str(element.get("element_type") or "") == "image" and not self._is_meaningful_image_element(element)
        ]
        if len(fragmented_images) < 12:
            return None

        anchors: list[tuple[float, dict[str, Any], list[float], str]] = []
        for element in fragmented_images:
            bbox = self._normalize_bbox(element.get("bbox"))
            if bbox is None:
                continue
            title = self._build_element_title(page_elements, element)
            candidate_text = self._build_element_search_text(element, title)
            candidate_tokens = self._tokens(candidate_text)
            overlap = len(query_tokens & candidate_tokens) * 6.0 + len(focus_tokens & candidate_tokens) * 4.0
            if overlap <= 0:
                continue
            anchors.append((overlap, element, bbox, candidate_text))
        anchors.sort(key=lambda item: item[0], reverse=True)
        if not anchors:
            return None

        seed_bboxes = [item[2] for item in anchors[: min(8, len(anchors))]]
        seed_bbox = self._union_bboxes(seed_bboxes)
        if seed_bbox is None:
            return None

        region_elements = self._collect_neighboring_page_elements(page_elements, seed_bbox)
        region_bbox = self._union_bboxes(
            [bbox for element in region_elements if (bbox := self._normalize_bbox(element.get("bbox"))) is not None]
        )
        if region_bbox is None:
            return None

        anchor_texts = [
            self._truncate_text(item[3], max_characters=60)
            for item in anchors[:5]
            if self._is_meaningful_anchor_text(item[3])
        ]
        section_hint = str(citation.get("section_hint") or "").strip()
        title_seed = f"{section_hint} 페이지 근거" if section_hint else " / ".join(anchor_texts[:3]) or f"p.{page_number} 페이지 근거"
        title = self._truncate_text(title_seed, max_characters=90)
        asset = {
            "type": "page_region",
            "element_id": "",
            "page_number": page_number,
            "bbox": region_bbox,
            "preview_bbox": region_bbox,
            "title": title,
            "text": self._truncate_text("\n".join(anchor_texts[:4]), max_characters=180),
        }
        return anchors[0][0] + len(region_elements), asset

    def _build_synthetic_table_asset(
        self,
        *,
        citation: dict[str, Any],
        query_tokens: set[str],
        focus_tokens: set[str],
        signals: dict[str, Any],
    ) -> tuple[float, dict[str, Any]] | None:
        table_markdown, row_count, column_count = self._extract_markdown_table_from_citation(citation)
        if not table_markdown:
            return None

        candidate_text = self._format_quote_text(
            "\n".join(
                [
                    str(citation.get("section_hint") or ""),
                    str(citation.get("highlight_text") or ""),
                    str(citation.get("quote") or ""),
                    table_markdown,
                ]
            )
        )
        score = self._score_supporting_asset(
            element_type="table",
            candidate_text=candidate_text,
            bbox=None,
            query_tokens=query_tokens,
            focus_tokens=focus_tokens,
            signals=signals,
        ) + 6.0
        title = self._truncate_text(
            str(citation.get("section_hint") or "표 근거").strip() or "표 근거",
            max_characters=80,
        )
        asset = {
            "type": "table",
            "element_id": "",
            "page_number": int(citation.get("page_number") or 0),
            "bbox": None,
            "title": title,
            "text": self._truncate_text(candidate_text, max_characters=220),
            "markdown": table_markdown,
            "row_count": row_count,
            "column_count": column_count,
            "synthetic": True,
            "_debug": self._build_supporting_asset_score_debug(
                element_type="table",
                candidate_text=candidate_text,
                bbox=None,
                query_tokens=query_tokens,
                focus_tokens=focus_tokens,
                signals=signals,
            ),
        }
        if not self._asset_supports_citation(
            asset_text="\n".join([title, candidate_text, table_markdown]),
            asset_type="table",
            signals=signals,
            synthetic=True,
        ):
            return None
        return score, asset

    def _extract_markdown_table_from_citation(self, citation: dict[str, Any]) -> tuple[str, int, int]:
        source_text = ""
        for candidate in (
            citation.get("chunk_text"),
            citation.get("quote"),
            citation.get("highlight_text"),
        ):
            normalized = self._format_quote_text(str(candidate or ""))
            if normalized:
                source_text = normalized
                break
        if "|" not in source_text:
            return "", 0, 0

        lines = [line.strip() for line in source_text.splitlines() if line.strip()]
        header_cells: list[str] = []
        rows: list[list[str]] = []
        seen_rows: set[tuple[str, ...]] = set()
        for line in lines:
            normalized = re.sub(r"^\*+\s*", "", line).strip()
            if not normalized:
                continue
            if not header_cells and "|" in normalized and "series table" in normalized.lower():
                header_part = re.sub(r"\*+\[series table\]\*+", "", normalized, flags=re.IGNORECASE).strip(" :-")
                header_cells = [self._sanitize_markdown_table_cell(cell) for cell in header_part.split("|") if cell.strip()]
                continue
            if not header_cells and normalized.count("|") >= 3 and ":" not in normalized:
                header_cells = [self._sanitize_markdown_table_cell(cell) for cell in normalized.split("|") if cell.strip()]
                continue
            if header_cells and ":" in normalized and "|" in normalized:
                label, values_raw = normalized.split(":", 1)
                values = [self._sanitize_markdown_table_cell(cell) for cell in values_raw.split("|") if cell.strip()]
                if len(values) != len(header_cells):
                    continue
                label_text = self._sanitize_markdown_table_cell(label.strip("* ").strip())
                if not label_text:
                    continue
                row_key = tuple([label_text, *values])
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                rows.append(list(row_key))

        if not header_cells or len(rows) < 2:
            return "", 0, 0

        header = ["항목", *header_cells]
        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        markdown_lines.extend("| " + " | ".join(row) + " |" for row in rows[:10])
        return "\n".join(markdown_lines), len(rows), len(header)

    def _sanitize_markdown_table_cell(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").replace("|", " ").strip())

    def _is_meaningful_anchor_text(self, text: str) -> bool:
        normalized = self._format_quote_text(text)
        if len(normalized) < 2:
            return False
        if re.fullmatch(r"[\d.,%()\-\s]+", normalized):
            return False
        return True

    def _query_prefers_visual_asset(self, query: str) -> bool:
        normalized = str(query or "").lower()
        visual_phrases = (
            "이미지",
            "그림",
            "그래프",
            "차트",
            "도표",
            "캡처",
            "시각",
            "페이지",
            "슬라이드",
            "figure",
            "show me",
            "보여줘",
            "보여주",
        )
        return any(phrase in normalized for phrase in visual_phrases)

    def _query_prefers_tabular_asset(self, query: str) -> bool:
        normalized = str(query or "").lower()
        table_phrases = (
            "표",
            "테이블",
            "손익계산서",
            "재무제표",
            "수치",
            "숫자",
            "항목",
            "series table",
        )
        return any(phrase in normalized for phrase in table_phrases)

    def _text_has_table_structure(self, text: str) -> bool:
        normalized = self._format_quote_text(text)
        if not normalized:
            return False
        if "|" in normalized or "[표:" in normalized:
            return True
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        key_value_lines = 0
        for line in lines[:8]:
            if ":" in line:
                key_value_lines += 1
        return key_value_lines >= 2

    def _looks_like_table_section(self, section_hint: str) -> bool:
        normalized = self._format_quote_text(section_hint)
        if not normalized:
            return False
        tableish_terms = (
            "표",
            "현황",
            "실적",
            "비율",
            "재무상태표",
            "손익계산서",
            "추진 실적",
            "총회",
            "이사회",
        )
        return any(term in normalized for term in tableish_terms)

    def _page_contains_table(self, *, document_id: str, page_number: int) -> bool:
        if not document_id or page_number <= 0:
            return False
        cache_key = (document_id, page_number)
        if cache_key in self._page_table_presence_cache:
            return self._page_table_presence_cache[cache_key]
        page_elements = self._load_page_elements(document_id=document_id, page_number=page_number)
        has_table = any(str(element.get("element_type") or "") == "table" for element in page_elements)
        self._page_table_presence_cache[cache_key] = has_table
        return has_table

    def _load_page_elements(self, *, document_id: str, page_number: int) -> list[dict[str, Any]]:
        payload = self._load_document_payload(document_id)
        if not payload:
            return []
        pages = self._get_asset_pages(payload)
        for page in pages:
            if not isinstance(page, dict):
                continue
            try:
                current_page_number = int(page.get("page_number") or 0)
            except (TypeError, ValueError):
                current_page_number = 0
            if current_page_number != page_number:
                continue
            elements = page.get("elements") if isinstance(page.get("elements"), list) else []
            normalized = [item for item in elements if isinstance(item, dict)]
            normalized.sort(key=lambda item: int(item.get("order") or 0))
            return normalized
        return []

    def _get_asset_pages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        asset_pages = payload.get("asset_pages")
        if isinstance(asset_pages, list) and asset_pages:
            return [item for item in asset_pages if isinstance(item, dict)]
        pages = payload.get("pages")
        if isinstance(pages, list):
            return [item for item in pages if isinstance(item, dict)]
        return []

    def _load_document_payload(self, document_id: str) -> dict[str, Any] | None:
        normalized_id = str(document_id or "").strip()
        if not normalized_id:
            return None
        if normalized_id in self._document_payload_cache:
            return self._document_payload_cache[normalized_id]
        payload_path = self._find_document_payload_path(normalized_id)
        if payload_path is None:
            self._document_payload_cache[normalized_id] = None
            return None
        try:
            payload = json.loads(read_text_with_fallback(payload_path)[0])
        except Exception:
            payload = None
        self._document_payload_cache[normalized_id] = payload
        if payload is not None:
            self._document_payload_path_cache[normalized_id] = payload_path
            return payload
        self._document_payload_cache[normalized_id] = None
        return None

    def _find_document_payload_path(self, document_id: str) -> Path | None:
        normalized_id = str(document_id or "").strip()
        if not normalized_id:
            return None
        if normalized_id in self._document_payload_path_cache:
            return self._document_payload_path_cache[normalized_id]

        for root in self._structured_document_roots():
            if not root.exists():
                continue
            for path in root.glob("*.json"):
                try:
                    payload = json.loads(read_text_with_fallback(path)[0])
                except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if str(payload.get("document_id") or "").strip() != normalized_id:
                    continue
                self._document_payload_path_cache[normalized_id] = path
                return path

        self._document_payload_path_cache[normalized_id] = None
        return None

    def _build_element_title(self, page_elements: list[dict[str, Any]], element: dict[str, Any]) -> str:
        order = self._safe_int(element.get("order"))
        caption_text = self._find_nearby_caption(page_elements, target_order=order)
        if caption_text:
            return caption_text

        element_type = str(element.get("element_type") or "")
        if element_type == "table":
            heading_text = self._find_nearby_table_heading(page_elements, element=element)
            if heading_text:
                return heading_text

        metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
        text = self._format_quote_text(str(element.get("text") or metadata.get("mcid_text") or ""))
        if text:
            return self._truncate_text(text.splitlines()[0], max_characters=80)

        if element_type == "table":
            row_count = self._safe_int(metadata.get("row_count"))
            column_count = self._safe_int(metadata.get("column_count"))
            if row_count > 0 and column_count > 0:
                return f"표 {row_count}행 x {column_count}열"
            return "표"
        return "이미지"

    def _find_nearby_caption(self, page_elements: list[dict[str, Any]], *, target_order: int) -> str:
        best_text = ""
        best_gap = 9999
        for candidate in page_elements:
            candidate_text = self._format_quote_text(str(candidate.get("text") or ""))
            if not candidate_text:
                continue
            candidate_type = str(candidate.get("element_type") or "")
            if candidate_type != "caption" and not self._looks_like_caption(candidate_text):
                continue
            gap = abs(self._safe_int(candidate.get("order")) - target_order)
            if gap > 4 or gap >= best_gap:
                continue
            best_gap = gap
            best_text = candidate_text.splitlines()[0]
        return self._truncate_text(best_text, max_characters=100)

    def _find_nearby_table_heading(self, page_elements: list[dict[str, Any]], *, element: dict[str, Any]) -> str:
        target_bbox = self._normalize_bbox(element.get("bbox"))
        target_order = self._safe_int(element.get("order"))
        if target_bbox is None:
            return ""

        best_text = ""
        best_score = float("-inf")
        for candidate in page_elements:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("element_type") or "") != "text":
                continue

            candidate_order = self._safe_int(candidate.get("order"))
            if candidate_order >= target_order:
                continue
            order_gap = target_order - candidate_order
            if order_gap > 20:
                continue

            candidate_text = self._format_quote_text(str(candidate.get("text") or ""))
            if not candidate_text:
                continue
            first_line = candidate_text.splitlines()[0].strip()
            if not first_line:
                continue
            if "단위" in first_line:
                continue
            if len(first_line) > 40:
                continue
            if re.fullmatch(r"[\d\s,().:%-]+", first_line):
                continue

            candidate_bbox = self._normalize_bbox(candidate.get("bbox"))
            if candidate_bbox is None:
                continue
            vertical_gap = target_bbox[1] - candidate_bbox[3]
            if vertical_gap < -6.0 or vertical_gap > 90.0:
                continue
            horizontal_overlap = min(target_bbox[2], candidate_bbox[2]) - max(target_bbox[0], candidate_bbox[0])
            if horizontal_overlap < 30.0:
                continue

            score = 100.0 - vertical_gap - (order_gap * 6.0)
            if re.match(r"^[○●■□▪︎◦\-]", first_line):
                score += 18.0
            if any(token in first_line for token in ("표", "비율", "현황", "실적", "내역", "추이", "손익", "재무")):
                score += 8.0
            if score > best_score:
                best_score = score
                best_text = first_line
        return self._truncate_text(best_text, max_characters=100)

    def _looks_like_caption(self, text: str) -> bool:
        first_line = self._format_quote_text(text).splitlines()[0] if self._format_quote_text(text) else ""
        return bool(re.match(r"^(그림|figure|table|표)\s*\d*", first_line, flags=re.IGNORECASE))

    def _build_element_search_text(self, element: dict[str, Any], title: str) -> str:
        metadata = element.get("metadata") if isinstance(element.get("metadata"), dict) else {}
        parts = [
            title,
            str(element.get("text") or ""),
            str(metadata.get("mcid_text") or ""),
        ]
        if str(element.get("element_type") or "") == "table":
            parts.append(str(element.get("markdown") or ""))
        return self._format_quote_text("\n".join(part for part in parts if part))

    def _score_supporting_asset(
        self,
        *,
        element_type: str,
        candidate_text: str,
        bbox: list[float] | None,
        query_tokens: set[str],
        focus_tokens: set[str],
        signals: dict[str, Any],
        citation_section_hint: str = "",
        candidate_title: str = "",
        citation_page_number: int = 0,
        candidate_page_number: int = 0,
        document_level: bool = False,
    ) -> float:
        candidate_tokens = self._tokens(candidate_text)
        candidate_numeric_tokens = self._extract_numeric_tokens(candidate_text)
        candidate_temporal_tokens = self._extract_temporal_tokens(candidate_text)
        score = 1.0 if element_type == "table" else 0.5
        if element_type == "image":
            score += self._image_area_score(bbox)
        if candidate_tokens:
            score += len(query_tokens & candidate_tokens) * 2.0
            score += len(focus_tokens & candidate_tokens) * 7.0
        score += len(signals["numeric_tokens"] & candidate_numeric_tokens) * 10.0
        score += len(signals["temporal_tokens"] & candidate_temporal_tokens) * 8.0
        if element_type == "table" and ("|" in candidate_text or "[표:" in candidate_text):
            score += 4.0
        score += self._supporting_asset_context_bonus(
            citation_section_hint=citation_section_hint,
            candidate_title=candidate_title,
            citation_page_number=citation_page_number,
            candidate_page_number=candidate_page_number,
            document_level=document_level,
        )
        return score

    def _build_supporting_asset_score_debug(
        self,
        *,
        element_type: str,
        candidate_text: str,
        bbox: list[float] | None,
        query_tokens: set[str],
        focus_tokens: set[str],
        signals: dict[str, Any],
        citation_section_hint: str = "",
        candidate_title: str = "",
        citation_page_number: int = 0,
        candidate_page_number: int = 0,
        document_level: bool = False,
    ) -> dict[str, Any]:
        candidate_tokens = self._tokens(candidate_text)
        candidate_numeric_tokens = self._extract_numeric_tokens(candidate_text)
        candidate_temporal_tokens = self._extract_temporal_tokens(candidate_text)

        query_overlap = sorted(query_tokens & candidate_tokens)
        focus_overlap = sorted(focus_tokens & candidate_tokens)
        numeric_overlap = sorted(signals["numeric_tokens"] & candidate_numeric_tokens)
        temporal_overlap = sorted(signals["temporal_tokens"] & candidate_temporal_tokens)

        base_score = 1.0 if element_type == "table" else 0.5
        image_area_bonus = self._image_area_score(bbox) if element_type == "image" else 0.0
        query_score = len(query_overlap) * 2.0
        focus_score = len(focus_overlap) * 7.0
        numeric_score = len(numeric_overlap) * 10.0
        temporal_score = len(temporal_overlap) * 8.0
        structure_bonus = 4.0 if element_type == "table" and ("|" in candidate_text or "[표:" in candidate_text) else 0.0
        context_bonus = self._supporting_asset_context_bonus(
            citation_section_hint=citation_section_hint,
            candidate_title=candidate_title,
            citation_page_number=citation_page_number,
            candidate_page_number=candidate_page_number,
            document_level=document_level,
        )

        return {
            "element_type": element_type,
            "query_overlap_tokens": query_overlap,
            "focus_overlap_tokens": focus_overlap,
            "numeric_overlap_tokens": numeric_overlap,
            "temporal_overlap_tokens": temporal_overlap,
            "score_breakdown": {
                "base_score": base_score,
                "image_area_bonus": image_area_bonus,
                "query_overlap_score": query_score,
                "focus_overlap_score": focus_score,
                "numeric_overlap_score": numeric_score,
                "temporal_overlap_score": temporal_score,
                "table_structure_bonus": structure_bonus,
                "context_bonus": context_bonus,
                "final_score": round(
                    base_score
                    + image_area_bonus
                    + query_score
                    + focus_score
                    + numeric_score
                    + temporal_score
                    + structure_bonus
                    + context_bonus,
                    4,
                ),
            },
        }

    def _summarize_supporting_asset_signals(self, signals: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_text_preview": self._truncate_text(str(signals.get("source_text") or ""), max_characters=300),
            "tokens": sorted(signals.get("tokens") or []),
            "numeric_tokens": sorted(signals.get("numeric_tokens") or []),
            "temporal_tokens": sorted(signals.get("temporal_tokens") or []),
        }

    def _extract_supporting_asset_signals(self, *, query: str, citation: dict[str, Any]) -> dict[str, Any]:
        source_text = self._build_supporting_asset_focus_text(query=query, citation=citation)
        return {
            "source_text": source_text,
            "tokens": self._tokens(source_text),
            "numeric_tokens": self._extract_numeric_tokens(source_text),
            "temporal_tokens": self._extract_temporal_tokens(source_text),
        }

    def _build_supporting_asset_focus_text(self, *, query: str, citation: dict[str, Any]) -> str:
        section_hint = self._format_quote_text(str(citation.get("section_hint") or ""))
        highlight_text = self._format_quote_text(str(citation.get("highlight_text") or ""))
        quote_text = self._format_quote_text(str(citation.get("quote") or ""))
        chunk_text = self._format_quote_text(str(citation.get("chunk_text") or ""))

        query_tokens = self._tokens(query)
        section_tokens = self._tokens(section_hint)
        numeric_anchor_tokens = self._extract_numeric_tokens(highlight_text or quote_text)

        selected_segments: list[str] = []
        for seed in (section_hint, highlight_text):
            if seed and seed not in selected_segments:
                selected_segments.append(seed)

        focused_windows = self._extract_supporting_focus_windows(
            text="\n".join([quote_text, chunk_text]),
            anchor_terms=sorted((query_tokens | section_tokens), key=len, reverse=True),
        )
        for window in focused_windows:
            if window not in selected_segments:
                selected_segments.append(window)

        scored_segments: list[tuple[float, str]] = []
        for segment in self._split_supporting_focus_segments("\n".join([quote_text, chunk_text])):
            segment_tokens = self._tokens(segment)
            if not segment_tokens:
                continue
            token_overlap = len(query_tokens & segment_tokens)
            section_overlap = len(section_tokens & segment_tokens)
            numeric_overlap = len(numeric_anchor_tokens & self._extract_numeric_tokens(segment))
            score = token_overlap * 5.0 + section_overlap * 6.0 + numeric_overlap * 3.0
            if score <= 0:
                continue
            scored_segments.append((score, segment))
        scored_segments.sort(key=lambda item: item[0], reverse=True)

        for _, segment in scored_segments[:4]:
            if segment not in selected_segments:
                selected_segments.append(segment)

        if not selected_segments:
            return self._format_quote_text("\n".join([section_hint, highlight_text, quote_text, chunk_text]))
        return self._format_quote_text("\n".join(selected_segments))

    def _split_supporting_focus_segments(self, text: str) -> list[str]:
        normalized = self._format_quote_text(text)
        if not normalized:
            return []
        parts = re.split(r"(?:\n+|\s+##\s+)", normalized)
        return [segment for segment in (self._format_quote_text(part) for part in parts) if segment]

    def _extract_supporting_focus_windows(self, *, text: str, anchor_terms: list[str], window_size: int = 120) -> list[str]:
        normalized = self._format_quote_text(text)
        lowered = normalized.lower()
        snippets: list[str] = []
        seen_ranges: list[tuple[int, int]] = []
        for term in anchor_terms:
            normalized_term = str(term or "").strip().lower()
            if len(normalized_term) < 2:
                continue
            start_index = 0
            while True:
                hit = lowered.find(normalized_term, start_index)
                if hit < 0:
                    break
                left = max(0, hit - window_size)
                right = min(len(normalized), hit + len(normalized_term) + window_size)
                overlap = False
                for existing_left, existing_right in seen_ranges:
                    if not (right < existing_left or existing_right < left):
                        overlap = True
                        break
                if not overlap:
                    seen_ranges.append((left, right))
                    snippet = self._format_quote_text(normalized[left:right])
                    if snippet:
                        snippets.append(snippet)
                start_index = hit + len(normalized_term)
                if len(snippets) >= 4:
                    return snippets
        return snippets

    def _supporting_asset_context_bonus(
        self,
        *,
        citation_section_hint: str,
        candidate_title: str,
        citation_page_number: int,
        candidate_page_number: int,
        document_level: bool,
    ) -> float:
        bonus = 0.0
        normalized_section = self._format_quote_text(citation_section_hint)
        normalized_title = self._format_quote_text(candidate_title)
        section_tokens = self._tokens(normalized_section)
        title_tokens = self._tokens(normalized_title)

        if normalized_section and normalized_title:
            if normalized_section in normalized_title or normalized_title in normalized_section:
                bonus += 18.0
            bonus += min(4, len(section_tokens & title_tokens)) * 4.0

        if citation_page_number > 0 and candidate_page_number > 0:
            if citation_page_number == candidate_page_number:
                bonus += 12.0
            else:
                page_gap = abs(citation_page_number - candidate_page_number)
                bonus -= min(24.0, page_gap * 6.0)

        if document_level and citation_page_number > 0 and candidate_page_number != citation_page_number:
            bonus -= 10.0
        return bonus

    def _extract_numeric_tokens(self, text: str) -> set[str]:
        normalized = self._format_quote_text(text)
        matches = re.findall(r"\b\d[\d,]*(?:\.\d+)?%?\b", normalized)
        return {match.lower() for match in matches if match}

    def _extract_temporal_tokens(self, text: str) -> set[str]:
        normalized = self._format_quote_text(text)
        patterns = (
            r"\b[1-4]q(?:fy)?\d{2,4}\b",
            r"\bfy\d{2,4}\b",
            r"\bh[12]\s?\d{2,4}\b",
            r"\b20\d{2}\b",
            r"\b\d{2}년\b",
            r"\b\d{1,2}월\b",
        )
        matches: set[str] = set()
        for pattern in patterns:
            matches.update(match.lower() for match in re.findall(pattern, normalized, flags=re.IGNORECASE))
        return matches

    def _asset_supports_citation(
        self,
        *,
        asset_text: str,
        asset_type: str,
        signals: dict[str, Any],
        synthetic: bool = False,
    ) -> bool:
        normalized_asset_text = self._format_quote_text(asset_text)
        if not normalized_asset_text:
            return False

        asset_tokens = self._tokens(normalized_asset_text)
        asset_numeric_tokens = self._extract_numeric_tokens(normalized_asset_text)
        asset_temporal_tokens = self._extract_temporal_tokens(normalized_asset_text)
        token_overlap = len(signals["tokens"] & asset_tokens)
        numeric_overlap = len(signals["numeric_tokens"] & asset_numeric_tokens)
        temporal_overlap = len(signals["temporal_tokens"] & asset_temporal_tokens)

        if synthetic:
            return numeric_overlap > 0 and (token_overlap + temporal_overlap) >= 2
        if asset_type == "image":
            return token_overlap >= 2 or ((numeric_overlap > 0 or temporal_overlap > 0) and token_overlap >= 1)
        if asset_type == "table":
            if numeric_overlap > 0 and (token_overlap > 0 or temporal_overlap > 0):
                return True
            if temporal_overlap > 0 and token_overlap >= 2:
                return True
            if token_overlap >= 3:
                return True
            if token_overlap >= 2 and not signals["numeric_tokens"] and not signals["temporal_tokens"]:
                return True
            return False
        if numeric_overlap > 0 and (token_overlap > 0 or temporal_overlap > 0):
            return True
        if temporal_overlap > 0 and token_overlap >= 1:
            return True
        return token_overlap >= 2

    def _normalize_bbox(self, raw_bbox: Any) -> list[float] | None:
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            return None
        try:
            left, top, right, bottom = (float(value) for value in raw_bbox)
        except (TypeError, ValueError):
            return None
        if right <= left or bottom <= top:
            return None
        return [left, top, right, bottom]

    def _union_bboxes(self, bboxes: list[list[float]]) -> list[float] | None:
        if not bboxes:
            return None
        return [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ]

    def _expand_bbox(self, bbox: list[float], *, x_padding: float, y_padding: float) -> list[float]:
        return [
            bbox[0] - x_padding,
            bbox[1] - y_padding,
            bbox[2] + x_padding,
            bbox[3] + y_padding,
        ]

    def _bboxes_intersect(self, left: list[float], right: list[float]) -> bool:
        return not (
            left[2] < right[0]
            or right[2] < left[0]
            or left[3] < right[1]
            or right[3] < left[1]
        )

    def _collect_neighboring_page_elements(self, page_elements: list[dict[str, Any]], seed_bbox: list[float]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        current_bbox = list(seed_bbox)

        changed = True
        while changed:
            changed = False
            expanded_bbox = self._expand_bbox(current_bbox, x_padding=120.0, y_padding=60.0)
            for element in page_elements:
                element_id = str(element.get("element_id") or "")
                if element_id in selected_ids:
                    continue
                bbox = self._normalize_bbox(element.get("bbox"))
                if bbox is None or not self._bboxes_intersect(expanded_bbox, bbox):
                    continue
                selected.append(element)
                if element_id:
                    selected_ids.add(element_id)
                current_bbox = self._union_bboxes([current_bbox, bbox]) or current_bbox
                changed = True
        return selected

    def _image_area_score(self, bbox: list[float] | None) -> float:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return 0.0
        try:
            left, top, right, bottom = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return 0.0
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        area = width * height
        return min(6.0, area / 15000.0)

    def _is_meaningful_image_element(self, element: dict[str, Any]) -> bool:
        bbox = element.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        try:
            left, top, right, bottom = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return False
        width = max(0.0, right - left)
        height = max(0.0, bottom - top)
        area = width * height
        return width >= 40.0 and height >= 40.0 and area >= 4000.0

    def _truncate_text(self, text: str, *, max_characters: int) -> str:
        normalized = self._format_quote_text(text)
        if len(normalized) <= max_characters:
            return normalized
        return normalized[: max_characters - 3].rstrip() + "..."

    def _truncate_table_markdown(self, markdown: str, *, max_lines: int = 6, max_characters: int = 900) -> str:
        lines = [line.rstrip() for line in self._format_quote_text(markdown).splitlines() if line.strip()]
        preview = "\n".join(lines[:max_lines]).strip()
        if len(preview) <= max_characters:
            return preview
        return preview[: max_characters - 3].rstrip() + "..."

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _rewrite_answer_citation_markers(
        self,
        answer: str,
        *,
        raw_citations: list[dict[str, Any]],
        normalized_citations: list[dict[str, Any]],
    ) -> str:
        text = str(answer or "").strip()
        if not text:
            return text

        if not normalized_citations:
            return re.sub(r"\s*\[(\d+)\]", "", text).strip()

        rewritten = re.sub(r"\s*\[(\d+)\]", "", text)
        rewritten = re.sub(r"\n\s*\[[^\n\]]*:[^\n\]]*\]\s*$", "", rewritten)
        rewritten = re.sub(r"\n\s*참조 문서:[^\n]*$", "", rewritten)
        rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()
        return rewritten

    def _build_document_summaries(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for match in matches:
            metadata = match.get("metadata", {})
            document_id = str(metadata.get("document_id", ""))
            if not document_id or document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            summary = self.summary_map.get(document_id)
            if not summary:
                continue
            summaries.append(
                {
                    "document_id": document_id,
                    "source_name": metadata.get("source_name", ""),
                    "summary_text": summary.get("summary_text", ""),
                    "highlights": summary.get("highlights", []),
                }
            )
        return summaries[:3]

    def _split_sentences(self, text: str) -> list[str]:
        """문장 단위로 분리. 마크다운 표 블록은 행 단위로 쪼개지 않고 통째로 유지."""
        results: list[str] = []
        table_buffer: list[str] = []
        non_table_buffer: list[str] = []

        def _flush_table() -> None:
            if table_buffer:
                block = "\n".join(table_buffer).strip()
                if block:
                    results.append(block)
                table_buffer.clear()

        def _flush_non_table() -> None:
            if non_table_buffer:
                segment = "\n".join(non_table_buffer)
                for part in re.split(r"(?<=[.!?])\s+|\n+", segment):
                    part = part.strip()
                    if part:
                        results.append(part)
                non_table_buffer.clear()

        for line in text.splitlines():
            if re.match(r"^\s*\|", line) or re.match(r"^\[표: \d+행 × \d+열\]$", line.strip()):
                _flush_non_table()
                table_buffer.append(line)
            else:
                _flush_table()
                non_table_buffer.append(line)

        _flush_table()
        _flush_non_table()
        return results

    def _tokens(self, text: str) -> set[str]:
        normalized = str(text or "").lower()
        basic_tokens = [token for token in re.findall(r"[0-9A-Za-z가-힣]+", normalized) if token]
        if not basic_tokens:
            return set()

        tokens = set(basic_tokens)
        for token in basic_tokens:
            tokens.update(self._korean_token_variants(token))
        compact = re.sub(r"[^0-9a-zA-Z가-힣]+", "", normalized)
        if compact and 2 <= len(compact) <= 24:
            tokens.add(compact)

        for left, right in zip(basic_tokens, basic_tokens[1:]):
            if len(left) < 2 or len(right) < 2:
                continue
            combined = f"{left}{right}"
            if len(combined) <= 24:
                tokens.add(combined)

        return tokens

    def _korean_token_variants(self, token: str) -> set[str]:
        if not re.search(r"[가-힣]", token):
            return set()
        variants: set[str] = set()
        suffixes = (
            "으로",
            "로",
            "에서",
            "에게",
            "께서",
            "부터",
            "까지",
            "에는",
            "에",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "와",
            "과",
            "의",
            "도",
            "만",
        )
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                variants.add(token[: -len(suffix)])
        if token.endswith("명") and len(token) > 3:
            variants.add(token[:-1])
        return variants

    def _query_intent_tokens(self, query: str) -> set[str]:
        stop_tokens = (
            "표",
            "테이블",
            "수치",
            "숫자",
            "알려줘",
            "보여줘",
            "보여주",
            "궁금해",
            "대해",
            "대한",
            "관련",
            "내용",
            "요약",
        )
        intent_tokens: set[str] = set()
        for token in self._tokens(query):
            if len(token) < 2:
                continue
            if any(stop_token in token for stop_token in stop_tokens):
                continue
            intent_tokens.add(token)
        return intent_tokens

    def _citation_anchor_tokens(self, citation: dict[str, Any]) -> set[str]:
        evidence_text = self._format_quote_text(
            "\n".join(
                [
                    str(citation.get("highlight_text") or ""),
                    str(citation.get("quote") or ""),
                    str(citation.get("chunk_text") or ""),
                ]
            )
        )
        bracket_headings = [self._normalize_heading_anchor(match) for match in re.findall(r"\[([^\]]+)\]", evidence_text)]
        specific_headings = [heading for heading in bracket_headings if heading and " 및 " not in heading]
        heading_anchor_text = "\n".join(specific_headings or bracket_headings)
        evidence_without_headings = re.sub(r"\[[^\]]+\]", " ", evidence_text)

        text = self._format_quote_text(
            "\n".join(
                [
                    "" if specific_headings else str(citation.get("section_hint") or ""),
                    heading_anchor_text,
                    evidence_without_headings,
                ]
            )
        )
        generic_tokens = {
            "구분",
            "항목",
            "단위",
            "기준",
            "현황",
            "주요",
            "기타",
            "합계",
            "비고",
            "사업",
            "페이지",
        }
        anchors: set[str] = set()
        for token in self._tokens(text):
            if len(token) < 2 or token in generic_tokens:
                continue
            if re.fullmatch(r"\d+", token):
                continue
            anchors.add(token)
        return anchors

    def _normalize_heading_anchor(self, value: str) -> str:
        normalized = self._format_quote_text(value)
        normalized = re.sub(r"^\s*[0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.)]?\s*", "", normalized)
        normalized = re.sub(r"^\s*[가-힣A-Za-z][.)]\s*", "", normalized)
        return normalized.strip()

    def _asset_matches_citation_anchors(self, anchor_tokens: set[str], asset_text: str) -> bool:
        if not anchor_tokens:
            return True
        compact_asset = re.sub(r"[^0-9a-zA-Z가-힣]+", "", str(asset_text or "").lower())
        if not compact_asset:
            return False

        strong_anchors = {token for token in anchor_tokens if len(token) >= 4}
        if any(token in compact_asset for token in strong_anchors):
            return True

        matched = sum(1 for token in anchor_tokens if token in compact_asset)
        return matched >= min(3, len(anchor_tokens))

    def _count_intent_matches(self, intent_tokens: set[str], text: str) -> int:
        if not intent_tokens:
            return 0
        compact = re.sub(r"[^0-9a-zA-Z가-힣]+", "", str(text or "").lower())
        if not compact:
            return 0
        count = 0
        for token in intent_tokens:
            if token in compact:
                count += 1
        return count

    def _format_quote_text(self, text: str) -> str:
        normalized = re.sub(r"<br\s*/?>", "\n", str(text or ""), flags=re.IGNORECASE)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _contains_normalized(self, haystack: str, needle: str) -> bool:
        return self._format_quote_text(needle) in self._format_quote_text(haystack)

    def _best_highlight_for_chunk(self, query: str, *candidates: str) -> str:
        query_tokens = self._tokens(query)
        best_candidate = ""
        best_score = -1
        for candidate in candidates:
            text = self._format_quote_text(candidate)
            if not text:
                continue
            score = len(query_tokens & self._tokens(text))
            if score > best_score:
                best_score = score
                best_candidate = text
        return best_candidate

    def _extract_page_number(self, section_hint: object, text: str) -> int | None:
        candidates = [str(section_hint or ""), str(text or "")]
        for candidate in candidates:
            match = re.search(r"page\s*(\d+)", candidate, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def _structured_document_roots(self) -> list[Path]:
        roots = [self.project_root / "data" / "structured" / "documents"]
        ui_runs_root = self.project_root / "outputs" / "ui_runs"
        if ui_runs_root.exists():
            for path in sorted(ui_runs_root.glob("*/structured/documents")):
                roots.append(path)
            for path in sorted(ui_runs_root.glob("*/parsing/json")):
                roots.append(path)
        return roots


