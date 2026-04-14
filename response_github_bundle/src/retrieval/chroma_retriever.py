from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.indexing.chroma_store import ChromaIndexManager
from src.indexing.embedding_backends import EmbeddingBackend
from src.retrieval.document_summary import load_summary_map
from src.retrieval.openai_answerer import OpenAIAnswerSynthesizer, load_openai_settings


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

    def search(
        self,
        query: str,
        strategy: str = "rule_based",
        top_k: int = 5,
        *,
        document_id: str = "",
        source_name: str = "",
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        collection = (
            self.index_manager.rule_collection if strategy == "rule_based" else self.index_manager.semantic_collection
        )
        query_embedding = self.embeddings.embed_query(query)
        scoped_ids = {item for item in (document_ids or []) if item}
        if scoped_ids:
            document_id = ""
            source_name = ""
        require_filter = bool(scoped_ids or document_id or source_name)
        n_results = top_k
        if require_filter:
            try:
                n_results = max(top_k * 10, min(collection.count(), 50))
            except Exception:
                n_results = max(top_k * 10, 50)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
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
        strategy: str = "rule_based",
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
            for sentence in self._split_sentences(match.get("document", "")):
                sentence_tokens = self._tokens(sentence)
                overlap = len(query_tokens & sentence_tokens)
                if overlap == 0:
                    continue
                score = overlap * 10 - distance
                candidates.append(
                    {
                        "score": score,
                        "sentence": sentence,
                        "document": match.get("document", "") or "",
                        "metadata": match.get("metadata", {}),
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

        for root in self._structured_document_roots():
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json")):
                try:
                    payload = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                try:
                    data = json.loads(payload)
                except Exception:
                    continue

                current_document_id = str(data.get("document_id", ""))
                current_source_name = str(data.get("source_name", ""))
                if scoped_ids and current_document_id not in scoped_ids:
                    continue
                if document_id and current_document_id != document_id:
                    continue
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
                        page_number = self._extract_page_number(chunk.get("section_hint"), chunk_text)
                    else:
                        chunk_text = str(chunk.get("serialized_text", ""))
                        chunk_index = index
                        section_hint = str(chunk.get("section") or "") or None
                        page_number = chunk.get("page")
                    if not chunk_text.strip():
                        continue
                    overlap = len(query_tokens & self._tokens(chunk_text))
                    if query_tokens and overlap == 0:
                        continue
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
        source_tokens = self._tokens(str(metadata.get("source_name") or ""))
        section_tokens = self._tokens(str(metadata.get("section_hint") or ""))
        document_tokens = self._tokens(str(match.get("document") or ""))
        overlap = (
            len(query_tokens & source_tokens) * 6
            + len(query_tokens & section_tokens) * 3
            + len(query_tokens & document_tokens)
        )
        distance = float(match.get("distance", 9999.0) or 9999.0)
        return (-overlap, distance)

    def _build_evidence(self, matches: list[dict[str, Any]], ranked_sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        used_keys: set[tuple[str, str]] = set()

        for item in ranked_sentences[:5]:
            metadata = item.get("metadata", {})
            key = (str(metadata.get("document_id", "")), item["sentence"])
            if key in used_keys:
                continue
            used_keys.add(key)
            page_number = metadata.get("page_number") or self._lookup_page_number(
                str(metadata.get("document_id", "")),
                item["sentence"],
                item.get("document", "") or "",
            )
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
            page_number = metadata.get("page_number") or self._lookup_page_number(
                str(metadata.get("document_id", "")),
                match.get("document", "") or "",
            )
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
        """structured JSON의 markdown/chunks에서 텍스트와 가장 유사한 실제 페이지 번호를 반환."""
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

        best_page: int | None = None
        best_score = -1.0
        for root in self._structured_document_roots():
            if not root.exists():
                continue
            for path in root.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(data.get("document_id", "")) != document_id:
                    continue
                markdown_pages = self._extract_markdown_pages(data)
                if markdown_pages:
                    for page_number, page_text in markdown_pages:
                        text_tokens = self._tokens(page_text)
                        if not text_tokens:
                            continue

                        score = 0.0
                        if primary_text and primary_text in page_text:
                            score += 1500.0
                        elif primary_text and page_text in primary_text:
                            score += 900.0

                        for candidate in normalized_candidates[1:]:
                            if candidate and candidate in page_text:
                                score += 220.0

                        overlap = len(primary_tokens & text_tokens)
                        if overlap:
                            score += overlap * 10.0

                        if score > best_score:
                            best_score = score
                            best_page = page_number
                for item in data.get("chunks", []):
                    page = item.get("page") or item.get("page_number")
                    if not page:
                        continue
                    text = self._format_quote_text(str(item.get("serialized_text") or item.get("text") or ""))
                    if not text:
                        continue
                    text_tokens = self._tokens(text)
                    if not text_tokens:
                        continue

                    score = 0.0
                    if primary_text and primary_text in text:
                        score += 1000.0
                    elif primary_text and text in primary_text:
                        score += 700.0

                    for candidate in normalized_candidates[1:]:
                        if candidate and candidate in text:
                            score += 180.0

                    overlap = len(primary_tokens & text_tokens)
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
            page_number = metadata.get("page_number") or self._lookup_page_number(
                str(metadata.get("document_id", "")),
                item["sentence"],
                item.get("document", "") or "",
            )
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
            citations = [
                {
                    "source_number": 1,
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "page_number": metadata.get("page_number") or self._lookup_page_number(
                        str(metadata.get("document_id", "")),
                        answer_sentences[0],
                        matches[0].get("document", "") or "",
                    ),
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
                        "page_number": evidence_item.get("page_number"),
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
                        evidence_item.get("page_number")
                        if evidence_item.get("page_number") is not None
                        else resolved_metadata.get("page_number")
                        if resolved_metadata.get("page_number") is not None
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
                    "page_number": metadata.get("page_number"),
                    "chunk_strategy": metadata.get("strategy") or "",
                }
            )
        return self._dedupe_citations(fallback)

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
        return set(re.findall(r"[0-9A-Za-z가-힣]+", text.lower()))

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
        return roots


