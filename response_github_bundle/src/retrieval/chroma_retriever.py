from __future__ import annotations

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
        scoped_ids = [item for item in (document_ids or []) if item]
        if scoped_ids and document_id and document_id in scoped_ids and len(scoped_ids) > 1:
            primary = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"document_id": document_id},
            )
            if primary.get("ids", [[]])[0]:
                results = primary
            else:
                remaining_ids = [item for item in scoped_ids if item != document_id]
                remaining_filter = _build_document_id_filter(remaining_ids)
                results = (
                    collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        where=remaining_filter,
                    )
                    if remaining_filter
                    else {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                )
        else:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if scoped_ids:
                query_kwargs["where"] = _build_document_id_filter(scoped_ids)
            elif document_id:
                query_kwargs["where"] = {"document_id": document_id}
            elif source_name:
                query_kwargs["where"] = {"source_name": source_name}
            results = collection.query(**query_kwargs)
            if not results.get("ids", [[]])[0] and not query_kwargs.get("where"):
                results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        matches: list[dict[str, Any]] = []
        for index, matched_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][index]
            matches.append(
                {
                    "id": matched_id,
                    "document": results.get("documents", [[]])[0][index],
                    "distance": results.get("distances", [[]])[0][index],
                    "metadata": metadata,
                }
            )
        return matches

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
                "answer": "질문 검색 중 오류가 발생했습니다. 현재 환경에서는 임베딩 또는 네트워크 연결이 제한될 수 있습니다.",
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

        return {
            "query": query,
            "strategy": strategy,
            "answer": answer_payload.get("answer", ""),
            "matches": matches,
            "document_summaries": self._build_document_summaries(matches),
            "citations": answer_payload.get("citations", []),
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
                        "metadata": match.get("metadata", {}),
                        "distance": distance,
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def _build_evidence(self, matches: list[dict[str, Any]], ranked_sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        used_keys: set[tuple[str, str]] = set()

        for item in ranked_sentences[:5]:
            metadata = item.get("metadata", {})
            key = (str(metadata.get("document_id", "")), item["sentence"])
            if key in used_keys:
                continue
            used_keys.add(key)
            evidence.append(
                {
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "excerpt": item["sentence"],
                }
            )

        if evidence:
            return evidence

        for match in matches[:3]:
            metadata = match.get("metadata", {})
            evidence.append(
                {
                    "source_name": metadata.get("source_name", ""),
                    "document_id": metadata.get("document_id", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "excerpt": (match.get("document", "") or "")[:400],
                }
            )
        return evidence

    def _fallback_answer(self, matches: list[dict[str, Any]], ranked_sentences: list[dict[str, Any]]) -> dict[str, Any]:
        answer_sentences = [item["sentence"] for item in ranked_sentences[:3]]
        citations = []
        for item in ranked_sentences[:3]:
            metadata = item.get("metadata", {})
            citations.append(
                {
                    "source_name": metadata.get("source_name", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "quote": item["sentence"][:160],
                }
            )
        if not answer_sentences:
            answer_sentences = [matches[0]["document"][:400].strip()]
            metadata = matches[0].get("metadata", {})
            citations = [
                {
                    "source_name": metadata.get("source_name", ""),
                    "section_hint": metadata.get("section_hint", ""),
                    "quote": answer_sentences[0][:160],
                }
            ]
        return {
            "answer": " ".join(answer_sentences),
            "citations": citations,
        }

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
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[0-9A-Za-z가-힣]+", text.lower()))

    def _structured_document_roots(self) -> list[Path]:
        roots = [self.project_root / "data" / "structured" / "documents"]
        ui_runs_root = self.project_root / "outputs" / "ui_runs"
        if ui_runs_root.exists():
            for path in sorted(ui_runs_root.glob("*/structured/documents")):
                roots.append(path)
        return roots
