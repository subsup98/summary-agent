from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.indexing.chunking import build_rule_based_chunks
from src.indexing.embedding_backends import OpenAIEmbeddingBackend, OpenAIEmbeddingSettings, cosine_similarity
from src.shared.io import ensure_directory, iso_now, write_json, write_text


TARGET_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]


class EmbeddingModelComparisonRunner:
    def __init__(self, structured_documents_root: Path, reports_root: Path, api_key: str) -> None:
        self.structured_documents_root = structured_documents_root
        self.reports_root = reports_root
        self.api_key = api_key

    def run(self) -> dict[str, Any]:
        ensure_directory(self.reports_root)
        documents = self._load_documents()
        queries = self._build_queries(documents)
        summary = {
            "started_at": iso_now(),
            "structured_documents_root": self.structured_documents_root.as_posix(),
            "document_count": len(documents),
            "query_count": len(queries),
            "models": [],
        }

        for model_name in TARGET_MODELS:
            backend = OpenAIEmbeddingBackend(OpenAIEmbeddingSettings(api_key=self.api_key, model=model_name))
            document_vectors = self._build_document_vectors(documents, backend)
            metrics = self._evaluate_queries(queries, document_vectors, backend)
            summary["models"].append(
                {
                    "model": model_name,
                    **metrics,
                }
            )

        summary["finished_at"] = iso_now()
        write_json(self.reports_root / "embedding_model_comparison.json", summary)
        write_text(self.reports_root / "embedding_model_comparison.md", self._render_markdown(summary))
        return summary

    def _load_documents(self) -> list[dict[str, Any]]:
        documents = []
        for path in sorted(self.structured_documents_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            markdown = payload.get("markdown") or ""
            chunks = build_rule_based_chunks(markdown, chunk_size=900, overlap=150)
            documents.append(
                {
                    "document_id": payload.get("document_id"),
                    "source_name": payload.get("source_name"),
                    "sections": payload.get("sections", []),
                    "chunks": chunks,
                }
            )
        return documents

    def _build_queries(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        queries: list[dict[str, Any]] = []
        for document in documents:
            section_titles = [section.get("title") for section in document.get("sections", []) if section.get("title")]
            for title in section_titles[:3]:
                queries.append(
                    {
                        "query": f"{document['source_name']} {title}",
                        "target_document_id": document["document_id"],
                    }
                )
            if not section_titles:
                queries.append(
                    {
                        "query": str(document["source_name"]),
                        "target_document_id": document["document_id"],
                    }
                )
        return queries

    def _build_document_vectors(self, documents: list[dict[str, Any]], backend: OpenAIEmbeddingBackend) -> list[dict[str, Any]]:
        vectors = []
        for document in documents:
            texts = [chunk["text"] for chunk in document["chunks"][:12]]
            embeddings = backend.embed_documents(texts) if texts else []
            vectors.append(
                {
                    "document_id": document["document_id"],
                    "source_name": document["source_name"],
                    "chunk_embeddings": embeddings,
                }
            )
        return vectors

    def _evaluate_queries(
        self,
        queries: list[dict[str, Any]],
        document_vectors: list[dict[str, Any]],
        backend: OpenAIEmbeddingBackend,
    ) -> dict[str, Any]:
        hit_at_1 = 0
        hit_at_3 = 0
        reciprocal_rank_total = 0.0

        for query in queries:
            query_vector = backend.embed_query(query["query"])
            ranked = []
            for document in document_vectors:
                score = max((cosine_similarity(query_vector, vector) for vector in document["chunk_embeddings"]), default=0.0)
                ranked.append({"document_id": document["document_id"], "score": score})
            ranked.sort(key=lambda item: item["score"], reverse=True)

            top_ids = [item["document_id"] for item in ranked[:3]]
            target = query["target_document_id"]
            if ranked and ranked[0]["document_id"] == target:
                hit_at_1 += 1
            if target in top_ids:
                hit_at_3 += 1
            for index, item in enumerate(ranked, start=1):
                if item["document_id"] == target:
                    reciprocal_rank_total += 1.0 / index
                    break

        query_count = max(len(queries), 1)
        return {
            "hit_at_1": round(hit_at_1 / query_count, 4),
            "hit_at_3": round(hit_at_3 / query_count, 4),
            "mrr": round(reciprocal_rank_total / query_count, 4),
        }

    def _render_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Embedding Model Comparison",
            "",
            f"- Structured Documents Root: `{summary['structured_documents_root']}`",
            f"- Document Count: `{summary['document_count']}`",
            f"- Query Count: `{summary['query_count']}`",
            "",
            "| Model | Hit@1 | Hit@3 | MRR |",
            "| --- | --- | --- | --- |",
        ]
        for model in summary["models"]:
            lines.append(
                f"| {model['model']} | {model['hit_at_1']} | {model['hit_at_3']} | {model['mrr']} |"
            )
        return "\n".join(lines) + "\n"
