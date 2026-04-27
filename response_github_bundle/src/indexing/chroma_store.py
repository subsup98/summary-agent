from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

from src.indexing.chunking import build_rule_based_chunks, build_semantic_chunks, clamp_chunks_for_embedding, compare_chunk_strategies
from src.indexing.embedding_backends import EmbeddingBackend, normalize_model_token, resolve_embedding_backend
from src.shared.io import ensure_directory, iso_now, write_json, write_text
from src.shared.runtime_deps import ensure_local_dependency_path


ensure_local_dependency_path()

import chromadb  # type: ignore  # noqa: E402


ProgressCallback = Callable[[str, int, int], None]


class ChromaIndexManager:
    def __init__(self, project_root: Path, vector_root: Path | None = None, embedding_backend: EmbeddingBackend | None = None) -> None:
        self.project_root = project_root
        self.vector_root = vector_root or (project_root / "outputs" / "vector_index")
        self.registry_path = self.vector_root / "registry.json"
        self.report_path = self.vector_root / "reports" / "latest_run.json"
        self.report_markdown_path = self.vector_root / "reports" / "latest_run.md"
        ensure_directory(self.vector_root)
        ensure_directory(self.vector_root / "reports")
        self.client = chromadb.PersistentClient(path=str(self.vector_root / "chroma_db"))
        self.embedding_backend = embedding_backend or resolve_embedding_backend()
        self.embedding_model = self.embedding_backend.model_name
        backend_token = normalize_model_token(self.embedding_model)
        self.rule_collection = self.client.get_or_create_collection(name=f"parsed_documents_rule_{backend_token}")
        self.semantic_collection = self.client.get_or_create_collection(name=f"parsed_documents_semantic_{backend_token}")
        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def find_duplicate_source(self, file_path: Path) -> dict[str, Any] | None:
        source_hash = self._hash_file(file_path)
        return self.find_duplicate_source_by_hash(source_hash)

    def find_duplicate_source_by_hash(self, source_hash: str) -> dict[str, Any] | None:
        registry = self._load_registry()
        for record in registry["documents"]:
            if record.get("source_hash") == source_hash:
                return record
        return None

    def ingest_single_document(
        self,
        payload: dict[str, Any],
        source_root: Path | None = None,
    ) -> dict[str, Any]:
        """파싱 직후 단일 문서를 즉시 벡터 인덱싱한다. 스레드 안전."""
        with self._lock:
            registry = self._load_registry()
            try:
                result = self._ingest_payload(payload, registry, source_root=source_root)
            except Exception as error:
                return {
                    "status": "failed",
                    "record": {
                        "document_id": payload.get("document_id"),
                        "source_name": payload.get("source_name"),
                        "status": "failed",
                        "error": str(error),
                    },
                    "comparison": None,
                }
            write_json(self.registry_path, registry)
        return result

    def ingest_structured_documents(
        self,
        structured_documents_root: Path,
        source_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        registry = self._load_registry()
        documents = sorted(structured_documents_root.glob("*.json"))
        summary: dict[str, Any] = {
            "started_at": iso_now(),
            "structured_documents_root": structured_documents_root.as_posix(),
            "total_documents": len(documents),
            "indexed_documents": 0,
            "duplicate_documents": 0,
            "failed_documents": 0,
            "documents": [],
            "comparisons": [],
        }

        for index, path in enumerate(documents, start=1):
            if progress_callback:
                progress_callback(f"벡터 인덱싱 중: {path.name}", index, len(documents))

            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                result = self._ingest_payload(payload, registry, source_root=source_root)
            except Exception as error:
                summary["failed_documents"] += 1
                summary["documents"].append(
                    {
                        "document_id": payload.get("document_id"),
                        "source_name": payload.get("source_name"),
                        "status": "failed",
                        "error": str(error),
                    }
                )
                continue

            summary["documents"].append(result["record"])
            if result["status"] == "indexed":
                summary["indexed_documents"] += 1
                summary["comparisons"].append(result["comparison"])
            else:
                summary["duplicate_documents"] += 1

        summary["finished_at"] = iso_now()
        summary["registry_document_count"] = len(registry["documents"])
        write_json(self.registry_path, registry)
        write_json(self.report_path, summary)
        write_text(self.report_markdown_path, self._render_summary(summary))
        return summary

    def delete_document(self, document_id: str) -> dict[str, Any]:
        deleted_rule = 0
        deleted_semantic = 0
        try:
            existing = self.rule_collection.get(where={"document_id": document_id})
            ids_to_delete = existing.get("ids") or []
            if ids_to_delete:
                self.rule_collection.delete(ids=ids_to_delete)
                deleted_rule = len(ids_to_delete)
        except Exception:
            pass
        try:
            existing = self.semantic_collection.get(where={"document_id": document_id})
            ids_to_delete = existing.get("ids") or []
            if ids_to_delete:
                self.semantic_collection.delete(ids=ids_to_delete)
                deleted_semantic = len(ids_to_delete)
        except Exception:
            pass

        registry = self._load_registry()
        before_count = len(registry["documents"])
        registry["documents"] = [r for r in registry["documents"] if r.get("document_id") != document_id]
        removed_from_registry = before_count - len(registry["documents"])
        if removed_from_registry:
            registry["updated_at"] = iso_now()
            write_json(self.registry_path, registry)

        return {
            "document_id": document_id,
            "deleted_rule_chunks": deleted_rule,
            "deleted_semantic_chunks": deleted_semantic,
            "removed_from_registry": removed_from_registry,
        }

    def get_status(self) -> dict[str, Any]:
        registry = self._load_registry()
        return {
            "exists": self.registry_path.exists(),
            "vector_root": self.vector_root.as_posix(),
            "document_count": len(registry["documents"]),
            "updated_at": registry.get("updated_at"),
            "report_path": self.report_path.as_posix() if self.report_path.exists() else None,
            "embedding_model": self.embedding_model,
        }

    def _ingest_payload(
        self,
        payload: dict[str, Any],
        registry: dict[str, Any],
        source_root: Path | None = None,
    ) -> dict[str, Any]:
        markdown = (payload.get("markdown") or "").strip()
        content_hash = self._hash_text(markdown)
        source_path = self._resolve_source_path(payload, source_root)
        source_hash = self._hash_file(source_path) if source_path and source_path.exists() else None

        duplicate = self._find_duplicate(registry, source_hash=source_hash, content_hash=content_hash)
        if duplicate and self._is_indexed_for_current_model(duplicate["record"]):
            return {
                "status": "duplicate",
                "record": {
                    "document_id": payload.get("document_id"),
                    "source_name": payload.get("source_name"),
                    "status": "duplicate",
                    "duplicate_reason": duplicate["reason"],
                    "existing_document_id": duplicate["record"].get("document_id"),
                    "existing_source_name": duplicate["record"].get("source_name"),
                },
                "comparison": None,
            }

        rule_chunks = build_rule_based_chunks(markdown, chunk_size=900, overlap=150)
        semantic_chunks = self._get_semantic_chunks(payload, markdown)
        comparison = compare_chunk_strategies(payload.get("document_id", ""), rule_chunks, semantic_chunks)

        self._upsert_chunks(self.rule_collection, payload, content_hash, rule_chunks)
        self._upsert_chunks(self.semantic_collection, payload, content_hash, semantic_chunks)

        registry_record = duplicate["record"] if duplicate else {}
        indexed_models = self._get_indexed_models(registry_record)
        if self.embedding_model not in indexed_models:
            indexed_models.append(self.embedding_model)
        registry_record.update(
            {
                "document_id": payload.get("document_id"),
                "source_name": payload.get("source_name"),
                "source_path": source_path.as_posix() if source_path else payload.get("source_path"),
                "source_hash": source_hash,
                "content_hash": content_hash,
                "indexed_at": iso_now(),
                "rule_chunk_count": len(rule_chunks),
                "semantic_chunk_count": len(semantic_chunks),
                "embedding_model": self.embedding_model,
                "indexed_models": indexed_models,
            }
        )
        if not duplicate:
            registry["documents"].append(registry_record)
        registry["updated_at"] = iso_now()

        return {
            "status": "indexed",
            "record": {
                "document_id": payload.get("document_id"),
                "source_name": payload.get("source_name"),
                "status": "indexed",
                "rule_chunk_count": len(rule_chunks),
                "semantic_chunk_count": len(semantic_chunks),
            },
            "comparison": comparison,
        }

    def _get_semantic_chunks(self, payload: dict[str, Any], markdown: str) -> list[dict[str, Any]]:
        raw_chunks = payload.get("semantic_chunks")
        if isinstance(raw_chunks, list):
            normalized_chunks: list[dict[str, Any]] = []
            for index, chunk in enumerate(raw_chunks):
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text") or "").strip()
                if not text:
                    continue
                normalized_chunks.append(
                    {
                        "strategy": "semantic",
                        "chunk_index": int(chunk.get("chunk_index", index) or index),
                        "text": text,
                        "char_count": int(chunk.get("char_count", len(text)) or len(text)),
                        "section_hint": chunk.get("section_hint"),
                    }
                )
            if normalized_chunks:
                return clamp_chunks_for_embedding(normalized_chunks, max_characters=6000, overlap=180)
        return build_semantic_chunks(markdown, embeddings=self.embedding_backend)

    def _upsert_chunks(
        self,
        collection: Any,
        payload: dict[str, Any],
        content_hash: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        if not chunks:
            return

        prepared_chunks = clamp_chunks_for_embedding(chunks, max_characters=6000, overlap=180)
        texts = [chunk["text"] for chunk in prepared_chunks]
        ids = [f"{content_hash}:{chunk['strategy']}:{chunk['chunk_index']}" for chunk in prepared_chunks]
        metadatas = [
            {
                "document_id": payload.get("document_id"),
                "source_name": payload.get("source_name"),
                "strategy": chunk["strategy"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "section_hint": chunk.get("section_hint") or "",
                "page_number": chunk.get("page_number") if chunk.get("page_number") is not None else "",
                "asset_page_number": (
                    chunk.get("asset_page_number") if chunk.get("asset_page_number") is not None else ""
                ),
                "content_hash": content_hash,
            }
            for chunk in prepared_chunks
        ]
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=self.embedding_backend.embed_documents(texts),
        )

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {
                "schema_version": "1.0",
                "updated_at": None,
                "documents": [],
            }
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _find_duplicate(
        self,
        registry: dict[str, Any],
        *,
        source_hash: str | None,
        content_hash: str,
    ) -> dict[str, Any] | None:
        for record in registry["documents"]:
            if source_hash and record.get("source_hash") == source_hash:
                return {"reason": "source_hash", "record": record}
            if record.get("content_hash") == content_hash:
                return {"reason": "content_hash", "record": record}
        return None

    def _get_indexed_models(self, record: dict[str, Any]) -> list[str]:
        indexed_models = record.get("indexed_models")
        if isinstance(indexed_models, list):
            return [str(model) for model in indexed_models if model]
        legacy_model = record.get("embedding_model")
        return [str(legacy_model)] if legacy_model else []

    def _is_indexed_for_current_model(self, record: dict[str, Any]) -> bool:
        return self.embedding_model in self._get_indexed_models(record)

    def _resolve_source_path(self, payload: dict[str, Any], source_root: Path | None) -> Path | None:
        raw_source_path = payload.get("source_path")
        if not raw_source_path:
            return None
        candidate = Path(str(raw_source_path))
        if candidate.is_absolute():
            return candidate
        if source_root is not None:
            return (source_root / candidate).resolve()
        return (self.project_root / candidate).resolve()

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _hash_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _render_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Vector Index Run Summary",
            "",
            f"- Structured Documents Root: `{summary['structured_documents_root']}`",
            f"- Started At: `{summary['started_at']}`",
            f"- Finished At: `{summary['finished_at']}`",
            f"- Total Documents: `{summary['total_documents']}`",
            f"- Indexed Documents: `{summary['indexed_documents']}`",
            f"- Duplicate Documents: `{summary['duplicate_documents']}`",
            f"- Failed Documents: `{summary['failed_documents']}`",
            "",
            "| Document | Status | Rule Chunks | Semantic Chunks |",
            "| --- | --- | --- | --- |",
        ]

        for document in summary["documents"]:
            lines.append(
                "| {source_name} | {status} | {rule_chunk_count} | {semantic_chunk_count} |".format(
                    source_name=document.get("source_name", ""),
                    status=document.get("status", ""),
                    rule_chunk_count=document.get("rule_chunk_count", ""),
                    semantic_chunk_count=document.get("semantic_chunk_count", ""),
                )
            )

        return "\n".join(lines) + "\n"
