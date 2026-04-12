from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib import error, request

from src.indexing.chunking import DeterministicTextEmbeddings
from src.shared.retry import call_with_retry
from src.shared.runtime_deps import ensure_local_dependency_path


ensure_local_dependency_path()

from langchain_core.embeddings import Embeddings  # type: ignore  # noqa: E402


OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_BATCH_TOKEN_LIMIT = 6000


def _open_url(req: request.Request, timeout: int) -> Any:
    return request.urlopen(req, timeout=timeout)


class EmbeddingBackend(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class OpenAIEmbeddingSettings:
    api_key: str | None
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def load_openai_embedding_settings() -> OpenAIEmbeddingSettings:
    file_values = _load_env_file_values()
    return OpenAIEmbeddingSettings(
        api_key=os.environ.get("OPENAI_API_KEY") or file_values.get("OPENAI_API_KEY"),
        model=os.environ.get("OPENAI_EMBEDDING_MODEL") or file_values.get("OPENAI_EMBEDDING_MODEL") or DEFAULT_OPENAI_EMBEDDING_MODEL,
    )


class DeterministicEmbeddingBackend(Embeddings):
    def __init__(self) -> None:
        self.model_name = "local-deterministic-v1"
        self._embeddings = DeterministicTextEmbeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


class OpenAIEmbeddingBackend(Embeddings):
    def __init__(self, settings: OpenAIEmbeddingSettings | None = None) -> None:
        self.settings = settings or load_openai_embedding_settings()
        self.model_name = self.settings.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for batch in self._split_batches(texts):
            embeddings.extend(self._embed(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._embed([text])
        return embeddings[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured for embeddings")
        payload = {
            "model": self.settings.model,
            "input": texts,
            "encoding_format": "float",
        }
        req = request.Request(
            OPENAI_EMBEDDINGS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        def _do_request() -> dict:
            try:
                with _open_url(req, timeout=120) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                exc.reason = f"OpenAI embeddings error: {exc.code} {detail}"
                raise

        body = call_with_retry(_do_request, context="OpenAIEmbeddingBackend")
        data = body.get("data", [])
        return [item["embedding"] for item in data]

    def _split_batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_token_estimate = 0
        for text in texts:
            estimated_tokens = self._estimate_tokens(text)
            if current_batch and current_token_estimate + estimated_tokens > OPENAI_EMBEDDING_BATCH_TOKEN_LIMIT:
                batches.append(current_batch)
                current_batch = []
                current_token_estimate = 0
            current_batch.append(text)
            current_token_estimate += estimated_tokens
        if current_batch:
            batches.append(current_batch)
        return batches

    def _estimate_tokens(self, text: str) -> int:
        # Stay conservative for Korean and mixed financial tables by treating a character as roughly one token.
        return max(len(text.strip()), 1)


def resolve_embedding_backend() -> EmbeddingBackend:
    settings = load_openai_embedding_settings()
    if settings.enabled:
        return OpenAIEmbeddingBackend(settings)
    return DeterministicEmbeddingBackend()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def normalize_model_token(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _load_env_file_values() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    values: dict[str, str] = {}
    for filename in (".env.local", ".env"):
        path = project_root / filename
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values
