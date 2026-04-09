from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from src.shared.retry import call_with_retry


OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5.2"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str | None
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def load_openai_settings() -> OpenAISettings:
    file_values = _load_env_file_values()
    return OpenAISettings(
        api_key=os.environ.get("OPENAI_API_KEY") or file_values.get("OPENAI_API_KEY"),
        model=os.environ.get("OPENAI_MODEL") or file_values.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
    )


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


class OpenAIAnswerSynthesizer:
    def __init__(self, settings: OpenAISettings | None = None) -> None:
        self.settings = settings or load_openai_settings()

    def is_enabled(self) -> bool:
        return self.settings.enabled

    def answer(self, query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        prompt = self._build_prompt(query, evidence)
        payload = {
            "model": self.settings.model,
            "input": prompt,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
        }
        raw = self._post(payload)
        text = self._extract_text(raw)
        return self._parse_json_response(text)

    def _build_prompt(self, query: str, evidence: list[dict[str, Any]]) -> str:
        evidence_lines: list[str] = []
        for index, item in enumerate(evidence, start=1):
            evidence_lines.append(
                "\n".join(
                    [
                        f"[SOURCE {index}]",
                        f"source_name: {item.get('source_name', '')}",
                        f"document_id: {item.get('document_id', '')}",
                        f"section_hint: {item.get('section_hint', '')}",
                        f"excerpt: {item.get('excerpt', '')}",
                    ]
                )
            )

        return (
            "You are answering questions only from supplied document evidence.\n"
            "Return strict JSON only with this shape:\n"
            '{"answer":"string","citations":[{"source_number":1,"source_name":"string","section_hint":"string","quote":"string"}]}\n'
            "Rules:\n"
            "- Use only the evidence provided.\n"
            "- If evidence is insufficient, say so in answer.\n"
            "- Every citation must correspond to one supplied source.\n"
            "- Keep quotes short.\n\n"
            f"Question:\n{query}\n\n"
            "Evidence:\n"
            + "\n\n".join(evidence_lines)
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            OPENAI_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        def _do_request() -> dict[str, Any]:
            try:
                with request.urlopen(req, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                # Attach detail to the exception so retry logic can re-raise with context
                exc.reason = f"OpenAI API error: {exc.code} {detail}"
                raise

        return call_with_retry(_do_request, context="OpenAIAnswerSynthesizer")

    def _extract_text(self, payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        parts: list[str] = []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        if parts:
            return "\n".join(parts).strip()
        if payload.get("output_text"):
            return str(payload["output_text"]).strip()
        raise RuntimeError("OpenAI response did not include output text")

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise RuntimeError("OpenAI response was not valid JSON")
            return json.loads(match.group(0))
