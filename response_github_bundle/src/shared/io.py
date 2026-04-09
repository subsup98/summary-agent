from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    ensure_directory(path.parent)
    path.write_bytes(content)


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1"), "latin-1"


def make_artifact_stem(path: Path) -> str:
    normalized_name = re.sub(r"\s+", "_", path.stem.strip())
    safe_name = re.sub(r'[<>:"/\\\\|?*]', "_", normalized_name)
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}--{digest}"
