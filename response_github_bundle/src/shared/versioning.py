from __future__ import annotations

import json
from typing import Any

from src.shared.constants import VERSION_FILE


def load_version_info() -> dict[str, Any]:
    if not VERSION_FILE.exists():
        return {
            "version": "0.0.0",
            "release_date": None,
            "scope": "unknown",
            "notes": [],
        }

    return json.loads(VERSION_FILE.read_text(encoding="utf-8"))
