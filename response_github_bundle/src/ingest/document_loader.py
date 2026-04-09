from __future__ import annotations

from pathlib import Path

from src.shared.constants import SUPPORTED_EXTENSIONS


class DocumentLoader:
    def discover_documents(self, root: Path) -> list[Path]:
        paths: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower() == "readme.md":
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            paths.append(path)
        return sorted(paths)
