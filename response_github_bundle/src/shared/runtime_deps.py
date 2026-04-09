from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def ensure_local_dependency_path(directory_name: str = ".deps_vector") -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    dependency_root = project_root / directory_name
    if not dependency_root.exists():
        return None

    dependency_path = str(dependency_root)
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    return dependency_root
