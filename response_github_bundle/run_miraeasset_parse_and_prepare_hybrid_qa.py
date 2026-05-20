from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ui.review_server import ReviewSessionManager, UploadedDocument


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manual_compare"
    / "miraeasset_main_pipeline"
    / "source"
    / "미래에셋증권 3분기 실적보고서.pdf"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "miraeasset_q3_parse_hybrid_run"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Mirae Asset Q3 report and print the command for hybrid QA.")
    parser.add_argument("--source", default=DEFAULT_SOURCE_PATH.as_posix(), help="PDF/document path to parse.")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source document not found: {source_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manager = ReviewSessionManager(project_root=PROJECT_ROOT)
    try:
        upload = UploadedDocument(filename=source_path.name, content=source_path.read_bytes())
        session = manager.create_run([upload])
    finally:
        manager.close()

    session_root = Path(str(session.get("session_root") or ""))
    structured_root = session_root / "structured" / "documents"
    structured_paths = sorted(structured_root.glob("*.json"))
    if not structured_paths:
        raise RuntimeError(f"Parsing finished but no structured JSON was found under {structured_root}")

    payload_path = structured_paths[0]
    manifest = {
        "source_path": source_path.as_posix(),
        "session_root": session_root.as_posix(),
        "structured_payload": payload_path.as_posix(),
        "hybrid_index_command": (
            f".\\.venv\\Scripts\\python.exe .\\run_miraeasset_curated_qa.py "
            f"--strategy hybrid --rebuild --payload-path \"{payload_path}\" --interactive --top-k 5"
        ),
    }
    manifest_path = OUTPUT_ROOT / "latest_parse_for_hybrid_qa.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest": manifest_path.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
