from __future__ import annotations

import json
from pathlib import Path

from src.ui.review_server import ReviewSessionManager, UploadedDocument


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(
    r"C:\Users\yongseop.im\Desktop\all_docs\금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc"
)
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_main_pipeline_run"


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manager = ReviewSessionManager(project_root=PROJECT_ROOT)
    try:
        upload = UploadedDocument(filename=SOURCE_PATH.name, content=SOURCE_PATH.read_bytes())
        session = manager.create_run([upload])
    finally:
        manager.close()

    session_path = RUN_ROOT / "session.json"
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    structured_root = Path(str(session.get("session_root") or "")) / "structured"
    markdown_paths = sorted(structured_root.rglob("*.md"))
    payload = {
        "source_path": SOURCE_PATH.as_posix(),
        "session_root": session.get("session_root"),
        "session_json_path": session_path.as_posix(),
        "latest_run_markdown_path": session.get("latest_run_markdown_path"),
        "markdown_paths": [path.as_posix() for path in markdown_paths],
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
