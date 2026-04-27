from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.io import ensure_directory  # noqa: E402
from src.ui.review_server import ReviewSessionManager  # noqa: E402


OFFICE_EXTENSIONS = {".doc", ".docx", ".hwp", ".hwpx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOC/HWP 계열 문서를 PDF로 변환한 뒤 structtree-actualtext 와 pymupdf4llm 결과를 비교합니다."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="비교할 office 문서 경로들 (.doc, .docx, .hwp, .hwpx)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="결과를 저장할 루트 디렉터리. 기본값: outputs/office_pdf_strategy_compare/<timestamp>",
    )
    return parser.parse_args()


def resolve_input_paths(raw_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        resolved.append(candidate)
    return resolved


def build_default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / "office_pdf_strategy_compare" / stamp


def render_all_in_one_dashboard(manifest: dict, compare_root: Path) -> Path:
    cards: list[str] = []
    for document in manifest.get("documents", []):
        raw_doc_path = str(document.get("document_path") or "").strip()
        if not raw_doc_path:
            continue
        doc_path = Path(raw_doc_path)
        try:
            relative_doc_path = doc_path.relative_to(compare_root).as_posix()
        except ValueError:
            relative_doc_path = doc_path.as_posix()
        source_name = str(document.get("source_name") or "")
        source_extension = str(document.get("source_extension") or "")
        producer = str((document.get("pdf_metadata") or {}).get("producer") or "n/a")
        selected_strategy = str((document.get("pdf_metadata") or {}).get("selected_strategy") or "n/a")
        cards.append(
            """
<section class="doc-card">
  <div class="doc-head">
    <div>
      <h2>{source_name}</h2>
      <p>{source_extension} · metadata selected: <code>{selected_strategy}</code></p>
      <p class="producer">producer: <code>{producer}</code></p>
    </div>
    <a class="open-link" href="{relative_doc_path}" target="_blank" rel="noopener noreferrer">Open only this compare</a>
  </div>
  <iframe src="{relative_doc_path}" loading="lazy" title="{source_name}"></iframe>
</section>
""".format(
                source_name=source_name,
                source_extension=source_extension,
                selected_strategy=selected_strategy,
                producer=producer,
                relative_doc_path=relative_doc_path,
            )
        )

    if not cards:
        cards.append("<p class='empty'>No document compare pages were generated.</p>")

    dashboard_path = compare_root / "all_in_one.html"
    dashboard_path.write_text(
        """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Office PDF Compare Dashboard</title>
  <style>
    :root {{
      --bg: #eef3f8;
      --panel: #ffffff;
      --line: #d6dfeb;
      --text: #172033;
      --muted: #52637c;
      --accent: #0f5cc0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,92,192,0.08), transparent 32%),
        linear-gradient(180deg, #f7f9fc 0%, var(--bg) 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1720px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      background: rgba(255,255,255,0.85);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px 26px;
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1.2;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }}
    .doc-list {{
      display: grid;
      gap: 18px;
    }}
    .doc-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
    }}
    .doc-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .doc-head h2 {{
      margin: 0 0 8px;
      font-size: 22px;
      line-height: 1.3;
    }}
    .doc-head p {{
      margin: 0 0 4px;
      color: var(--muted);
      font-size: 14px;
    }}
    .producer {{
      word-break: break-word;
    }}
    .open-link {{
      flex: 0 0 auto;
      text-decoration: none;
      color: white;
      background: linear-gradient(135deg, #0f5cc0, #1982ff);
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 600;
      white-space: nowrap;
    }}
    iframe {{
      width: 100%;
      min-height: 1180px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: white;
    }}
    code {{
      background: #eff4fb;
      border-radius: 6px;
      padding: 2px 6px;
    }}
    .empty {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    @media (max-width: 900px) {{
      .doc-head {{
        flex-direction: column;
      }}
      iframe {{
        min-height: 900px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Office PDF Strategy Compare Dashboard</h1>
      <p>DOC/HWP를 PDF로 변환한 뒤 <code>structtree-actualtext</code> 와 <code>pymupdf4llm</code> 결과를 한 번에 훑어볼 수 있는 대시보드입니다.</p>
    </section>
    <section class="doc-list">
      {cards}
    </section>
  </main>
</body>
</html>
""".format(cards="".join(cards)),
        encoding="utf-8",
    )
    return dashboard_path


def main() -> int:
    args = parse_args()
    input_paths = resolve_input_paths(args.paths)

    missing_paths = [path for path in input_paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"[missing] {path}")
        return 1

    office_paths = [path for path in input_paths if path.suffix.lower() in OFFICE_EXTENSIONS]
    skipped_paths = [path for path in input_paths if path.suffix.lower() not in OFFICE_EXTENSIONS]

    if skipped_paths:
        for path in skipped_paths:
            print(f"[skipped:unsupported-extension] {path}")
    if not office_paths:
        print("No office documents to compare.")
        return 1

    session_root = args.output_root.resolve() if args.output_root else build_default_output_root()
    ensure_directory(session_root)

    manager = ReviewSessionManager(project_root=PROJECT_ROOT)
    try:
        manifest = manager._build_office_pdf_strategy_compare(
            session_root=session_root,
            saved_uploads=[{"stored_path": path.as_posix()} for path in office_paths],
        )
    finally:
        manager.close()

    if not manifest:
        print("Comparison manifest was not created.")
        return 1

    compare_root = session_root / "reports" / "office_pdf_strategy_compare"
    dashboard_path = render_all_in_one_dashboard(manifest, compare_root)

    print(f"manifest: {session_root / 'reports' / 'office_pdf_strategy_compare' / 'manifest.json'}")
    print(f"index: {session_root / 'reports' / 'office_pdf_strategy_compare' / 'index.html'}")
    print(f"dashboard: {dashboard_path}")
    print(f"document_count: {manifest.get('document_count', 0)}")

    for document in manifest.get("documents", []):
        source_name = str(document.get("source_name") or "")
        conversion_succeeded = bool(document.get("conversion_succeeded"))
        error = str(document.get("error") or "")
        print(
            f"- {source_name}: conversion={'ok' if conversion_succeeded else 'failed'}"
            + (f", error={error}" if error else "")
        )
        for strategy in document.get("strategies", []):
            print(
                "  * {name}: applied={applied}, chars={chars}, issues={issues}".format(
                    name=str(strategy.get("strategy_name") or ""),
                    applied=str(strategy.get("applied_strategy") or ""),
                    chars=str(strategy.get("char_count") or 0),
                    issues=len(strategy.get("issues") or []),
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
