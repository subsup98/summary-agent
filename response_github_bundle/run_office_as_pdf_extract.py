from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifiers.document_classifier import classify_document  # noqa: E402
from src.parsers.pdf.markdown_extractor import PdfMarkdownExtractor  # noqa: E402
from src.shared.io import ensure_directory, make_artifact_stem, write_json  # noqa: E402
from src.ui.review_server import ReviewSessionManager  # noqa: E402


OFFICE_EXTENSIONS = {".doc", ".docx", ".hwp", ".hwpx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Office 문서를 PDF로 변환한 뒤, PDF 파이프라인(metadata-selected)을 강제로 적용해 markdown을 추출합니다."
    )
    parser.add_argument("paths", nargs="+", help="입력 문서 경로")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="결과 저장 루트. 기본값: outputs/office_as_pdf_extract/<timestamp>",
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
    return PROJECT_ROOT / "outputs" / "office_as_pdf_extract" / stamp


def main() -> int:
    args = parse_args()
    input_paths = resolve_input_paths(args.paths)
    output_root = args.output_root.resolve() if args.output_root else build_default_output_root()
    ensure_directory(output_root)

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"[missing] {path}")
        return 1

    manager = ReviewSessionManager(project_root=PROJECT_ROOT)
    extractor = PdfMarkdownExtractor(enable_omitted_picture_ocr=False)
    results: list[dict[str, object]] = []

    try:
        converted_root = PROJECT_ROOT / "outputs" / "tmp_office_as_pdf_extract" / output_root.name
        markdown_root = output_root / "markdown"
        ensure_directory(converted_root)
        ensure_directory(markdown_root)

        for input_path in input_paths:
            artifact_stem = make_artifact_stem(input_path)
            pdf_path = input_path
            conversion_used = False

            if input_path.suffix.lower() in OFFICE_EXTENSIONS:
                pdf_path = converted_root / f"{artifact_stem}.pdf"
                conversion_used = manager.convert_source_to_pdf(input_path, pdf_path)
                if not conversion_used or not pdf_path.exists():
                    results.append(
                        {
                            "source_path": input_path.as_posix(),
                            "status": "conversion_failed",
                        }
                    )
                    continue
            elif input_path.suffix.lower() != ".pdf":
                results.append(
                    {
                        "source_path": input_path.as_posix(),
                        "status": "unsupported_extension",
                    }
                )
                continue

            classification = classify_document(pdf_path)
            with fitz.open(pdf_path) as document:
                result = extractor.extract(
                    pdf_path,
                    document,
                    classification,
                    strategy_name="metadata-selected",
                )

            markdown_path = markdown_root / f"{artifact_stem}__metadata-selected.md"
            metadata_path = markdown_root / f"{artifact_stem}__metadata-selected.json"
            markdown_path.write_text(result.markdown.rstrip() + "\n", encoding="utf-8")
            write_json(
                metadata_path,
                {
                    "source_path": input_path.as_posix(),
                    "pdf_path": pdf_path.as_posix(),
                    "conversion_used": conversion_used,
                    "selected_strategy": classification.pdf_parser_strategy,
                    "applied_strategy": result.applied_strategy,
                    "metadata": result.metadata,
                    "elapsed_ms": result.elapsed_ms,
                    "issue_count": len(result.issues),
                    "issues": [issue.__dict__ for issue in result.issues],
                    "char_count": len(result.markdown),
                    "line_count": len(result.markdown.splitlines()),
                },
            )
            results.append(
                {
                    "source_path": input_path.as_posix(),
                    "pdf_path": pdf_path.as_posix(),
                    "markdown_path": markdown_path.as_posix(),
                    "metadata_path": metadata_path.as_posix(),
                    "status": "ok",
                    "selected_strategy": classification.pdf_parser_strategy,
                    "applied_strategy": result.applied_strategy,
                    "char_count": len(result.markdown),
                    "line_count": len(result.markdown.splitlines()),
                }
            )
    finally:
        manager.close()

    manifest_path = output_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_root": output_root.as_posix(),
            "documents": results,
        },
    )

    print(f"manifest: {manifest_path}")
    for item in results:
        status = item.get("status")
        print(f"- {item.get('source_path')}: {status}")
        if status == "ok":
            print(f"  selected={item.get('selected_strategy')} applied={item.get('applied_strategy')}")
            print(f"  markdown={item.get('markdown_path')}")
            print(f"  metadata={item.get('metadata_path')}")

    return 0 if all(item.get("status") == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
