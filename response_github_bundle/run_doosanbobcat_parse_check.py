from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from src.pipeline.parsing_pipeline import ParsingPipeline, ParsingPipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PDF = PROJECT_ROOT / "outputs" / "ui_runs" / "review-20260420-092501" / "source" / "한화투자증권 두산밥캣 기업분석 리포트.pdf"
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_parse_check" / "doosanbobcat_20260421"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    os.environ["OPENAI_API_KEY"] = ""

    if not SOURCE_PDF.exists():
        raise FileNotFoundError(f"Source PDF not found: {SOURCE_PDF}")

    _reset_dir(RUN_ROOT)
    source_root = RUN_ROOT / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    copied_source = source_root / SOURCE_PDF.name
    shutil.copy2(SOURCE_PDF, copied_source)

    config = ParsingPipelineConfig(
        source_root=source_root,
        interim_root=RUN_ROOT / "interim",
        structured_root=RUN_ROOT / "structured",
        outputs_root=RUN_ROOT / "parsing",
        comparisons_root=RUN_ROOT / "comparisons",
        reports_root=RUN_ROOT / "reports",
        enable_omitted_picture_ocr=False,
        max_workers=1,
    )
    summary = ParsingPipeline(config).run()

    result = {
        "run_root": RUN_ROOT.as_posix(),
        "source_pdf": copied_source.as_posix(),
        "summary_path": (RUN_ROOT / "parsing" / "pipeline_summary.json").as_posix(),
        "latest_run_markdown_path": summary.get("latest_run_markdown_path"),
        "parsing_review_index_path": summary.get("parsing_review_index_path"),
        "documents": summary.get("documents", []),
    }
    result_path = RUN_ROOT / "result_paths.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
