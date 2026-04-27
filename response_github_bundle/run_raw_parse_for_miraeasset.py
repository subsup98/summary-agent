from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from src.parsers.common.models import ParsedDocument
from src.parsers.common.serialization import build_document_summary, extract_markdown_sections
from src.parsers.pdf.pdf_parser import PdfParser
from src.pipeline.parsing_pipeline import ParsingPipeline, ParsingPipelineConfig
from src.shared.io import iso_now


PROJECT_ROOT = Path(__file__).resolve().parent
DESKTOP = Path.home() / "Desktop"
INPUT_FILES = {
    "3분기": DESKTOP / "all_docs" / "미래에셋증권 3분기 실적보고서.pdf",
    "4분기": DESKTOP / "all_docs" / "미래에셋증권 4분기 실적보고서.pdf",
}
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_raw_parse" / "miraeasset_q3_q4"


class RawPdfParser(PdfParser):
    parser_name = "pdf-metadata-router-raw"

    def parse(self, path: Path, classification) -> ParsedDocument:
        parsed = super().parse(path, classification)
        raw_markdown = str(parsed.metadata.get("markdown_raw") or parsed.markdown or "").strip()
        if not raw_markdown:
            raw_markdown = parsed.markdown

        parsed.markdown = raw_markdown
        parsed.sections = extract_markdown_sections(raw_markdown)
        parsed.blocks = []
        parsed.chunks = []
        parsed.metadata["postprocess_logs"] = []
        parsed.metadata["rule_version"] = None
        parsed.metadata["postprocess_bypassed"] = True
        parsed.metadata["raw_parse_runner"] = "run_raw_parse_for_miraeasset.py"
        parsed.created_at = iso_now()
        parsed.summary = build_document_summary(parsed)
        return parsed


class RawPdfParsingPipeline(ParsingPipeline):
    def _make_parsers(self) -> dict[str, object]:
        parsers = super()._make_parsers()
        parsers["pdf"] = RawPdfParser(enable_omitted_picture_ocr=False)
        return parsers


def _prepare_inputs(source_root: Path) -> None:
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    for label, source_path in INPUT_FILES.items():
        if not source_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source_path}")
        shutil.copy2(source_path, source_root / source_path.name)


def _clean_run_root() -> None:
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    os.environ["OPENAI_API_KEY"] = ""

    _clean_run_root()
    source_root = RUN_ROOT / "source"
    interim_root = RUN_ROOT / "data" / "interim"
    structured_root = RUN_ROOT / "data" / "structured"
    outputs_root = RUN_ROOT / "outputs"
    reports_root = outputs_root / "reports"
    comparisons_root = RUN_ROOT / "data" / "comparisons"

    _prepare_inputs(source_root)

    config = ParsingPipelineConfig(
        source_root=source_root,
        interim_root=interim_root,
        structured_root=structured_root,
        outputs_root=outputs_root,
        comparisons_root=comparisons_root,
        reports_root=reports_root,
        enable_omitted_picture_ocr=False,
        max_workers=2,
    )
    summary = RawPdfParsingPipeline(config).run()

    markdown_dir = outputs_root / "markdown"
    desktop_outputs: dict[str, str] = {}
    for label, source_path in INPUT_FILES.items():
        source_name = source_path.name
        matches = [document for document in summary.get("documents", []) if Path(document.get("source_path", "")).name == source_name]
        if not matches:
            raise RuntimeError(f"결과 레코드를 찾을 수 없습니다: {source_name}")
        markdown_path = Path(matches[0]["markdown_path"])
        desktop_path = DESKTOP / f"{label} 파싱.md"
        shutil.copy2(markdown_path, desktop_path)
        desktop_outputs[label] = desktop_path.as_posix()

    result = {
        "run_root": RUN_ROOT.as_posix(),
        "summary_path": (outputs_root / "pipeline_summary.json").as_posix(),
        "parsing_review_index_path": summary.get("parsing_review_index_path"),
        "overlay_review_index_path": summary.get("overlay_review_index_path"),
        "desktop_outputs": desktop_outputs,
    }
    (RUN_ROOT / "result_paths.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
