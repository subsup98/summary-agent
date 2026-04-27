from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz

from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser
from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "outputs" / "manual_raw_parse" / "miraeasset_q3_q4" / "source"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tests" / "structtree_table_markdown"
EXPECTED_STRATEGY = "structtree-actualtext"
EXPECTED_MARKDOWN_SNIPPETS = [
    "| 3Q24 | 4Q24 | 1Q25 | 2Q25 | 3Q25 |",
    "| 순영업수익 | 572.8 | 393.9 | 539.3 | 713.8 | 594.9 | 1,848.0 |",
    "| ( 연결 ) 지배주주 자기자본 | 11,327 | 12,100 | 12,173 | 12,256 | 12,735 | 12,735 |",
]


def resolve_pdf_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        return path

    if not DEFAULT_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Sample PDF directory not found: {DEFAULT_SOURCE_DIR}")

    preferred = sorted(DEFAULT_SOURCE_DIR.glob("*3*.pdf"))
    if preferred:
        return preferred[0]

    fallback = sorted(DEFAULT_SOURCE_DIR.glob("*.pdf"))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(f"No PDF files found in: {DEFAULT_SOURCE_DIR}")


def is_separator_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def find_markdown_tables(markdown: str) -> list[str]:
    tables: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines) - 1:
        current = lines[index].strip()
        if current.startswith("|") and is_separator_row(lines[index + 1]):
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            tables.append("\n".join(table_lines).strip())
            continue
        index += 1
    return tables


def find_html_tables(markdown: str) -> list[str]:
    return [match.group(0).strip() for match in re.finditer(r"<table>.*?</table>", markdown, flags=re.DOTALL | re.IGNORECASE)]


def find_all_tables(markdown: str) -> list[str]:
    return [*find_markdown_tables(markdown), *find_html_tables(markdown)]


def preview_tables(tables: list[str], limit: int = 4) -> list[list[str]]:
    previews: list[list[str]] = []
    for table in tables[:limit]:
        previews.append(table.splitlines()[:6])
    return previews


def normalize_assertion_text(text: str) -> str:
    normalized = text.replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"\(\s+", "(", normalized)
    normalized = re.sub(r"\s+\)", ")", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_structtree_markdown(pdf_path: Path) -> tuple[str, dict[str, Any]]:
    with fitz.open(pdf_path) as document:
        markdown, metadata = PowerPointStructTreeExtractor().extract_markdown(document)
    return markdown, metadata


def extract_parser_result(pdf_path: Path) -> dict[str, Any]:
    classification = classify_document(pdf_path)
    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classification)
    markdown_metadata = parsed.metadata.get("markdown_metadata", {})
    return {
        "parser_name": parsed.parser_name,
        "selected_strategy": markdown_metadata.get("selected_strategy"),
        "applied_strategy": parsed.metadata.get("markdown_source"),
        "markdown": parsed.markdown,
        "page_table_element_counts": [
            sum(1 for element in page.elements if element.element_type == "table")
            for page in parsed.pages
        ],
        "issue_codes": [issue.code for issue in parsed.issues],
    }


def build_checks(
    structtree_markdown: str,
    structtree_metadata: dict[str, Any],
    parser_result: dict[str, Any],
) -> list[dict[str, Any]]:
    structtree_tables = find_all_tables(structtree_markdown)
    parser_tables = find_all_tables(parser_result["markdown"])
    normalized_parser_markdown = normalize_assertion_text(parser_result["markdown"])
    checks = [
        {
            "name": "structtree extractor used",
            "passed": bool(structtree_metadata.get("used")),
            "details": structtree_metadata,
        },
        {
            "name": "structtree table_count >= 2",
            "passed": int(structtree_metadata.get("table_count", 0)) >= 2,
            "details": {"table_count": structtree_metadata.get("table_count", 0)},
        },
        {
            "name": "structtree markdown tables >= 2",
            "passed": len(structtree_tables) >= 2,
            "details": {"table_blocks": len(structtree_tables)},
        },
        {
            "name": "parser selected strategy is structtree-actualtext",
            "passed": parser_result["selected_strategy"] == EXPECTED_STRATEGY,
            "details": {"selected_strategy": parser_result["selected_strategy"]},
        },
        {
            "name": "parser applied strategy is structtree-actualtext",
            "passed": parser_result["applied_strategy"] == EXPECTED_STRATEGY,
            "details": {"applied_strategy": parser_result["applied_strategy"]},
        },
        {
            "name": "parser markdown tables >= 2",
            "passed": len(parser_tables) >= 2,
            "details": {"table_blocks": len(parser_tables)},
        },
    ]

    for snippet in EXPECTED_MARKDOWN_SNIPPETS:
        normalized_snippet = normalize_assertion_text(snippet)
        checks.append(
            {
                "name": f"markdown contains snippet: {snippet[:40]}",
                "passed": normalized_snippet in normalized_parser_markdown,
                "details": {
                    "snippet": snippet,
                    "normalized_snippet": normalized_snippet,
                },
            }
        )

    return checks


def write_artifacts(
    output_dir: Path,
    structtree_markdown: str,
    parser_markdown: str,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structtree_markdown.md").write_text(structtree_markdown, encoding="utf-8")
    (output_dir / "parser_markdown.md").write_text(parser_markdown, encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression check for StructTree markdown tables.")
    parser.add_argument("--pdf", help="Optional PDF path. Defaults to the saved Mirae Asset 3Q sample.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for test artifacts.",
    )
    args = parser.parse_args()

    pdf_path = resolve_pdf_path(args.pdf)
    output_dir = Path(args.output_dir).expanduser().resolve()

    structtree_markdown, structtree_metadata = extract_structtree_markdown(pdf_path)
    parser_result = extract_parser_result(pdf_path)
    checks = build_checks(structtree_markdown, structtree_metadata, parser_result)
    failed_checks = [check["name"] for check in checks if not check["passed"]]

    structtree_tables = find_all_tables(structtree_markdown)
    parser_tables = find_all_tables(parser_result["markdown"])
    summary = {
        "pdf_path": pdf_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "structtree_metadata": structtree_metadata,
        "structtree_table_block_count": len(structtree_tables),
        "structtree_table_previews": preview_tables(structtree_tables),
        "parser_name": parser_result["parser_name"],
        "parser_selected_strategy": parser_result["selected_strategy"],
        "parser_applied_strategy": parser_result["applied_strategy"],
        "parser_table_block_count": len(parser_tables),
        "parser_table_previews": preview_tables(parser_tables),
        "page_table_element_counts": parser_result["page_table_element_counts"],
        "issue_codes": parser_result["issue_codes"],
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }

    write_artifacts(output_dir, structtree_markdown, parser_result["markdown"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed_checks:
        print("\nFAILED CHECKS:", file=sys.stderr)
        for name in failed_checks:
            print(f"- {name}", file=sys.stderr)
        return 1

    print("\nPASS: StructTree markdown table regression check succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
