from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor  # noqa: E402
from src.shared.io import ensure_directory, make_artifact_stem, write_json  # noqa: E402
from src.ui.review_server import ReviewSessionManager  # noqa: E402


OFFICE_EXTENSIONS = {".doc", ".docx", ".hwp", ".hwpx"}


@dataclass
class StructTable:
    page_number: int
    xref: int
    rows: list[list[dict[str, Any]]]


@dataclass
class FitzTable:
    page_number: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StructTree 구조에 PDF text table 값을 채운 하이브리드 markdown을 생성합니다.")
    parser.add_argument("source_path", help="입력 office 또는 pdf 문서 경로")
    parser.add_argument("--label", required=True, help="출력 파일 이름 라벨. 예: doctest, hwptest")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "office_struct_hybrid",
        help="산출물 저장 루트",
    )
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def collect_struct_tables(document: fitz.Document, extractor: PowerPointStructTreeExtractor) -> list[StructTable]:
    catalog_xref = document.pdf_catalog()
    struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
    if struct_root[0] != "xref":
        return []

    root_xref = int(struct_root[1].split()[0])
    page_map = {document.page_xref(i): i + 1 for i in range(document.page_count)}
    table_xrefs: dict[int, list[list[dict[str, Any]]]] = {}
    table_page_map: dict[int, int] = {}
    extractor._find_tables(document, root_xref, page_map, None, table_xrefs)
    extractor._map_table_pages(document, root_xref, page_map, None, table_page_map, set(table_xrefs))

    return [
        StructTable(page_number=table_page_map.get(xref, 0), xref=xref, rows=rows)
        for xref, rows in sorted(table_xrefs.items(), key=lambda item: (table_page_map.get(item[0], 9999), item[0]))
    ]


def collect_fitz_tables(document: fitz.Document) -> list[FitzTable]:
    tables: list[FitzTable] = []
    for page_index in range(document.page_count):
        page = document[page_index]
        try:
            found = page.find_tables().tables
        except Exception:
            continue
        for table in found:
            try:
                bbox = tuple(float(value) for value in table.bbox)
                rows = table.extract()
            except Exception:
                continue
            normalized_rows = [[(cell or "").replace("\n", "<br>").strip() for cell in row] for row in rows]
            tables.append(FitzTable(page_number=page_index + 1, bbox=bbox, rows=normalized_rows))
    return tables


def normalize_struct_rows(rows: list[list[dict[str, Any]]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    normalized: list[list[str]] = []
    for row in rows:
        values = [str(cell.get("text") or "").replace("\n", "<br>").strip() for cell in row]
        values.extend([""] * max(0, width - len(values)))
        normalized.append(values)
    return normalized


def overlap_score(struct_rows: list[list[dict[str, Any]]], fitz_rows: list[list[str]]) -> tuple[int, int, int]:
    struct_row_count = len(struct_rows)
    fitz_row_count = len(fitz_rows)
    struct_width = max((len(row) for row in struct_rows), default=0)
    fitz_width = max((len(row) for row in fitz_rows), default=0)
    return (
        -abs(struct_row_count - fitz_row_count),
        -abs(struct_width - fitz_width),
        min(struct_row_count, fitz_row_count) * min(struct_width, fitz_width),
    )


def select_best_fitz_table(struct_table: StructTable, fitz_tables: list[FitzTable], used_keys: set[tuple[int, int]]) -> FitzTable | None:
    same_page = [table for table in fitz_tables if table.page_number == struct_table.page_number]
    if not same_page:
        return None

    ranked = sorted(
        enumerate(same_page),
        key=lambda item: (
            overlap_score(struct_table.rows, item[1].rows),
            -item[1].bbox[1],
            -item[1].bbox[0],
        ),
        reverse=True,
    )
    for local_index, table in ranked:
        key = (table.page_number, local_index)
        if key not in used_keys:
            used_keys.add(key)
            return table
    return ranked[0][1] if ranked else None


def merge_struct_with_fitz(struct_table: StructTable, fitz_table: FitzTable | None) -> list[list[str]]:
    merged = normalize_struct_rows(struct_table.rows)
    if fitz_table is None:
        return merged

    fitz_rows = fitz_table.rows
    target_width = max((len(row) for row in merged), default=0)
    if target_width == 0:
        return merged

    for row_index, row in enumerate(merged):
        if row_index >= len(fitz_rows):
            continue
        fitz_row = list(fitz_rows[row_index])
        if len(fitz_row) < target_width:
            fitz_row.extend([""] * (target_width - len(fitz_row)))
        elif len(fitz_row) > target_width:
            fitz_row = fitz_row[:target_width]

        for cell_index, value in enumerate(row):
            if value.strip():
                continue
            candidate = fitz_row[cell_index].strip()
            if candidate:
                row[cell_index] = candidate
    return merged


def rows_to_markdown(rows: list[list[str]]) -> str:
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""
    normalized: list[list[str]] = []
    for row in rows:
        padded = list(row) + [""] * max(0, width - len(row))
        normalized.append([cell.replace("|", "\\|") for cell in padded])
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:] or [[""] * width]
    return "\n".join("| " + " | ".join(line) + " |" for line in [header, separator, *body])


def main() -> int:
    args = parse_args()
    source_path = resolve_path(args.source_path)
    output_root = args.output_root.resolve()
    ensure_directory(output_root)

    if not source_path.exists():
        print(f"missing: {source_path}")
        return 1

    manager = ReviewSessionManager(project_root=PROJECT_ROOT)
    extractor = PowerPointStructTreeExtractor()
    artifact_stem = make_artifact_stem(source_path)
    converted_root = PROJECT_ROOT / "outputs" / "tmp_office_struct_hybrid" / args.label
    ensure_directory(converted_root)

    pdf_path = source_path
    converted = False
    try:
        if source_path.suffix.lower() in OFFICE_EXTENSIONS:
            pdf_path = converted_root / f"{artifact_stem}.pdf"
            converted = manager.convert_source_to_pdf(source_path, pdf_path)
            if not converted or not pdf_path.exists():
                print(f"conversion_failed: {source_path}")
                return 1

        with fitz.open(pdf_path) as document:
            struct_tables = collect_struct_tables(document, extractor)
            fitz_tables = collect_fitz_tables(document)

        used_fitz_keys: set[tuple[int, int]] = set()
        lines = [
            f"# Hybrid Table Export - {args.label}",
            "",
            f"- source: `{source_path.as_posix()}`",
            f"- pdf: `{pdf_path.as_posix()}`",
            f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- struct_tables: `{len(struct_tables)}`",
            f"- fitz_tables: `{len(fitz_tables)}`",
            "",
            "> StructTree의 행/열 구조를 기준으로 보고, 비어 있는 셀은 같은 페이지의 PDF table text로 보강했습니다.",
            "",
        ]
        summary: list[dict[str, Any]] = []

        for index, struct_table in enumerate(struct_tables, start=1):
            matched = select_best_fitz_table(struct_table, fitz_tables, used_fitz_keys)
            merged_rows = merge_struct_with_fitz(struct_table, matched)
            struct_row_lengths = [len(row) for row in struct_table.rows]
            fitz_row_lengths = [len(row) for row in matched.rows] if matched else []
            lines.append(f"## Table {index} - Page {struct_table.page_number}")
            lines.append("")
            lines.append(f"- struct_xref: `{struct_table.xref}`")
            lines.append(f"- struct_row_lengths: `{struct_row_lengths}`")
            lines.append(f"- fitz_row_lengths: `{fitz_row_lengths}`")
            lines.append(f"- fitz_bbox: `{list(matched.bbox) if matched else None}`")
            lines.append("")
            lines.append(rows_to_markdown(merged_rows))
            lines.append("")
            summary.append(
                {
                    "table_index": index,
                    "page_number": struct_table.page_number,
                    "struct_xref": struct_table.xref,
                    "struct_row_lengths": struct_row_lengths,
                    "fitz_row_lengths": fitz_row_lengths,
                    "fitz_bbox": list(matched.bbox) if matched else None,
                    "filled_cell_count": sum(1 for row in merged_rows for cell in row if cell.strip()),
                }
            )

        markdown_path = output_root / f"{args.label}.md"
        summary_path = output_root / f"{args.label}.json"
        markdown_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        write_json(
            summary_path,
            {
                "label": args.label,
                "source_path": source_path.as_posix(),
                "pdf_path": pdf_path.as_posix(),
                "converted": converted,
                "struct_table_count": len(struct_tables),
                "fitz_table_count": len(fitz_tables),
                "tables": summary,
            },
        )
        print(f"markdown: {markdown_path}")
        print(f"summary: {summary_path}")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
