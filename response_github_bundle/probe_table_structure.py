"""미래에셋 PDF의 StructTree에서 Table/TR/TD/TH 계층이 실제로 존재하는지 탐색하는 스크립트."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


TABLE_ROLES = {"Table", "TR", "TD", "TH", "THead", "TBody", "TFoot"}
BLOCK_ROLES = {
    "Document", "Part", "Sect", "Div", "P", "L", "LI",
    "Title", "H1", "H2", "H3", "H4", "H5", "H6",
    "Textbox", "Caption", "Note",
    "Table", "TR", "TD", "TH", "THead", "TBody", "TFoot",
}


@dataclass
class TableCell:
    row_index: int
    col_index: int
    role: str  # TD or TH
    text: str
    xref: int


@dataclass
class TableNode:
    page_number: int
    xref: int
    rows: list[list[TableCell]]


def get_name_value(document: fitz.Document, xref: int, key: str) -> str | None:
    value_type, value = document.xref_get_key(xref, key)
    if value_type == "name":
        return value.lstrip("/")
    return None


def get_string_value(document: fitz.Document, xref: int, key: str) -> str | None:
    value_type, value = document.xref_get_key(xref, key)
    if value_type == "string":
        return value
    return None


def parse_k_value(document: fitz.Document, xref: int) -> list[tuple[str, int]]:
    value_type, value = document.xref_get_key(xref, "K")
    if value_type == "xref":
        return [("xref", int(value.split()[0]))]
    if value_type != "array":
        return []
    refs = [("xref", int(match)) for match in re.findall(r"(\d+) 0 R", value)]
    if refs:
        return refs
    return [("int", int(match)) for match in re.findall(r"(?<!\d)(\d+)(?!\d)", value)]


def collect_text_recursive(document: fitz.Document, xref: int, depth: int = 0) -> str:
    """노드와 그 하위 노드에서 ActualText를 재귀적으로 수집."""
    texts: list[str] = []
    actual_text = get_string_value(document, xref, "ActualText") or get_string_value(document, xref, "Alt")
    if actual_text:
        texts.append(actual_text.replace("\ufeff", "").strip())

    for kind, child_xref in parse_k_value(document, xref):
        if kind == "xref":
            child_text = collect_text_recursive(document, child_xref, depth + 1)
            if child_text:
                texts.append(child_text)

    return " ".join(texts).strip()


def walk_for_tables(
    document: fitz.Document,
    xref: int,
    page_map: dict[int, int],
    inherited_page: int | None,
    tables: list[TableNode],
) -> None:
    """StructTree를 재귀 탐색하며 Table 노드를 찾고 TR/TD 계층을 파싱."""
    role = get_name_value(document, xref, "S")

    page_number = inherited_page
    page_value = document.xref_get_key(xref, "Pg")
    if page_value[0] == "xref":
        page_xref = int(page_value[1].split()[0])
        page_number = page_map.get(page_xref, inherited_page)

    if role == "Table" and page_number is not None:
        table = parse_table_node(document, xref, page_number)
        if table and table.rows:
            tables.append(table)
        return  # Table 내부는 parse_table_node에서 처리했으므로 더 내려가지 않음

    children = parse_k_value(document, xref)
    for kind, child_xref in children:
        if kind == "xref":
            walk_for_tables(document, child_xref, page_map, page_number, tables)


def parse_table_node(document: fitz.Document, table_xref: int, page_number: int) -> TableNode | None:
    """Table xref에서 TR/TD 계층을 파싱하여 행/열 구조를 복원."""
    rows: list[list[TableCell]] = []
    children = parse_k_value(document, table_xref)

    for kind, child_xref in children:
        if kind != "xref":
            continue
        child_role = get_name_value(document, child_xref, "S")

        if child_role == "TR":
            cells = parse_row_node(document, child_xref, len(rows))
            if cells:
                rows.append(cells)
        elif child_role in {"THead", "TBody", "TFoot"}:
            # THead/TBody/TFoot는 TR을 감싸는 그룹
            group_children = parse_k_value(document, child_xref)
            for gk, gx in group_children:
                if gk == "xref":
                    gr = get_name_value(document, gx, "S")
                    if gr == "TR":
                        cells = parse_row_node(document, gx, len(rows))
                        if cells:
                            rows.append(cells)

    return TableNode(page_number=page_number, xref=table_xref, rows=rows)


def parse_row_node(document: fitz.Document, tr_xref: int, row_index: int) -> list[TableCell]:
    """TR 노드에서 TD/TH 셀들을 파싱."""
    cells: list[TableCell] = []
    children = parse_k_value(document, tr_xref)

    for col_index, (kind, child_xref) in enumerate(children):
        if kind != "xref":
            continue
        child_role = get_name_value(document, child_xref, "S")
        if child_role in {"TD", "TH"}:
            cell_text = collect_text_recursive(document, child_xref)
            cells.append(TableCell(
                row_index=row_index,
                col_index=col_index,
                role=child_role,
                text=cell_text,
                xref=child_xref,
            ))

    return cells


def table_to_markdown(table: TableNode) -> str:
    """TableNode를 마크다운 테이블로 변환."""
    if not table.rows:
        return ""

    col_count = max(len(row) for row in table.rows)
    lines: list[str] = []

    for row_idx, row in enumerate(table.rows):
        cells = [cell.text.replace("|", "\\|").replace("\n", " ") for cell in row]
        # 부족한 열 패딩
        while len(cells) < col_count:
            cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="StructTree에서 Table/TR/TD 계층을 탐색하여 표 구조를 복원합니다.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--pages", type=str, default="", help="특정 페이지만 (예: 1,3,5-8)")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    pdf_path = args.pdf_path.expanduser().resolve()
    document = fitz.open(pdf_path)

    try:
        catalog_xref = document.pdf_catalog()
        struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
        if struct_root[0] != "xref":
            print("이 PDF에는 StructTree가 없습니다.")
            return

        root_xref = int(struct_root[1].split()[0])
        page_map = {document.page_xref(i): i + 1 for i in range(document.page_count)}

        tables: list[TableNode] = []
        walk_for_tables(document, root_xref, page_map, None, tables)

        # 페이지 필터
        if args.pages:
            page_filter: set[int] = set()
            for part in args.pages.split(","):
                part = part.strip()
                if "-" in part:
                    s, e = part.split("-", 1)
                    for p in range(int(s), int(e) + 1):
                        page_filter.add(p)
                else:
                    page_filter.add(int(part))
            tables = [t for t in tables if t.page_number in page_filter]

        print(f"=== StructTree Table 구조 탐색 결과 ===")
        print(f"PDF: {pdf_path.name}")
        print(f"발견된 Table 노드: {len(tables)}개\n")

        result_tables = []

        for idx, table in enumerate(tables, 1):
            print(f"--- Table {idx} (Page {table.page_number}, xref={table.xref}) ---")
            print(f"행: {len(table.rows)}개, 최대 열: {max(len(r) for r in table.rows) if table.rows else 0}개")

            # 구조 트리 출력
            for row_idx, row in enumerate(table.rows):
                role_label = "TH" if any(c.role == "TH" for c in row) else "TD"
                cells_preview = [f"[{c.role}] {c.text[:30]}" for c in row]
                print(f"  Row {row_idx}: {' | '.join(cells_preview)}")

            print()
            md = table_to_markdown(table)
            print(md)
            print()

            result_tables.append({
                "table_index": idx,
                "page_number": table.page_number,
                "xref": table.xref,
                "row_count": len(table.rows),
                "col_count": max(len(r) for r in table.rows) if table.rows else 0,
                "markdown": md,
                "cells": [
                    {
                        "row": cell.row_index,
                        "col": cell.col_index,
                        "role": cell.role,
                        "text": cell.text,
                        "xref": cell.xref,
                    }
                    for row in table.rows
                    for cell in row
                ],
            })

        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps({"source": pdf_path.name, "tables": result_tables}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nJSON 저장: {args.json_output}")

    finally:
        document.close()


if __name__ == "__main__":
    main()
