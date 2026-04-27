"""
기존 파이프라인 + StructTree 테이블 구조화를 추가한 테스트 스크립트.

기존 코드는 건드리지 않고, StructTreeExtractor를 서브클래싱하여
Table/TR/TD 계층을 마크다운 테이블로 변환하는 로직만 추가합니다.

사용법:
    cd response_github_bundle
    python run_table_aware_parse.py "C:/Users/yongseop.im/Desktop/all_docs/미래에셋증권 3분기 실적보고서.pdf"
    python run_table_aware_parse.py "C:/Users/yongseop.im/Desktop/all_docs/미래에셋증권 3분기 실적보고서.pdf" --pages 5,16
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import fitz

from src.parsers.pdf.structtree_extractor import (
    BLOCK_ROLES,
    HEADING_ROLES,
    PowerPointStructTreeExtractor,
    StructTextRun,
)


# ---------------------------------------------------------------------------
# StructTree 테이블 구조 파서 (probe_table_structure.py에서 가져온 로직)
# ---------------------------------------------------------------------------

def _get_name_value(doc: fitz.Document, xref: int, key: str) -> str | None:
    vt, v = doc.xref_get_key(xref, key)
    return v.lstrip("/") if vt == "name" else None


def _get_string_value(doc: fitz.Document, xref: int, key: str) -> str | None:
    vt, v = doc.xref_get_key(xref, key)
    return v if vt == "string" else None


def _parse_k(doc: fitz.Document, xref: int) -> list[tuple[str, int]]:
    vt, v = doc.xref_get_key(xref, "K")
    if vt == "xref":
        return [("xref", int(v.split()[0]))]
    if vt != "array":
        return []
    refs = [("xref", int(m)) for m in re.findall(r"(\d+) 0 R", v)]
    if refs:
        return refs
    return [("int", int(m)) for m in re.findall(r"(?<!\d)(\d+)(?!\d)", v)]


def _collect_text(doc: fitz.Document, xref: int) -> str:
    texts: list[str] = []
    at = _get_string_value(doc, xref, "ActualText") or _get_string_value(doc, xref, "Alt")
    if at:
        texts.append(at.replace("\ufeff", "").strip())
    for kind, child_xref in _parse_k(doc, xref):
        if kind == "xref":
            t = _collect_text(doc, child_xref)
            if t:
                texts.append(t)
    return " ".join(texts).strip()


def _parse_row(doc: fitz.Document, tr_xref: int) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for kind, child_xref in _parse_k(doc, tr_xref):
        if kind != "xref":
            continue
        role = _get_name_value(doc, child_xref, "S")
        if role in {"TD", "TH"}:
            cells.append({"role": role, "text": _collect_text(doc, child_xref), "xref": child_xref})
    return cells


def _parse_table(doc: fitz.Document, table_xref: int) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for kind, child_xref in _parse_k(doc, table_xref):
        if kind != "xref":
            continue
        role = _get_name_value(doc, child_xref, "S")
        if role == "TR":
            cells = _parse_row(doc, child_xref)
            if cells:
                rows.append(cells)
        elif role in {"THead", "TBody", "TFoot"}:
            for gk, gx in _parse_k(doc, child_xref):
                if gk == "xref" and _get_name_value(doc, gx, "S") == "TR":
                    cells = _parse_row(doc, gx)
                    if cells:
                        rows.append(cells)
    return rows


def _table_rows_to_markdown(rows: list[list[dict[str, Any]]]) -> str:
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        cells = [c["text"].replace("|", "\\|").replace("\n", " ") for c in row]
        while len(cells) < col_count:
            cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
        if row_idx == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 테이블 구조를 인식하는 StructTreeExtractor 서브클래스
# ---------------------------------------------------------------------------

class TableAwareStructTreeExtractor(PowerPointStructTreeExtractor):
    """extract_markdown에서 Table role을 만나면 마크다운 테이블로 변환."""

    def extract_markdown(self, document: fitz.Document) -> tuple[str, dict[str, Any]]:
        # StructTree 루트 찾기
        catalog_xref = document.pdf_catalog()
        struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
        if struct_root[0] != "xref":
            return "", {"used": False, "reason": "no-struct-tree"}

        root_xref = int(struct_root[1].split()[0])
        page_map = {document.page_xref(i): i + 1 for i in range(document.page_count)}

        # 테이블 xref 수집 (어떤 xref가 Table인지 미리 파악)
        table_xrefs: dict[int, list[list[dict[str, Any]]]] = {}
        self._find_tables(document, root_xref, page_map, None, table_xrefs)

        # 테이블에 속하는 모든 xref 수집 (run에서 제외하기 위해)
        table_member_xrefs: set[int] = set()
        for table_xref in table_xrefs:
            self._collect_descendant_xrefs(document, table_xref, table_member_xrefs)

        # 기존 extract_runs 실행
        runs = self.extract_runs(document)
        if not runs and not table_xrefs:
            return "", {"used": False, "reason": "no-actualtext-runs"}

        # 테이블 소속 run의 block_id 수집
        table_block_ids: set[int] = set()
        for run in runs:
            if run.block_id in table_member_xrefs:
                table_block_ids.add(run.block_id)

        # 테이블 xref → 페이지 매핑
        table_page_map: dict[int, int] = {}
        self._map_table_pages(document, root_xref, page_map, None, table_page_map, set(table_xrefs.keys()))

        # 마크다운 생성 (테이블 소속 run은 건너뛰고, Table 노드에서 구조화된 마크다운 삽입)
        markdown_lines: list[str] = []
        current_page: int | None = None
        current_block_id: int | None = None
        current_block_role: str | None = None
        current_fragments: list[str] = []
        block_count = 0
        emitted_tables: set[int] = set()
        table_count = 0

        def flush_block() -> None:
            nonlocal current_fragments, block_count
            if current_page is None or not current_fragments:
                current_fragments = []
                return
            block_text = self._normalize_text(self._join_fragments(current_fragments))
            current_fragments = []
            if not block_text:
                return
            block_count += 1
            if current_block_role in HEADING_ROLES:
                heading_level = 1 if current_block_role == "Title" else int(current_block_role[-1])
                markdown_lines.append(f"{'#' * min(max(heading_level, 1), 6)} {block_text}")
            elif current_block_role == "LI":
                block_text = re.sub(r"^[\-\u2022\u25AA\u25CF\uf0a7+\s]+", "", block_text)
                markdown_lines.append(f"- {block_text}")
            else:
                markdown_lines.append(block_text)
            markdown_lines.append("")

        def emit_page_tables(page_number: int) -> None:
            nonlocal table_count
            for txref, rows in table_xrefs.items():
                if txref in emitted_tables:
                    continue
                if table_page_map.get(txref) != page_number:
                    continue
                emitted_tables.add(txref)
                md = _table_rows_to_markdown(rows)
                if md:
                    table_count += 1
                    markdown_lines.append(md)
                    markdown_lines.append("")

        for run in runs:
            # 테이블 소속 run은 건너뛰기
            if run.block_id in table_member_xrefs:
                # 이 run의 페이지에 해당하는 테이블을 아직 안 넣었으면 삽입
                if run.page_number != current_page:
                    flush_block()
                    current_page = run.page_number
                    markdown_lines.append(f"# Page {run.page_number}")
                    markdown_lines.append("")
                    current_block_id = None
                    current_block_role = None
                emit_page_tables(run.page_number)
                continue

            if run.page_number != current_page:
                flush_block()
                # 이전 페이지의 남은 테이블 emit
                if current_page is not None:
                    emit_page_tables(current_page)
                current_page = run.page_number
                markdown_lines.append(f"# Page {run.page_number}")
                markdown_lines.append("")
                current_block_id = None
                current_block_role = None

            if run.block_id != current_block_id:
                flush_block()
                current_block_id = run.block_id
                current_block_role = run.block_role

            current_fragments.append(run.text)

        flush_block()
        if current_page is not None:
            emit_page_tables(current_page)

        # 어떤 페이지에도 emit 안 된 테이블 처리
        for txref, rows in table_xrefs.items():
            if txref not in emitted_tables:
                page = table_page_map.get(txref)
                if page:
                    emitted_tables.add(txref)
                    md = _table_rows_to_markdown(rows)
                    if md:
                        table_count += 1
                        markdown_lines.append(f"# Page {page}")
                        markdown_lines.append("")
                        markdown_lines.append(md)
                        markdown_lines.append("")

        markdown = "\n".join(markdown_lines).strip()
        metadata = {
            "used": bool(markdown),
            "source": "structtree-actualtext-table-aware",
            "run_count": len(runs),
            "block_count": block_count,
            "table_count": table_count,
            "pages": sorted({run.page_number for run in runs}),
        }
        return markdown, metadata

    def _find_tables(
        self,
        doc: fitz.Document,
        xref: int,
        page_map: dict[int, int],
        inherited_page: int | None,
        result: dict[int, list[list[dict[str, Any]]]],
    ) -> None:
        role = _get_name_value(doc, xref, "S")
        page_number = inherited_page
        pv = doc.xref_get_key(xref, "Pg")
        if pv[0] == "xref":
            px = int(pv[1].split()[0])
            page_number = page_map.get(px, inherited_page)

        if role == "Table":
            rows = _parse_table(doc, xref)
            if rows:
                result[xref] = rows
            return

        for kind, child_xref in _parse_k(doc, xref):
            if kind == "xref":
                self._find_tables(doc, child_xref, page_map, page_number, result)

    def _map_table_pages(
        self,
        doc: fitz.Document,
        xref: int,
        page_map: dict[int, int],
        inherited_page: int | None,
        result: dict[int, int],
        target_xrefs: set[int],
    ) -> None:
        page_number = inherited_page
        pv = doc.xref_get_key(xref, "Pg")
        if pv[0] == "xref":
            px = int(pv[1].split()[0])
            page_number = page_map.get(px, inherited_page)

        if xref in target_xrefs and page_number is not None:
            result[xref] = page_number

        for kind, child_xref in _parse_k(doc, xref):
            if kind == "xref":
                self._map_table_pages(doc, child_xref, page_map, page_number, result, target_xrefs)

    def _collect_descendant_xrefs(self, doc: fitz.Document, xref: int, result: set[int]) -> None:
        result.add(xref)
        for kind, child_xref in _parse_k(doc, xref):
            if kind == "xref":
                self._collect_descendant_xrefs(doc, child_xref, result)


# ---------------------------------------------------------------------------
# 테스트 실행
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="기존 파이프라인의 structtree markdown에 테이블 구조화를 추가하여 비교 테스트합니다."
    )
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--pages", type=str, default="", help="특정 페이지만 (예: 5,16)")
    args = parser.parse_args()

    pdf_path = args.pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_path}")

    output_dir = Path(__file__).resolve().parent / "outputs" / "table_aware_test"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    try:
        # 기존 방식
        original_extractor = PowerPointStructTreeExtractor()
        original_md, original_meta = original_extractor.extract_markdown(document)

        # 테이블 구조화 방식
        table_aware_extractor = TableAwareStructTreeExtractor()
        table_aware_md, table_aware_meta = table_aware_extractor.extract_markdown(document)
    finally:
        document.close()

    # 페이지 필터링
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

        def filter_pages(md: str) -> str:
            lines = md.split("\n")
            result: list[str] = []
            include = False
            for line in lines:
                if line.startswith("# Page "):
                    try:
                        pn = int(line.split()[-1])
                        include = pn in page_filter
                    except ValueError:
                        include = False
                if include:
                    result.append(line)
            return "\n".join(result).strip()

        original_md = filter_pages(original_md)
        table_aware_md = filter_pages(table_aware_md)

    # 결과 저장
    original_path = output_dir / "original.md"
    table_aware_path = output_dir / "table_aware.md"
    meta_path = output_dir / "metadata.json"

    original_path.write_text(original_md, encoding="utf-8")
    table_aware_path.write_text(table_aware_md, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "source": pdf_path.name,
                "original_metadata": original_meta,
                "table_aware_metadata": table_aware_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"=== 비교 결과 ===")
    print(f"PDF: {pdf_path.name}")
    print(f"")
    print(f"[기존] run_count={original_meta.get('run_count')}, block_count={original_meta.get('block_count')}")
    print(f"[개선] run_count={table_aware_meta.get('run_count')}, block_count={table_aware_meta.get('block_count')}, table_count={table_aware_meta.get('table_count')}")
    print(f"")
    print(f"기존 마크다운 길이: {len(original_md)} chars")
    print(f"개선 마크다운 길이: {len(table_aware_md)} chars")
    print(f"")
    print(f"저장 위치:")
    print(f"  기존: {original_path}")
    print(f"  개선: {table_aware_path}")
    print(f"  메타: {meta_path}")


if __name__ == "__main__":
    main()
