"""
PDF 텍스트를 bbox 위치 기반으로 배치한 마크다운을 생성합니다.
y좌표로 줄을 나누고, x좌표로 공백을 넣어서 대략적인 위치를 표현합니다.
StructTree 테이블은 마크다운 테이블로 삽입합니다.

사용법:
    python run_layout_md.py "경로/파일.pdf"
    python run_layout_md.py "경로/파일.pdf" --pages 5,16
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import median
from typing import Any

import fitz


# ---------------------------------------------------------------------------
# StructTree 테이블 파싱
# ---------------------------------------------------------------------------

def _get_name(doc: fitz.Document, xref: int, key: str) -> str | None:
    vt, v = doc.xref_get_key(xref, key)
    return v.lstrip("/") if vt == "name" else None

def _get_str(doc: fitz.Document, xref: int, key: str) -> str | None:
    vt, v = doc.xref_get_key(xref, key)
    return v if vt == "string" else None

def _parse_k(doc: fitz.Document, xref: int) -> list[tuple[str, int]]:
    vt, v = doc.xref_get_key(xref, "K")
    if vt == "xref":
        return [("xref", int(v.split()[0]))]
    if vt != "array":
        return []
    refs = [("xref", int(m)) for m in re.findall(r"(\d+) 0 R", v)]
    return refs if refs else [("int", int(m)) for m in re.findall(r"(?<!\d)(\d+)(?!\d)", v)]

def _collect_text(doc: fitz.Document, xref: int) -> str:
    texts: list[str] = []
    at = _get_str(doc, xref, "ActualText") or _get_str(doc, xref, "Alt")
    if at:
        texts.append(at.replace("\ufeff", "").strip())
    for kind, cx in _parse_k(doc, xref):
        if kind == "xref":
            t = _collect_text(doc, cx)
            if t:
                texts.append(t)
    return " ".join(texts).strip()

def _parse_table(doc: fitz.Document, xref: int) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for kind, cx in _parse_k(doc, xref):
        if kind != "xref":
            continue
        role = _get_name(doc, cx, "S")
        if role == "TR":
            cells = []
            for ck, ccx in _parse_k(doc, cx):
                if ck == "xref" and _get_name(doc, ccx, "S") in {"TD", "TH"}:
                    cells.append({"role": _get_name(doc, ccx, "S"), "text": _collect_text(doc, ccx)})
            if cells:
                rows.append(cells)
        elif role in {"THead", "TBody", "TFoot"}:
            for gk, gx in _parse_k(doc, cx):
                if gk == "xref" and _get_name(doc, gx, "S") == "TR":
                    cells = []
                    for ck, ccx in _parse_k(doc, gx):
                        if ck == "xref" and _get_name(doc, ccx, "S") in {"TD", "TH"}:
                            cells.append({"role": _get_name(doc, ccx, "S"), "text": _collect_text(doc, ccx)})
                    if cells:
                        rows.append(cells)
    return rows

def find_all_tables(doc: fitz.Document) -> dict[int, list[dict]]:
    catalog = doc.pdf_catalog()
    sr = doc.xref_get_key(catalog, "StructTreeRoot")
    if sr[0] != "xref":
        return {}
    root_xref = int(sr[1].split()[0])
    page_map = {doc.page_xref(i): i + 1 for i in range(doc.page_count)}
    tables_by_page: dict[int, list[dict]] = {}

    def walk(xref: int, inherited_page: int | None) -> None:
        role = _get_name(doc, xref, "S")
        page_num = inherited_page
        pv = doc.xref_get_key(xref, "Pg")
        if pv[0] == "xref":
            page_num = page_map.get(int(pv[1].split()[0]), inherited_page)
        if role == "Table" and page_num:
            rows = _parse_table(doc, xref)
            if rows:
                tables_by_page.setdefault(page_num, []).append({"rows": rows})
            return
        for kind, cx in _parse_k(doc, xref):
            if kind == "xref":
                walk(cx, page_num)

    walk(root_xref, None)
    return tables_by_page

def table_to_markdown(rows: list[list[dict]]) -> str:
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    lines: list[str] = []
    for ri, row in enumerate(rows):
        cells = [c["text"].replace("|", "\\|").replace("\n", " ") for c in row]
        while len(cells) < col_count:
            cells.append("")
        lines.append("| " + " | ".join(cells) + " |")
        if ri == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 위치 기반 마크다운 렌더링
# ---------------------------------------------------------------------------

COLUMNS = 120  # 출력 너비 (문자 수)


def extract_blocks(page: fitz.Page) -> list[dict]:
    """페이지에서 텍스트 블록 + bbox 추출."""
    blocks: list[dict] = []
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", ())
        if len(bbox) != 4:
            continue
        lines: list[str] = []
        for line in block.get("lines", []):
            lt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if lt:
                lines.append(lt)
        text = "\n".join(lines).strip()
        if text:
            blocks.append({
                "bbox": [float(v) for v in bbox],
                "text": text,
            })
    return blocks


def is_in_table_area(block_bbox: list[float], table_bboxes: list[tuple]) -> bool:
    """블록이 테이블 영역 안에 있는지 확인."""
    by0, by1 = block_bbox[1], block_bbox[3]
    bx0, bx1 = block_bbox[0], block_bbox[2]
    for tb in table_bboxes:
        if by0 >= tb[1] - 3 and by1 <= tb[3] + 3 and bx0 >= tb[0] - 10 and bx1 <= tb[2] + 10:
            return True
    return False


def estimate_table_bbox_from_blocks(
    table_rows: list[list[dict]],
    page_blocks: list[dict],
) -> tuple[float, float, float, float] | None:
    cell_keys = set()
    for row in table_rows:
        for cell in row:
            k = re.sub(r"\s+", "", cell["text"])
            if k:
                cell_keys.add(k)
    if not cell_keys:
        return None
    matched: list[list[float]] = []
    for b in page_blocks:
        bk = re.sub(r"\s+", "", b["text"])
        if bk in cell_keys:
            matched.append(b["bbox"])
    if not matched:
        return None
    return (
        min(b[0] for b in matched) - 5,
        min(b[1] for b in matched) - 5,
        max(b[2] for b in matched) + 5,
        max(b[3] for b in matched) + 5,
    )


def render_positioned_line(items: list[dict], page_width: float) -> str:
    """같은 y줄에 있는 텍스트 조각들을 x좌표 기반으로 한 줄에 배치."""
    items_sorted = sorted(items, key=lambda it: it["x0"])
    line_chars = [" "] * COLUMNS
    cursor = 0

    for item in items_sorted:
        text = item["text"].replace("\n", " ").strip()
        if not text:
            continue
        target_col = int(round((item["x0"] / page_width) * (COLUMNS - 1)))
        start = max(cursor, min(COLUMNS - 1, target_col))
        for i, ch in enumerate(text):
            col = start + i
            if col >= COLUMNS:
                break
            line_chars[col] = ch
        cursor = min(COLUMNS, start + len(text) + 1)

    return "".join(line_chars).rstrip()


def render_page_layout_md(
    page: fitz.Page,
    page_number: int,
    page_tables: list[dict],
) -> str:
    """한 페이지의 위치 기반 마크다운 생성."""
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    blocks = extract_blocks(page)

    # 테이블 bbox 추정
    table_bboxes: list[tuple] = []
    table_y_positions: list[tuple[float, str]] = []  # (y위치, 마크다운)

    for tbl in page_tables:
        tb = estimate_table_bbox_from_blocks(tbl["rows"], blocks)
        if tb:
            table_bboxes.append(tb)
            md = table_to_markdown(tbl["rows"])
            table_y_positions.append((tb[1], md))

    # 테이블 영역 밖 블록만 수집
    text_items: list[dict] = []
    for b in blocks:
        if is_in_table_area(b["bbox"], table_bboxes):
            continue
        text_items.append({
            "x0": b["bbox"][0],
            "y0": b["bbox"][1],
            "y1": b["bbox"][3],
            "text": b["text"],
        })

    # y좌표 기반으로 줄 그룹핑
    all_items = sorted(text_items, key=lambda it: (it["y0"], it["x0"]))

    if not all_items and not table_y_positions:
        return ""

    # 높이 기반 줄 간격 계산
    heights = [max(1.0, it["y1"] - it["y0"]) for it in all_items] if all_items else [12.0]
    line_tolerance = max(3.0, median(heights) * 0.4)

    # 줄 그룹핑
    rows: list[dict] = []
    for item in all_items:
        if rows and abs(item["y0"] - rows[-1]["y"]) <= line_tolerance:
            rows[-1]["items"].append(item)
        else:
            rows.append({"y": item["y0"], "items": [item]})

    # 테이블도 y위치에 삽입
    table_entries = sorted(table_y_positions, key=lambda t: t[0])

    # 모든 요소를 y순서로 합치기
    output_lines: list[str] = []
    output_lines.append(f"{'=' * COLUMNS}")
    output_lines.append(f"  Page {page_number}".center(COLUMNS))
    output_lines.append(f"{'=' * COLUMNS}")
    output_lines.append("")

    row_idx = 0
    tbl_idx = 0
    prev_y: float | None = None

    positive_gaps = []
    for i in range(1, len(rows)):
        gap = rows[i]["y"] - rows[i - 1]["y"]
        if gap > line_tolerance:
            positive_gaps.append(gap)
    base_gap = median(positive_gaps) if positive_gaps else max(12.0, median(heights) * 1.5)

    while row_idx < len(rows) or tbl_idx < len(table_entries):
        row_y = rows[row_idx]["y"] if row_idx < len(rows) else float("inf")
        tbl_y = table_entries[tbl_idx][0] if tbl_idx < len(table_entries) else float("inf")

        if tbl_y <= row_y:
            # 테이블 삽입
            if prev_y is not None:
                gap = tbl_y - prev_y
                if gap > base_gap * 1.5:
                    blank_count = max(1, min(4, int(round(gap / base_gap)) - 1))
                    output_lines.extend([""] * blank_count)
            output_lines.append("")
            output_lines.append(table_entries[tbl_idx][1])
            output_lines.append("")
            prev_y = tbl_y
            tbl_idx += 1
        else:
            # 텍스트 줄 삽입
            if prev_y is not None:
                gap = row_y - prev_y
                if gap > base_gap * 1.5:
                    blank_count = max(1, min(4, int(round(gap / base_gap)) - 1))
                    output_lines.extend([""] * blank_count)
            line = render_positioned_line(rows[row_idx]["items"], pw)
            output_lines.append(line)
            prev_y = row_y
            row_idx += 1

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 위치 기반 마크다운 레이아웃 생성")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--pages", type=str, default="", help="페이지 필터 (예: 5,16)")
    parser.add_argument("--columns", type=int, default=120, help="출력 너비 (문자 수)")
    args = parser.parse_args()

    global COLUMNS
    COLUMNS = args.columns

    pdf_path = args.pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_path}")

    page_filter: set[int] | None = None
    if args.pages:
        page_filter = set()
        for part in args.pages.split(","):
            part = part.strip()
            if "-" in part:
                s, e = part.split("-", 1)
                for p in range(int(s), int(e) + 1):
                    page_filter.add(p)
            else:
                page_filter.add(int(part))

    tables_by_page = {}
    document = fitz.open(pdf_path)
    try:
        tables_by_page = find_all_tables(document)
        all_pages_md: list[str] = []

        for page_index in range(document.page_count):
            page_number = page_index + 1
            if page_filter and page_number not in page_filter:
                continue
            page = document[page_index]
            page_tables = tables_by_page.get(page_number, [])
            page_md = render_page_layout_md(page, page_number, page_tables)
            if page_md:
                all_pages_md.append(page_md)
    finally:
        document.close()

    result = "\n\n".join(all_pages_md)

    output_dir = Path(__file__).resolve().parent / "outputs" / "layout_md"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_layout.md"
    output_path.write_text(result, encoding="utf-8")
    print(f"output: {output_path}")
    print(f"pages: {len(all_pages_md)}")
    print(f"length: {len(result)} chars")


if __name__ == "__main__":
    main()
