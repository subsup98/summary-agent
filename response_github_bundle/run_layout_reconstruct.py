"""
PDF의 모든 텍스트를 원래 위치에 배치한 HTML을 생성합니다.
StructTree 테이블 구조가 있으면 해당 영역에 마크다운 테이블을 오버레이합니다.

사용법:
    python run_layout_reconstruct.py "경로/파일.pdf"
    python run_layout_reconstruct.py "경로/파일.pdf" --pages 5,16
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import fitz


# ---------------------------------------------------------------------------
# StructTree 테이블 파싱 (probe_table_structure.py 로직 재사용)
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
    """page_number → [{"xref", "rows", "bbox"}] 매핑 반환."""
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
                tables_by_page.setdefault(page_num, []).append({"xref": xref, "rows": rows})
            return
        for kind, cx in _parse_k(doc, xref):
            if kind == "xref":
                walk(cx, page_num)

    walk(root_xref, None)
    return tables_by_page


# ---------------------------------------------------------------------------
# 페이지 텍스트 블록 추출 (bbox 포함)
# ---------------------------------------------------------------------------

def extract_text_blocks(page: fitz.Page, page_number: int) -> list[dict]:
    """get_text("dict")에서 블록별 bbox + 텍스트 추출."""
    blocks: list[dict] = []
    page_dict = page.get_text("dict", sort=True)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", ())
        if len(bbox) != 4:
            continue
        lines: list[dict] = []
        for line in block.get("lines", []):
            spans_data: list[dict] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    spans_data.append({
                        "text": text,
                        "size": round(span.get("size", 10), 1),
                        "flags": span.get("flags", 0),
                        "font": span.get("font", ""),
                        "bbox": list(span.get("bbox", ())),
                    })
            if spans_data:
                line_text = "".join(s["text"] for s in spans_data).strip()
                if line_text:
                    lines.append({
                        "text": line_text,
                        "bbox": list(line.get("bbox", ())),
                        "spans": spans_data,
                    })
        if lines:
            block_text = "\n".join(l["text"] for l in lines)
            blocks.append({
                "page_number": page_number,
                "bbox": [float(v) for v in bbox],
                "text": block_text,
                "lines": lines,
                "font_size": max((s["size"] for l in lines for s in l["spans"]), default=10),
            })
    return blocks


# ---------------------------------------------------------------------------
# 테이블 bbox 추정 (테이블 셀 텍스트와 visible block 매칭)
# ---------------------------------------------------------------------------

def estimate_table_bbox(
    table_rows: list[list[dict]],
    page_blocks: list[dict],
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    """테이블 셀 텍스트를 page block과 매칭해서 bbox 추정."""
    cell_texts = set()
    for row in table_rows:
        for cell in row:
            t = re.sub(r"\s+", "", cell["text"])
            if t:
                cell_texts.add(t)
    if not cell_texts:
        return None

    matched_bboxes: list[list[float]] = []
    for block in page_blocks:
        block_key = re.sub(r"\s+", "", block["text"])
        if block_key in cell_texts:
            matched_bboxes.append(block["bbox"])

    if not matched_bboxes:
        return None

    x0 = max(0, min(b[0] for b in matched_bboxes) - 5)
    y0 = max(0, min(b[1] for b in matched_bboxes) - 5)
    x1 = min(page_width, max(b[2] for b in matched_bboxes) + 5)
    y1 = min(page_height, max(b[3] for b in matched_bboxes) + 5)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# HTML 생성
# ---------------------------------------------------------------------------

def render_table_html(rows: list[list[dict]]) -> str:
    """테이블을 HTML <table>로 렌더."""
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    html_lines = ['<table>']
    for ri, row in enumerate(rows):
        html_lines.append('  <tr>')
        for ci, cell in enumerate(row):
            tag = "th" if cell["role"] == "TH" else "td"
            html_lines.append(f'    <{tag}>{html.escape(cell["text"])}</{tag}>')
        for _ in range(col_count - len(row)):
            html_lines.append('    <td></td>')
        html_lines.append('  </tr>')
    html_lines.append('</table>')
    return "\n".join(html_lines)


def build_html(
    pdf_path: Path,
    document: fitz.Document,
    page_filter: set[int] | None,
    output_dir: Path,
) -> str:
    tables_by_page = find_all_tables(document)
    matrix = fitz.Matrix(2.0, 2.0)
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    page_sections: list[str] = []

    for page_index in range(document.page_count):
        page_number = page_index + 1
        if page_filter and page_number not in page_filter:
            continue

        page = document[page_index]
        pw = float(page.rect.width)
        ph = float(page.rect.height)

        # 페이지 이미지 생성
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_name = f"page-{page_number:03d}.png"
        (asset_dir / img_name).write_bytes(pixmap.tobytes("png"))

        # 텍스트 블록 추출
        blocks = extract_text_blocks(page, page_number)

        # 이 페이지의 테이블 정보
        page_tables = tables_by_page.get(page_number, [])

        # 테이블 bbox 추정 & 테이블 영역에 속하는 블록 마킹
        table_overlays: list[str] = []
        table_block_indices: set[int] = set()

        for tbl in page_tables:
            tbl_bbox = estimate_table_bbox(tbl["rows"], blocks, pw, ph)
            if not tbl_bbox:
                continue

            # 테이블 영역에 겹치는 블록 마킹
            for bi, block in enumerate(blocks):
                bb = block["bbox"]
                # y 기준으로 테이블 영역 안에 있는지 체크
                if bb[1] >= tbl_bbox[1] - 2 and bb[3] <= tbl_bbox[3] + 2:
                    if bb[0] >= tbl_bbox[0] - 10 and bb[2] <= tbl_bbox[2] + 10:
                        table_block_indices.add(bi)

            # 테이블 HTML 오버레이
            left = tbl_bbox[0] / pw * 100
            top = tbl_bbox[1] / ph * 100
            width = (tbl_bbox[2] - tbl_bbox[0]) / pw * 100
            height = (tbl_bbox[3] - tbl_bbox[1]) / ph * 100

            table_html = render_table_html(tbl["rows"])
            table_overlays.append(f"""
<div class="table-overlay" style="left:{left:.2f}%;top:{top:.2f}%;width:{width:.2f}%;max-height:{height + 2:.2f}%">
  {table_html}
</div>""")

        # 텍스트 블록 오버레이 (테이블 소속 제외)
        text_overlays: list[str] = []
        for bi, block in enumerate(blocks):
            if bi in table_block_indices:
                continue
            bb = block["bbox"]
            left = bb[0] / pw * 100
            top = bb[1] / ph * 100
            width = (bb[2] - bb[0]) / pw * 100
            height = (bb[3] - bb[1]) / ph * 100
            font_size = block["font_size"]
            # 폰트 크기 비율 조정 (PDF pt → 화면 비율)
            fs_pct = font_size / ph * 100

            text_escaped = html.escape(block["text"]).replace("\n", "<br>")
            text_overlays.append(f"""
<div class="text-block" style="left:{left:.2f}%;top:{top:.2f}%;width:{width:.2f}%;font-size:{fs_pct:.3f}vw"
     title="{html.escape(block['text'][:100])}">
  {text_escaped}
</div>""")

        page_sections.append(f"""
<section class="page-card" id="page-{page_number}">
  <div class="page-header">
    <h2>Page {page_number}</h2>
    <span class="page-meta">blocks: {len(blocks)} | tables: {len(page_tables)} | table-blocks-replaced: {len(table_block_indices)}</span>
  </div>
  <div class="page-compare">
    <div class="page-col">
      <h3>원본 PDF</h3>
      <div class="page-stage">
        <img src="assets/{img_name}" alt="Page {page_number}">
      </div>
    </div>
    <div class="page-col">
      <h3>재구성 레이아웃</h3>
      <div class="page-stage reconstructed">
        <img src="assets/{img_name}" alt="Page {page_number}" class="ghost-img">
        {"".join(text_overlays)}
        {"".join(table_overlays)}
      </div>
    </div>
  </div>
</section>""")

    page_count = len(page_sections)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF Layout Reconstruction</title>
  <style>
    :root {{
      --bg: #f4f1ec;
      --card: #fff;
      --border: rgba(0,0,0,0.1);
      --table-bg: rgba(30, 133, 100, 0.08);
      --table-border: rgba(30, 133, 100, 0.35);
      --text-bg: rgba(32, 124, 202, 0.06);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
      padding: 1rem;
    }}
    .hero {{
      text-align: center;
      padding: 1.5rem 1rem;
      margin-bottom: 1rem;
    }}
    .hero h1 {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
    .hero p {{ color: #666; font-size: 0.9rem; }}

    .controls {{
      text-align: center;
      margin-bottom: 1rem;
      display: flex;
      justify-content: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}
    .controls button {{
      padding: 0.4rem 0.8rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--card);
      cursor: pointer;
      font-size: 0.85rem;
    }}
    .controls button.active {{
      background: #333;
      color: #fff;
      border-color: #333;
    }}

    .page-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 1.5rem;
      overflow: hidden;
      box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }}
    .page-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.8rem 1rem;
      border-bottom: 1px solid var(--border);
      background: #fafaf8;
    }}
    .page-header h2 {{ font-size: 1rem; }}
    .page-meta {{ font-size: 0.8rem; color: #888; }}

    .page-compare {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
    }}
    .page-compare.single-view {{
      grid-template-columns: 1fr;
    }}
    .page-col {{
      border-right: 1px solid var(--border);
    }}
    .page-col:last-child {{ border-right: none; }}
    .page-col h3 {{
      text-align: center;
      padding: 0.5rem;
      font-size: 0.85rem;
      color: #666;
      border-bottom: 1px solid var(--border);
      background: #fdfcfa;
    }}

    .page-stage {{
      position: relative;
      padding: 0;
      background: white;
    }}
    .page-stage img {{
      display: block;
      width: 100%;
      height: auto;
    }}

    /* 재구성 뷰 */
    .reconstructed {{
      background: #fff;
    }}
    .reconstructed .ghost-img {{
      opacity: 0.08;
    }}
    .show-ghost .ghost-img {{
      opacity: 0.35;
    }}

    .text-block {{
      position: absolute;
      color: #1a1a1a;
      line-height: 1.3;
      padding: 1px 2px;
      background: var(--text-bg);
      border-radius: 2px;
      overflow: hidden;
      pointer-events: auto;
      cursor: default;
    }}
    .text-block:hover {{
      background: rgba(32, 124, 202, 0.15);
      z-index: 10;
    }}

    .table-overlay {{
      position: absolute;
      background: var(--table-bg);
      border: 2px solid var(--table-border);
      border-radius: 6px;
      padding: 4px;
      overflow: auto;
      z-index: 5;
    }}
    .table-overlay table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.65vw;
      line-height: 1.2;
    }}
    .table-overlay th,
    .table-overlay td {{
      border: 1px solid rgba(30, 133, 100, 0.3);
      padding: 2px 4px;
      text-align: left;
      white-space: nowrap;
    }}
    .table-overlay th {{
      background: rgba(30, 133, 100, 0.12);
      font-weight: 600;
    }}

    /* 모드 토글 */
    .hide-text .text-block {{ display: none; }}
    .hide-table .table-overlay {{ display: none; }}

    @media (max-width: 900px) {{
      .page-compare {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>PDF Layout Reconstruction</h1>
    <p>{html.escape(pdf_path.name)} &mdash; {page_count} pages</p>
  </header>

  <div class="controls">
    <button id="btn-ghost" onclick="toggleGhost()">배경 PDF 진하게</button>
    <button id="btn-text" class="active" onclick="toggleText()">텍스트 블록</button>
    <button id="btn-table" class="active" onclick="toggleTable()">구조화 테이블</button>
  </div>

  {"".join(page_sections)}

  <script>
    function toggleGhost() {{
      document.querySelectorAll('.reconstructed').forEach(el => el.classList.toggle('show-ghost'));
      document.getElementById('btn-ghost').classList.toggle('active');
    }}
    function toggleText() {{
      document.querySelectorAll('.reconstructed').forEach(el => el.classList.toggle('hide-text'));
      document.getElementById('btn-text').classList.toggle('active');
    }}
    function toggleTable() {{
      document.querySelectorAll('.reconstructed').forEach(el => el.classList.toggle('hide-table'));
      document.getElementById('btn-table').classList.toggle('active');
    }}
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 텍스트를 원래 위치에 배치한 레이아웃 HTML 생성")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--pages", type=str, default="", help="페이지 필터 (예: 1,3,5-8)")
    args = parser.parse_args()

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

    output_dir = Path(__file__).resolve().parent / "outputs" / "layout_reconstruct"
    output_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    try:
        html_content = build_html(pdf_path, document, page_filter, output_dir)
    finally:
        document.close()

    html_path = output_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"output: {html_path}")


if __name__ == "__main__":
    main()
