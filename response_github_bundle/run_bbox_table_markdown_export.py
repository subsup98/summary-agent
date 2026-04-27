from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "outputs" / "manual_raw_parse" / "miraeasset_q3_q4" / "source"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bbox_table_markdown"


def resolve_pdf_path() -> Path:
    candidates = sorted(DEFAULT_SOURCE_DIR.glob("*3*.pdf"))
    if candidates:
        return candidates[0]
    candidates = sorted(DEFAULT_SOURCE_DIR.glob("*.pdf"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No sample PDF found in {DEFAULT_SOURCE_DIR}")


def render_markdown_table(rows: list[list[dict[str, Any]]], extractor: PowerPointStructTreeExtractor) -> str:
    return extractor._table_rows_to_markdown(rows)


def render_irregular_html_table(rows: list[list[dict[str, Any]]]) -> str:
    # Page 16 right-bottom table: bbox inspection showed a 5-column visual grid.
    # StructTree row lengths [3, 5, 4, 5] map to:
    # - header row with colspans [2, 2, 1]
    # - second body row missing the first column because the first row spans 2 rows
    header = [cell["text"] for cell in rows[0]]
    body1 = [cell["text"] for cell in rows[1]]
    body2 = [cell["text"] for cell in rows[2]]
    body3 = [cell["text"] for cell in rows[3]]

    lines = [
        '<table>',
        "  <thead>",
        "    <tr>",
        f"      <th colspan=\"2\">{header[0]}</th>",
        f"      <th colspan=\"2\">{header[1]}</th>",
        f"      <th>{header[2]}</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
        "    <tr>",
        f"      <td rowspan=\"2\">{body1[0]}</td>",
        f"      <td>{body1[1]}</td>",
        f"      <td>{body1[2]}</td>",
        f"      <td>{body1[3]}</td>",
        f"      <td>{body1[4]}</td>",
        "    </tr>",
        "    <tr>",
        f"      <td>{body2[0]}</td>",
        f"      <td>{body2[1]}</td>",
        f"      <td>{body2[2]}</td>",
        f"      <td>{body2[3]}</td>",
        "    </tr>",
        "    <tr>",
        f"      <td>{body3[0]}</td>",
        f"      <td>{body3[1]}</td>",
        f"      <td>{body3[2]}</td>",
        f"      <td>{body3[3]}</td>",
        f"      <td>{body3[4]}</td>",
        "    </tr>",
        "  </tbody>",
        "</table>",
    ]
    return "\n".join(lines)


def main() -> None:
    pdf_path = resolve_pdf_path()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PowerPointStructTreeExtractor()

    with fitz.open(pdf_path) as document:
        catalog_xref = document.pdf_catalog()
        struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
        if struct_root[0] != "xref":
            raise RuntimeError("StructTreeRoot not found.")
        root_xref = int(struct_root[1].split()[0])
        page_map = {document.page_xref(i): i + 1 for i in range(document.page_count)}

        table_xrefs: dict[int, list[list[dict[str, Any]]]] = {}
        extractor._find_tables(document, root_xref, page_map, None, table_xrefs)

        table_page_map: dict[int, int] = {}
        extractor._map_table_pages(document, root_xref, page_map, None, table_page_map, set(table_xrefs.keys()))

    ordered_tables = sorted(table_xrefs.items(), key=lambda item: (table_page_map.get(item[0], 9999), item[0]))

    md_lines = [
        f"# {pdf_path.name} - StructTree + BBox Verified Tables",
        "",
        "Page numbers below are PDF page numbers.",
        "",
    ]
    summary: list[dict[str, Any]] = []

    for index, (table_xref, rows) in enumerate(ordered_tables, start=1):
        page_number = table_page_map.get(table_xref)
        row_lengths = [len(row) for row in rows]
        md_lines.append(f"## Table {index} - Page {page_number}")
        md_lines.append("")

        if row_lengths == [3, 5, 4, 5]:
            md_lines.append(render_irregular_html_table(rows))
            rendering = "html-irregular"
        else:
            md_lines.append(render_markdown_table(rows, extractor))
            rendering = "markdown-regular"

        md_lines.append("")
        summary.append(
            {
                "table_index": index,
                "page_number": page_number,
                "xref": table_xref,
                "row_lengths": row_lengths,
                "rendering": rendering,
            }
        )

    markdown_path = output_dir / "miraeasset_q3_tables.md"
    summary_path = output_dir / "summary.json"
    markdown_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"markdown_path": markdown_path.as_posix(), "summary_path": summary_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
