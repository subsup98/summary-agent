from __future__ import annotations

import json
import re
import struct
from collections import Counter
from pathlib import Path

import olefile

from src.classifiers.document_classifier import classify_document
from src.parsers.office.doc_parser import DocParser


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(r"C:\Users\yongseop.im\Desktop\all_docs\금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc")
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_tdef_compare"

SPRMS = {
    "sprmTTableHeader": b"\x04\x34",
    "sprmTTlp": b"\x0A\x74",
    "sprmTDyaRowHeight": b"\x07\x94",
    "sprmTDefTable": b"\x08\xD6",
}


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x0b", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def find_all(data: bytes, needle: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        index = data.find(needle, start)
        if index < 0:
            return offsets
        offsets.append(index)
        start = index + 1


def reconstruct_full_text(path: Path) -> tuple[bytes, str, str]:
    ole = olefile.OleFileIO(str(path))
    try:
        word_document = ole.openstream("WordDocument").read()
        flags = struct.unpack("<H", word_document[0x0A:0x0C])[0]
        table_stream_name = "1Table" if (flags & 0x0200) else "0Table"
        table_stream = ole.openstream(table_stream_name).read()

        fc_clx = struct.unpack("<I", word_document[0x01A2:0x01A6])[0]
        lcb_clx = struct.unpack("<I", word_document[0x01A6:0x01AA])[0]
        clx_data = table_stream[fc_clx : fc_clx + lcb_clx]

        pos = 0
        full_text = ""
        while pos < len(clx_data):
            token = clx_data[pos]
            if token == 0x01:
                cb = struct.unpack("<H", clx_data[pos + 1 : pos + 3])[0]
                pos += 3 + cb
                continue
            if token != 0x02:
                break

            lcb = struct.unpack("<I", clx_data[pos + 1 : pos + 5])[0]
            pcd_data = clx_data[pos + 5 : pos + 5 + lcb]
            piece_count = (lcb - 4) // 12
            parts: list[str] = []
            for index in range(piece_count):
                cp_start = struct.unpack("<I", pcd_data[index * 4 : index * 4 + 4])[0]
                cp_end = struct.unpack("<I", pcd_data[(index + 1) * 4 : (index + 1) * 4 + 4])[0]
                pcd_offset = (piece_count + 1) * 4 + index * 8
                fc_compressed = struct.unpack("<I", pcd_data[pcd_offset + 2 : pcd_offset + 6])[0]
                is_unicode = not bool(fc_compressed & 0x40000000)
                fc = fc_compressed & 0x3FFFFFFF
                char_count = cp_end - cp_start
                if is_unicode:
                    text_bytes = word_document[fc : fc + char_count * 2]
                    text = text_bytes.decode("utf-16le", errors="replace")
                else:
                    fc = fc // 2
                    text_bytes = word_document[fc : fc + char_count]
                    text = text_bytes.decode("cp1252", errors="replace")
                parts.append(text)
            full_text = "".join(parts)
            break

        return word_document, table_stream_name, full_text
    finally:
        ole.close()


def parse_tables_from_full_text(full_text: str) -> tuple[list[list[list[str]]], list[str], list[dict[str, object]]]:
    cell = chr(7)
    para = chr(13)

    current_text: list[str] = []
    current_row_cells: list[str] = []
    current_table_rows: list[list[str]] = []
    tables: list[list[list[str]]] = []
    paragraphs: list[str] = []
    row_records: list[dict[str, object]] = []

    index = 0
    active_table_index = 0
    while index < len(full_text):
        ch = full_text[index]

        if ch == cell:
            cell_content = "".join(current_text).strip()
            cell_content = cell_content.replace(para, " ").strip()
            cell_content = re.sub(r"\s*SHAPE\s+\\?\*?\s*MERGEFORMAT\s*", "", cell_content).strip()
            cell_content = re.sub(r"\s+", " ", cell_content)
            current_row_cells.append(cell_content)
            current_text = []

            if index + 1 < len(full_text) and full_text[index + 1] == para:
                if current_row_cells:
                    if not current_table_rows:
                        active_table_index += 1
                    row_records.append(
                        {
                            "table_index": active_table_index,
                            "row_index_in_table": len(current_table_rows) + 1,
                            "cells": list(current_row_cells),
                            "cell_count": len(current_row_cells),
                        }
                    )
                    current_table_rows.append(list(current_row_cells))
                current_row_cells = []
                index += 2
                continue

        elif ch == para:
            if not current_row_cells and not current_text:
                index += 1
                continue

            if current_row_cells:
                current_text.append(" ")
            else:
                para_text = "".join(current_text).strip()
                para_text = re.sub(r"\s*SHAPE\s+\\?\*?\s*MERGEFORMAT\s*", "", para_text).strip()
                para_text = re.sub(r"\s+", " ", para_text)
                if para_text and len(para_text) > 1:
                    if current_table_rows:
                        tables.append(current_table_rows)
                        current_table_rows = []
                    paragraphs.append(para_text)
                current_text = []
        else:
            current_text.append(ch)

        index += 1

    if current_table_rows:
        tables.append(current_table_rows)

    return tables, paragraphs, row_records


def parse_tdef_records(word_document: bytes) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for offset in find_all(word_document, SPRMS["sprmTDefTable"]):
        if offset + 5 > len(word_document):
            continue
        cb = struct.unpack("<H", word_document[offset + 2 : offset + 4])[0]
        number_of_columns = word_document[offset + 4]
        if not (1 <= number_of_columns <= 63):
            continue

        operand = word_document[offset + 4 : offset + 4 + cb - 1]
        required = 1 + 2 * (number_of_columns + 1)
        if len(operand) < required:
            continue

        rgdxa_center = [
            struct.unpack("<h", operand[1 + (index * 2) : 1 + (index * 2) + 2])[0]
            for index in range(number_of_columns + 1)
        ]
        records.append(
            {
                "offset": offset,
                "cb": cb,
                "number_of_columns": number_of_columns,
                "rgdxa_center": rgdxa_center,
                "has_table_header_marker_nearby": bool(
                    word_document[max(0, offset - 128) : offset + 256].find(SPRMS["sprmTTableHeader"]) >= 0
                ),
            }
        )
    return records


def attach_tdef_metadata_to_rows(
    row_records: list[dict[str, object]],
    tdef_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    pointer = 0

    for row in row_records:
        cell_count = int(row["cell_count"])
        chosen_index: int | None = None
        for index in range(pointer, len(tdef_records)):
            if int(tdef_records[index]["number_of_columns"]) == cell_count:
                chosen_index = index
                break
        if chosen_index is None and pointer < len(tdef_records):
            chosen_index = pointer

        chosen = tdef_records[chosen_index] if chosen_index is not None else None
        if chosen_index is not None:
            pointer = chosen_index + 1

        enriched_row = dict(row)
        if chosen is not None:
            boundaries = list(chosen["rgdxa_center"])
            widths = [boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)]
            enriched_row["tdef_match"] = {
                "offset": chosen["offset"],
                "number_of_columns": chosen["number_of_columns"],
                "rgdxa_center": boundaries,
                "column_widths": widths,
                "has_table_header_marker_nearby": chosen["has_table_header_marker_nearby"],
                "exact_column_match": int(chosen["number_of_columns"]) == cell_count,
            }
        else:
            enriched_row["tdef_match"] = None
        enriched.append(enriched_row)

    return enriched


def infer_header_rows(table_rows: list[dict[str, object]]) -> int:
    if not table_rows:
        return 0
    if len(table_rows) == 1:
        return 1
    first = table_rows[0]
    second = table_rows[1]
    if first.get("tdef_match", {}).get("has_table_header_marker_nearby"):
        return 1
    first_numeric = sum(bool(re.search(r"\d", cell)) for cell in first["cells"])
    second_numeric = sum(bool(re.search(r"\d", cell)) for cell in second["cells"])
    if first_numeric <= max(1, len(first["cells"]) // 3) and second_numeric <= max(1, len(second["cells"]) // 3):
        return 2
    return 1


def render_basic_table_markdown(table_index: int, rows: list[list[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    padded = [row + [""] * (max_cols - len(row)) for row in rows]
    lines = [f"## Table {table_index}", ""]
    lines.append("| " + " | ".join(cell or " " for cell in padded[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in padded[0]) + " |")
    for row in padded[1:]:
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_enriched_table_markdown(table_index: int, rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""

    max_cols = max(len(row["cells"]) for row in rows)
    padded_cells = [row["cells"] + [""] * (max_cols - len(row["cells"])) for row in rows]
    header_rows = infer_header_rows(rows)

    lines = [f"## Table {table_index}", ""]
    lines.append(f"- inferred_header_rows: `{header_rows}`")
    lines.append(f"- row_count: `{len(rows)}`")
    lines.append(f"- max_column_count: `{max_cols}`")

    exact_matches = sum(bool(row.get("tdef_match", {}).get("exact_column_match")) for row in rows)
    lines.append(f"- exact_tdef_matches: `{exact_matches}/{len(rows)}`")
    header_marked_rows = sum(bool(row.get("tdef_match", {}).get("has_table_header_marker_nearby")) for row in rows)
    lines.append(f"- rows_with_header_marker_nearby: `{header_marked_rows}`")
    lines.append("")

    lines.append("| " + " | ".join(cell or " " for cell in padded_cells[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in padded_cells[0]) + " |")
    for row in padded_cells[1:]:
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")

    lines.append("")
    lines.append("```text")
    for row in rows:
        match = row.get("tdef_match") or {}
        widths = match.get("column_widths") or []
        lines.append(
            "row_{row_idx}: cells={cells} exact={exact} header_nearby={header} widths={widths}".format(
                row_idx=row["row_index_in_table"],
                cells=row["cell_count"],
                exact=match.get("exact_column_match"),
                header=match.get("has_table_header_marker_nearby"),
                widths=widths,
            )
        )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def group_rows_by_table(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    grouped: list[list[dict[str, object]]] = []
    current_index = None
    current_rows: list[dict[str, object]] = []
    for row in rows:
        table_index = row["table_index"]
        if current_index is None or table_index == current_index:
            current_index = table_index
            current_rows.append(row)
            continue
        grouped.append(current_rows)
        current_index = table_index
        current_rows = [row]
    if current_rows:
        grouped.append(current_rows)
    return grouped


def build_comparison_report(
    pipeline_raw_method: str,
    tables: list[list[list[str]]],
    enriched_tables: list[list[dict[str, object]]],
    tdef_records: list[dict[str, object]],
    parsed,
) -> str:
    lines = [
        "# DOC OLE TDef Compare",
        "",
        f"- source: `{SOURCE_PATH.as_posix()}`",
        f"- pipeline_raw_method: `{pipeline_raw_method}`",
        f"- pipeline_block_types: `{dict(Counter(block['type'] for block in parsed.blocks))}`",
        f"- basic_ole_table_count: `{len(tables)}`",
        f"- tdef_record_count: `{len(tdef_records)}`",
        "",
        "## Verdict",
        "",
        "- `sprmTDefTable`-aware reconstruction is better than text-only row splitting for column geometry and row-level metadata.",
        "- It still is not a full Word logical tree, because rows are matched heuristically rather than through full PAPX/property application.",
        "",
    ]

    exact_matches = sum(
        bool(row.get("tdef_match", {}).get("exact_column_match"))
        for table in enriched_tables
        for row in table
    )
    total_rows = sum(len(table) for table in enriched_tables)
    lines.append("## Match Quality")
    lines.append("")
    lines.append(f"- exact row-to-TDef column matches: `{exact_matches}/{total_rows}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    parser = DocParser()
    issues: list = []
    pipeline_raw = parser._try_powershell_word(SOURCE_PATH, issues)
    pipeline_raw_method = "powershell-word-com" if pipeline_raw else ""
    if not pipeline_raw:
        pipeline_raw = parser._try_ole_extraction(SOURCE_PATH, issues)
        pipeline_raw_method = "ole-best-effort" if pipeline_raw else "unavailable"

    parsed = parser.parse(SOURCE_PATH, classify_document(SOURCE_PATH))
    word_document, table_stream_name, full_text = reconstruct_full_text(SOURCE_PATH)
    tables, paragraphs, row_records = parse_tables_from_full_text(full_text)
    tdef_records = parse_tdef_records(word_document)
    enriched_rows = attach_tdef_metadata_to_rows(row_records, tdef_records)
    enriched_tables = group_rows_by_table(enriched_rows)

    (RUN_ROOT / "pipeline_raw_pre_postprocess.txt").write_text(
        normalize_text(pipeline_raw) + ("\n" if pipeline_raw else ""),
        encoding="utf-8",
    )
    (RUN_ROOT / "pipeline_postprocessed_markdown.md").write_text(parsed.markdown, encoding="utf-8")
    (RUN_ROOT / "ole_piece_table_full_text.txt").write_text(full_text, encoding="utf-8")
    (RUN_ROOT / "tdef_enriched_rows.json").write_text(
        json.dumps(enriched_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "tdef_records.json").write_text(
        json.dumps(tdef_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    basic_lines = ["# Basic OLE Table Reconstruction", ""]
    for index, table in enumerate(tables, start=1):
        basic_lines.append(render_basic_table_markdown(index, table))
    (RUN_ROOT / "ole_basic_tables.md").write_text("\n".join(basic_lines), encoding="utf-8")

    enriched_lines = [
        "# TDef-Enriched OLE Reconstruction",
        "",
        f"- selected_table_stream: `{table_stream_name}`",
        f"- paragraph_count: `{len(paragraphs)}`",
        "",
    ]
    for index, table_rows in enumerate(enriched_tables, start=1):
        enriched_lines.append(render_enriched_table_markdown(index, table_rows))
    (RUN_ROOT / "ole_tdef_enriched_tables.md").write_text("\n".join(enriched_lines), encoding="utf-8")

    comparison_report = build_comparison_report(pipeline_raw_method, tables, enriched_tables, tdef_records, parsed)
    (RUN_ROOT / "comparison_report.md").write_text(comparison_report, encoding="utf-8")

    manifest = {
        "source_path": SOURCE_PATH.as_posix(),
        "selected_table_stream": table_stream_name,
        "pipeline_raw_method": pipeline_raw_method,
        "basic_ole_table_count": len(tables),
        "basic_ole_row_count": len(row_records),
        "tdef_record_count": len(tdef_records),
        "enriched_table_count": len(enriched_tables),
        "pipeline_block_type_counts": dict(Counter(block["type"] for block in parsed.blocks)),
        "comparison_report_path": (RUN_ROOT / "comparison_report.md").as_posix(),
        "basic_tables_path": (RUN_ROOT / "ole_basic_tables.md").as_posix(),
        "enriched_tables_path": (RUN_ROOT / "ole_tdef_enriched_tables.md").as_posix(),
        "pipeline_markdown_path": (RUN_ROOT / "pipeline_postprocessed_markdown.md").as_posix(),
        "issues": [issue.__dict__ for issue in issues],
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
