from __future__ import annotations

import json
import re
import struct
from collections import Counter
from pathlib import Path

import olefile


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(r"C:\Users\yongseop.im\Desktop\all_docs\금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc")
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_actual_rows"

SPRMS = {
    "sprmTDefTable": b"\x08\xD6",
    "sprmTTableHeader": b"\x04\x34",
}


def reconstruct_text_and_maps(path: Path) -> tuple[bytes, str, str, dict[int, int]]:
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
        fc_to_cp: dict[int, int] = {}

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
                if not is_unicode:
                    fc = fc // 2

                piece_chars: list[str] = []
                for cp in range(cp_start, cp_end):
                    char_offset = fc + ((cp - cp_start) * 2 if is_unicode else (cp - cp_start))
                    if is_unicode:
                        char = word_document[char_offset : char_offset + 2].decode("utf-16le", errors="replace")
                    else:
                        char = word_document[char_offset : char_offset + 1].decode("cp1252", errors="replace")
                    piece_chars.append(char)
                    if char == "\x07":
                        fc_to_cp[char_offset] = cp
                parts.append("".join(piece_chars))
            full_text = "".join(parts)
            break

        return word_document, table_stream_name, full_text, fc_to_cp
    finally:
        ole.close()


def parse_plcf_bte_papx(word_document: bytes, table_stream: bytes) -> bytes:
    fc = struct.unpack("<I", word_document[0x102:0x106])[0]
    lcb = struct.unpack("<I", word_document[0x106:0x10A])[0]
    return table_stream[fc : fc + lcb]


def parse_tdef_from_grpprl(grpprl: bytes) -> dict[str, object] | None:
    index = grpprl.find(SPRMS["sprmTDefTable"])
    if index < 0 or index + 5 > len(grpprl):
        return None

    cb = struct.unpack("<H", grpprl[index + 2 : index + 4])[0]
    number_of_columns = grpprl[index + 4]
    operand_length = cb + 1
    operand = grpprl[index + 2 : index + 2 + operand_length]
    if len(operand) < 3 or not (1 <= number_of_columns <= 63):
        return None

    payload = operand[2:]
    required = 1 + 2 * (number_of_columns + 1)
    if len(payload) < required:
        return None

    rgdxa_center = [
        struct.unpack("<h", payload[1 + (offset * 2) : 1 + (offset * 2) + 2])[0]
        for offset in range(number_of_columns + 1)
    ]
    column_widths = [rgdxa_center[i + 1] - rgdxa_center[i] for i in range(number_of_columns)]
    return {
        "number_of_columns": number_of_columns,
        "rgdxa_center": rgdxa_center,
        "column_widths": column_widths,
    }


def parse_actual_row_entries(path: Path, word_document: bytes, fc_to_cp: dict[int, int]) -> list[dict[str, object]]:
    ole = olefile.OleFileIO(str(path))
    try:
        flags = struct.unpack("<H", word_document[0x0A:0x0C])[0]
        table_stream_name = "1Table" if (flags & 0x0200) else "0Table"
        table_stream = ole.openstream(table_stream_name).read()
    finally:
        ole.close()

    plcf_bte_papx = parse_plcf_bte_papx(word_document, table_stream)
    if not plcf_bte_papx:
        return []

    entry_count = (len(plcf_bte_papx) - 4) // 8
    if entry_count <= 0:
        return []

    data_offset = 4 * (entry_count + 1)
    rows: list[dict[str, object]] = []

    for index in range(entry_count):
        raw = struct.unpack("<I", plcf_bte_papx[data_offset + index * 4 : data_offset + (index + 1) * 4])[0]
        pn = raw & 0x003FFFFF
        fkp = word_document[pn * 512 : (pn + 1) * 512]
        if len(fkp) < 512:
            continue
        cpara = fkp[-1]
        if not (1 <= cpara <= 0x1D):
            continue

        rgfc = [struct.unpack("<I", fkp[offset * 4 : (offset + 1) * 4])[0] for offset in range(cpara + 1)]

        for para_index in range(cpara):
            bx_offset = (cpara + 1) * 4 + para_index * 13
            b_offset = fkp[bx_offset]
            if b_offset == 0:
                continue
            papx_offset = b_offset * 2
            if papx_offset >= 511:
                continue

            cb = fkp[papx_offset]
            if cb == 0:
                if papx_offset + 1 >= len(fkp):
                    continue
                cb_prime = fkp[papx_offset + 1]
                total_length = 2 + (2 * cb_prime)
            else:
                total_length = 1 + (2 * cb - 1)
            grpprl = fkp[papx_offset : papx_offset + total_length]

            tdef = parse_tdef_from_grpprl(grpprl)
            if not tdef:
                continue

            row_end_fc = rgfc[para_index]
            row_end_cp = fc_to_cp.get(row_end_fc)
            if row_end_cp is None:
                continue

            rows.append(
                {
                    "row_end_fc": row_end_fc,
                    "row_end_cp": row_end_cp,
                    "papx_pn": pn,
                    "papx_entry_index": para_index,
                    "papx_offset_in_fkp": papx_offset,
                    "has_header_marker": grpprl.find(SPRMS["sprmTTableHeader"]) >= 0,
                    **tdef,
                }
            )

    rows.sort(key=lambda row: int(row["row_end_cp"]))
    return rows


def extract_cells_for_actual_row(full_text: str, row_end_cp: int, number_of_columns: int) -> tuple[int | None, list[str], list[int]]:
    cell_mark = "\x07"
    mark_positions: list[int] = []
    cursor = row_end_cp
    while cursor >= 0 and len(mark_positions) < number_of_columns:
        if full_text[cursor] == cell_mark:
            mark_positions.append(cursor)
        cursor -= 1

    if len(mark_positions) < number_of_columns:
        return None, [], []

    mark_positions.reverse()

    previous_mark = full_text.rfind(cell_mark, 0, mark_positions[0])
    row_start_cp = previous_mark + 1 if previous_mark >= 0 else 0

    cells: list[str] = []
    cell_mark_cps: list[int] = []
    cell_start = row_start_cp
    for mark_cp in mark_positions:
        raw = full_text[cell_start:mark_cp].replace("\r", "\n")
        raw = re.sub(r"\s*SHAPE\s+\\?\*?\s*MERGEFORMAT\s*", "", raw).strip()
        raw = re.sub(r"\s+", " ", raw)
        cells.append(raw)
        cell_mark_cps.append(mark_cp)
        cell_start = mark_cp + 1

    return row_start_cp, cells, cell_mark_cps


def render_rows_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# DOC Actual Row Reconstruction",
        "",
        f"- source: `{SOURCE_PATH.as_posix()}`",
        f"- actual_row_count: `{len(rows)}`",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        lines.append(f"## Row {index}")
        lines.append("")
        lines.append(f"- row_end_fc: `{row['row_end_fc']}`")
        lines.append(f"- row_end_cp: `{row['row_end_cp']}`")
        lines.append(f"- row_start_cp: `{row.get('row_start_cp')}`")
        lines.append(f"- number_of_columns: `{row['number_of_columns']}`")
        lines.append(f"- has_header_marker: `{row['has_header_marker']}`")
        lines.append(f"- column_widths_twips: `{row['column_widths']}`")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "cells": row.get("cells") or [],
                    "cell_mark_cps": row.get("cell_mark_cps") or [],
                    "rgdxa_center": row["rgdxa_center"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    word_document, table_stream_name, full_text, fc_to_cp = reconstruct_text_and_maps(SOURCE_PATH)
    rows = parse_actual_row_entries(SOURCE_PATH, word_document, fc_to_cp)

    for row in rows:
        row_start_cp, cells, cell_mark_cps = extract_cells_for_actual_row(
            full_text,
            int(row["row_end_cp"]),
            int(row["number_of_columns"]),
        )
        row["row_start_cp"] = row_start_cp
        row["cells"] = cells
        row["cell_mark_cps"] = cell_mark_cps
        row["cell_count_from_marks"] = len(cells)

    (RUN_ROOT / "actual_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "actual_rows.md").write_text(render_rows_markdown(rows), encoding="utf-8")

    manifest = {
        "source_path": SOURCE_PATH.as_posix(),
        "selected_table_stream": table_stream_name,
        "actual_row_count": len(rows),
        "column_count_distribution": dict(Counter(int(row["number_of_columns"]) for row in rows)),
        "header_marker_row_count": sum(bool(row["has_header_marker"]) for row in rows),
        "actual_rows_json_path": (RUN_ROOT / "actual_rows.json").as_posix(),
        "actual_rows_markdown_path": (RUN_ROOT / "actual_rows.md").as_posix(),
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
