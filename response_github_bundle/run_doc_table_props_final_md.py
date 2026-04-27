from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import olefile


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(r"C:\Users\yongseop.im\Desktop\all_docs\금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc")
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_table_props_final"

OPCODES = {
    "PFInTable": b"\x16\x24",
    "PFTtp": b"\x17\x24",
    "PItap": b"\x49\x66",
    "PDtap": b"\x4A\x66",
    "PFInnerTableCell": b"\x4B\x24",
    "PFInnerTtp": b"\x4C\x24",
    "TDefTable": b"\x08\xD6",
    "TInsert": b"\x21\x76",
    "TDelete": b"\x22\x56",
    "TDxaCol": b"\x23\x76",
    "TMerge": b"\x24\x56",
    "TSplit": b"\x25\x56",
    "TVertMerge": b"\x2B\xD6",
    "TCellWidth": b"\x35\xD6",
}

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def find_all(data: bytes, needle: bytes) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        index = data.find(needle, start)
        if index < 0:
            return result
        result.append(index)
        start = index + 1


def reconstruct_piece_table(path: Path) -> tuple[bytes, bytes, str, list[dict[str, int | bool]]]:
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
        pieces: list[dict[str, int | bool]] = []
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
                fc_raw = fc_compressed & 0x3FFFFFFF
                fc_start = fc_raw if is_unicode else fc_raw // 2
                pieces.append(
                    {
                        "cp_start": cp_start,
                        "cp_end": cp_end,
                        "fc_start": fc_start,
                        "is_unicode": is_unicode,
                    }
                )

                chars: list[str] = []
                for cp in range(cp_start, cp_end):
                    char_offset = fc_start + ((cp - cp_start) * 2 if is_unicode else (cp - cp_start))
                    if is_unicode:
                        char = word_document[char_offset : char_offset + 2].decode("utf-16le", errors="replace")
                    else:
                        char = word_document[char_offset : char_offset + 1].decode("cp1252", errors="replace")
                    chars.append(char)
                parts.append("".join(chars))
            full_text = "".join(parts)
            break

        return word_document, table_stream, full_text, pieces
    finally:
        ole.close()


def fc_to_cp(fc: int, pieces: list[dict[str, int | bool]]) -> int | None:
    for piece in pieces:
        cp_start = int(piece["cp_start"])
        cp_end = int(piece["cp_end"])
        fc_start = int(piece["fc_start"])
        is_unicode = bool(piece["is_unicode"])
        if is_unicode:
            fc_end = fc_start + ((cp_end - cp_start) * 2)
            if fc_start <= fc <= fc_end and (fc - fc_start) % 2 == 0:
                return cp_start + ((fc - fc_start) // 2)
        else:
            fc_end = fc_start + (cp_end - cp_start)
            if fc_start <= fc <= fc_end:
                return cp_start + (fc - fc_start)
    return None


def parse_bool_property(blob: bytes, opcode: bytes) -> bool:
    positions = find_all(blob, opcode)
    if not positions:
        return False
    pos = positions[-1]
    if pos + 2 >= len(blob):
        return False
    return blob[pos + 2] != 0


def parse_int32_property(blob: bytes, opcode: bytes) -> int | None:
    positions = find_all(blob, opcode)
    if not positions:
        return None
    pos = positions[-1]
    if pos + 6 > len(blob):
        return None
    return struct.unpack("<i", blob[pos + 2 : pos + 6])[0]


def parse_tdef(blob: bytes) -> dict[str, object] | None:
    positions = find_all(blob, OPCODES["TDefTable"])
    if not positions:
        return None
    pos = positions[-1]
    if pos + 5 > len(blob):
        return None
    cb = struct.unpack("<H", blob[pos + 2 : pos + 4])[0]
    number_of_columns = blob[pos + 4]
    payload = blob[pos + 4 : pos + 4 + cb - 1]
    required = 1 + (2 * (number_of_columns + 1))
    if len(payload) < required:
        return None

    rgdxa_center = [
        struct.unpack("<h", payload[1 + (index * 2) : 1 + (index * 2) + 2])[0]
        for index in range(number_of_columns + 1)
    ]
    column_widths = [rgdxa_center[i + 1] - rgdxa_center[i] for i in range(number_of_columns)]
    return {
        "number_of_columns": number_of_columns,
        "rgdxa_center": rgdxa_center,
        "column_widths": column_widths,
    }


def parse_papx_entries(word_document: bytes, table_stream: bytes, pieces: list[dict[str, int | bool]]) -> list[dict]:
    fc_bte = struct.unpack("<I", word_document[0x102:0x106])[0]
    lcb_bte = struct.unpack("<I", word_document[0x106:0x10A])[0]
    plcf_bte_papx = table_stream[fc_bte : fc_bte + lcb_bte]
    entry_count = (len(plcf_bte_papx) - 4) // 8
    if entry_count <= 0:
        return []

    data_offset = 4 * (entry_count + 1)
    entries: list[dict] = []
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
            blob = fkp[papx_offset : papx_offset + total_length]

            fc_start = rgfc[para_index]
            fc_end = rgfc[para_index + 1]
            cp_start = fc_to_cp(fc_start, pieces)
            cp_end = fc_to_cp(fc_end, pieces)
            if cp_start is None or cp_end is None:
                continue

            p_itap = parse_int32_property(blob, OPCODES["PItap"])
            p_dtap = parse_int32_property(blob, OPCODES["PDtap"]) or 0
            in_table = parse_bool_property(blob, OPCODES["PFInTable"])
            depth = (p_itap or 0) + p_dtap if in_table else 0

            entries.append(
                {
                    "fc_start": fc_start,
                    "fc_end": fc_end,
                    "cp_start": cp_start,
                    "cp_end": cp_end,
                    "papx_pn": pn,
                    "papx_entry_index": para_index,
                    "blob": blob.hex(),
                    "in_table": in_table,
                    "depth": depth,
                    "p_itap": p_itap,
                    "p_dtap": p_dtap,
                    "is_ttp": parse_bool_property(blob, OPCODES["PFTtp"]),
                    "is_inner_cell": parse_bool_property(blob, OPCODES["PFInnerTableCell"]),
                    "is_inner_ttp": parse_bool_property(blob, OPCODES["PFInnerTtp"]),
                    "tdef": parse_tdef(blob),
                    "modifier_flags": {
                        "TInsert": bool(find_all(blob, OPCODES["TInsert"])),
                        "TDelete": bool(find_all(blob, OPCODES["TDelete"])),
                        "TDxaCol": bool(find_all(blob, OPCODES["TDxaCol"])),
                        "TMerge": bool(find_all(blob, OPCODES["TMerge"])),
                        "TSplit": bool(find_all(blob, OPCODES["TSplit"])),
                        "TVertMerge": bool(find_all(blob, OPCODES["TVertMerge"])),
                        "TCellWidth": bool(find_all(blob, OPCODES["TCellWidth"])),
                    },
                }
            )
    entries.sort(key=lambda item: (item["cp_start"], item["cp_end"]))
    return entries


def slice_clean(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = CONTROL_CHAR_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s*SHAPE\s+\\?\*?\s*MERGEFORMAT\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_row_cells(full_text: str, row_end_cp: int, number_of_columns: int) -> tuple[int | None, list[str], list[int]]:
    mark_positions: list[int] = []
    cursor = row_end_cp
    while cursor >= 0 and len(mark_positions) < number_of_columns:
        if full_text[cursor] == "\x07":
            mark_positions.append(cursor)
        cursor -= 1
    if len(mark_positions) < number_of_columns:
        return None, [], []

    mark_positions.reverse()
    previous_mark = full_text.rfind("\x07", 0, mark_positions[0])
    row_start_cp = previous_mark + 1 if previous_mark >= 0 else 0

    cells: list[str] = []
    cell_start = row_start_cp
    for mark_cp in mark_positions:
        cells.append(slice_clean(full_text[cell_start:mark_cp]))
        cell_start = mark_cp + 1
    return row_start_cp, cells, mark_positions


def build_rows(entries: list[dict], full_text: str) -> list[dict]:
    rows: list[dict] = []
    for entry in entries:
        if not entry["in_table"]:
            continue
        if int(entry["depth"]) != 1:
            continue
        if not entry["is_ttp"]:
            continue
        if not entry["tdef"]:
            continue

        tdef = entry["tdef"]
        row_start_cp, cells, cell_mark_cps = extract_row_cells(full_text, int(entry["cp_start"]), int(tdef["number_of_columns"]))
        rows.append(
            {
                "row_start_cp": row_start_cp,
                "row_end_cp": entry["cp_start"],
                "depth": entry["depth"],
                "number_of_columns": tdef["number_of_columns"],
                "rgdxa_center": tdef["rgdxa_center"],
                "column_widths": tdef["column_widths"],
                "cells": cells,
                "cell_mark_cps": cell_mark_cps,
                "modifier_flags": entry["modifier_flags"],
            }
        )
    rows.sort(key=lambda row: row["row_end_cp"])
    return rows


def group_rows_into_tables(rows: list[dict], entries: list[dict]) -> list[list[dict]]:
    by_cp = {row["row_end_cp"]: row for row in rows}
    row_paragraphs = [entry for entry in entries if entry["cp_start"] in by_cp]
    row_paragraphs.sort(key=lambda item: item["cp_start"])

    groups: list[list[dict]] = []
    current: list[dict] = []

    for index, paragraph in enumerate(row_paragraphs):
        row = by_cp[paragraph["cp_start"]]
        if not current:
            current = [row]
        else:
            prev = row_paragraphs[index - 1]
            between = [
                entry
                for entry in entries
                if entry["cp_start"] > prev["cp_start"] and entry["cp_start"] < paragraph["cp_start"]
            ]
            has_out_of_table_break = any((not entry["in_table"]) or int(entry["depth"]) != int(paragraph["depth"]) for entry in between)
            if has_out_of_table_break:
                groups.append(current)
                current = [row]
            else:
                current.append(row)
    if current:
        groups.append(current)
    return groups


def postprocess_tables(tables: list[list[dict]]) -> list[list[dict]]:
    cleaned_tables: list[list[dict]] = []
    for table in tables:
        cleaned_rows: list[dict] = []
        for row in table:
            cleaned_cells = [slice_clean(str(cell or "")) for cell in row["cells"]]
            if not any(cleaned_cells):
                continue
            cleaned_row = dict(row)
            cleaned_row["cells"] = cleaned_cells
            cleaned_rows.append(cleaned_row)
        if not cleaned_rows:
            continue
        cleaned_tables.append(cleaned_rows)
    return cleaned_tables


def build_master_boundaries(group: list[dict]) -> list[int]:
    values: list[int] = []
    for row in group:
        for boundary in row["rgdxa_center"]:
            if boundary not in values:
                values.append(boundary)
    values.sort()
    return values


def map_row_to_master(row: dict, master_boundaries: list[int]) -> list[str]:
    output = [""] * (len(master_boundaries) - 1)
    for index, text in enumerate(row["cells"]):
        if index + 1 >= len(row["rgdxa_center"]):
            break
        left = row["rgdxa_center"][index]
        right = row["rgdxa_center"][index + 1]
        overlaps: list[int] = []
        for master_index in range(len(master_boundaries) - 1):
            overlap = min(right, master_boundaries[master_index + 1]) - max(left, master_boundaries[master_index])
            if overlap > 0:
                overlaps.append(master_index)
        if not overlaps:
            continue
        output[overlaps[0]] = text
    return output


def render_table(table_index: int, group: list[dict]) -> str:
    master_boundaries = build_master_boundaries(group)
    rendered_rows = [map_row_to_master(row, master_boundaries) for row in group]
    header = rendered_rows[0]

    lines = [f"## Table {table_index}", ""]
    lines.append(f"- row_count: `{len(group)}`")
    lines.append(f"- depth: `{group[0]['depth']}`")
    lines.append(f"- master_column_count: `{len(master_boundaries) - 1}`")
    lines.append(f"- master_boundaries_twips: `{master_boundaries}`")

    active_modifiers = sorted(
        {
            name
            for row in group
            for name, enabled in row["modifier_flags"].items()
            if enabled
        }
    )
    lines.append(f"- active_row_modifiers: `{active_modifiers}`")
    lines.append("")
    lines.append("| " + " | ".join(cell or " " for cell in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rendered_rows[1:]:
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")

    lines.append("")
    lines.append("```text")
    for row in group:
        lines.append(
            "cp={start}-{end} ncols={ncols} widths={widths}".format(
                start=row["row_start_cp"],
                end=row["row_end_cp"],
                ncols=row["number_of_columns"],
                widths=row["column_widths"],
            )
        )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_document(tables: list[list[dict]], entries: list[dict]) -> str:
    nested_paragraphs = [
        entry for entry in entries if entry["in_table"] and int(entry["depth"]) > 1
    ]
    any_modifiers = sorted(
        {
            name
            for entry in entries
            if entry["tdef"]
            for name, enabled in entry["modifier_flags"].items()
            if enabled
        }
    )

    lines = [
        "# DOC Table Markdown (Paragraph/Table Property Driven)",
        "",
        f"- Source: `{SOURCE_PATH.as_posix()}`",
        f"- Table count: `{len(tables)}`",
        f"- Nested table paragraph count: `{len(nested_paragraphs)}`",
        f"- Table-row modifiers found after `sprmTDefTable`: `{any_modifiers}`",
        "",
        "> This file is built by checking paragraph PAPX for `sprmPFInTable`, `sprmPItap`, `sprmPDtap`, and `sprmPFTtp`,",
        "> then reading row table properties from the row mark PAPX, especially `sprmTDefTable`.",
        "> Postprocessing removes control characters like BS and drops fully empty rows/tables.",
        "",
    ]

    for index, table in enumerate(tables, start=1):
        lines.append(render_table(index, table))

    return "\n".join(lines)


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    word_document, table_stream, full_text, pieces = reconstruct_piece_table(SOURCE_PATH)
    entries = parse_papx_entries(word_document, table_stream, pieces)
    rows = build_rows(entries, full_text)
    tables = postprocess_tables(group_rows_into_tables(rows, entries))

    final_markdown = render_document(tables, entries)
    final_path = RUN_ROOT / "final_tables_from_properties.md"
    final_path.write_text(final_markdown, encoding="utf-8")

    manifest = {
        "source_path": SOURCE_PATH.as_posix(),
        "paragraph_entry_count": len(entries),
        "depth1_row_count": len(rows),
        "table_count": len(tables),
        "nested_table_paragraph_count": sum(1 for entry in entries if entry["in_table"] and int(entry["depth"]) > 1),
        "final_markdown_path": final_path.as_posix(),
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
