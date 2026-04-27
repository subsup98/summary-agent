from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import olefile


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(r"C:\Users\yongseop.im\Desktop\all_docs\금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc")
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_raw_props"

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
    return pos + 2 < len(blob) and blob[pos + 2] != 0


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
    return {
        "number_of_columns": number_of_columns,
        "rgdxa_center": rgdxa_center,
    }


def snippet(text: str, cp_start: int, cp_end: int) -> str:
    raw = text[cp_start:cp_end].replace("\r", " ")
    raw = CONTROL_CHAR_RE.sub(" ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:200]


def describe_modifier_flags(flags: dict[str, bool]) -> str:
    enabled = [name for name, is_enabled in flags.items() if is_enabled]
    if not enabled:
        return "No follow-up table modifier SPRMs are present. Only the base row definition is visible."
    return f"Follow-up table modifier SPRMs are present: {', '.join(enabled)}"


def describe_tdef(tdef: dict[str, object] | None) -> list[str]:
    if not tdef:
        return ["`sprmTDefTable` is absent, so this paragraph does not carry a direct row layout definition."]
    number_of_columns = tdef.get("number_of_columns")
    rgdxa_center = tdef.get("rgdxa_center") or []
    column_widths: list[int] = []
    if isinstance(rgdxa_center, list) and len(rgdxa_center) >= 2:
        try:
            column_widths = [
                int(rgdxa_center[index + 1]) - int(rgdxa_center[index])
                for index in range(len(rgdxa_center) - 1)
            ]
        except Exception:
            column_widths = []
    lines = [
        f"`sprmTDefTable` is present. The base column count for this row is `{number_of_columns}`.",
        f"`rgdxaCenter` is `{rgdxa_center}` and represents the table left edge plus each cell's right boundary.",
    ]
    if column_widths:
        lines.append(f"The derived column widths are `{column_widths}`.")
    return lines


def describe_entry(entry: dict) -> list[str]:
    lines: list[str] = []
    if entry["in_table"]:
        depth_text = f"This paragraph is inside a table. The current table depth is `{entry['depth']}`."
        if entry["depth"] and entry["depth"] > 1:
            depth_text += " This indicates a nested table region."
        elif entry["depth"] == 1:
            depth_text += " This is a top-level table region."
        lines.append(depth_text)
    else:
        lines.append("This paragraph is not marked as being inside a table.")

    if entry["is_ttp"]:
        lines.append("`sprmPFTtp=true`, so this paragraph is a row end (TTP mark).")
    else:
        lines.append("`sprmPFTtp=false`, so this paragraph is not marked as a row end.")

    if entry["is_inner_cell"]:
        lines.append("`sprmPFInnerTableCell=true`, so this paragraph is related to an inner-table cell boundary.")
    if entry["is_inner_ttp"]:
        lines.append("`sprmPFInnerTtp=true`, so this paragraph is related to an inner-table row boundary.")

    if entry["p_itap"] is not None:
        lines.append(f"`sprmPItap={entry['p_itap']}`. This is the paragraph's table depth base value.")
    if entry["p_dtap"] is not None:
        lines.append(f"`sprmPDtap={entry['p_dtap']}`. This is the table depth delta value.")

    lines.extend(describe_tdef(entry["tdef"]))
    lines.append(describe_modifier_flags(entry["modifier_flags"]))
    return lines


def decode_segments(blob: bytes) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    bool_names = {
        "PFInTable": "paragraph is inside a table",
        "PFTtp": "paragraph is a row end (TTP)",
        "PFInnerTableCell": "paragraph is an inner-table cell boundary",
        "PFInnerTtp": "paragraph is an inner-table row boundary",
    }
    int_names = {
        "PItap": "paragraph table depth base value",
        "PDtap": "table depth delta value",
    }

    for name, label in bool_names.items():
        for pos in find_all(blob, OPCODES[name]):
            if pos + 3 <= len(blob):
                raw = blob[pos : pos + 3]
                value = raw[2] != 0
                segments.append(
                    {
                        "offset": str(pos),
                        "hex": raw.hex(),
                        "meaning": f"{name} = {value} ({label})",
                    }
                )

    for name, label in int_names.items():
        for pos in find_all(blob, OPCODES[name]):
            if pos + 6 <= len(blob):
                raw = blob[pos : pos + 6]
                value = struct.unpack("<i", raw[2:6])[0]
                segments.append(
                    {
                        "offset": str(pos),
                        "hex": raw.hex(),
                        "meaning": f"{name} = {value} ({label})",
                    }
                )

    for pos in find_all(blob, OPCODES["TDefTable"]):
        if pos + 5 > len(blob):
            continue
        cb = struct.unpack("<H", blob[pos + 2 : pos + 4])[0]
        end = min(len(blob), pos + 4 + cb)
        raw = blob[pos:end]
        tdef = parse_tdef(blob)
        if tdef:
            meaning = (
                f"TDefTable: columns={tdef['number_of_columns']}, "
                f"rgdxaCenter={tdef['rgdxa_center']}"
            )
        else:
            meaning = "TDefTable found, but operand could not be fully decoded"
        segments.append({"offset": str(pos), "hex": raw.hex(), "meaning": meaning})

    for name in ("TInsert", "TDelete", "TDxaCol", "TMerge", "TSplit", "TVertMerge", "TCellWidth"):
        for pos in find_all(blob, OPCODES[name]):
            end = min(len(blob), pos + 12)
            raw = blob[pos:end]
            segments.append(
                {
                    "offset": str(pos),
                    "hex": raw.hex(),
                    "meaning": f"{name} modifier present",
                }
            )

    segments.sort(key=lambda item: int(item["offset"]))
    return segments


def parse_entries(word_document: bytes, table_stream: bytes, full_text: str, pieces: list[dict[str, int | bool]]) -> list[dict]:
    fc_bte = struct.unpack("<I", word_document[0x102:0x106])[0]
    lcb_bte = struct.unpack("<I", word_document[0x106:0x10A])[0]
    plcf = table_stream[fc_bte : fc_bte + lcb_bte]
    entry_count = (len(plcf) - 4) // 8
    data_offset = 4 * (entry_count + 1)
    results: list[dict] = []

    for index in range(entry_count):
        raw = struct.unpack("<I", plcf[data_offset + index * 4 : data_offset + (index + 1) * 4])[0]
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
            total_length = (2 + (2 * fkp[papx_offset + 1])) if cb == 0 else (1 + (2 * cb - 1))
            blob = fkp[papx_offset : papx_offset + total_length]

            fc_start = rgfc[para_index]
            fc_end = rgfc[para_index + 1]
            cp_start = fc_to_cp(fc_start, pieces)
            cp_end = fc_to_cp(fc_end, pieces)
            if cp_start is None or cp_end is None:
                continue

            entry = {
                "fc_start": fc_start,
                "fc_end": fc_end,
                "cp_start": cp_start,
                "cp_end": cp_end,
                "papx_pn": pn,
                "papx_entry_index": para_index,
                "in_table": parse_bool_property(blob, OPCODES["PFInTable"]),
                "is_ttp": parse_bool_property(blob, OPCODES["PFTtp"]),
                "is_inner_cell": parse_bool_property(blob, OPCODES["PFInnerTableCell"]),
                "is_inner_ttp": parse_bool_property(blob, OPCODES["PFInnerTtp"]),
                "p_itap": parse_int32_property(blob, OPCODES["PItap"]),
                "p_dtap": parse_int32_property(blob, OPCODES["PDtap"]),
                "tdef": parse_tdef(blob),
                "modifier_flags": {
                    name: bool(find_all(blob, OPCODES[name]))
                    for name in ("TInsert", "TDelete", "TDxaCol", "TMerge", "TSplit", "TVertMerge", "TCellWidth")
                },
                "decoded_values": {
                    "sprmPFInTable": parse_bool_property(blob, OPCODES["PFInTable"]),
                    "sprmPFTtp": parse_bool_property(blob, OPCODES["PFTtp"]),
                    "sprmPFInnerTableCell": parse_bool_property(blob, OPCODES["PFInnerTableCell"]),
                    "sprmPFInnerTtp": parse_bool_property(blob, OPCODES["PFInnerTtp"]),
                    "sprmPItap": parse_int32_property(blob, OPCODES["PItap"]),
                    "sprmPDtap": parse_int32_property(blob, OPCODES["PDtap"]),
                    "sprmTDefTable": parse_tdef(blob),
                    "sprmTInsert": bool(find_all(blob, OPCODES["TInsert"])),
                    "sprmTDelete": bool(find_all(blob, OPCODES["TDelete"])),
                    "sprmTDxaCol": bool(find_all(blob, OPCODES["TDxaCol"])),
                    "sprmTMerge": bool(find_all(blob, OPCODES["TMerge"])),
                    "sprmTSplit": bool(find_all(blob, OPCODES["TSplit"])),
                    "sprmTVertMerge": bool(find_all(blob, OPCODES["TVertMerge"])),
                    "sprmTCellWidth": bool(find_all(blob, OPCODES["TCellWidth"])),
                },
                "decoded_segments": decode_segments(blob),
                "raw_blob_hex": blob.hex(),
                "text_snippet": snippet(full_text, cp_start, cp_end),
            }
            entry["depth"] = ((entry["p_itap"] or 0) + (entry["p_dtap"] or 0)) if entry["in_table"] else 0
            results.append(entry)

    results.sort(key=lambda item: (item["cp_start"], item["cp_end"]))
    return results


def render(entries: list[dict]) -> str:
    lines = [
        "# DOC OLE Raw Table Property Dump",
        "",
        "- This is a raw property dump from OLE/PAPX.",
        "- No table regrouping, no markdown table reconstruction, no row merging.",
        "- Each item is a paragraph/property record in source order.",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        if not entry["in_table"] and not entry["tdef"]:
            continue
        lines.append(f"## Entry {index}")
        lines.append("")
        lines.append(f"- cp: `{entry['cp_start']}..{entry['cp_end']}`")
        lines.append(f"- fc: `{entry['fc_start']}..{entry['fc_end']}`")
        lines.append(f"- in_table: `{entry['in_table']}`")
        lines.append(f"- depth: `{entry['depth']}`")
        lines.append(f"- is_ttp: `{entry['is_ttp']}`")
        lines.append(f"- is_inner_cell: `{entry['is_inner_cell']}`")
        lines.append(f"- is_inner_ttp: `{entry['is_inner_ttp']}`")
        lines.append(f"- p_itap: `{entry['p_itap']}`")
        lines.append(f"- p_dtap: `{entry['p_dtap']}`")
        lines.append(f"- tdef: `{entry['tdef']}`")
        lines.append(f"- modifiers: `{entry['modifier_flags']}`")
        lines.append(f"- snippet: `{entry['text_snippet']}`")
        lines.append("")
        lines.append("### Human Description")
        lines.append("")
        for description in describe_entry(entry):
            lines.append(f"- {description}")
        lines.append("")
        lines.append("### Decoded Values")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(entry["decoded_values"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("### Byte Mapping")
        lines.append("")
        if entry["decoded_segments"]:
            for segment in entry["decoded_segments"]:
                lines.append(
                    f"- offset `{segment['offset']}`: `{segment['hex']}` -> {segment['meaning']}"
                )
        else:
            lines.append("- No decodable table-related opcodes were found.")
        lines.append("")
        lines.append("### Raw Hex")
        lines.append("")
        lines.append("```text")
        lines.append(entry["raw_blob_hex"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    word_document, table_stream, full_text, pieces = reconstruct_piece_table(SOURCE_PATH)
    entries = parse_entries(word_document, table_stream, full_text, pieces)

    json_path = RUN_ROOT / "raw_table_props.json"
    md_path = RUN_ROOT / "raw_table_props.md"
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render(entries), encoding="utf-8")

    manifest = {
        "source_path": SOURCE_PATH.as_posix(),
        "entry_count": len(entries),
        "json_path": json_path.as_posix(),
        "markdown_path": md_path.as_posix(),
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
