from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path

import olefile


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(r"C:\Users\yongseop.im\Desktop\all_docs\금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc")
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_deep_probe"

SPRMS = {
    "sprmTTableHeader": b"\x04\x34",
    "sprmTTlp": b"\x0A\x74",
    "sprmTDyaRowHeight": b"\x07\x94",
    "sprmTDefTable": b"\x08\xD6",
    "sprmTInsert": b"\x21\x76",
    "sprmTDelete": b"\x22\x56",
}


def find_all(data: bytes, needle: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        index = data.find(needle, start)
        if index < 0:
            return offsets
        offsets.append(index)
        start = index + 1


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
            }
        )
    return records


def build_report(signature_offsets: dict[str, list[int]], records: list[dict[str, object]]) -> str:
    lines = [
        "# DOC OLE Deep Probe",
        "",
        f"- source: `{SOURCE_PATH.as_posix()}`",
        "",
        "## Signature Counts",
        "",
    ]
    for name, offsets in signature_offsets.items():
        lines.append(f"- {name}: `{len(offsets)}`")

    lines.extend(["", "## TDefTable Summary", ""])
    column_counter = Counter(int(record["number_of_columns"]) for record in records)
    for number_of_columns, count in sorted(column_counter.items()):
        lines.append(f"- columns={number_of_columns}: `{count}` rows")

    lines.extend(["", "## Unique Layout Samples", ""])
    seen: set[tuple[int, tuple[int, ...]]] = set()
    shown = 0
    for record in records:
        key = (
            int(record["number_of_columns"]),
            tuple(int(value) for value in record["rgdxa_center"]),
        )
        if key in seen:
            continue
        seen.add(key)
        shown += 1
        lines.append(f"### Layout {shown}")
        lines.append("")
        lines.append(f"- offset: `{record['offset']}`")
        lines.append(f"- number_of_columns: `{record['number_of_columns']}`")
        lines.append(f"- cell_boundaries_twips: `{record['rgdxa_center']}`")
        lines.append("")
        if shown >= 20:
            break

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `sprmTDefTable` present means row-level cell definitions are embedded in the DOC binary.",
            "- `sprmTTableHeader` present means header-row metadata also exists in the file.",
            "- This confirms a deeper OLE-based parser is feasible beyond plain text extraction.",
            "- It does not by itself give a ready-made logical tree; we still need PAPX/row-property application to map these definitions back to row marks and document positions.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    ole = olefile.OleFileIO(str(SOURCE_PATH))
    try:
        word_document = ole.openstream("WordDocument").read()
        flags = struct.unpack("<H", word_document[0x0A:0x0C])[0]
        table_stream_name = "1Table" if (flags & 0x0200) else "0Table"
    finally:
        ole.close()

    signature_offsets = {name: find_all(word_document, value) for name, value in SPRMS.items()}
    records = parse_tdef_records(word_document)

    report_path = RUN_ROOT / "deep_probe_report.md"
    report_path.write_text(build_report(signature_offsets, records), encoding="utf-8")

    manifest = {
        "source_path": SOURCE_PATH.as_posix(),
        "selected_table_stream": table_stream_name,
        "signature_counts": {name: len(offsets) for name, offsets in signature_offsets.items()},
        "valid_tdef_record_count": len(records),
        "column_count_distribution": dict(Counter(int(record["number_of_columns"]) for record in records)),
        "report_path": report_path.as_posix(),
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "tdef_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
