from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import olefile

from src.classifiers.document_classifier import classify_document
from src.parsers.office.doc_parser import DocParser


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_raw_parse" / "doc_raw_all"


def discover_doc_files() -> list[Path]:
    return sorted(PROJECT_ROOT.glob("outputs/ui_runs/*/source/*.doc"))


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "doc"


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x0b", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def reconstruct_full_text(path: Path) -> str:
    ole = olefile.OleFileIO(path)
    try:
        wd = ole.openstream("WordDocument").read()

        flags = struct.unpack("<H", wd[0x0A:0x0C])[0]
        table_stream_name = "1Table" if (flags & 0x0200) else "0Table"
        table_stream = ole.openstream(table_stream_name).read()

        fc_clx = struct.unpack("<I", wd[0x01A2:0x01A6])[0]
        lcb_clx = struct.unpack("<I", wd[0x01A6:0x01AA])[0]
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
                    text_bytes = wd[fc : fc + char_count * 2]
                    text = text_bytes.decode("utf-16le", errors="replace")
                else:
                    fc = fc // 2
                    text_bytes = wd[fc : fc + char_count]
                    text = text_bytes.decode("cp1252", errors="replace")
                parts.append(text)
            full_text = "".join(parts)
            break
        return full_text
    finally:
        ole.close()


def parse_tables_from_full_text(full_text: str) -> tuple[list[list[list[str]]], list[str]]:
    cell = chr(7)
    para = chr(13)

    current_text: list[str] = []
    current_row_cells: list[str] = []
    current_table_rows: list[list[str]] = []
    all_tables: list[list[list[str]]] = []
    non_table_paragraphs: list[str] = []

    index = 0
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
                        all_tables.append(current_table_rows)
                        current_table_rows = []
                    non_table_paragraphs.append(para_text)
                current_text = []
        else:
            current_text.append(ch)

        index += 1

    if current_table_rows:
        all_tables.append(current_table_rows)

    return all_tables, non_table_paragraphs


def pick_header_row_count(table: list[list[str]]) -> int:
    if not table:
        return 0
    if len(table) == 1:
        return 1
    first = " ".join(cell for cell in table[0] if cell).strip()
    second = " ".join(cell for cell in table[1] if cell).strip() if len(table) > 1 else ""
    if len(table) > 1 and len(table[0]) == len(table[1]) and first and second:
        first_numeric = sum(bool(re.search(r"\d", cell)) for cell in table[0])
        second_numeric = sum(bool(re.search(r"\d", cell)) for cell in table[1])
        if first_numeric <= max(1, len(table[0]) // 3) and second_numeric <= max(1, len(table[1]) // 3):
            return 2
    return 1


def render_table_markdown(table: list[list[str]], title: str) -> str:
    if not table:
        return ""

    max_cols = max(len(row) for row in table)
    padded = [row + [""] * (max_cols - len(row)) for row in table]
    header_row_count = pick_header_row_count(padded)

    lines = [f"## {title}", ""]
    lines.append(f"- inferred_header_rows: `{header_row_count}`")
    lines.append(f"- row_count: `{len(table)}`")
    lines.append(f"- column_count: `{max_cols}`")
    lines.append("")

    primary_header = padded[0]
    lines.append("| " + " | ".join(cell or " " for cell in primary_header) + " |")
    lines.append("| " + " | ".join("---" for _ in primary_header) + " |")

    for row in padded[1:]:
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")

    if header_row_count > 1:
        lines.append("")
        lines.append("```text")
        for index in range(header_row_count):
            lines.append(f"header_row_{index + 1}: " + " | ".join(cell or " " for cell in padded[index]))
        lines.append("```")

    lines.append("")
    return "\n".join(lines)


def extract_pipeline_raw(path: Path, parser: DocParser) -> tuple[str, str, list[dict[str, str]]]:
    issues: list = []
    com_raw = parser._try_powershell_word(path, issues)
    if com_raw:
        return "powershell-word-com", com_raw, [issue.__dict__ for issue in issues]

    ole_raw = parser._try_ole_extraction(path, issues)
    if ole_raw:
        return "ole-best-effort", ole_raw, [issue.__dict__ for issue in issues]

    return "unavailable", "", [issue.__dict__ for issue in issues]


def build_output_dir(path: Path) -> Path:
    run_name = path.parent.parent.name
    base_name = slugify(path.stem)
    output_dir = RUN_ROOT / f"{run_name}__{base_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_one(path: Path) -> dict[str, object]:
    parser = DocParser()
    classification = classify_document(path)
    output_dir = build_output_dir(path)

    extraction_method, pipeline_raw_text, issues = extract_pipeline_raw(path, parser)
    full_text = reconstruct_full_text(path) if olefile.isOleFile(path) else ""
    tables, paragraphs = parse_tables_from_full_text(full_text) if full_text else ([], [])

    pipeline_raw_path = output_dir / "pipeline_raw_pre_postprocess.txt"
    ole_full_text_path = output_dir / "ole_piece_table_full_text.txt"
    ole_tables_path = output_dir / "ole_piece_table_tables.md"

    pipeline_raw_path.write_text(normalize_text(pipeline_raw_text) + ("\n" if pipeline_raw_text else ""), encoding="utf-8")
    ole_full_text_path.write_text(full_text, encoding="utf-8")

    table_doc_lines = [
        "# DOC OLE Table Reconstruction",
        "",
        f"- source_file: `{path.name}`",
        f"- source_path: `{path.as_posix()}`",
        f"- inferred_table_count: `{len(tables)}`",
        f"- non_table_paragraph_count: `{len(paragraphs)}`",
        "",
        "## Notes",
        "",
        "- This is inferred from OLE piece-table text plus DOC control characters.",
        "- It is not a true Word logical tree export.",
        "- Header rows below are inferred from the first one or two rows.",
        "",
        "## Non-table Paragraphs",
        "",
    ]
    for paragraph in paragraphs:
        table_doc_lines.append(paragraph)
        table_doc_lines.append("")
    for index, table in enumerate(tables, start=1):
        table_doc_lines.append(render_table_markdown(table, f"Table {index}"))
    ole_tables_path.write_text("\n".join(table_doc_lines), encoding="utf-8")

    result = {
        "source_path": path.as_posix(),
        "classification": classification.__dict__,
        "pipeline_extraction_method": extraction_method,
        "pipeline_raw_pre_postprocess_path": pipeline_raw_path.as_posix(),
        "ole_piece_table_full_text_path": ole_full_text_path.as_posix(),
        "ole_piece_table_tables_path": ole_tables_path.as_posix(),
        "table_count": len(tables),
        "paragraph_count": len(paragraphs),
        "issues": issues,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    doc_files = discover_doc_files()
    if not doc_files:
        raise FileNotFoundError("No .doc files found under outputs/ui_runs/*/source")

    results = [process_one(path) for path in doc_files]
    summary = {
        "generated_at_root": RUN_ROOT.as_posix(),
        "document_count": len(results),
        "documents": results,
    }
    (RUN_ROOT / "index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
