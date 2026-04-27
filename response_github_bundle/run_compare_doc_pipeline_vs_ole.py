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
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_pipeline_vs_ole"


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x0b", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def reconstruct_full_text(path: Path) -> tuple[str, str]:
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
        return full_text, table_stream_name
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


def infer_header_row_count(table: list[list[str]]) -> int:
    if not table:
        return 0
    if len(table) == 1:
        return 1
    max_cols = max(len(row) for row in table)
    padded = [row + [""] * (max_cols - len(row)) for row in table]
    first_numeric = sum(bool(re.search(r"\d", cell)) for cell in padded[0])
    second_numeric = sum(bool(re.search(r"\d", cell)) for cell in padded[1])
    if first_numeric <= max(1, max_cols // 3) and second_numeric <= max(1, max_cols // 3):
        return 2
    return 1


def render_table_markdown(table: list[list[str]], title: str) -> str:
    if not table:
        return ""

    max_cols = max(len(row) for row in table)
    padded = [row + [""] * (max_cols - len(row)) for row in table]
    header_rows = infer_header_row_count(padded)

    lines = [f"## {title}", ""]
    lines.append(f"- inferred_header_rows: `{header_rows}`")
    lines.append(f"- row_count: `{len(table)}`")
    lines.append(f"- column_count: `{max_cols}`")
    lines.append("")
    lines.append("| " + " | ".join(cell or " " for cell in padded[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in padded[0]) + " |")
    for row in padded[1:]:
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")
    if header_rows > 1:
        lines.append("")
        lines.append("```text")
        for index in range(header_rows):
            lines.append(f"header_row_{index + 1}: " + " | ".join(cell or " " for cell in padded[index]))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def summarize_pipeline_blocks(blocks: list[dict[str, object]]) -> list[str]:
    lines = ["# Pipeline Block Summary", ""]
    type_counts = Counter(str(block.get("type")) for block in blocks)
    for block_type, count in sorted(type_counts.items()):
        lines.append(f"- {block_type}: `{count}`")
    lines.append("")
    for index, block in enumerate(blocks, start=1):
        block_type = str(block.get("type"))
        lines.append(f"## Block {index}")
        lines.append("")
        lines.append(f"- type: `{block_type}`")
        if block.get("section"):
            lines.append(f"- section: `{block.get('section')}`")
        if block_type == "row_table":
            header_rows = block.get("header_rows") or []
            rows = block.get("rows") or []
            lines.append(f"- header_rows: `{len(header_rows)}`")
            lines.append(f"- data_rows: `{len(rows)}`")
            lines.append("```text")
            for row_index, row in enumerate(header_rows, start=1):
                lines.append(f"header_row_{row_index}: " + " | ".join(str(cell) for cell in row))
            for row_index, row in enumerate(rows, start=1):
                lines.append(f"row_{row_index}: " + " | ".join(str(cell) for cell in row))
            lines.append("```")
        elif block_type == "kv_table":
            lines.append("```json")
            lines.append(json.dumps(block.get("fields") or {}, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            content = str(block.get("content") or "")
            if content:
                lines.append("```text")
                lines.append(content)
                lines.append("```")
        lines.append("")
    return lines


def build_tree_assessment(tables: list[list[list[str]]], parsed_blocks: list[dict[str, object]]) -> list[str]:
    pipeline_row_tables = [block for block in parsed_blocks if block.get("type") == "row_table"]
    pipeline_kv_tables = [block for block in parsed_blocks if block.get("type") == "kv_table"]

    lines = ["# Tree Structure Assessment", ""]
    lines.append("## Verdict")
    lines.append("")
    lines.append("- Current pipeline: true table tree reconstruction is not supported for `.doc`.")
    lines.append("- Current OLE comparison path: row and cell boundaries can be inferred, but a formal logical tree is not recovered.")
    lines.append("")
    lines.append("## Why")
    lines.append("")
    lines.append("- The active `.doc` parser only extracts text candidates from OLE streams and then applies heuristic block postprocessing.")
    lines.append("- The OLE comparison here rebuilds piece-table text and splits on control characters, which gives table-like rows and cells.")
    lines.append("- Neither path currently parses Word table property structures such as merged-cell geometry, nested tables, or semantic header metadata.")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(f"- OLE inferred tables: `{len(tables)}`")
    lines.append(f"- Pipeline `row_table` blocks: `{len(pipeline_row_tables)}`")
    lines.append(f"- Pipeline `kv_table` blocks: `{len(pipeline_kv_tables)}`")
    lines.append("- OLE tables often preserve first-row and some second-row header text, but this is inferred from position, not from a true node tree.")
    lines.append("- Pipeline blocks flatten most content into paragraph/title/kv_table/row_table heuristics after raw extraction.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("- We can say `table-like structure` is partially recoverable from OLE text.")
    lines.append("- We cannot say `logical tree / authoritative table header hierarchy` is currently recoverable with this implementation.")
    return lines


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_PATH}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    parser = DocParser()
    classification = classify_document(SOURCE_PATH)
    issues: list = []
    pipeline_raw = parser._try_powershell_word(SOURCE_PATH, issues)
    raw_method = "powershell-word-com" if pipeline_raw else ""
    if not pipeline_raw:
        pipeline_raw = parser._try_ole_extraction(SOURCE_PATH, issues)
        raw_method = "ole-best-effort" if pipeline_raw else "unavailable"

    parsed = parser.parse(SOURCE_PATH, classification)
    full_text, table_stream_name = reconstruct_full_text(SOURCE_PATH)
    tables, paragraphs = parse_tables_from_full_text(full_text)

    (RUN_ROOT / "pipeline_raw_pre_postprocess.txt").write_text(
        normalize_text(pipeline_raw) + ("\n" if pipeline_raw else ""),
        encoding="utf-8",
    )
    (RUN_ROOT / "ole_piece_table_full_text.txt").write_text(full_text, encoding="utf-8")
    (RUN_ROOT / "pipeline_postprocessed_markdown.md").write_text(parsed.markdown, encoding="utf-8")
    (RUN_ROOT / "pipeline_blocks_summary.md").write_text(
        "\n".join(summarize_pipeline_blocks(parsed.blocks)),
        encoding="utf-8",
    )

    table_lines = [
        "# OLE Piece Table Reconstruction",
        "",
        f"- source: `{SOURCE_PATH.as_posix()}`",
        f"- selected_table_stream: `{table_stream_name}`",
        f"- inferred_table_count: `{len(tables)}`",
        f"- non_table_paragraph_count: `{len(paragraphs)}`",
        "",
        "## Paragraphs",
        "",
    ]
    for paragraph in paragraphs:
        table_lines.append(paragraph)
        table_lines.append("")
    for index, table in enumerate(tables, start=1):
        table_lines.append(render_table_markdown(table, f"Table {index}"))
    (RUN_ROOT / "ole_piece_table_tables.md").write_text("\n".join(table_lines), encoding="utf-8")

    (RUN_ROOT / "tree_structure_assessment.md").write_text(
        "\n".join(build_tree_assessment(tables, parsed.blocks)),
        encoding="utf-8",
    )

    manifest = {
        "source_path": SOURCE_PATH.as_posix(),
        "pipeline_raw_method": raw_method,
        "classification": classification.__dict__,
        "pipeline_status": parsed.status,
        "pipeline_parser_name": parsed.parser_name,
        "pipeline_raw_pre_postprocess_path": (RUN_ROOT / "pipeline_raw_pre_postprocess.txt").as_posix(),
        "pipeline_postprocessed_markdown_path": (RUN_ROOT / "pipeline_postprocessed_markdown.md").as_posix(),
        "pipeline_blocks_summary_path": (RUN_ROOT / "pipeline_blocks_summary.md").as_posix(),
        "ole_piece_table_full_text_path": (RUN_ROOT / "ole_piece_table_full_text.txt").as_posix(),
        "ole_piece_table_tables_path": (RUN_ROOT / "ole_piece_table_tables.md").as_posix(),
        "tree_structure_assessment_path": (RUN_ROOT / "tree_structure_assessment.md").as_posix(),
        "ole_inferred_table_count": len(tables),
        "pipeline_block_type_counts": dict(Counter(str(block.get("type")) for block in parsed.blocks)),
        "issues": [issue.__dict__ for issue in issues],
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
