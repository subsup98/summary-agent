from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ACTUAL_ROWS_PATH = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_actual_rows" / "actual_rows.json"
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_report_ready"

CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def clean_cell(text: str) -> str:
    cleaned = CONTROL_CHAR_RE.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_rows() -> list[dict]:
    raw_rows = json.loads(ACTUAL_ROWS_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for row in raw_rows:
        cells = [clean_cell(cell) for cell in row["cells"]]
        nonempty_cells = [cell for cell in cells if cell]
        row = dict(row)
        row["cells"] = cells
        row["nonempty_cells"] = nonempty_cells
        row["nonempty_count"] = len(nonempty_cells)
        rows.append(row)
    return rows


def group_rows(rows: list[dict]) -> list[list[dict]]:
    rows = [row for row in rows if row["nonempty_count"] > 0]
    groups: list[list[dict]] = []
    current: list[dict] = []

    def same_group(prev: dict, curr: dict) -> bool:
        gap = int(curr["row_start_cp"]) - int(prev["row_end_cp"])
        prev_left, prev_right = prev["rgdxa_center"][0], prev["rgdxa_center"][-1]
        curr_left, curr_right = curr["rgdxa_center"][0], curr["rgdxa_center"][-1]
        outer_close = abs(prev_left - curr_left) <= 200 and abs(prev_right - curr_right) <= 400
        similar_cols = abs(int(prev["number_of_columns"]) - int(curr["number_of_columns"])) <= 1
        return gap <= 1400 and (outer_close or similar_cols)

    for row in rows:
        if not current:
            current = [row]
            continue
        if same_group(current[-1], row):
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    if current:
        groups.append(current)
    return groups


def classify_group(group: list[dict]) -> str:
    if len(group) == 1 and group[0]["nonempty_count"] <= 1:
        return "heading"
    if len(group) <= 2 and sum(row["nonempty_count"] for row in group) <= 3:
        return "heading"
    return "table"


def merge_heading_groups(groups: list[list[dict]]) -> list[dict]:
    merged: list[dict] = []
    pending_headings: list[str] = []

    for group in groups:
        kind = classify_group(group)
        if kind == "heading":
            for row in group:
                for cell in row["nonempty_cells"]:
                    if cell not in pending_headings:
                        pending_headings.append(cell)
            continue
        merged.append({"kind": "table", "headings": pending_headings[:], "rows": group})
        pending_headings = []

    if pending_headings:
        merged.append({"kind": "heading_only", "headings": pending_headings[:]})
    return merged


def build_master_boundaries(rows: list[dict]) -> list[int]:
    boundaries: list[int] = []
    for row in rows:
        for boundary in row["rgdxa_center"]:
            if boundary not in boundaries:
                boundaries.append(boundary)
    boundaries.sort()
    return boundaries


def map_row(row: dict, master_boundaries: list[int]) -> list[str]:
    output = [""] * (len(master_boundaries) - 1)
    for index, text in enumerate(row["cells"]):
        if not text:
            continue
        if index + 1 >= len(row["rgdxa_center"]):
            break
        left = row["rgdxa_center"][index]
        right = row["rgdxa_center"][index + 1]
        hits: list[int] = []
        for master_index in range(len(master_boundaries) - 1):
            overlap = min(right, master_boundaries[master_index + 1]) - max(left, master_boundaries[master_index])
            if overlap > 0:
                hits.append(master_index)
        if hits:
            output[hits[0]] = text
    return output


def pick_header(rows: list[list[str]]) -> list[str]:
    first = rows[0]
    nonempty = sum(bool(cell) for cell in first)
    if nonempty >= 2:
        return first
    if len(rows) > 1:
        return rows[1]
    return first


def render_table_block(index: int, block: dict) -> str:
    rows = block["rows"]
    master_boundaries = build_master_boundaries(rows)
    rendered = [map_row(row, master_boundaries) for row in rows]
    header = pick_header(rendered)
    lines: list[str] = []

    for heading in block["headings"]:
        lines.append(f"### {heading}")
    if block["headings"]:
        lines.append("")

    lines.append(f"## Table {index}")
    lines.append("")
    lines.append(f"- row_count: `{len(rows)}`")
    lines.append(f"- column_count: `{len(master_boundaries) - 1}`")
    lines.append("")
    lines.append("| " + " | ".join(cell or " " for cell in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    header_used = False
    for row in rendered:
        if not header_used and row == header:
            header_used = True
            continue
        if any(cell for cell in row):
            lines.append("| " + " | ".join(cell or " " for cell in row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_document(blocks: list[dict]) -> str:
    lines = [
        "# DOC Report-Ready Markdown",
        "",
        "- Source: `C:/Users/yongseop.im/Desktop/all_docs/금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc`",
        "- This version is postprocessed for readability.",
        "- Control characters and empty rows were removed.",
        "- Short heading fragments are promoted to headings and attached to the next table.",
        "",
    ]

    table_index = 1
    for block in blocks:
        if block["kind"] == "table":
            lines.append(render_table_block(table_index, block))
            table_index += 1
        elif block["kind"] == "heading_only":
            for heading in block["headings"]:
                lines.append(f"### {heading}")
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    groups = group_rows(rows)
    blocks = merge_heading_groups(groups)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = RUN_ROOT / "report_ready_tables.md"
    output_path.write_text(render_document(blocks), encoding="utf-8")

    manifest = {
        "source_path": "C:/Users/yongseop.im/Desktop/all_docs/금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc",
        "group_count_before_heading_merge": len(groups),
        "block_count_after_heading_merge": len(blocks),
        "report_ready_markdown_path": output_path.as_posix(),
    }
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
