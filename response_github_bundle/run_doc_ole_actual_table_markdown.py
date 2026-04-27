from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ACTUAL_ROWS_PATH = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_actual_rows" / "actual_rows.json"
RUN_ROOT = PROJECT_ROOT / "outputs" / "manual_compare" / "doc_ole_final_markdown"


def load_rows() -> list[dict]:
    return json.loads(ACTUAL_ROWS_PATH.read_text(encoding="utf-8"))


def group_rows(rows: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []

    def same_table(prev: dict, curr: dict) -> bool:
        prev_left, prev_right = prev["rgdxa_center"][0], prev["rgdxa_center"][-1]
        curr_left, curr_right = curr["rgdxa_center"][0], curr["rgdxa_center"][-1]
        gap = int(curr["row_start_cp"]) - int(prev["row_end_cp"])
        same_outer = abs(prev_left - curr_left) <= 160 and abs(prev_right - curr_right) <= 320
        if same_outer and gap <= 120:
            return True
        if gap <= 8 and abs(prev_right - curr_right) <= 320:
            return True
        return False

    for row in rows:
        if not current:
            current = [row]
            continue
        if same_table(current[-1], row):
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    if current:
        groups.append(current)
    return groups


def build_master_boundaries(group: list[dict]) -> list[int]:
    boundaries: list[int] = []
    for row in group:
        for value in row["rgdxa_center"]:
            if value not in boundaries:
                boundaries.append(value)
    boundaries.sort()
    return boundaries


def map_row_to_master_columns(row: dict, master_boundaries: list[int]) -> list[str]:
    master_col_count = len(master_boundaries) - 1
    output = [""] * master_col_count

    row_boundaries = row["rgdxa_center"]
    row_cells = row["cells"]
    for index, text in enumerate(row_cells):
        if index + 1 >= len(row_boundaries):
            break
        left = row_boundaries[index]
        right = row_boundaries[index + 1]

        start_index = None
        end_index = None
        for master_index in range(master_col_count):
            master_left = master_boundaries[master_index]
            master_right = master_boundaries[master_index + 1]
            overlap = min(right, master_right) - max(left, master_left)
            if overlap > 0:
                if start_index is None:
                    start_index = master_index
                end_index = master_index

        if start_index is None:
            continue
        output[start_index] = text
        for fill_index in range(start_index + 1, (end_index or start_index) + 1):
            output[fill_index] = ""

    return output


def render_group_markdown(table_index: int, group: list[dict]) -> str:
    master_boundaries = build_master_boundaries(group)
    if len(master_boundaries) < 2:
        return ""

    master_col_count = len(master_boundaries) - 1
    rendered_rows = [map_row_to_master_columns(row, master_boundaries) for row in group]

    header = rendered_rows[0]
    lines = [f"## Table {table_index}", ""]
    lines.append(f"- row_count: `{len(group)}`")
    lines.append(f"- master_column_count: `{master_col_count}`")
    lines.append(f"- master_boundaries_twips: `{master_boundaries}`")
    lines.append(f"- row_cp_range: `{group[0]['row_start_cp']}..{group[-1]['row_end_cp']}`")
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


def main() -> None:
    rows = load_rows()
    groups = group_rows(rows)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    lines = [
        "# DOC OLE Final Markdown",
        "",
        "- Source: `C:/Users/yongseop.im/Desktop/all_docs/금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc`",
        f"- Actual row count: `{len(rows)}`",
        f"- Grouped table count: `{len(groups)}`",
        "",
        "> This markdown is built from actual DOC row properties (`sprmTDefTable`) plus row-end mark positions.",
        "> Merged cells are flattened into markdown columns because markdown has no colspan/rowspan.",
        "",
    ]

    manifest = {
        "source_path": "C:/Users/yongseop.im/Desktop/all_docs/금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc",
        "actual_row_count": len(rows),
        "grouped_table_count": len(groups),
        "final_markdown_path": (RUN_ROOT / "final_tables.md").as_posix(),
    }

    for index, group in enumerate(groups, start=1):
        lines.append(render_group_markdown(index, group))

    (RUN_ROOT / "final_tables.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
