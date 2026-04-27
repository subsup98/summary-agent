from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "requested_md_bundle"

DATE_RE = re.compile(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$")
INT_RE = re.compile(r"^\d{1,4}$")
EXPLICIT_KV_RE = re.compile(r"^(?P<label>[^:\n]{1,40})\s*:\s*(?P<value>.*?)\s*$")
SECTION_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
DATE_INT_LINE_RE = re.compile(r"^(?P<date>\d{4}[./-]\d{1,2}[./-]\d{1,2})\s+(?P<count>\d{1,4})$")


@dataclass
class Record:
    values: dict[str, str] = field(default_factory=dict)
    list_values: dict[str, list[str]] = field(default_factory=dict)


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"\s+", " ", label.strip())
    return cleaned


def normalize_item(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:[-*]\s+|[.\-]+\s*)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def join_cell(value: str, items: list[str]) -> str:
    if items:
        return "<br>".join(item for item in items if item)
    return value.strip()


def render_table(headers: list[str], records: list[Record]) -> list[str]:
    if not headers or not records:
        return []

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        row: list[str] = []
        for header in headers:
            row.append(join_cell(record.values.get(header, ""), record.list_values.get(header, [])))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_explicit_record(lines: list[str], start: int) -> tuple[Record | None, list[str], int]:
    index = start
    labels: list[str] = []
    record = Record()
    active_list_label: str | None = None

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            if record.values or record.list_values:
                index += 1
                break
            return None, [], start
        if SECTION_HEADING_RE.match(stripped):
            break

        kv_match = EXPLICIT_KV_RE.match(stripped)
        if kv_match:
            label = normalize_label(kv_match.group("label"))
            value = kv_match.group("value").strip()
            labels.append(label)
            if value:
                record.values[label] = re.sub(r"\s+", " ", value)
                active_list_label = None
            else:
                record.values[label] = ""
                active_list_label = label
            index += 1
            continue

        if stripped in {"[", "]"}:
            index += 1
            continue

        if active_list_label and stripped.startswith(("*", "-", "•")):
            record.list_values.setdefault(active_list_label, []).append(normalize_item(stripped))
            index += 1
            continue

        if active_list_label and stripped and not EXPLICIT_KV_RE.match(stripped):
            normalized = normalize_item(stripped)
            if normalized:
                record.list_values.setdefault(active_list_label, []).append(normalized)
                index += 1
                continue

        break

    if len(labels) < 2:
        return None, [], start
    return record, labels, index


def build_implicit_record(lines: list[str], start: int, schema: list[str]) -> tuple[Record | None, int]:
    if len(schema) < 3:
        return None, start

    match = DATE_INT_LINE_RE.match(lines[start].strip())
    if not match:
        return None, start

    record = Record(
        values={
            schema[0]: match.group("date"),
            schema[1]: match.group("count"),
        }
    )
    index = start + 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    items: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            break
        if SECTION_HEADING_RE.match(stripped):
            break
        if DATE_INT_LINE_RE.match(stripped):
            break
        if EXPLICIT_KV_RE.match(stripped):
            break
        if stripped.startswith(("*", "-", "•")):
            items.append(normalize_item(stripped))
        else:
            normalized = normalize_item(stripped)
            if normalized:
                items.append(normalized)
        index += 1

    if not items:
        return None, start
    record.list_values[schema[2]] = items
    return record, index


def recover_record_tables(markdown: str) -> tuple[str, list[dict[str, object]]]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    index = 0
    active_schema: list[str] | None = None
    changes: list[dict[str, object]] = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        explicit_record, labels, next_index = build_explicit_record(lines, index)
        if explicit_record is not None:
            schema = labels
            records = [explicit_record]
            cursor = next_index
            implicit_count = 0

            while cursor < len(lines):
                while cursor < len(lines) and not lines[cursor].strip():
                    cursor += 1
                candidate, candidate_next = build_implicit_record(lines, cursor, schema)
                if candidate is None:
                    candidate2, labels2, candidate2_next = build_explicit_record(lines, cursor)
                    if candidate2 is None or labels2 != schema:
                        break
                    records.append(candidate2)
                    cursor = candidate2_next
                    continue
                records.append(candidate)
                implicit_count += 1
                cursor = candidate_next

            if len(records) >= 2:
                out.extend(render_table(schema, records))
                out.append("")
                changes.append(
                    {
                        "start_line": index + 1,
                        "end_line": cursor,
                        "schema": schema,
                        "record_count": len(records),
                        "implicit_record_count": implicit_count,
                    }
                )
                active_schema = schema
                index = cursor
                continue

        if active_schema is not None:
            probe = index
            while probe < len(lines) and not lines[probe].strip():
                out.append(lines[probe])
                probe += 1
            if probe != index:
                index = probe
                if index >= len(lines):
                    break
            implicit_record, implicit_next = build_implicit_record(lines, index, active_schema)
            if implicit_record is not None:
                records = [implicit_record]
                cursor = implicit_next
                while cursor < len(lines):
                    while cursor < len(lines) and not lines[cursor].strip():
                        cursor += 1
                    candidate, candidate_next = build_implicit_record(lines, cursor, active_schema)
                    if candidate is None:
                        break
                    records.append(candidate)
                    cursor = candidate_next
                if len(records) >= 1:
                    out.extend(render_table(active_schema, records))
                    out.append("")
                    changes.append(
                        {
                            "start_line": index + 1,
                            "end_line": cursor,
                            "schema": active_schema,
                            "record_count": len(records),
                            "implicit_record_count": len(records),
                        }
                    )
                    index = cursor
                    continue

        out.append(line)
        if stripped and SECTION_HEADING_RE.match(stripped):
            active_schema = None
        index += 1

    return "\n".join(out).strip() + "\n", changes


def write_preview(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    source = input_path.read_text(encoding="utf-8")
    preview, changes = recover_record_tables(source)
    output_path.write_text(preview, encoding="utf-8")

    report_lines = [
        f"# Preview Report: {input_path.name}",
        "",
        f"- source: `{input_path.as_posix()}`",
        f"- output: `{output_path.as_posix()}`",
        f"- change_count: {len(changes)}",
        "",
    ]
    for idx, change in enumerate(changes, start=1):
        report_lines.extend(
            [
                f"## Change {idx}",
                f"- lines: {change['start_line']}..{change['end_line']}",
                f"- schema: {', '.join(change['schema'])}",
                f"- record_count: {change['record_count']}",
                f"- implicit_record_count: {change['implicit_record_count']}",
                "",
            ]
        )
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview generalized record-set recovery on existing markdown outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where preview markdown files will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            output_dir / "hwp_existing_pipeline_recordset_preview.md",
            output_dir / "hwp_existing_pipeline_recordset_preview_report.md",
            output_dir / "hwp_existing_pipeline.md",
        ),
        (
            output_dir / "hwp_converted_pdf_pymupdf4llm_recordset_preview.md",
            output_dir / "hwp_converted_pdf_pymupdf4llm_recordset_preview_report.md",
            output_dir / "hwp_converted_pdf_pymupdf4llm.md",
        ),
    ]

    for output_path, report_path, input_path in jobs:
        write_preview(input_path=input_path, output_path=output_path, report_path=report_path)
        print(output_path.as_posix())
        print(report_path.as_posix())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
