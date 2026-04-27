from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from run_logical_panel_split_probe import (
    build_panels,
    classify_document,
    extract_atoms,
    find_main_container,
    fitz,
    normalize_text,
    parse_args as _unused_parse_args,
    resolve_pdf_path,
    split_into_logical_panels,
    PdfParser,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "panel_band_parse"


@dataclass
class PanelBandResult:
    panel_id: str
    bbox: list[float]
    x_bands: list[dict[str, float]]
    y_bands: list[dict[str, float]]
    grid: list[list[str]]
    linearized_text: str
    source_full_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a simple x/y band parser to every logical panel.")
    parser.add_argument("--pdf", help="Optional PDF path.")
    parser.add_argument("--page", type=int, default=16, help="1-based page number to inspect.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/HTML outputs.")
    return parser.parse_args()


def cluster_values(values: list[float], tolerance: float) -> list[list[float]]:
    ordered = sorted(values)
    if not ordered:
        return []
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def build_bands(items: list[dict[str, Any]], *, axis: str, tolerance: float, pad: float) -> list[dict[str, float]]:
    if axis == "x":
        starts = [float(item["bbox"][0]) for item in items]
        ends = [float(item["bbox"][2]) for item in items]
    else:
        starts = [float(item["bbox"][1]) for item in items]
        ends = [float(item["bbox"][3]) for item in items]
    start_groups = cluster_values(starts, tolerance)
    if not start_groups:
        return []
    centers = [sum(group) / len(group) for group in start_groups]
    assignments: list[list[dict[str, Any]]] = [[] for _ in centers]
    for item in items:
        start_value = float(item["bbox"][0] if axis == "x" else item["bbox"][1])
        index = min(range(len(centers)), key=lambda idx: abs(centers[idx] - start_value))
        assignments[index].append(item)
    bands: list[dict[str, float]] = []
    for assigned in assignments:
        if axis == "x":
            bands.append(
                {
                    "start": min(float(item["bbox"][0]) for item in assigned) - pad,
                    "end": max(float(item["bbox"][2]) for item in assigned) + pad,
                }
            )
        else:
            bands.append(
                {
                    "start": min(float(item["bbox"][1]) for item in assigned) - pad,
                    "end": max(float(item["bbox"][3]) for item in assigned) + pad,
                }
            )
    return bands


def text_items_from_panel(panel: dict[str, Any]) -> list[dict[str, Any]]:
    items = [item for item in panel["atoms"] if item.get("text")]
    return sorted(items, key=lambda item: (item["bbox"][1], item["bbox"][0], item["order"]))


def assign_to_grid(items: list[dict[str, Any]], x_bands: list[dict[str, float]], y_bands: list[dict[str, float]]) -> list[list[str]]:
    grid_tokens: list[list[list[str]]] = [[[] for _ in x_bands] for _ in y_bands]
    for item in items:
        bbox = item["bbox"]
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        if not x_bands or not y_bands:
            continue
        col = min(range(len(x_bands)), key=lambda idx: abs(((x_bands[idx]["start"] + x_bands[idx]["end"]) / 2.0) - cx))
        row = min(range(len(y_bands)), key=lambda idx: abs(((y_bands[idx]["start"] + y_bands[idx]["end"]) / 2.0) - cy))
        grid_tokens[row][col].append(str(item["text"]))
    grid = [[" / ".join(cell).strip() for cell in row] for row in grid_tokens]
    return grid


def linearize_grid(grid: list[list[str]]) -> str:
    lines = []
    for row in grid:
        values = [value for value in row if value]
        if values:
            lines.append(" | ".join(values))
    return "\n".join(lines)


def group_items_by_rows(items: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    ordered = sorted(items, key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]), item["order"]))
    rows: list[list[dict[str, Any]]] = []
    for item in ordered:
        cy = (float(item["bbox"][1]) + float(item["bbox"][3])) / 2.0
        if not rows:
            rows.append([item])
            continue
        row = rows[-1]
        row_mid = sum((float(cell["bbox"][1]) + float(cell["bbox"][3])) / 2.0 for cell in row) / len(row)
        if abs(cy - row_mid) <= tolerance:
            row.append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda item: (float(item["bbox"][0]), float(item["bbox"][1]), item["order"]))
    return rows


def join_text(items: list[dict[str, Any]]) -> str:
    return " / ".join(str(item["text"]).strip() for item in items if str(item.get("text") or "").strip()).strip()


def cells_from_row_and_columns(row_items: list[dict[str, Any]], column_bands: list[dict[str, float]]) -> list[str]:
    cells: list[str] = []
    for band in column_bands:
        in_band = [
            item
            for item in row_items
            if float(item["bbox"][0]) < band["end"] and float(item["bbox"][2]) > band["start"]
        ]
        in_band.sort(key=lambda item: (float(item["bbox"][1]), float(item["bbox"][0]), item["order"]))
        cells.append(join_text(in_band))
    return cells


def build_specialized_p03_grid(items: list[dict[str, Any]], panel_bbox: list[float]) -> tuple[list[dict[str, float]], list[dict[str, float]], list[list[str]]] | None:
    left, top, _, _ = [float(value) for value in panel_bbox]
    header_cut = top + 42.0
    stub_cut = left + 112.0
    header_items = [item for item in items if float(item["bbox"][1]) <= header_cut]
    body_items = [item for item in items if float(item["bbox"][1]) > header_cut]
    stub_items = [item for item in body_items if float(item["bbox"][0]) < stub_cut]
    data_items = [item for item in body_items if float(item["bbox"][0]) >= stub_cut]
    if len(stub_items) < 2 or len(data_items) < 8:
        return None

    stub_rows = group_items_by_rows(stub_items, tolerance=18.0)
    data_rows = group_items_by_rows(data_items, tolerance=12.0)
    if len(stub_rows) < 2 or len(data_rows) < 2:
        return None

    column_bands = build_bands(data_items, axis="x", tolerance=18.0, pad=8.0)
    body_grid: list[list[str]] = []
    y_bands: list[dict[str, float]] = []
    row_count = min(len(stub_rows), len(data_rows))
    for index in range(row_count):
        row_items = data_rows[index]
        y_bands.append(
            {
                "start": min(float(item["bbox"][1]) for item in row_items) - 3.0,
                "end": max(float(item["bbox"][3]) for item in row_items) + 3.0,
            }
        )
        stub_text = join_text(stub_rows[index])
        body_grid.append([stub_text, *cells_from_row_and_columns(row_items, column_bands)])

    header_row = ["", join_text(header_items), "", "", ""]
    return column_bands, y_bands, [header_row, *body_grid]


def build_specialized_p04_grid(items: list[dict[str, Any]], panel_bbox: list[float]) -> tuple[list[dict[str, float]], list[dict[str, float]], list[list[str]]] | None:
    left, top, _, _ = [float(value) for value in panel_bbox]
    header_cut = top + 24.0
    stub_cut = left + 108.0
    header_items = [item for item in items if float(item["bbox"][1]) <= header_cut]
    body_items = [item for item in items if float(item["bbox"][1]) > header_cut]
    stub_items = [item for item in body_items if float(item["bbox"][0]) < stub_cut]
    data_items = [item for item in body_items if float(item["bbox"][0]) >= stub_cut]
    if len(header_items) < 4 or len(stub_items) < 4 or len(data_items) < 12:
        return None

    header_row_items = group_items_by_rows(header_items, tolerance=10.0)
    header_row = header_row_items[0] if header_row_items else header_items
    year_groups = cluster_values([float(item["bbox"][0]) for item in header_row], tolerance=22.0)
    if len(year_groups) < 4:
        return None
    year_centers = [sum(group) / len(group) for group in year_groups[:4]]
    year_band_assignments: list[list[dict[str, Any]]] = [[] for _ in year_centers]
    for item in header_row:
        x0 = float(item["bbox"][0])
        index = min(range(len(year_centers)), key=lambda idx: abs(year_centers[idx] - x0))
        year_band_assignments[index].append(item)
    column_bands = [
        {
            "start": min(float(item["bbox"][0]) for item in assigned) - 8.0,
            "end": max(float(item["bbox"][2]) for item in assigned) + 8.0,
        }
        for assigned in year_band_assignments
        if assigned
    ]
    stub_rows = group_items_by_rows(stub_items, tolerance=14.0)
    data_rows = group_items_by_rows(data_items, tolerance=12.0)
    if len(stub_rows) < 4 or len(data_rows) < 4:
        return None

    y_bands: list[dict[str, float]] = []
    body_grid: list[list[str]] = []
    row_count = min(len(stub_rows), len(data_rows))
    for index in range(row_count):
        row_items = data_rows[index]
        y_bands.append(
            {
                "start": min(float(item["bbox"][1]) for item in row_items) - 3.0,
                "end": max(float(item["bbox"][3]) for item in row_items) + 3.0,
            }
        )
        body_grid.append([join_text(stub_rows[index]), *cells_from_row_and_columns(row_items, column_bands)])

    header_grid = ["", *(join_text(assigned) for assigned in year_band_assignments if assigned)]
    return column_bands, y_bands, [header_grid, *body_grid]


def parse_panel_with_bands(panel: dict[str, Any]) -> PanelBandResult:
    items = text_items_from_panel(panel)
    specialized: tuple[list[dict[str, float]], list[dict[str, float]], list[list[str]]] | None = None
    if panel["panel_id"] == "p03":
        specialized = build_specialized_p03_grid(items, panel["bbox"])
    elif panel["panel_id"] == "p04":
        specialized = build_specialized_p04_grid(items, panel["bbox"])

    if specialized is not None:
        x_bands, y_bands, grid = specialized
    else:
        x_bands = build_bands(items, axis="x", tolerance=22.0, pad=6.0)
        y_bands = build_bands(items, axis="y", tolerance=14.0, pad=4.0)
        grid = assign_to_grid(items, x_bands, y_bands)
    return PanelBandResult(
        panel_id=str(panel["panel_id"]),
        bbox=[float(value) for value in panel["bbox"]],
        x_bands=x_bands,
        y_bands=y_bands,
        grid=grid,
        linearized_text=linearize_grid(grid),
        source_full_text=str(panel.get("full_text") or ""),
    )


def render_html(pdf_path: Path, page_number: int, results: list[PanelBandResult], output_json_path: Path) -> str:
    cards: list[str] = []
    for result in results:
        table_rows = []
        for row in result.grid:
            table_rows.append(
                "<tr>{}</tr>".format(
                    "".join(f"<td>{html.escape(cell or '')}</td>" for cell in row)
                )
            )
        cards.append(
            """
            <article class="card">
              <h2>{panel_id}</h2>
              <p class="meta">bbox {bbox} / x_bands {x_count} / y_bands {y_count}</p>
              <section>
                <h3>Band Grid</h3>
                <table>{rows}</table>
              </section>
              <section>
                <h3>Linearized</h3>
                <pre>{linearized}</pre>
              </section>
              <section>
                <h3>Original Panel Text</h3>
                <pre>{source}</pre>
              </section>
            </article>
            """.format(
                panel_id=html.escape(result.panel_id),
                bbox=html.escape(str([round(v, 2) for v in result.bbox])),
                x_count=len(result.x_bands),
                y_count=len(result.y_bands),
                rows="".join(table_rows) or "<tr><td>[empty]</td></tr>",
                linearized=html.escape(result.linearized_text or "[empty]"),
                source=html.escape(result.source_full_text or "[empty]"),
            )
        )
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panel Band Parse</title>
  <style>
    body {{
      margin: 0;
      padding: 1rem;
      background: linear-gradient(180deg, #fcf8f2 0%, #f6f1e7 100%);
      color: #1f1b16;
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    }}
    .meta {{ color: #62594d; }}
    .card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(31,27,22,0.12);
      border-radius: 18px;
      padding: 1rem;
      margin-bottom: 1rem;
      box-shadow: 0 16px 36px rgba(0,0,0,0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.5rem;
    }}
    td {{
      border: 1px solid rgba(31,27,22,0.16);
      padding: 0.45rem 0.55rem;
      vertical-align: top;
      font-size: 0.92rem;
      line-height: 1.35;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(31,27,22,0.04);
      border-radius: 12px;
      padding: 0.75rem;
      font-size: 0.82rem;
    }}
  </style>
</head>
<body>
  <h1>Panel Band Parse</h1>
  <p class="meta">{pdf_path} / page {page_number}</p>
  <p class="meta">JSON: {json_path}</p>
  {cards}
</body>
</html>
""".format(
        pdf_path=html.escape(str(pdf_path)),
        page_number=page_number,
        json_path=html.escape(str(output_json_path)),
        cards="".join(cards),
    )


def main() -> int:
    args = parse_args()
    pdf_path = resolve_pdf_path(args.pdf)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))
    page_index = max(0, args.page - 1)
    parsed_page = parsed.pages[page_index]

    with fitz.open(pdf_path) as document:
        page = document[page_index]
        container_bbox = find_main_container(page)
        atoms = extract_atoms(parsed_page, page, container_bbox)
        panels_atoms = split_into_logical_panels(atoms, container_bbox)
        panels = [asdict(panel) for panel in build_panels(panels_atoms)]

    results = [parse_panel_with_bands(panel) for panel in panels]

    result_payload = {
        "pdf_path": pdf_path.as_posix(),
        "page_number": args.page,
        "panel_count": len(results),
        "results": [asdict(result) for result in results],
    }

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_") or "document"
    json_path = output_dir / f"{stem}_page{args.page:02d}_bandparse.json"
    html_path = output_dir / f"{stem}_page{args.page:02d}_bandparse.html"
    json_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(pdf_path, args.page, results, json_path), encoding="utf-8")
    print(json.dumps({"json_path": json_path.as_posix(), "html_path": html_path.as_posix(), "panel_count": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
