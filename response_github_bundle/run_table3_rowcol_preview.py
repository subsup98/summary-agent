from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "outputs" / "manual_raw_parse" / "miraeasset_q3_q4" / "source"
OUTPUT_HTML = PROJECT_ROOT / "outputs" / "hybrid_struct_bbox_tables" / "table3_rowcol_preview.html"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "hybrid_struct_bbox_tables" / "table3_rowcol_preview.json"


@dataclass
class BoxToken:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def resolve_pdf_path() -> Path:
    candidates = sorted(DEFAULT_SOURCE_DIR.glob("*3*.pdf"))
    if candidates:
        return candidates[0]
    candidates = sorted(DEFAULT_SOURCE_DIR.glob("*.pdf"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No sample PDF found in {DEFAULT_SOURCE_DIR}")


def clean_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace("( ", "(").replace(" )", ")")
    value = value.replace(" %", "%")
    return value


def cluster_edges(values: list[float], tolerance: float) -> list[list[float]]:
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


def group_tokens_by_rows(tokens: list[BoxToken], tolerance: float = 14.0) -> list[list[BoxToken]]:
    ordered = sorted(tokens, key=lambda token: (token.y0, token.x0))
    rows: list[list[BoxToken]] = []
    for token in ordered:
        if not rows:
            rows.append([token])
            continue
        row = rows[-1]
        row_mid = sum(item.cy for item in row) / len(row)
        if abs(token.cy - row_mid) <= tolerance:
            row.append(token)
        else:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda token: token.x0)
    return rows


def split_table_regions(tokens: list[BoxToken]) -> tuple[list[BoxToken], list[BoxToken], list[BoxToken]]:
    header = [token for token in tokens if token.y0 < 360]
    stub = [token for token in tokens if token.x0 < 445 and token.y0 >= 360]
    data = [token for token in tokens if token.x0 >= 445 and token.y0 >= 360]
    return header, stub, data


def build_column_bands(data_rows: list[list[BoxToken]]) -> list[dict[str, float]]:
    tokens = [token for row in data_rows for token in row]
    start_groups = cluster_edges([token.x0 for token in tokens], tolerance=20.0)
    centers = [sum(group) / len(group) for group in start_groups]
    if len(centers) < 4:
        raise RuntimeError(f"Could not derive 4 x-start clusters: {centers}")

    chosen_centers = centers[:4]
    assignments: list[list[BoxToken]] = [[] for _ in chosen_centers]
    for token in tokens:
        index = min(range(len(chosen_centers)), key=lambda idx: abs(token.x0 - chosen_centers[idx]))
        assignments[index].append(token)

    bands: list[dict[str, float]] = []
    for assigned in assignments:
        bands.append(
            {
                "start": min(token.x0 for token in assigned) - 8,
                "end": max(token.x1 for token in assigned) + 8,
            }
        )
    return bands


def build_row_bands(stub_rows: list[list[BoxToken]], data_rows: list[list[BoxToken]]) -> list[dict[str, float]]:
    if len(data_rows) >= 4:
        band_rows = [data_rows[0], data_rows[1], data_rows[2] + data_rows[3]]
    else:
        band_rows = data_rows[:3]
    stub_labels = [clean_text(" ".join(token.text for token in stub_rows[0])), clean_text(" ".join(token.text for token in stub_rows[1]))]

    bands: list[dict[str, float]] = []
    for index, row in enumerate(band_rows):
        y0 = min(token.y0 for token in row)
        y1 = max(token.y1 for token in row)
        label = stub_labels[0] if index < 2 else stub_labels[1]
        bands.append({"start": y0 - 2, "end": y1 + 2, "stub": label})
    return bands


def collect_tokens_in_band(tokens: list[BoxToken], start: float, end: float) -> list[BoxToken]:
    return [token for token in tokens if start <= token.cy <= end]


def row_to_cells(tokens: list[BoxToken], column_bands: list[dict[str, float]]) -> list[str]:
    cells: list[str] = []
    for band in column_bands:
        items = [token for token in tokens if token.x0 < band["end"] and token.x1 > band["start"]]
        items.sort(key=lambda token: (token.y0, token.x0))
        cells.append(clean_text(" ".join(token.text for token in items)))
    return cells


def main() -> None:
    pdf_path = resolve_pdf_path()
    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))
    page = parsed.pages[15]

    tokens = [
        BoxToken(
            text=clean_text(element.text or ""),
            x0=float(element.bbox[0]),
            y0=float(element.bbox[1]),
            x1=float(element.bbox[2]),
            y1=float(element.bbox[3]),
        )
        for element in page.elements
        if element.element_type == "image"
        and (element.text or "").strip()
        and element.bbox
        and element.bbox[0] >= 390
        and 340 <= element.bbox[1] <= 446
    ]
    tokens.sort(key=lambda token: (token.y0, token.x0))

    header_tokens, stub_tokens, data_tokens = split_table_regions(tokens)
    stub_rows = group_tokens_by_rows(stub_tokens, tolerance=18.0)
    data_rows = group_tokens_by_rows(data_tokens, tolerance=12.0)

    column_bands = build_column_bands(data_rows)
    row_bands = build_row_bands(stub_rows, data_rows)

    body_rows: list[list[str]] = []
    for row_band in row_bands:
        row_tokens = collect_tokens_in_band(data_tokens, row_band["start"], row_band["end"])
        cells = [row_band["stub"], *row_to_cells(row_tokens, column_bands)]
        body_rows.append(cells)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Table 3 Row/Column Preview</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: #f6f4ef;
      color: #222;
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    }}
    .wrap {{
      max-width: 980px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      color: #e77b22;
    }}
    p {{
      margin: 0 0 20px;
      color: #666;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.08);
    }}
    th, td {{
      border: 2px solid #f2efe8;
      padding: 16px 14px;
      text-align: left;
      vertical-align: middle;
      line-height: 1.35;
      font-size: 16px;
    }}
    th {{
      background: #8d8d8d;
      color: #fff;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Table 3 Row/Column Preview</h1>
    <p>Page 16 / start-end band based reconstruction</p>
    <table>
      <thead>
        <tr>
          <th colspan="2">목표</th>
          <th colspan="2">이행</th>
          <th>이행률</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{body_rows[0][0]}</td>
          <td>{body_rows[0][1]}</td>
          <td>{body_rows[0][2]}</td>
          <td>{body_rows[0][3]}</td>
          <td>{body_rows[0][4]}</td>
        </tr>
        <tr>
          <td>{body_rows[1][0]}</td>
          <td>{body_rows[1][1]}</td>
          <td>{body_rows[1][2]}</td>
          <td>{body_rows[1][3]}</td>
          <td>{body_rows[1][4]}</td>
        </tr>
        <tr>
          <td>{body_rows[2][0]}</td>
          <td>{body_rows[2][1]}</td>
          <td>{body_rows[2][2]}</td>
          <td>{body_rows[2][3]}</td>
          <td>{body_rows[2][4]}</td>
        </tr>
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "pdf_path": pdf_path.as_posix(),
                "header_tokens": [token.__dict__ for token in header_tokens],
                "stub_rows": [[token.__dict__ for token in row] for row in stub_rows],
                "data_rows": [[token.__dict__ for token in row] for row in data_rows],
                "row_bands": row_bands,
                "column_bands": column_bands,
                "body_rows": body_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"html_path": OUTPUT_HTML.as_posix(), "json_path": OUTPUT_JSON.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
