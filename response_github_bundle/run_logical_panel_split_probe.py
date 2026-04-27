from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz

from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logical_panel_split_probe"


@dataclass
class Atom:
    kind: str
    order: int
    text: str
    bbox: list[float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass
class Panel:
    panel_id: str
    bbox: list[float]
    atom_count: int
    reading_mode: str
    full_text: str
    atoms: list[dict[str, Any]]


def normalize_text(text: str) -> str:
    cleaned = str(text or "").replace("\ufeff", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def bbox_union(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def point_in_bbox(x: float, y: float, bbox: list[float], *, margin: float = 0.0) -> bool:
    return bbox[0] - margin <= x <= bbox[2] + margin and bbox[1] - margin <= y <= bbox[3] + margin


def cluster_positions(values: list[float], tolerance: float) -> list[list[float]]:
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


def resolve_pdf_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        return path
    candidates = sorted(Path.home().joinpath("Desktop", "all_docs").glob("*3*pdf"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No default PDF found. Pass --pdf explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a large visual container into logical panels using whitespace.")
    parser.add_argument("--pdf", help="Optional PDF path.")
    parser.add_argument("--page", type=int, default=16, help="1-based page number to inspect.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/HTML outputs.")
    return parser.parse_args()


def find_main_container(page: fitz.Page) -> list[float]:
    page_area = float(page.rect.width * page.rect.height)
    candidates: list[list[float]] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        fill = drawing.get("fill")
        if rect is None or fill is None:
            continue
        bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        area = bbox_area(bbox)
        if page_area * 0.18 <= area <= page_area * 0.8:
            candidates.append(bbox)
    if not candidates:
        return [0.0, 0.0, float(page.rect.width), float(page.rect.height)]
    return max(candidates, key=bbox_area)


def extract_atoms(parsed_page: Any, page: fitz.Page, container_bbox: list[float]) -> list[Atom]:
    atoms: list[Atom] = []
    order = 1
    for element in parsed_page.elements:
        bbox = list(element.bbox or [])
        if len(bbox) != 4:
            continue
        text = normalize_text(element.text or "")
        if not text:
            continue
        if not point_in_bbox((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0, container_bbox, margin=2.0):
            continue
        atoms.append(Atom(kind=str(element.element_type), order=order, text=text, bbox=[float(v) for v in bbox]))
        order += 1

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        fill = drawing.get("fill")
        if rect is None or fill is None:
            continue
        bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        if not point_in_bbox((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0, container_bbox, margin=2.0):
            continue
        area = bbox_area(bbox)
        if area < 600 or area > bbox_area(container_bbox) * 0.4:
            continue
        atoms.append(Atom(kind="drawing_box", order=order, text="", bbox=bbox))
        order += 1

    return sorted(atoms, key=lambda item: (item.bbox[1], item.bbox[0], item.order))


def gap_split(atoms: list[Atom], container_bbox: list[float], *, axis: str, min_ratio: float) -> tuple[list[Atom], list[Atom], float] | None:
    text_atoms = [atom for atom in atoms if atom.text]
    if len(text_atoms) < 8:
        return None
    ordered = sorted(text_atoms, key=lambda item: item.cx if axis == "x" else item.cy)
    span = max(1.0, (container_bbox[2] - container_bbox[0]) if axis == "x" else (container_bbox[3] - container_bbox[1]))
    best_gap = 0.0
    best_threshold = None
    for index in range(1, len(ordered)):
        prev = ordered[index - 1]
        curr = ordered[index]
        if axis == "x":
            gap = curr.bbox[0] - prev.bbox[2]
            threshold = (prev.bbox[2] + curr.bbox[0]) / 2.0
            left = [item for item in atoms if item.cx <= threshold]
            right = [item for item in atoms if item.cx > threshold]
        else:
            gap = curr.bbox[1] - prev.bbox[3]
            threshold = (prev.bbox[3] + curr.bbox[1]) / 2.0
            left = [item for item in atoms if item.cy <= threshold]
            right = [item for item in atoms if item.cy > threshold]
        if gap / span >= min_ratio and len(left) >= 4 and len(right) >= 4 and gap > best_gap:
            best_gap = gap
            best_threshold = threshold
            best_pair = (left, right)
    if best_threshold is None:
        return None
    return best_pair[0], best_pair[1], best_threshold


def infer_reading_mode(atoms: list[Atom]) -> str:
    text_atoms = [atom for atom in atoms if atom.text]
    row_clusters = cluster_positions([atom.cy for atom in text_atoms], tolerance=14.0)
    col_clusters = cluster_positions([atom.cx for atom in text_atoms], tolerance=24.0)
    if len(col_clusters) >= len(row_clusters) + 2:
        return "column_major"
    return "row_major"


def serialize_atoms(atoms: list[Atom], reading_mode: str) -> str:
    text_atoms = [atom for atom in atoms if atom.text]
    if not text_atoms:
        return ""
    if reading_mode == "column_major":
        centers = [sum(group) / len(group) for group in cluster_positions([atom.cx for atom in text_atoms], tolerance=24.0)]
        cols: dict[int, list[Atom]] = {}
        for atom in text_atoms:
            idx = min(range(len(centers)), key=lambda i: abs(centers[i] - atom.cx))
            cols.setdefault(idx, []).append(atom)
        parts: list[str] = []
        for idx in sorted(cols):
            for atom in sorted(cols[idx], key=lambda item: (item.bbox[1], item.bbox[0], item.order)):
                parts.append(atom.text)
        return " / ".join(parts)
    centers = [sum(group) / len(group) for group in cluster_positions([atom.cy for atom in text_atoms], tolerance=14.0)]
    rows: dict[int, list[Atom]] = {}
    for atom in text_atoms:
        idx = min(range(len(centers)), key=lambda i: abs(centers[i] - atom.cy))
        rows.setdefault(idx, []).append(atom)
    parts: list[str] = []
    for idx in sorted(rows):
        for atom in sorted(rows[idx], key=lambda item: (item.bbox[0], item.bbox[1], item.order)):
            parts.append(atom.text)
    return " / ".join(parts)


def split_into_logical_panels(atoms: list[Atom], container_bbox: list[float]) -> list[list[Atom]]:
    horizontal = gap_split(atoms, container_bbox, axis="y", min_ratio=0.05)
    if horizontal is None:
        return [atoms]
    top, bottom, _ = horizontal
    top_bbox = bbox_union([atom.bbox for atom in top])
    bottom_bbox = bbox_union([atom.bbox for atom in bottom])
    top_vertical = gap_split(top, top_bbox, axis="x", min_ratio=0.05)
    bottom_vertical = gap_split(bottom, bottom_bbox, axis="x", min_ratio=0.05)
    panels: list[list[Atom]] = []
    if top_vertical is not None:
        panels.extend([top_vertical[0], top_vertical[1]])
    else:
        panels.append(top)
    if bottom_vertical is not None:
        panels.extend([bottom_vertical[0], bottom_vertical[1]])
    else:
        panels.append(bottom)
    return [sorted(panel, key=lambda atom: (atom.bbox[1], atom.bbox[0], atom.order)) for panel in panels if len(panel) >= 4]


def build_panels(panels_atoms: list[list[Atom]]) -> list[Panel]:
    panels: list[Panel] = []
    for index, atoms in enumerate(sorted(panels_atoms, key=lambda group: (bbox_union([atom.bbox for atom in group])[1], bbox_union([atom.bbox for atom in group])[0])), start=1):
        bbox = bbox_union([atom.bbox for atom in atoms])
        reading_mode = infer_reading_mode(atoms)
        panels.append(
            Panel(
                panel_id=f"p{index:02d}",
                bbox=[round(v, 2) for v in bbox],
                atom_count=len(atoms),
                reading_mode=reading_mode,
                full_text=serialize_atoms(atoms, reading_mode),
                atoms=[
                    {
                        "kind": atom.kind,
                        "order": atom.order,
                        "bbox": [round(v, 2) for v in atom.bbox],
                        "text": atom.text,
                    }
                    for atom in atoms
                ],
            )
        )
    return panels


def render_html(pdf_path: Path, page_number: int, page_width: float, page_height: float, container_bbox: list[float], panels: list[Panel], output_json_path: Path) -> str:
    overlays: list[str] = []
    colors = ["#c74e2a", "#1b805f", "#265d8b", "#7b4aaa", "#70604e"]
    cards: list[str] = []
    container_left = container_bbox[0] / max(page_width, 1.0) * 100.0
    container_top = container_bbox[1] / max(page_height, 1.0) * 100.0
    container_width = (container_bbox[2] - container_bbox[0]) / max(page_width, 1.0) * 100.0
    container_height = (container_bbox[3] - container_bbox[1]) / max(page_height, 1.0) * 100.0
    overlays.append(
        '<div class="container-box" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%">container</div>'.format(
            left=container_left,
            top=container_top,
            width=container_width,
            height=container_height,
        )
    )
    for index, panel in enumerate(panels):
        color = colors[index % len(colors)]
        left = panel.bbox[0] / max(page_width, 1.0) * 100.0
        top = panel.bbox[1] / max(page_height, 1.0) * 100.0
        width = (panel.bbox[2] - panel.bbox[0]) / max(page_width, 1.0) * 100.0
        height = (panel.bbox[3] - panel.bbox[1]) / max(page_height, 1.0) * 100.0
        payload = html.escape(json.dumps(asdict(panel), ensure_ascii=False))
        overlays.append(
            '<button class="panel-box" data-panel="{payload}" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;--panel-color:{color}">{label}</button>'.format(
                payload=payload,
                left=left,
                top=top,
                width=width,
                height=height,
                color=color,
                label=html.escape(panel.panel_id),
            )
        )
        cards.append(
            """
            <article class="card">
              <div class="card-head">
                <strong>{panel_id}</strong>
                <span class="meta">atoms {atom_count}</span>
                <span class="meta">read {reading_mode}</span>
              </div>
              <p class="meta">bbox {bbox}</p>
              <p class="full-text">{full_text}</p>
            </article>
            """.format(
                panel_id=html.escape(panel.panel_id),
                atom_count=panel.atom_count,
                reading_mode=html.escape(panel.reading_mode),
                bbox=html.escape(str(panel.bbox)),
                full_text=html.escape(panel.full_text or "[no text]"),
            )
        )
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Logical Panel Split Probe</title>
  <style>
    :root {{
      --paper: #f6f1e7;
      --ink: #1f1b16;
      --line: rgba(31, 27, 22, 0.12);
      --panel: rgba(255, 251, 245, 0.94);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #fcf8f2 0%, var(--paper) 100%);
      color: var(--ink);
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(24rem, 1fr);
      gap: 1rem;
      padding: 1rem;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 1rem;
      box-shadow: 0 18px 44px rgba(0, 0, 0, 0.08);
    }}
    .stage {{
      position: relative;
      width: 100%;
      aspect-ratio: {page_width:.4f} / {page_height:.4f};
      background: white;
      border-radius: 16px;
      overflow: hidden;
      border: 1px dashed rgba(0, 0, 0, 0.16);
    }}
    .container-box {{
      position: absolute;
      border: 2px dashed #8a8177;
      background: rgba(138, 129, 119, 0.08);
      color: #5b5248;
      font-size: 0.72rem;
      padding: 0.2rem 0.35rem;
    }}
    .panel-box {{
      position: absolute;
      border: 2px solid var(--panel-color);
      background: color-mix(in srgb, var(--panel-color) 14%, transparent);
      color: #111;
      font-size: 0.74rem;
      padding: 0.18rem 0.35rem;
      cursor: pointer;
      overflow: hidden;
    }}
    .sidebar {{
      max-height: calc(100vh - 2rem);
      overflow: auto;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 0.85rem;
      margin-bottom: 0.75rem;
      background: rgba(255,255,255,0.86);
    }}
    .card-head {{
      display: flex;
      gap: 0.5rem;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 0.35rem;
    }}
    .meta {{
      color: #62594d;
      font-size: 0.82rem;
    }}
    .full-text {{
      margin: 0.45rem 0 0;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(31, 27, 22, 0.04);
      border-radius: 12px;
      padding: 0.75rem;
      font-size: 0.78rem;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <h1>Logical Panel Split Probe</h1>
      <p>{pdf_path} / page {page_number}</p>
      <p>JSON: {json_path}</p>
      <div class="stage">{overlays}</div>
    </section>
    <aside class="panel sidebar">
      <h2>Panels</h2>
      {cards}
      <h3>Selected Panel</h3>
      <pre id="detail">패널을 클릭하면 상세 JSON이 여기에 표시됩니다.</pre>
    </aside>
  </div>
  <script>
    const detail = document.getElementById('detail');
    for (const button of document.querySelectorAll('.panel-box')) {{
      button.addEventListener('click', () => {{
        detail.textContent = JSON.stringify(JSON.parse(button.dataset.panel), null, 2);
      }});
    }}
  </script>
</body>
</html>
""".format(
        page_width=page_width,
        page_height=page_height,
        pdf_path=html.escape(str(pdf_path)),
        page_number=page_number,
        json_path=html.escape(str(output_json_path)),
        overlays="\n".join(overlays),
        cards="\n".join(cards),
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
        panels = build_panels(panels_atoms)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

    result = {
        "pdf_path": pdf_path.as_posix(),
        "page_number": args.page,
        "container_bbox": [round(v, 2) for v in container_bbox],
        "panel_count": len(panels),
        "panels": [asdict(panel) for panel in panels],
    }

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_") or "document"
    json_path = output_dir / f"{stem}_page{args.page:02d}_panels.json"
    html_path = output_dir / f"{stem}_page{args.page:02d}_panels.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(pdf_path, args.page, page_width, page_height, container_bbox, panels, json_path), encoding="utf-8")
    print(json.dumps({"json_path": json_path.as_posix(), "html_path": html_path.as_posix(), "panel_count": len(panels)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
