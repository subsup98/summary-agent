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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drawing_box_probe"


@dataclass
class TextToken:
    element_type: str
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
class DrawingBox:
    box_id: str
    bbox: list[float]
    drawing_type: str
    fill: list[float] | None
    fill_opacity: float | None
    stroke: list[float] | None
    stroke_width: float | None
    area: float
    assigned_texts: list[dict[str, Any]]
    full_text: str


def normalize_text(text: str) -> str:
    cleaned = str(text or "").replace("\ufeff", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


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
    parser = argparse.ArgumentParser(description="Visualize drawing boxes and assigned text tokens on a PDF page.")
    parser.add_argument("--pdf", help="Optional PDF path.")
    parser.add_argument("--page", type=int, default=16, help="1-based page number to inspect.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/HTML outputs.")
    return parser.parse_args()


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def point_in_bbox(x: float, y: float, bbox: list[float], *, margin: float = 0.0) -> bool:
    return bbox[0] - margin <= x <= bbox[2] + margin and bbox[1] - margin <= y <= bbox[3] + margin


def contains_bbox(outer: list[float], inner: list[float], *, margin: float = 0.0) -> bool:
    return (
        inner[0] >= outer[0] - margin
        and inner[1] >= outer[1] - margin
        and inner[2] <= outer[2] + margin
        and inner[3] <= outer[3] + margin
    )


def extract_candidate_boxes(page: fitz.Page) -> list[dict[str, Any]]:
    page_area = float(page.rect.width * page.rect.height)
    candidates: list[dict[str, Any]] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        area = bbox_area(bbox)
        if area <= 0:
            continue
        if area > page_area * 0.82:
            continue
        if area < page_area * 0.0012:
            continue
        fill = drawing.get("fill")
        fill_opacity = drawing.get("fill_opacity")
        stroke = drawing.get("color")
        stroke_width = drawing.get("width")
        drawing_type = str(drawing.get("type") or "")
        has_fill = fill is not None and (fill_opacity is None or float(fill_opacity) > 0.01)
        has_stroke = stroke is not None and (stroke_width or 0) > 0
        if not has_fill and not has_stroke:
            continue
        candidates.append(
            {
                "bbox": bbox,
                "drawing_type": drawing_type,
                "fill": [float(value) for value in fill] if fill is not None else None,
                "fill_opacity": float(fill_opacity) if fill_opacity is not None else None,
                "stroke": [float(value) for value in stroke] if stroke is not None else None,
                "stroke_width": float(stroke_width) if stroke_width is not None else None,
                "area": area,
            }
        )
    return sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0], item["area"]))


def remove_redundant_boxes(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        bbox = candidate["bbox"]
        redundant = False
        for existing in kept:
            existing_bbox = existing["bbox"]
            same_fill = existing.get("fill") == candidate.get("fill")
            if same_fill and contains_bbox(existing_bbox, bbox, margin=1.5):
                redundant = True
                break
        if not redundant:
            kept.append(candidate)
    return kept


def extract_text_tokens(parsed_page: Any) -> list[TextToken]:
    tokens: list[TextToken] = []
    for element in parsed_page.elements:
        bbox = list(element.bbox or [])
        if len(bbox) != 4:
            continue
        text = normalize_text(element.text or "")
        if not text:
            continue
        tokens.append(
            TextToken(
                element_type=str(element.element_type),
                order=int(element.order),
                text=text,
                bbox=[float(value) for value in bbox],
            )
        )
    return sorted(tokens, key=lambda item: (item.bbox[1], item.bbox[0], item.order))


def assign_texts_to_boxes(boxes: list[dict[str, Any]], tokens: list[TextToken]) -> list[DrawingBox]:
    results: list[DrawingBox] = []
    for index, box in enumerate(boxes, start=1):
        bbox = box["bbox"]
        assigned = [
            token
            for token in tokens
            if point_in_bbox(token.cx, token.cy, bbox, margin=1.5)
        ]
        assigned.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.order))
        full_text = " / ".join(token.text for token in assigned)
        results.append(
            DrawingBox(
                box_id=f"b{index:02d}",
                bbox=[round(value, 2) for value in bbox],
                drawing_type=str(box.get("drawing_type") or ""),
                fill=box.get("fill"),
                fill_opacity=box.get("fill_opacity"),
                stroke=box.get("stroke"),
                stroke_width=box.get("stroke_width"),
                area=round(float(box.get("area") or 0.0), 2),
                assigned_texts=[
                    {
                        "element_type": token.element_type,
                        "order": token.order,
                        "bbox": [round(value, 2) for value in token.bbox],
                        "text": token.text,
                    }
                    for token in assigned
                ],
                full_text=full_text,
            )
        )
    return results


def render_html(
    *,
    pdf_path: Path,
    page_number: int,
    page_width: float,
    page_height: float,
    boxes: list[DrawingBox],
    output_json_path: Path,
) -> str:
    overlays: list[str] = []
    cards: list[str] = []
    for box in boxes:
        left = box.bbox[0] / max(page_width, 1.0) * 100.0
        top = box.bbox[1] / max(page_height, 1.0) * 100.0
        width = max(0.4, (box.bbox[2] - box.bbox[0]) / max(page_width, 1.0) * 100.0)
        height = max(0.4, (box.bbox[3] - box.bbox[1]) / max(page_height, 1.0) * 100.0)
        fill = box.fill or [0.45, 0.45, 0.45]
        color = "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, int(fill[0] * 255))),
            max(0, min(255, int(fill[1] * 255))),
            max(0, min(255, int(fill[2] * 255))),
        )
        payload = html.escape(json.dumps(asdict(box), ensure_ascii=False))
        overlays.append(
            '<button class="overlay" data-box="{payload}" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;--box-color:{color}">{label}</button>'.format(
                payload=payload,
                left=left,
                top=top,
                width=width,
                height=height,
                color=color,
                label=html.escape(box.box_id),
            )
        )
        cards.append(
            """
            <article class="card">
              <div class="card-head">
                <strong>{box_id}</strong>
                <span class="meta">texts {text_count}</span>
                <span class="meta">area {area}</span>
              </div>
              <p class="meta">bbox {bbox}</p>
              <p class="full-text">{full_text}</p>
            </article>
            """.format(
                box_id=html.escape(box.box_id),
                text_count=len(box.assigned_texts),
                area=html.escape(str(box.area)),
                bbox=html.escape(str(box.bbox)),
                full_text=html.escape(box.full_text or "[no assigned text]"),
            )
        )

    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drawing Box Probe</title>
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
    .overlay {{
      position: absolute;
      border: 2px solid var(--box-color);
      background: color-mix(in srgb, var(--box-color) 16%, transparent);
      color: #111;
      font-size: 0.72rem;
      padding: 0.2rem 0.35rem;
      overflow: hidden;
      cursor: pointer;
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
      margin-bottom: 0.4rem;
    }}
    .meta {{
      color: #62594d;
      font-size: 0.82rem;
    }}
    .full-text {{
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0.4rem 0 0;
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
      <h1>Drawing Box Probe</h1>
      <p>{pdf_path} / page {page_number}</p>
      <p>JSON: {json_path}</p>
      <div class="stage">
        {overlays}
      </div>
    </section>
    <aside class="panel sidebar">
      <h2>Boxes</h2>
      {cards}
      <h3>Selected Box</h3>
      <pre id="detail">박스를 클릭하면 상세 JSON이 여기에 표시됩니다.</pre>
    </aside>
  </div>
  <script>
    const detail = document.getElementById('detail');
    for (const button of document.querySelectorAll('.overlay')) {{
      button.addEventListener('click', () => {{
        detail.textContent = JSON.stringify(JSON.parse(button.dataset.box), null, 2);
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
    if page_index >= len(parsed.pages):
        raise ValueError(f"page out of range: {args.page}")
    parsed_page = parsed.pages[page_index]

    with fitz.open(pdf_path) as document:
        page = document[page_index]
        candidate_boxes = remove_redundant_boxes(extract_candidate_boxes(page))
        tokens = extract_text_tokens(parsed_page)
        boxes = assign_texts_to_boxes(candidate_boxes, tokens)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

    result = {
        "pdf_path": pdf_path.as_posix(),
        "page_number": args.page,
        "box_count": len(boxes),
        "boxes": [asdict(box) for box in boxes],
    }

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_") or "document"
    json_path = output_dir / f"{stem}_page{args.page:02d}_boxes.json"
    html_path = output_dir / f"{stem}_page{args.page:02d}_boxes.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(
        render_html(
            pdf_path=pdf_path,
            page_number=args.page,
            page_width=page_width,
            page_height=page_height,
            boxes=boxes,
            output_json_path=json_path,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"json_path": json_path.as_posix(), "html_path": html_path.as_posix(), "box_count": len(boxes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
