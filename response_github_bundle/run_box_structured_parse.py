from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz

from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "box_structured_parse"


@dataclass
class Token:
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
class Box:
    box_id: str
    bbox: list[float]
    fill: list[float] | None
    area: float
    texts: list[Token]


@dataclass
class StructuredBlock:
    block_id: str
    bbox: list[float]
    source: str
    box_ids: list[str]
    token_count: int
    reading_mode: str
    full_text: str
    tokens: list[dict[str, Any]]


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


def bboxes_close(left: list[float], right: list[float], *, x_pad: float, y_pad: float) -> bool:
    return not (
        left[2] + x_pad < right[0]
        or right[2] + x_pad < left[0]
        or left[3] + y_pad < right[1]
        or right[3] + y_pad < left[1]
    )


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
    parser = argparse.ArgumentParser(description="Experimental structured parse using drawing boxes plus residual text clusters.")
    parser.add_argument("--pdf", help="Optional PDF path.")
    parser.add_argument("--page", type=int, default=16, help="1-based page number to inspect.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/HTML outputs.")
    return parser.parse_args()


def extract_tokens(parsed_page: Any) -> list[Token]:
    tokens: list[Token] = []
    for element in parsed_page.elements:
        bbox = list(element.bbox or [])
        if len(bbox) != 4:
            continue
        text = normalize_text(element.text or "")
        if not text:
            continue
        tokens.append(
            Token(
                element_type=str(element.element_type),
                order=int(element.order),
                text=text,
                bbox=[float(value) for value in bbox],
            )
        )
    return sorted(tokens, key=lambda item: (item.bbox[1], item.bbox[0], item.order))


def extract_boxes(page: fitz.Page, tokens: list[Token]) -> tuple[list[Box], list[float] | None]:
    page_area = float(page.rect.width * page.rect.height)
    raw: list[dict[str, Any]] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        bbox = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
        area = bbox_area(bbox)
        if area > page_area * 0.82 or area < page_area * 0.0012:
            continue
        fill = drawing.get("fill")
        fill_opacity = drawing.get("fill_opacity")
        if fill is None or (fill_opacity is not None and float(fill_opacity) <= 0.01):
            continue
        raw.append({"bbox": bbox, "fill": [float(v) for v in fill], "area": area})

    raw.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["area"]))
    container_bbox = None
    boxes: list[Box] = []
    for item in raw:
        bbox = item["bbox"]
        area = item["area"]
        if area > page_area * 0.45:
            container_bbox = bbox
            continue
        assigned = [token for token in tokens if point_in_bbox(token.cx, token.cy, bbox, margin=1.5)]
        if not assigned:
            continue
        boxes.append(
            Box(
                box_id=f"b{len(boxes)+1:02d}",
                bbox=[round(v, 2) for v in bbox],
                fill=item["fill"],
                area=round(area, 2),
                texts=assigned,
            )
        )
    return boxes, container_bbox


def connected_components(boxes: list[Box]) -> list[list[Box]]:
    visited: set[int] = set()
    groups: list[list[Box]] = []
    for index, box in enumerate(boxes):
        if index in visited:
            continue
        stack = [index]
        visited.add(index)
        group: list[Box] = []
        while stack:
            current_index = stack.pop()
            current = boxes[current_index]
            group.append(current)
            for candidate_index, candidate in enumerate(boxes):
                if candidate_index in visited:
                    continue
                x_pad = 16.0
                y_pad = 12.0
                if bboxes_close(current.bbox, candidate.bbox, x_pad=x_pad, y_pad=y_pad):
                    visited.add(candidate_index)
                    stack.append(candidate_index)
        groups.append(sorted(group, key=lambda item: (item.bbox[1], item.bbox[0])))
    return sorted(groups, key=lambda group: (bbox_union([box.bbox for box in group])[1], bbox_union([box.bbox for box in group])[0]))


def residual_token_clusters(tokens: list[Token], boxes: list[Box], container_bbox: list[float] | None) -> list[list[Token]]:
    assigned_orders = {token.order for box in boxes for token in box.texts}
    residual = [token for token in tokens if token.order not in assigned_orders]
    if container_bbox is not None:
        residual = [token for token in residual if point_in_bbox(token.cx, token.cy, container_bbox, margin=2.0)]
    residual = [token for token in residual if token.bbox[1] >= 130.0]
    visited: set[int] = set()
    groups: list[list[Token]] = []
    for index, token in enumerate(residual):
        if index in visited:
            continue
        stack = [index]
        visited.add(index)
        group: list[Token] = []
        while stack:
            current_index = stack.pop()
            current = residual[current_index]
            group.append(current)
            for candidate_index, candidate in enumerate(residual):
                if candidate_index in visited:
                    continue
                if bboxes_close(current.bbox, candidate.bbox, x_pad=28.0, y_pad=18.0):
                    visited.add(candidate_index)
                    stack.append(candidate_index)
        if len(group) >= 3:
            groups.append(sorted(group, key=lambda item: (item.bbox[1], item.bbox[0], item.order)))
    return groups


def reading_mode_for_tokens(tokens: list[Token]) -> str:
    row_clusters = cluster_positions([token.cy for token in tokens], tolerance=14.0)
    col_clusters = cluster_positions([token.cx for token in tokens], tolerance=24.0)
    if len(col_clusters) >= len(row_clusters) + 2:
        return "column_major"
    return "row_major"


def serialize_tokens(tokens: list[Token], reading_mode: str) -> str:
    if reading_mode == "column_major":
        col_centers = [sum(group) / len(group) for group in cluster_positions([token.cx for token in tokens], tolerance=24.0)]
        columns: dict[int, list[Token]] = defaultdict(list)
        for token in tokens:
            index = min(range(len(col_centers)), key=lambda idx: abs(col_centers[idx] - token.cx))
            columns[index].append(token)
        parts: list[str] = []
        for index in sorted(columns):
            for token in sorted(columns[index], key=lambda item: (item.bbox[1], item.bbox[0], item.order)):
                parts.append(token.text)
        return " / ".join(parts)

    row_centers = [sum(group) / len(group) for group in cluster_positions([token.cy for token in tokens], tolerance=14.0)]
    rows: dict[int, list[Token]] = defaultdict(list)
    for token in tokens:
        index = min(range(len(row_centers)), key=lambda idx: abs(row_centers[idx] - token.cy))
        rows[index].append(token)
    parts: list[str] = []
    for index in sorted(rows):
        for token in sorted(rows[index], key=lambda item: (item.bbox[0], item.bbox[1], item.order)):
            parts.append(token.text)
    return " / ".join(parts)


def build_structured_blocks(box_groups: list[list[Box]], residual_groups: list[list[Token]]) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []
    index = 1
    for group in box_groups:
        tokens = sorted({token.order: token for box in group for token in box.texts}.values(), key=lambda item: (item.bbox[1], item.bbox[0], item.order))
        reading_mode = reading_mode_for_tokens(tokens)
        blocks.append(
            StructuredBlock(
                block_id=f"blk{index:02d}",
                bbox=[round(v, 2) for v in bbox_union([box.bbox for box in group])],
                source="drawing_boxes",
                box_ids=[box.box_id for box in group],
                token_count=len(tokens),
                reading_mode=reading_mode,
                full_text=serialize_tokens(tokens, reading_mode),
                tokens=[
                    {
                        "order": token.order,
                        "element_type": token.element_type,
                        "bbox": [round(v, 2) for v in token.bbox],
                        "text": token.text,
                    }
                    for token in tokens
                ],
            )
        )
        index += 1

    for group in residual_groups:
        reading_mode = reading_mode_for_tokens(group)
        blocks.append(
            StructuredBlock(
                block_id=f"blk{index:02d}",
                bbox=[round(v, 2) for v in bbox_union([token.bbox for token in group])],
                source="residual_text_cluster",
                box_ids=[],
                token_count=len(group),
                reading_mode=reading_mode,
                full_text=serialize_tokens(group, reading_mode),
                tokens=[
                    {
                        "order": token.order,
                        "element_type": token.element_type,
                        "bbox": [round(v, 2) for v in token.bbox],
                        "text": token.text,
                    }
                    for token in group
                ],
            )
        )
        index += 1

    return sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))


def render_html(pdf_path: Path, page_number: int, page_width: float, page_height: float, blocks: list[StructuredBlock], output_json_path: Path) -> str:
    overlays: list[str] = []
    cards: list[str] = []
    colors = {"drawing_boxes": "#265d8b", "residual_text_cluster": "#c74e2a"}
    for block in blocks:
        left = block.bbox[0] / max(page_width, 1.0) * 100.0
        top = block.bbox[1] / max(page_height, 1.0) * 100.0
        width = max(0.5, (block.bbox[2] - block.bbox[0]) / max(page_width, 1.0) * 100.0)
        height = max(0.5, (block.bbox[3] - block.bbox[1]) / max(page_height, 1.0) * 100.0)
        color = colors.get(block.source, "#70604e")
        payload = html.escape(json.dumps(asdict(block), ensure_ascii=False))
        overlays.append(
            '<button class="overlay" data-block="{payload}" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;--block-color:{color}">{label}</button>'.format(
                payload=payload,
                left=left,
                top=top,
                width=width,
                height=height,
                color=color,
                label=html.escape(block.block_id),
            )
        )
        cards.append(
            """
            <article class="card">
              <div class="card-head">
                <strong>{block_id}</strong>
                <span class="pill" style="background:{color}">{source}</span>
                <span class="meta">tokens {token_count}</span>
                <span class="meta">read {reading_mode}</span>
              </div>
              <p class="meta">bbox {bbox} / boxes {box_ids}</p>
              <p class="full-text">{full_text}</p>
            </article>
            """.format(
                block_id=html.escape(block.block_id),
                color=color,
                source=html.escape(block.source),
                token_count=block.token_count,
                reading_mode=html.escape(block.reading_mode),
                bbox=html.escape(str(block.bbox)),
                box_ids=html.escape(", ".join(block.box_ids) if block.box_ids else "-"),
                full_text=html.escape(block.full_text or "[no text]"),
            )
        )

    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Box Structured Parse</title>
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
      border: 2px solid var(--block-color);
      background: color-mix(in srgb, var(--block-color) 14%, transparent);
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
    .pill {{
      color: white;
      padding: 0.15rem 0.5rem;
      border-radius: 999px;
      font-size: 0.73rem;
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
      <h1>Box Structured Parse</h1>
      <p>{pdf_path} / page {page_number}</p>
      <p>JSON: {json_path}</p>
      <div class="stage">{overlays}</div>
    </section>
    <aside class="panel sidebar">
      <h2>Blocks</h2>
      {cards}
      <h3>Selected Block</h3>
      <pre id="detail">블록을 클릭하면 상세 JSON이 여기에 표시됩니다.</pre>
    </aside>
  </div>
  <script>
    const detail = document.getElementById('detail');
    for (const button of document.querySelectorAll('.overlay')) {{
      button.addEventListener('click', () => {{
        detail.textContent = JSON.stringify(JSON.parse(button.dataset.block), null, 2);
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
        tokens = extract_tokens(parsed_page)
        boxes, container_bbox = extract_boxes(page, tokens)
        box_groups = connected_components(boxes)
        residual_groups = residual_token_clusters(tokens, boxes, container_bbox)
        blocks = build_structured_blocks(box_groups, residual_groups)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

    result = {
        "pdf_path": pdf_path.as_posix(),
        "page_number": args.page,
        "block_count": len(blocks),
        "blocks": [asdict(block) for block in blocks],
    }

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_") or "document"
    json_path = output_dir / f"{stem}_page{args.page:02d}_structured.json"
    html_path = output_dir / f"{stem}_page{args.page:02d}_structured.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(pdf_path, args.page, page_width, page_height, blocks, json_path), encoding="utf-8")
    print(json.dumps({"json_path": json_path.as_posix(), "html_path": html_path.as_posix(), "block_count": len(blocks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
