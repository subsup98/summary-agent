from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.classifiers.document_classifier import classify_document
from src.parsers.pdf.pdf_parser import PdfParser


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "region_type_probe"

REGION_TYPES = ("table", "chart", "kpi_card", "paragraph", "unknown")


@dataclass
class ProbeElement:
    element_type: str
    order: int
    text: str
    bbox: list[float]

    @property
    def x0(self) -> float:
        return float(self.bbox[0])

    @property
    def y0(self) -> float:
        return float(self.bbox[1])

    @property
    def x1(self) -> float:
        return float(self.bbox[2])

    @property
    def y1(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class RegionCandidate:
    region_id: str
    bbox: list[float]
    element_count: int
    dominant_element_type: str
    reading_mode: str
    full_text: str
    features: dict[str, Any]
    scores: dict[str, float]
    region_type: str
    confidence: float
    members: list[dict[str, Any]]


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


def overlaps_or_is_close(left: ProbeElement, right: ProbeElement, *, x_gap: float, y_gap: float) -> bool:
    horizontal_gap = max(left.x0, right.x0) - min(left.x1, right.x1)
    vertical_gap = max(left.y0, right.y0) - min(left.y1, right.y1)
    return horizontal_gap <= x_gap and vertical_gap <= y_gap


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


def normalize_cluster_centers(values: list[float], tolerance: float) -> list[float]:
    return [sum(group) / len(group) for group in cluster_positions(values, tolerance)]


def infer_region_type(features: dict[str, Any]) -> tuple[str, dict[str, float], float]:
    scores = {name: 0.0 for name in REGION_TYPES}

    if features["row_cluster_count"] >= 3:
        scores["table"] += 1.5
    if features["col_cluster_count"] >= 3:
        scores["table"] += 1.5
    if features["number_ratio"] >= 0.35:
        scores["table"] += 1.2
    if features["header_like_count"] >= 1:
        scores["table"] += 0.8
    if features["year_count"] >= 3:
        scores["table"] += 0.6
    if features["grid_density"] >= 0.55:
        scores["table"] += 1.0

    if features["year_count"] >= 3:
        scores["chart"] += 1.8
    if features["percent_count"] >= 2:
        scores["chart"] += 1.3
    if features["legend_like_count"] >= 2:
        scores["chart"] += 1.1
    if features["large_numeric_count"] >= 3:
        scores["chart"] += 0.8
    if features["row_cluster_count"] >= 2 and features["col_cluster_count"] >= 4:
        scores["chart"] += 0.8

    if features["large_numeric_count"] >= 2:
        scores["kpi_card"] += 1.7
    if features["unit_count"] >= 2:
        scores["kpi_card"] += 1.1
    if features["short_label_count"] >= 2:
        scores["kpi_card"] += 0.8
    if features["number_ratio"] >= 0.25:
        scores["kpi_card"] += 0.6
    if features["element_count"] <= 30:
        scores["kpi_card"] += 0.5

    if features["sentence_like_count"] >= 2:
        scores["paragraph"] += 1.6
    if features["long_text_count"] >= 2:
        scores["paragraph"] += 1.2
    if features["number_ratio"] <= 0.2:
        scores["paragraph"] += 0.8
    if features["line_break_like_count"] >= 3:
        scores["paragraph"] += 0.6

    scores["unknown"] += 0.2
    if max(scores.values()) < 1.5:
        scores["unknown"] += 1.4
    if features["element_count"] <= 2:
        scores["unknown"] += 0.8

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = round(max(0.0, min(0.99, 0.45 + (best_score - second_score) * 0.12)), 3)
    return best_type, scores, confidence


def infer_reading_mode(features: dict[str, Any], region_type: str, group_bbox: list[float]) -> str:
    width = max(1.0, group_bbox[2] - group_bbox[0])
    height = max(1.0, group_bbox[3] - group_bbox[1])
    if region_type == "kpi_card" and features["col_cluster_count"] >= 2:
        return "column_major"
    if region_type == "chart" and width > height * 1.1 and features["col_cluster_count"] >= 4:
        return "column_major"
    return "row_major"


def build_full_text(elements: list[ProbeElement], reading_mode: str) -> str:
    if not elements:
        return ""

    ordered_parts: list[str] = []
    if reading_mode == "column_major":
        column_centers = normalize_cluster_centers([item.cx for item in elements], tolerance=28.0)
        buckets: dict[int, list[ProbeElement]] = defaultdict(list)
        for item in elements:
            index = min(range(len(column_centers)), key=lambda idx: abs(column_centers[idx] - item.cx))
            buckets[index].append(item)
        for index in sorted(buckets):
            column_items = sorted(buckets[index], key=lambda item: (item.y0, item.x0, item.order))
            ordered_parts.extend(normalize_text(item.text) for item in column_items if normalize_text(item.text))
        return " / ".join(ordered_parts)

    row_centers = normalize_cluster_centers([item.cy for item in elements], tolerance=14.0)
    buckets = defaultdict(list)
    for item in elements:
        index = min(range(len(row_centers)), key=lambda idx: abs(row_centers[idx] - item.cy))
        buckets[index].append(item)
    for index in sorted(buckets):
        row_items = sorted(buckets[index], key=lambda item: (item.x0, item.y0, item.order))
        ordered_parts.extend(normalize_text(item.text) for item in row_items if normalize_text(item.text))
    return " / ".join(ordered_parts)


def extract_features(elements: list[ProbeElement]) -> dict[str, Any]:
    texts = [normalize_text(item.text) for item in elements if normalize_text(item.text)]
    number_like = [text for text in texts if re.search(r"\d", text)]
    percent_like = [text for text in texts if "%" in text]
    year_like = [text for text in texts if re.fullmatch(r"(19|20)\d{2}", text)]
    unit_like = [text for text in texts if any(token in text for token in ("원", "억원", "조", "만주", "주", "%"))]
    legend_like = [text for text in texts if any(token in text for token in ("총액", "비율", "배당", "소각", "환원"))]
    header_like = [text for text in texts if any(token in text for token in ("목표", "이행", "이행률", "기준일", "연도"))]
    sentence_like = [text for text in texts if len(text) >= 18 and (" " in text or any(token in text for token in ("정책", "목표", "대비", "시행")))]
    long_text = [text for text in texts if len(text) >= 30]
    large_numeric = [text for text in texts if re.search(r"\d{3,}", text)]
    short_label = [text for text in texts if 1 <= len(text) <= 8 and not re.fullmatch(r"[\d,./%]+", text)]

    row_clusters = cluster_positions([item.cy for item in elements], tolerance=14.0)
    col_clusters = cluster_positions([item.cx for item in elements], tolerance=26.0)

    text_count = sum(1 for item in elements if item.element_type == "text")
    image_count = sum(1 for item in elements if item.element_type == "image")
    bbox = bbox_union([item.bbox for item in elements])
    region_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    covered_area = sum(item.area for item in elements)
    grid_density = min(1.0, covered_area / region_area)

    return {
        "element_count": len(elements),
        "text_count": text_count,
        "image_count": image_count,
        "number_ratio": round(len(number_like) / max(1, len(texts)), 3),
        "percent_count": len(percent_like),
        "year_count": len(year_like),
        "unit_count": len(unit_like),
        "legend_like_count": len(legend_like),
        "header_like_count": len(header_like),
        "sentence_like_count": len(sentence_like),
        "long_text_count": len(long_text),
        "large_numeric_count": len(large_numeric),
        "short_label_count": len(short_label),
        "line_break_like_count": sum(1 for text in texts if " " in text and len(text) <= 20),
        "row_cluster_count": len(row_clusters),
        "col_cluster_count": len(col_clusters),
        "grid_density": round(grid_density, 3),
    }


def build_region_candidates(page_elements: list[ProbeElement]) -> list[list[ProbeElement]]:
    usable = [item for item in page_elements if item.bbox and len(item.bbox) == 4]
    visited: set[int] = set()
    groups: list[list[ProbeElement]] = []

    for index, element in enumerate(usable):
        if index in visited:
            continue
        queue = [index]
        visited.add(index)
        group: list[ProbeElement] = []
        while queue:
            current_index = queue.pop()
            current = usable[current_index]
            group.append(current)
            for candidate_index, candidate in enumerate(usable):
                if candidate_index in visited:
                    continue
                x_gap = max(28.0, min(120.0, max(current.width, candidate.width) * 1.8))
                y_gap = max(18.0, min(90.0, max(current.height, candidate.height) * 2.4))
                if overlaps_or_is_close(current, candidate, x_gap=x_gap, y_gap=y_gap):
                    visited.add(candidate_index)
                    queue.append(candidate_index)
        groups.append(sorted(group, key=lambda item: (item.y0, item.x0, item.order)))

    refined: list[list[ProbeElement]] = []
    for group in groups:
        if should_force_quadrant_split(group):
            refined.extend(split_large_group_by_quadrants(group))
        else:
            refined.extend(split_group_recursively(group))
    return sorted(refined, key=lambda items: (bbox_union([item.bbox for item in items])[1], bbox_union([item.bbox for item in items])[0]))


def should_force_quadrant_split(group: list[ProbeElement]) -> bool:
    if len(group) < 120:
        return False
    bbox = bbox_union([item.bbox for item in group])
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width >= 500 and height >= 220


def split_large_group_by_quadrants(group: list[ProbeElement]) -> list[list[ProbeElement]]:
    bbox = bbox_union([item.bbox for item in group])
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])

    centers = [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    ]
    assignments: dict[int, list[ProbeElement]] = defaultdict(list)
    for item in group:
        nx = (item.cx - bbox[0]) / width
        ny = (item.cy - bbox[1]) / height
        best_index = min(
            range(len(centers)),
            key=lambda idx: math.dist((nx, ny), centers[idx]),
        )
        assignments[best_index].append(item)

    result: list[list[ProbeElement]] = []
    for index in range(len(centers)):
        cluster = sorted(assignments.get(index, []), key=lambda item: (item.y0, item.x0, item.order))
        if len(cluster) >= 6:
            result.extend(split_group_recursively(cluster))
    return result if result else [group]


def split_group_recursively(group: list[ProbeElement], *, depth: int = 0) -> list[list[ProbeElement]]:
    if len(group) < 12 or depth >= 3:
        return [group]

    split = find_gap_split(group)
    if split is None:
        return [group]

    axis, threshold = split
    if axis == "x":
        left = [item for item in group if item.cx <= threshold]
        right = [item for item in group if item.cx > threshold]
        if len(left) < 3 or len(right) < 3:
            return [group]
        return [*split_group_recursively(left, depth=depth + 1), *split_group_recursively(right, depth=depth + 1)]

    top = [item for item in group if item.cy <= threshold]
    bottom = [item for item in group if item.cy > threshold]
    if len(top) < 3 or len(bottom) < 3:
        return [group]
    return [*split_group_recursively(top, depth=depth + 1), *split_group_recursively(bottom, depth=depth + 1)]


def find_gap_split(group: list[ProbeElement]) -> tuple[str, float] | None:
    x_sorted = sorted(group, key=lambda item: item.cx)
    y_sorted = sorted(group, key=lambda item: item.cy)
    bbox = bbox_union([item.bbox for item in group])
    width = max(1.0, bbox[2] - bbox[0])
    height = max(1.0, bbox[3] - bbox[1])

    best_axis = ""
    best_gap = 0.0
    best_threshold = 0.0

    for ordered, axis, min_ratio in ((x_sorted, "x", 0.07), (y_sorted, "y", 0.07)):
        for index in range(1, len(ordered)):
            prev_item = ordered[index - 1]
            next_item = ordered[index]
            if axis == "x":
                gap = next_item.x0 - prev_item.x1
                ratio = gap / width
                threshold = (prev_item.x1 + next_item.x0) / 2.0
            else:
                gap = next_item.y0 - prev_item.y1
                ratio = gap / height
                threshold = (prev_item.y1 + next_item.y0) / 2.0
            if ratio >= min_ratio and gap > best_gap:
                best_axis = axis
                best_gap = gap
                best_threshold = threshold

    if not best_axis:
        return None
    return best_axis, best_threshold


def collapse_nested_regions(candidates: list[RegionCandidate]) -> list[RegionCandidate]:
    kept: list[RegionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.bbox[0], item.bbox[1], -item.element_count)):
        bbox = candidate.bbox
        nested = False
        for existing in kept:
            existing_bbox = existing.bbox
            if (
                bbox[0] >= existing_bbox[0] - 6.0
                and bbox[1] >= existing_bbox[1] - 6.0
                and bbox[2] <= existing_bbox[2] + 6.0
                and bbox[3] <= existing_bbox[3] + 6.0
                and existing.element_count >= candidate.element_count
            ):
                nested = True
                break
        if not nested:
            kept.append(candidate)
    return kept


def summarize_region(group: list[ProbeElement], index: int) -> RegionCandidate:
    bbox = bbox_union([item.bbox for item in group])
    features = extract_features(group)
    region_type, scores, confidence = infer_region_type(features)
    reading_mode = infer_reading_mode(features, region_type, bbox)
    dominant_element_type = Counter(item.element_type for item in group).most_common(1)[0][0]
    full_text = build_full_text(group, reading_mode)
    return RegionCandidate(
        region_id=f"r{index:02d}",
        bbox=[round(value, 2) for value in bbox],
        element_count=len(group),
        dominant_element_type=dominant_element_type,
        reading_mode=reading_mode,
        full_text=full_text,
        features=features,
        scores={name: round(score, 3) for name, score in scores.items()},
        region_type=region_type,
        confidence=confidence,
        members=[
            {
                "element_type": item.element_type,
                "order": item.order,
                "bbox": [round(value, 2) for value in item.bbox],
                "text": normalize_text(item.text)[:120],
            }
            for item in group
        ],
    )


def render_html_report(*, pdf_path: Path, page_number: int, page_width: float, page_height: float, regions: list[RegionCandidate], output_json_path: Path) -> str:
    colors = {
        "table": "#265d8b",
        "chart": "#c74e2a",
        "kpi_card": "#1b805f",
        "paragraph": "#7b4aaa",
        "unknown": "#70604e",
    }
    cards = []
    overlays = []
    for region in regions:
        left = max(0.0, min(100.0, region.bbox[0] / max(1.0, page_width) * 100.0))
        top = max(0.0, min(100.0, region.bbox[1] / max(1.0, page_height) * 100.0))
        width = max(0.4, min(100.0, (region.bbox[2] - region.bbox[0]) / max(1.0, page_width) * 100.0))
        height = max(0.4, min(100.0, (region.bbox[3] - region.bbox[1]) / max(1.0, page_height) * 100.0))
        payload = html.escape(json.dumps(asdict(region), ensure_ascii=False))
        color = colors.get(region.region_type, colors["unknown"])
        overlays.append(
            '<button class="overlay" data-region="{payload}" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;--region-color:{color}">{label}</button>'.format(
                payload=payload,
                left=left,
                top=top,
                width=width,
                height=height,
                color=color,
                label=html.escape(region.region_id),
            )
        )
        cards.append(
            """
            <article class="card">
              <div class="card-head">
                <strong>{region_id}</strong>
                <span class="pill" style="background:{color}">{region_type}</span>
                <span class="confidence">conf {confidence:.2f}</span>
              </div>
              <p class="preview">{full_text}</p>
              <p class="meta">bbox {bbox} / elements {element_count} / dominant {dominant} / read {reading_mode}</p>
              <pre>{scores}</pre>
            </article>
            """.format(
                region_id=html.escape(region.region_id),
                color=html.escape(color),
                region_type=html.escape(region.region_type),
                confidence=region.confidence,
                full_text=html.escape(region.full_text or "[no text]"),
                bbox=html.escape(str(region.bbox)),
                element_count=region.element_count,
                dominant=html.escape(region.dominant_element_type),
                reading_mode=html.escape(region.reading_mode),
                scores=html.escape(json.dumps(region.scores, ensure_ascii=False, indent=2)),
            )
        )

    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Region Type Probe</title>
  <style>
    :root {{
      --paper: #f6f1e7;
      --ink: #1f1b16;
      --line: rgba(31, 27, 22, 0.12);
      --panel: rgba(255, 251, 245, 0.92);
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
      grid-template-columns: minmax(0, 1.6fr) minmax(22rem, 0.9fr);
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
      background: linear-gradient(180deg, #ffffff 0%, #fbf8f4 100%);
      border: 1px dashed rgba(0, 0, 0, 0.14);
      border-radius: 16px;
      overflow: hidden;
    }}
    .overlay {{
      position: absolute;
      display: flex;
      align-items: flex-start;
      justify-content: flex-start;
      padding: 0.2rem 0.35rem;
      border: 2px solid var(--region-color);
      background: color-mix(in srgb, var(--region-color) 18%, transparent);
      color: #111;
      font-size: 0.72rem;
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
      background: rgba(255,255,255,0.84);
      margin-bottom: 0.75rem;
    }}
    .card-head {{
      display: flex;
      gap: 0.45rem;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 0.45rem;
    }}
    .pill {{
      color: white;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
      font-size: 0.74rem;
      text-transform: uppercase;
    }}
    .confidence, .meta {{
      color: #5f564b;
      font-size: 0.82rem;
    }}
    .preview {{
      margin: 0.3rem 0 0.45rem;
      line-height: 1.4;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(31, 27, 22, 0.04);
      border-radius: 12px;
      padding: 0.65rem;
      font-size: 0.78rem;
    }}
    .detail {{
      margin-top: 1rem;
      border-top: 1px solid var(--line);
      padding-top: 1rem;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel">
      <h1>Region Type Probe</h1>
      <p>{pdf_path} / page {page_number}</p>
      <p>JSON: {json_path}</p>
      <div class="stage">
        {overlays}
      </div>
    </section>
    <aside class="panel sidebar">
      <h2>Regions</h2>
      {cards}
      <section class="detail">
        <h3>Selected Region</h3>
        <pre id="detail">Overlay를 클릭하면 상세 JSON이 여기에 표시됩니다.</pre>
      </section>
    </aside>
  </div>
  <script>
    const detail = document.getElementById('detail');
    for (const button of document.querySelectorAll('.overlay')) {{
      button.addEventListener('click', () => {{
        detail.textContent = JSON.stringify(JSON.parse(button.dataset.region), null, 2);
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
    parser = argparse.ArgumentParser(description="Probe page regions and classify them into table/chart/kpi/paragraph/unknown.")
    parser.add_argument("--pdf", help="Optional PDF path.")
    parser.add_argument("--page", type=int, default=16, help="1-based page number to inspect.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON/HTML outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = resolve_pdf_path(args.pdf)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = PdfParser(enable_omitted_picture_ocr=False).parse(pdf_path, classify_document(pdf_path))
    page_index = max(0, args.page - 1)
    if page_index >= len(parsed.pages):
        raise ValueError(f"page out of range: {args.page}")
    page = parsed.pages[page_index]

    page_elements = [
        ProbeElement(
            element_type=str(element.element_type),
            order=int(element.order),
            text=str(element.text or ""),
            bbox=[float(value) for value in (element.bbox or [])],
        )
        for element in page.elements
        if element.bbox and len(element.bbox) == 4
    ]

    grouped = build_region_candidates(page_elements)
    summarized = [summarize_region(group, index + 1) for index, group in enumerate(grouped) if len(group) >= 3]
    regions = collapse_nested_regions(summarized)

    result = {
        "pdf_path": pdf_path.as_posix(),
        "page_number": args.page,
        "parser_name": parsed.parser_name,
        "selected_strategy": parsed.metadata.get("markdown_metadata", {}).get("selected_strategy"),
        "markdown_source": parsed.metadata.get("markdown_source"),
        "page_element_count": len(page_elements),
        "region_count": len(regions),
        "regions": [asdict(region) for region in regions],
    }

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("_") or "document"
    json_path = output_dir / f"{stem}_page{args.page:02d}_regions.json"
    html_path = output_dir / f"{stem}_page{args.page:02d}_regions.html"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    page_width = max((item.x1 for item in page_elements), default=1.0)
    page_height = max((item.y1 for item in page_elements), default=1.0)
    html_path.write_text(
        render_html_report(
            pdf_path=pdf_path,
            page_number=args.page,
            page_width=page_width,
            page_height=page_height,
            regions=regions,
            output_json_path=json_path,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"json_path": json_path.as_posix(), "html_path": html_path.as_posix(), "region_count": len(regions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
