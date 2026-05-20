from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz

import run_drawing_guided_page_structure as guided


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drawing_guided_pages_04_16"
PDF_NAME = "\ubbf8\ub798\uc5d0\uc14b\uc99d\uad8c 3\ubd84\uae30 \uc2e4\uc801\ubcf4\uace0\uc11c.pdf"
PDF_PATH = Path.home() / "Desktop" / "all_docs" / PDF_NAME
PAGE_JSON_PATHS = [
    OUTPUT_DIR / "page_04_drawing_guided_structure.json",
    OUTPUT_DIR / "page_16_drawing_guided_structure.json",
]


def main() -> None:
    with fitz.open(PDF_PATH) as document:
        for json_path in PAGE_JSON_PATHS:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            add_x_coordinate_structure(document, payload)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            structured_path = json_path.with_name(json_path.stem.replace("_structure", "_x_coordinate_structure") + json_path.suffix)
            structured_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(structured_path.as_posix())

    summary_path = OUTPUT_DIR / "x_coordinate_structure_summary.md"
    summary_path.write_text(render_summary(PAGE_JSON_PATHS), encoding="utf-8")
    print(summary_path.as_posix())


def add_x_coordinate_structure(document: fitz.Document, payload: dict[str, Any]) -> None:
    page_number = int(payload["page_number"])
    page = document[page_number - 1]
    tokens = guided.extract_tokens(document, page, page_number)

    for region in payload.get("regions", []):
        rect = rect_from_bbox(region.get("bbox") or [])
        region_tokens = guided.tokens_in_rect(tokens, rect) if rect else []
        predicted_type = (region.get("type_classification") or {}).get("predicted_type") or region.get("type")
        region["x_coordinate_structure"] = build_x_coordinate_structure(region_tokens, predicted_type)


def rect_from_bbox(bbox: list[Any]) -> guided.Rect | None:
    if len(bbox) != 4:
        return None
    return guided.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def build_x_coordinate_structure(tokens: list[Any], region_type: str | None) -> dict[str, Any]:
    if not tokens:
        return {
            "strategy": "x_coordinate_columns",
            "reason": "no_tokens_in_region",
            "columns": [],
            "y_bands": [],
        }

    y_bands = build_y_bands(tokens, region_type or "")
    columns: list[dict[str, Any]] = []
    for band_index, band_tokens in enumerate(y_bands):
        band_columns = build_columns_for_band(band_tokens, region_type or "")
        for column in band_columns:
            column["band_index"] = band_index
            columns.append(column)

    return {
        "strategy": "y_band_then_x_anchor_columns",
        "region_type_hint": region_type,
        "y_bands": [
            {
                "band_index": index,
                "y_range": y_range_for_tokens(band_tokens),
                "token_count": len(band_tokens),
            }
            for index, band_tokens in enumerate(y_bands)
        ],
        "columns": columns,
    }


def build_y_bands(tokens: list[Any], region_type: str) -> list[list[Any]]:
    sorted_tokens = sorted(tokens, key=lambda token: (token.box.y0, token.box.x0))
    if region_type == "chart":
        return [sorted_tokens]
    if region_type == "mixed_chart_panel":
        # Guard against mixed panels: line charts, legends, stacked bars, and notes
        # can share x positions but belong to different vertical bands.
        return cluster_tokens_by_y_gap(sorted_tokens, gap=28.0)
    if region_type in {"table"}:
        return [sorted_tokens]
    if region_type in {"highlight_card", "notes"}:
        return cluster_tokens_by_y_gap(sorted_tokens, gap=18.0)
    return cluster_tokens_by_y_gap(sorted_tokens, gap=34.0)


def cluster_tokens_by_y_gap(tokens: list[Any], *, gap: float) -> list[list[Any]]:
    if not tokens:
        return []
    rows: list[list[Any]] = []
    current: list[Any] = [tokens[0]]
    last_y = tokens[0].box.cy
    for token in tokens[1:]:
        if token.box.cy - last_y > gap:
            rows.append(current)
            current = [token]
        else:
            current.append(token)
        last_y = token.box.cy
    rows.append(current)
    return rows


def build_columns_for_band(tokens: list[Any], region_type: str) -> list[dict[str, Any]]:
    anchors = find_x_anchors(tokens, region_type)
    if len(anchors) >= 2:
        return build_anchor_columns(tokens, anchors)
    return build_cluster_columns(tokens)


def find_x_anchors(tokens: list[Any], region_type: str) -> list[Any]:
    axis_pattern = re.compile(r"^(?:20\d{2}|3\s?Q\d{2}|3Q\d{2}|Q\d{2}|YTD|QoQ)$", re.IGNORECASE)
    candidates = [token for token in tokens if axis_pattern.fullmatch(normalize_anchor_text(token.text))]
    if len(candidates) <= 1 and region_type in {"kpi_pair_panel", "kpi_panel"}:
        candidates = find_large_number_tokens(tokens)
    if len(candidates) <= 1:
        return []
    # Prefer lower axis labels when a band contains both data labels and axis labels.
    max_y = max(token.box.cy for token in candidates)
    lower = [token for token in candidates if max_y - token.box.cy <= 24.0]
    chosen = lower if len(lower) >= 2 else candidates
    return sorted(chosen, key=lambda token: token.box.cx)


def normalize_anchor_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def find_large_number_tokens(tokens: list[Any]) -> list[Any]:
    pattern = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$")
    return sorted([token for token in tokens if pattern.fullmatch(token.text)], key=lambda token: token.box.cx)


def build_anchor_columns(tokens: list[Any], anchors: list[Any]) -> list[dict[str, Any]]:
    anchor_xs = [anchor.box.cx for anchor in anchors]
    boundaries: list[tuple[float, float]] = []
    for index, anchor in enumerate(anchors):
        left = (anchor_xs[index - 1] + anchor.box.cx) / 2.0 if index > 0 else anchor.box.cx - 80.0
        right = (anchor.box.cx + anchor_xs[index + 1]) / 2.0 if index < len(anchors) - 1 else anchor.box.cx + 80.0
        boundaries.append((left, right))

    columns: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        left, right = boundaries[index]
        column_tokens = [
            token
            for token in tokens
            if left <= token.box.cx < right
        ]
        columns.append(column_record(index, anchor.text, left, right, column_tokens, anchor=anchor))
    return columns


def build_cluster_columns(tokens: list[Any]) -> list[dict[str, Any]]:
    sorted_tokens = sorted(tokens, key=lambda token: token.box.cx)
    clusters: list[list[Any]] = []
    for token in sorted_tokens:
        if not clusters:
            clusters.append([token])
            continue
        cluster = clusters[-1]
        cluster_center = sum(item.box.cx for item in cluster) / len(cluster)
        if abs(token.box.cx - cluster_center) <= 42.0:
            cluster.append(token)
        else:
            clusters.append([token])

    columns = []
    for index, cluster in enumerate(clusters):
        left = min(token.box.x0 for token in cluster)
        right = max(token.box.x1 for token in cluster)
        label = infer_column_label(cluster, index)
        columns.append(column_record(index, label, left, right, cluster))
    return columns


def column_record(
    index: int,
    label: str,
    left: float,
    right: float,
    tokens: list[Any],
    *,
    anchor: Any | None = None,
) -> dict[str, Any]:
    ordered = sorted(tokens, key=lambda token: (token.box.y0, token.box.x0))
    return {
        "column_index": index,
        "anchor": label,
        "x_range": [round(left, 3), round(right, 3)],
        "anchor_bbox": token_bbox(anchor) if anchor is not None else None,
        "items_top_to_bottom": [
            {
                "text": token.text,
                "bbox": token_bbox(token),
                "cx": round(token.box.cx, 3),
                "cy": round(token.box.cy, 3),
            }
            for token in ordered
        ],
        "joined_text": " | ".join(token.text for token in ordered),
    }


def infer_column_label(tokens: list[Any], index: int) -> str:
    for token in sorted(tokens, key=lambda item: (item.box.y0, item.box.x0)):
        if re.search(r"\d", token.text):
            return token.text
    return f"column_{index + 1}"


def token_bbox(token: Any | None) -> list[float] | None:
    if token is None:
        return None
    return [round(token.box.x0, 3), round(token.box.y0, 3), round(token.box.x1, 3), round(token.box.y1, 3)]


def y_range_for_tokens(tokens: list[Any]) -> list[float]:
    if not tokens:
        return []
    return [round(min(token.box.y0 for token in tokens), 3), round(max(token.box.y1 for token in tokens), 3)]


def render_summary(page_paths: list[Path]) -> str:
    lines = [
        "# X Coordinate Region Structure Summary",
        "",
        "| page | region | type | bands | columns | first columns |",
        "|---:|---|---|---:|---:|---|",
    ]
    for path in page_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for region in payload.get("regions", []):
            structure = region.get("x_coordinate_structure") or {}
            columns = structure.get("columns") or []
            first = "; ".join(f"{col.get('anchor')}: {col.get('joined_text')}" for col in columns[:3])
            predicted = (region.get("type_classification") or {}).get("predicted_type") or region.get("type")
            lines.append(
                f"| {payload.get('page_number')} | {region.get('id')} | {predicted} | "
                f"{len(structure.get('y_bands') or [])} | {len(columns)} | {first} |"
            )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
