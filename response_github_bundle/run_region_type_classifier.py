from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drawing_guided_pages_04_16"
PAGE_JSON_PATHS = [
    OUTPUT_DIR / "page_04_drawing_guided_structure.json",
    OUTPUT_DIR / "page_16_drawing_guided_structure.json",
]


TYPE_LABELS = (
    "table",
    "chart",
    "kpi_panel",
    "kpi_pair_panel",
    "mixed_chart_panel",
    "highlight_card",
    "notes",
    "unknown",
)


def main() -> None:
    output_paths: list[str] = []
    for path in PAGE_JSON_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        classify_page_regions(payload)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        classified_path = path.with_name(path.stem.replace("_structure", "_classified_structure") + path.suffix)
        classified_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        output_paths.append(classified_path.as_posix())

    summary_path = OUTPUT_DIR / "region_type_classification_summary.md"
    summary_path.write_text(render_summary(PAGE_JSON_PATHS), encoding="utf-8")
    print(json.dumps({"classified_json": output_paths, "summary": summary_path.as_posix()}, ensure_ascii=False, indent=2))


def classify_page_regions(payload: dict[str, Any]) -> None:
    page_number = int(payload.get("page_number") or 0)
    for region in payload.get("regions", []):
        features = extract_region_features(region, page_number)
        scores = score_region_type(features)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = confidence_from_scores(best_score, second_score)
        region["type_classification"] = {
            "predicted_type": best_type if best_score > 0 else "unknown",
            "confidence": confidence,
            "scores": {key: round(value, 3) for key, value in ranked},
            "features": features,
        }


def extract_region_features(region: dict[str, Any], page_number: int) -> dict[str, Any]:
    markdown = str(region.get("markdown") or "")
    bbox = region.get("bbox") or [0, 0, 0, 0]
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    evidence = region.get("evidence") or {}

    tokens = re.findall(r"[A-Za-z가-힣0-9.,%()'’~+\-]+", markdown)
    numeric_tokens = [token for token in tokens if re.search(r"\d", token)]
    percent_tokens = [token for token in tokens if "%" in token]
    axis_labels = [token for token in tokens if re.fullmatch(r"(?:20\d{2}|3Q\d{2}|Q\d{2}|YTD|QoQ)", token)]
    unit_tokens = [token for token in tokens if token in {"억", "원", "조", "만원", "주", "%p"} or re.search(r"(?:억|조|원|주|%p)", token)]
    large_numbers = [token for token in numeric_tokens if re.fullmatch(r"\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\)?", token)]
    bullet_count = len(re.findall(r"(?m)^\s*[-▪•]", markdown))
    note_marker_count = len(re.findall(r"(?m)^\s*\d+\.", markdown)) + len(re.findall(r"(?m)^\s*\d+\)", markdown))
    markdown_table_rows = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    pipe_table_row_count = len(markdown_table_rows)
    sentence_like_lines = [
        line
        for line in markdown.splitlines()
        if len(line.strip()) >= 18 and not line.strip().startswith("|") and not re.fullmatch(r"[\d,.\s%()'’~+\-]+", line.strip())
    ]

    matching_fills = evidence.get("matching_fills") or []
    bar_fills = evidence.get("bar_fills") or []
    cell_fill_count = int(evidence.get("cell_fill_count") or 0)
    matching_orange_lines = evidence.get("matching_orange_lines") or []
    fill_count = count_evidence_fills(evidence)
    orange_line_count = count_evidence_lines(evidence, matching_orange_lines)

    return {
        "page_number": page_number,
        "bbox_width": round(width, 3),
        "bbox_height": round(height, 3),
        "bbox_y0": round(y0, 3),
        "token_count": int(region.get("token_count") or len(tokens)),
        "markdown_chars": len(markdown),
        "numeric_token_count": len(numeric_tokens),
        "numeric_token_ratio": round(len(numeric_tokens) / max(1, len(tokens)), 3),
        "percent_token_count": len(percent_tokens),
        "axis_label_count": len(axis_labels),
        "unit_token_count": len(unit_tokens),
        "large_number_count": len(large_numbers),
        "bullet_count": bullet_count,
        "note_marker_count": note_marker_count,
        "sentence_line_count": len(sentence_like_lines),
        "pipe_table_row_count": pipe_table_row_count,
        "cell_fill_count": cell_fill_count,
        "bar_fill_count": len(bar_fills),
        "fill_count": fill_count,
        "orange_line_count": orange_line_count,
        "has_card_rect": bool(matching_fills and y0 >= 300 and width >= 300 and height >= 70),
        "has_markdown_table": pipe_table_row_count >= 3 and any("---" in line for line in markdown_table_rows),
        "has_axis_labels": len(axis_labels) >= 2,
        "has_legend_terms": any(term in markdown for term in ("Trading", "Brokerage", "WM", "IB", "Legend")),
    }


def count_evidence_fills(evidence: dict[str, Any]) -> int:
    count = 0
    for key in ("matching_fills", "bar_fills"):
        value = evidence.get(key)
        if isinstance(value, list):
            count += len(value)
    if evidence.get("cell_fill_count"):
        count += int(evidence["cell_fill_count"])
    return count


def count_evidence_lines(evidence: dict[str, Any], matching_orange_lines: list[Any]) -> int:
    count = len(matching_orange_lines)
    for key in ("top_line", "bottom_line", "separator_line"):
        if evidence.get(key):
            count += 1
    return count


def score_region_type(features: dict[str, Any]) -> dict[str, float]:
    table = 0.0
    table += 5.0 if features["has_markdown_table"] else 0.0
    table += min(6.0, features["cell_fill_count"] / 3.0)
    table += min(3.0, features["pipe_table_row_count"] / 2.0)
    table += 1.0 if features["orange_line_count"] else 0.0

    chart = 0.0
    chart += 3.0 if features["has_axis_labels"] else 0.0
    chart += min(5.0, features["bar_fill_count"] * 0.8)
    chart += min(3.0, features["percent_token_count"] * 0.6)
    chart += 1.5 if features["has_legend_terms"] else 0.0
    chart += 1.0 if features["numeric_token_ratio"] >= 0.45 else 0.0

    mixed_chart = 0.0
    mixed_chart += 3.0 if features["has_axis_labels"] else 0.0
    mixed_chart += min(5.0, features["fill_count"] * 0.35)
    mixed_chart += 2.0 if features["has_legend_terms"] else 0.0
    mixed_chart += 1.5 if features["percent_token_count"] >= 4 else 0.0
    mixed_chart += 1.0 if features["large_number_count"] >= 2 else 0.0

    kpi_panel = 0.0
    kpi_panel += min(5.0, features["large_number_count"] * 0.8)
    kpi_panel += min(3.0, features["unit_token_count"] * 0.35)
    kpi_panel += 2.0 if features["bbox_width"] > 250 and features["bbox_height"] > 90 else 0.0
    kpi_panel -= 2.0 if features["has_markdown_table"] else 0.0

    kpi_pair = 0.0
    kpi_pair += 4.0 if 2 <= features["large_number_count"] <= 6 else 0.0
    kpi_pair += 2.0 if features["unit_token_count"] >= 2 else 0.0
    kpi_pair += 1.5 if features["fill_count"] <= 4 else 0.0
    kpi_pair += 1.0 if features["bbox_height"] < 110 else 0.0
    kpi_pair -= 2.0 if features["percent_token_count"] >= 4 else 0.0

    highlight_card = 0.0
    highlight_card += 4.0 if features["has_card_rect"] else 0.0
    highlight_card += min(4.0, features["sentence_line_count"] * 1.2)
    highlight_card += min(3.0, features["bullet_count"] * 1.5)
    highlight_card += 1.0 if features["bbox_y0"] >= 300 else 0.0

    notes = 0.0
    notes += min(8.0, features["note_marker_count"] * 2.5)
    notes += 2.0 if features["bbox_y0"] >= 440 else 0.0
    notes += 1.0 if features["bbox_height"] < 50 else 0.0

    return {
        "table": max(0.0, table),
        "chart": max(0.0, chart),
        "kpi_panel": max(0.0, kpi_panel),
        "kpi_pair_panel": max(0.0, kpi_pair),
        "mixed_chart_panel": max(0.0, mixed_chart),
        "highlight_card": max(0.0, highlight_card),
        "notes": max(0.0, notes),
        "unknown": 0.1,
    }


def confidence_from_scores(best_score: float, second_score: float) -> float:
    if best_score <= 0:
        return 0.0
    margin = max(0.0, best_score - second_score)
    confidence = 0.45 + min(0.45, margin / max(1.0, best_score) * 0.7) + min(0.1, best_score / 100.0)
    return round(min(0.99, confidence), 3)


def render_summary(page_paths: list[Path]) -> str:
    lines = [
        "# Region Type Classification Summary",
        "",
        "| page | region | previous_type | predicted_type | confidence | top_scores |",
        "|---:|---|---|---|---:|---|",
    ]
    for path in page_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for region in payload.get("regions", []):
            classification = region.get("type_classification") or {}
            scores = classification.get("scores") or {}
            top_scores = ", ".join(f"{key}={value}" for key, value in list(scores.items())[:3])
            lines.append(
                "| {page} | {region} | {previous} | {predicted} | {confidence} | {scores} |".format(
                    page=payload.get("page_number"),
                    region=region.get("id"),
                    previous=region.get("type"),
                    predicted=classification.get("predicted_type"),
                    confidence=classification.get("confidence"),
                    scores=top_scores,
                )
            )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
