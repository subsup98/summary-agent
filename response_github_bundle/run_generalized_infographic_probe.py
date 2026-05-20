from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import fitz

from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generalized_infographic_probe"
PAGE_NUMBER = 16
TOP_LIMIT = 315.0


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def padded(self, amount: float) -> "Box":
        return Box(self.x0 - amount, self.y0 - amount, self.x1 + amount, self.y1 + amount)

    def intersects(self, other: "Box") -> bool:
        return self.x0 <= other.x1 and self.x1 >= other.x0 and self.y0 <= other.y1 and self.y1 >= other.y0

    def contains_center(self, other: "Box", *, pad: float = 0.0) -> bool:
        return (
            self.x0 - pad <= other.cx <= self.x1 + pad
            and self.y0 - pad <= other.cy <= self.y1 + pad
        )

    def union(self, other: "Box") -> "Box":
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def to_list(self) -> list[float]:
        return [round(self.x0, 3), round(self.y0, 3), round(self.x1, 3), round(self.y1, 3)]


@dataclass(frozen=True)
class Token:
    text: str
    box: Box
    block_id: int
    block_role: str
    leaf_role: str
    mcids: tuple[int, ...]
    xrefs: tuple[int | None, ...]


def resolve_pdf_path() -> Path:
    desktop_docs = Path.home() / "Desktop" / "all_docs"
    exact_name = "\ubbf8\ub798\uc5d0\uc14b\uc99d\uad8c 3\ubd84\uae30 \uc2e4\uc801\ubcf4\uace0\uc11c.pdf"
    exact_path = desktop_docs / exact_name
    if exact_path.exists():
        return exact_path
    candidates = sorted(desktop_docs.glob("*.pdf"))
    for candidate in candidates:
        name = candidate.name
        if "3" in name and candidate.stat().st_size > 2_000_000:
            return candidate
    raise FileNotFoundError("Could not locate the Q3 Mirae Asset PDF under Desktop/all_docs")


def extract_tokens(document: fitz.Document, page: fitz.Page, page_number: int) -> list[Token]:
    extractor = PowerPointStructTreeExtractor()
    runs = [run for run in extractor.extract_runs(document) if run.page_number == page_number]
    mcid_boxes = extract_mcid_boxes_from_image_draws(page)
    tokens: list[Token] = []

    for run in runs:
        text = normalize_text(run.text)
        if not text or text.startswith("/Users/"):
            continue
        boxes = [mcid_boxes[mcid]["bbox"] for mcid in run.mcids if mcid in mcid_boxes]
        if not boxes:
            continue
        box = union_boxes(boxes)
        if box.y0 < 0 or box.y0 > TOP_LIMIT:
            continue
        tokens.append(
            Token(
                text=text,
                box=box,
                block_id=run.block_id,
                block_role=run.block_role,
                leaf_role=run.leaf_role,
                mcids=tuple(run.mcids),
                xrefs=tuple(mcid_boxes[mcid].get("xref") for mcid in run.mcids if mcid in mcid_boxes),
            )
        )
    return dedupe_tokens(tokens)


def extract_mcid_boxes_from_image_draws(page: fitz.Page) -> dict[int, dict[str, Any]]:
    resource_map = build_image_resource_map(page)
    draw_ops = extract_image_draw_operations(page, set(resource_map))
    try:
        image_infos = sorted(page.get_image_info(xrefs=True), key=lambda item: item.get("number", 0))
    except Exception:
        image_infos = []

    result: dict[int, dict[str, Any]] = {}
    for draw_op, image_info in zip(draw_ops, image_infos):
        mcid = draw_op.get("mcid")
        bbox = image_info.get("bbox") or ()
        if not isinstance(mcid, int) or len(bbox) != 4:
            continue
        box = Box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if box.y0 < 0 or box.y0 > TOP_LIMIT:
            continue
        item = result.setdefault(
            mcid,
            {
                "bbox": box,
                "xref": as_int(image_info.get("xref"))
                or as_int(resource_map.get(str(draw_op.get("xobject_name")), {}).get("xref")),
                "xobject_name": draw_op.get("xobject_name"),
            },
        )
        item["bbox"] = item["bbox"].union(box)
    return result


def build_image_resource_map(page: fitz.Page) -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for item in page.get_images(full=True):
        if len(item) < 8:
            continue
        name = str(item[7] or "")
        if not name:
            continue
        resources[name] = {"name": name, "xref": as_int(item[0])}
    return resources


def extract_image_draw_operations(page: fitz.Page, image_names: set[str]) -> list[dict[str, Any]]:
    if not image_names:
        return []
    try:
        contents = page.read_contents().decode("latin-1", errors="replace")
    except Exception:
        return []

    token_pattern = re.compile(r"/MCID\s+(\d+)|\b(BDC|BMC|EMC)\b|/([A-Za-z0-9_.+-]+)\s+Do")
    stack: list[int | None] = []
    pending_mcid: int | None = None
    operations: list[dict[str, Any]] = []

    for match in token_pattern.finditer(contents):
        mcid_value, marker, xobject_name = match.group(1), match.group(2), match.group(3)
        if mcid_value is not None:
            pending_mcid = int(mcid_value)
            continue
        if marker in {"BDC", "BMC"}:
            stack.append(pending_mcid)
            pending_mcid = None
            continue
        if marker == "EMC":
            if stack:
                stack.pop()
            pending_mcid = None
            continue
        if xobject_name and xobject_name in image_names:
            current_mcid = next((value for value in reversed(stack) if value is not None), None)
            operations.append({"xobject_name": xobject_name, "mcid": current_mcid})
    return operations


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\ufeff", "")).strip()


def union_boxes(boxes: list[Box]) -> Box:
    box = boxes[0]
    for item in boxes[1:]:
        box = box.union(item)
    return box


def as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe_tokens(tokens: list[Token]) -> list[Token]:
    kept: list[Token] = []
    for token in sorted(tokens, key=lambda item: (item.box.y0, item.box.x0, item.text)):
        duplicate = False
        for existing in kept:
            if (
                token.text == existing.text
                and abs(token.box.cx - existing.box.cx) <= 2.0
                and abs(token.box.cy - existing.box.cy) <= 2.0
                and token.mcids == existing.mcids
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(token)
    return kept


def extract_drawing_boxes(page: fitz.Page) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for index, drawing in enumerate(page.get_drawings()):
        rect = drawing.get("rect")
        if rect is None:
            continue
        box = Box(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        if box.y1 < 80 or box.y0 > TOP_LIMIT:
            continue
        if box.width < 35 or box.height < 18:
            continue
        if box.area < 1_200:
            continue
        if box.width > page.rect.width * 0.96 or box.height > page.rect.height * 0.60:
            continue
        boxes.append(
            {
                "index": index,
                "bbox": box,
                "fill": color_to_list(drawing.get("fill")),
                "stroke": color_to_list(drawing.get("color")),
                "width": drawing.get("width"),
                "item_count": len(drawing.get("items") or []),
            }
        )
    boxes.extend(extract_large_image_boxes(page))
    boxes.extend(build_drawing_hull_boxes(boxes, page))
    return merge_similar_drawing_boxes(boxes)


def extract_large_image_boxes(page: fitz.Page) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    try:
        image_infos = page.get_image_info(xrefs=True)
    except Exception:
        return boxes
    for index, info in enumerate(image_infos):
        bbox = info.get("bbox") or ()
        if len(bbox) != 4:
            continue
        box = Box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if box.y1 < 80 or box.y0 > TOP_LIMIT:
            continue
        if box.width < 120 or box.height < 35:
            continue
        if box.width > page.rect.width * 0.90 or box.height > page.rect.height * 0.50:
            continue
        boxes.append(
            {
                "index": index,
                "bbox": box,
                "fill": None,
                "stroke": None,
                "width": None,
                "item_count": 1,
                "source_kind": "large-image",
                "xref": as_int(info.get("xref")),
            }
        )
    return boxes


def build_drawing_hull_boxes(drawings: list[dict[str, Any]], page: fitz.Page) -> list[dict[str, Any]]:
    small_visuals = [
        drawing["bbox"]
        for drawing in drawings
        if drawing.get("source_kind") != "large-image"
        and drawing["bbox"].width <= 90
        and drawing["bbox"].height <= 70
    ]
    if len(small_visuals) < 2:
        return []

    hull = union_boxes(small_visuals)
    if hull.width < 180 or hull.height < 30:
        return []
    panel_box = Box(
        max(0.0, hull.x0 - 8.0),
        max(80.0, hull.y0 - 92.0),
        min(float(page.rect.width), hull.x1 + 8.0),
        min(TOP_LIMIT, hull.y1 + 18.0),
    )
    return [
        {
            "index": -1,
            "bbox": panel_box,
            "fill": None,
            "stroke": None,
            "width": None,
            "item_count": len(small_visuals),
            "source_kind": "drawing-hull",
        }
    ]


def color_to_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [round(float(item), 4) for item in value]
    except TypeError:
        return None


def merge_similar_drawing_boxes(drawings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(drawings, key=lambda item: (item["bbox"].area, item["bbox"].y0, item["bbox"].x0), reverse=True)
    kept: list[dict[str, Any]] = []
    for drawing in ordered:
        box = drawing["bbox"]
        if any(box_overlap_ratio(box, existing["bbox"]) > 0.92 for existing in kept):
            continue
        kept.append(drawing)
    return sorted(kept, key=lambda item: (item["bbox"].x0, item["bbox"].y0))


def box_overlap_ratio(a: Box, b: Box) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = max(1.0, min(a.area, b.area))
    return overlap / smaller


def build_panels(tokens: list[Token], drawings: list[dict[str, Any]], page: fitz.Page) -> list[dict[str, Any]]:
    drawing_panels: list[dict[str, Any]] = []
    for drawing in drawings:
        box = drawing["bbox"]
        members = [token for token in tokens if box.contains_center(token.box, pad=3.0)]
        if len(members) < 3:
            continue
        drawing_panels.append(
            {
                "source": "drawing",
                "bbox": box,
                "tokens": members,
                "drawing": {
                    "index": drawing["index"],
                    "source_kind": drawing.get("source_kind", "drawing"),
                    "fill": drawing["fill"],
                    "stroke": drawing["stroke"],
                    "width": drawing["width"],
                    "item_count": drawing["item_count"],
                    "xref": drawing.get("xref"),
                },
            }
        )

    panels = remove_nested_panels(drawing_panels)
    assigned = {id(token) for panel in panels for token in panel["tokens"]}
    leftovers = [token for token in tokens if id(token) not in assigned and token.box.y0 >= 90.0]
    panels.extend(build_connected_component_panels(leftovers, page))
    panels = remove_duplicate_panels(panels)
    panels = merge_nearby_panels(panels)

    for panel in panels:
        panel["tokens"] = sorted(panel["tokens"], key=lambda item: (item.box.y0, item.box.x0))
        panel["rows"] = render_rows(panel["tokens"])
        panel["text"] = "\n".join(panel["rows"]).strip()
        panel["token_count"] = len(panel["tokens"])

    return sort_panels(panels)


def remove_nested_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for panel in sorted(panels, key=lambda item: item["bbox"].area, reverse=True):
        box = panel["bbox"]
        if any(box_overlap_ratio(box, existing["bbox"]) > 0.88 for existing in kept):
            continue
        kept.append(panel)
    return kept


def build_connected_component_panels(tokens: list[Token], page: fitz.Page) -> list[dict[str, Any]]:
    components: list[list[Token]] = []
    for token in sorted(tokens, key=lambda item: (item.box.x0, item.box.y0)):
        target_index: int | None = None
        expanded = token.box.padded(24.0)
        for index, component in enumerate(components):
            component_box = union_token_box(component).padded(28.0)
            if expanded.intersects(component_box):
                target_index = index
                break
        if target_index is None:
            components.append([token])
        else:
            components[target_index].append(token)

    changed = True
    while changed:
        changed = False
        merged: list[list[Token]] = []
        for component in components:
            component_box = union_token_box(component).padded(30.0)
            for existing in merged:
                if component_box.intersects(union_token_box(existing).padded(30.0)):
                    existing.extend(component)
                    changed = True
                    break
            else:
                merged.append(component)
        components = merged

    panels: list[dict[str, Any]] = []
    for component in components:
        if len(component) < 3:
            continue
        box = union_token_box(component).padded(8.0)
        if box.area < 1_500 or box.width > page.rect.width * 0.95:
            continue
        panels.append({"source": "coordinate-fallback", "bbox": box, "tokens": component})
    return panels


def merge_nearby_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed = True
    while changed:
        changed = False
        next_panels: list[dict[str, Any]] = []
        used: set[int] = set()
        for index, panel in enumerate(panels):
            if index in used:
                continue
            merged = panel
            for other_index in range(index + 1, len(panels)):
                if other_index in used:
                    continue
                other = panels[other_index]
                if should_merge_panels(merged, other):
                    merged = combine_panels(merged, other)
                    used.add(other_index)
                    changed = True
            used.add(index)
            next_panels.append(merged)
        panels = next_panels
    return panels


def should_merge_panels(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = left["bbox"]
    b = right["bbox"]
    left_kind = (left.get("drawing") or {}).get("source_kind")
    right_kind = (right.get("drawing") or {}).get("source_kind")
    horizontal_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vertical_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    y_overlap = axis_overlap_ratio(a.y0, a.y1, b.y0, b.y1)
    x_overlap = axis_overlap_ratio(a.x0, a.x1, b.x0, b.x1)

    if "drawing-hull" in {left_kind, right_kind} and x_overlap < 0.20:
        return False
    if y_overlap >= 0.35 and horizontal_gap <= 70.0:
        return True
    large_box: Box | None = None
    other_box: Box | None = None
    if left_kind == "large-image":
        large_box, other_box = a, b
    elif right_kind == "large-image":
        large_box, other_box = b, a
    if (
        large_box is not None
        and other_box is not None
        and other_box.y0 >= large_box.y1
        and x_overlap >= 0.25
        and 0.0 < vertical_gap <= 38.0
    ):
        return True
    return False


def axis_overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    smaller = max(1.0, min(a1 - a0, b1 - b0))
    return overlap / smaller


def combine_panels(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    source = left["source"] if left["source"] == "drawing" else right["source"]
    drawing = left.get("drawing") or right.get("drawing")
    return {
        "source": source,
        "bbox": left["bbox"].union(right["bbox"]),
        "tokens": list(dict.fromkeys([*left["tokens"], *right["tokens"]])),
        "drawing": drawing,
    }


def remove_duplicate_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(panels, key=lambda item: (0 if item["source"] == "drawing" else 1, -item["token_count"] if "token_count" in item else 0))
    kept: list[dict[str, Any]] = []
    for panel in ordered:
        box = panel["bbox"]
        token_ids = {id(token) for token in panel["tokens"]}
        duplicate = False
        for existing in kept:
            existing_ids = {id(token) for token in existing["tokens"]}
            if box_overlap_ratio(box, existing["bbox"]) > 0.70 or len(token_ids & existing_ids) >= max(2, len(token_ids) * 0.70):
                duplicate = True
                break
        if not duplicate:
            kept.append(panel)
    return kept


def union_token_box(tokens: list[Token]) -> Box:
    return Box(
        min(token.box.x0 for token in tokens),
        min(token.box.y0 for token in tokens),
        max(token.box.x1 for token in tokens),
        max(token.box.y1 for token in tokens),
    )


def sort_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not panels:
        return []
    columns: list[list[dict[str, Any]]] = []
    for panel in sorted(panels, key=lambda item: item["bbox"].x0):
        for column in columns:
            column_box = union_panel_box(column)
            if abs(panel["bbox"].x0 - column_box.x0) <= 80.0 or abs(panel["bbox"].cx - column_box.cx) <= max(80.0, min(panel["bbox"].width, column_box.width) * 0.45):
                column.append(panel)
                break
        else:
            columns.append([panel])

    ordered: list[dict[str, Any]] = []
    for column in sorted(columns, key=lambda col: union_panel_box(col).x0):
        ordered.extend(sorted(column, key=lambda item: item["bbox"].y0))

    for index, panel in enumerate(ordered, start=1):
        panel["panel_id"] = f"P{index}"
    return ordered


def union_panel_box(panels: list[dict[str, Any]]) -> Box:
    box = panels[0]["bbox"]
    for panel in panels[1:]:
        box = box.union(panel["bbox"])
    return box


def render_rows(tokens: list[Token]) -> list[str]:
    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda item: (item.box.cy, item.box.x0)):
        if not rows:
            rows.append([token])
            continue
        row = rows[-1]
        row_center = sum(item.box.cy for item in row) / len(row)
        tolerance = max(4.0, min(11.0, token.box.height * 0.75))
        if abs(token.box.cy - row_center) <= tolerance:
            row.append(token)
        else:
            rows.append([token])
    rendered: list[str] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.box.x0)
        parts: list[str] = []
        previous: Token | None = None
        for token in ordered:
            if previous is not None and token.box.x0 - previous.box.x1 > 18.0:
                parts.append(" | ")
            elif previous is not None:
                parts.append(" ")
            parts.append(token.text)
            previous = token
        rendered.append("".join(parts).strip())
    return rendered


def build_payload(pdf_path: Path) -> dict[str, Any]:
    with fitz.open(pdf_path) as document:
        page = document[PAGE_NUMBER - 1]
        tokens = extract_tokens(document, page, PAGE_NUMBER)
        drawings = extract_drawing_boxes(page)
        panels = build_panels(tokens, drawings, page)
        return {
            "pdf_path": pdf_path.as_posix(),
            "page_number": PAGE_NUMBER,
            "page_size": [round(float(page.rect.width), 3), round(float(page.rect.height), 3)],
            "method": "MCID/xref text runs plus drawing object panel candidates and coordinate fallback; no text-rule classification",
            "top_limit": TOP_LIMIT,
            "token_count": len(tokens),
            "drawing_candidate_count": len(drawings),
            "panel_count": len(panels),
            "panels": [
                {
                    "panel_id": panel["panel_id"],
                    "source": panel["source"],
                    "bbox": panel["bbox"].to_list(),
                    "token_count": panel["token_count"],
                    "drawing": panel.get("drawing"),
                    "rows": panel["rows"],
                    "text": panel["text"],
                    "tokens": [
                        {
                            "text": token.text,
                            "bbox": token.box.to_list(),
                            "mcids": list(token.mcids),
                            "xrefs": [xref for xref in token.xrefs if xref is not None],
                            "role": f"{token.block_role}/{token.leaf_role}",
                        }
                        for token in panel["tokens"]
                    ],
                    "mcids": sorted({mcid for token in panel["tokens"] for mcid in token.mcids}),
                    "xrefs": sorted({xref for token in panel["tokens"] for xref in token.xrefs if xref is not None}),
                }
                for panel in panels
            ],
            "drawing_candidates": [
                {
                    "index": drawing["index"],
                    "bbox": drawing["bbox"].to_list(),
                    "fill": drawing["fill"],
                    "stroke": drawing["stroke"],
                    "width": drawing["width"],
                    "item_count": drawing["item_count"],
                    "source_kind": drawing.get("source_kind", "drawing"),
                    "xref": drawing.get("xref"),
                }
                for drawing in drawings
            ],
        }


def write_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Page 16 Panel Segmentation Probe",
        "",
        f"- Method: {payload['method']}",
        f"- Tokens in top region: {payload['token_count']}",
        f"- Drawing candidates: {payload['drawing_candidate_count']}",
        f"- Panels: {payload['panel_count']}",
        "",
    ]
    for panel in payload["panels"]:
        lines.extend(
            [
                f"## {panel['panel_id']} ({panel['source']})",
                "",
                f"- BBox: `{panel['bbox']}`",
                f"- Token count: `{panel['token_count']}`",
                f"- MCIDs: `{panel['mcids'][:24]}{' ...' if len(panel['mcids']) > 24 else ''}`",
                f"- XRefs: `{panel['xrefs'][:12]}{' ...' if len(panel['xrefs']) > 12 else ''}`",
                "",
                "```text",
                panel["text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_html(payload: dict[str, Any]) -> str:
    page_width, _page_height = payload["page_size"]
    scale = 1.25
    canvas_width = int(page_width * scale)
    canvas_height = int(TOP_LIMIT * scale)
    overlays: list[str] = []
    colors = ["#d9485f", "#0f766e", "#2563eb", "#9333ea", "#b45309", "#047857"]
    for index, panel in enumerate(payload["panels"]):
        x0, y0, x1, y1 = panel["bbox"]
        color = colors[index % len(colors)]
        overlays.append(
            "<div class='panel' "
            f"style='left:{x0 * scale}px;top:{y0 * scale}px;width:{(x1 - x0) * scale}px;height:{(y1 - y0) * scale}px;border-color:{color};'>"
            f"<span style='background:{color};'>{escape(panel['panel_id'])}</span></div>"
        )
    token_marks: list[str] = []
    for panel in payload["panels"]:
        for token in panel["tokens"]:
            x0, y0, x1, y1 = token["bbox"]
            token_marks.append(
                "<div class='token' "
                f"style='left:{x0 * scale}px;top:{y0 * scale}px;width:{max(2, (x1 - x0) * scale)}px;height:{max(2, (y1 - y0) * scale)}px;'>"
                f"{escape(token['text'])}</div>"
            )
    panel_cards = "".join(
        "<section class='card'>"
        f"<h2>{escape(panel['panel_id'])} <span>{escape(panel['source'])}</span></h2>"
        f"<p>bbox={escape(str(panel['bbox']))} / tokens={panel['token_count']}</p>"
        f"<pre>{escape(panel['text'])}</pre>"
        "</section>"
        for panel in payload["panels"]
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Page 16 Panel Probe</title>
  <style>
    body {{ margin:0; padding:24px; font-family:Segoe UI, Malgun Gothic, sans-serif; background:#f6f7f9; color:#172033; }}
    h1 {{ margin:0 0 6px; font-size:24px; }}
    .meta {{ margin:0 0 18px; color:#5b667a; }}
    .layout {{ display:grid; grid-template-columns:{canvas_width}px minmax(360px, 1fr); gap:18px; align-items:start; }}
    .canvas {{ position:relative; width:{canvas_width}px; height:{canvas_height}px; background:#fff; border:1px solid #cfd6e3; overflow:hidden; }}
    .panel {{ position:absolute; border:2px solid; box-sizing:border-box; background:rgba(255,255,255,0.05); }}
    .panel span {{ position:absolute; left:-2px; top:-22px; color:#fff; font-size:12px; padding:2px 7px; }}
    .token {{ position:absolute; font-size:9px; color:#111827; outline:1px solid rgba(37,99,235,0.16); overflow:hidden; white-space:nowrap; }}
    .cards {{ display:grid; gap:12px; }}
    .card {{ background:#fff; border:1px solid #d9dfeb; padding:14px; }}
    .card h2 {{ margin:0 0 6px; font-size:16px; }}
    .card h2 span {{ color:#64748b; font-size:12px; font-weight:500; }}
    .card p {{ margin:0 0 10px; color:#64748b; font-size:12px; }}
    pre {{ margin:0; white-space:pre-wrap; line-height:1.5; font-size:13px; }}
  </style>
</head>
<body>
  <h1>Page 16 Panel Segmentation Probe</h1>
  <p class="meta">MCID/xref text runs + drawing candidates + coordinate fallback, no text-rule classification</p>
  <div class="layout">
    <div class="canvas">{''.join(token_marks)}{''.join(overlays)}</div>
    <div class="cards">{panel_cards}</div>
  </div>
</body>
</html>
"""


def main() -> None:
    pdf_path = resolve_pdf_path()
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(pdf_path)
    json_path = output_dir / "miraeasset_page16_panel_probe.json"
    md_path = output_dir / "miraeasset_page16_panel_probe.md"
    html_path = output_dir / "miraeasset_page16_panel_probe.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(write_markdown(payload), encoding="utf-8")
    html_path.write_text(write_html(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": json_path.as_posix(),
                "md_path": md_path.as_posix(),
                "html_path": html_path.as_posix(),
                "panel_count": payload["panel_count"],
                "drawing_candidate_count": payload["drawing_candidate_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
