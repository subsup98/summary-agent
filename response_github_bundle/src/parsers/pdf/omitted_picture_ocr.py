from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


OPTIONAL_DEPENDENCY_ROOT = Path(__file__).resolve().parents[3] / ".deps_parser_ext"
PAGE_HEADING_PATTERN = re.compile(r"^#\s+Page\s+(\d+)\s*$", re.IGNORECASE)
OMITTED_PICTURE_PATTERN = re.compile(
    r"\*\*==>\s*picture\s*\[(?P<width>\d+)\s*x\s*(?P<height>\d+)\]\s*intentionally omitted\s*<==\*\*",
    re.IGNORECASE,
)
MIN_OCR_PLACEHOLDER_WIDTH = 80
MIN_OCR_PLACEHOLDER_HEIGHT = 40
MIN_OCR_PLACEHOLDER_AREA = 6000


@dataclass
class OmittedPicturePlaceholder:
    page_number: int
    page_ordinal: int
    width: int
    height: int
    raw_text: str
    line_number: int


@dataclass
class OmittedPictureMatch:
    page_number: int
    page_ordinal: int
    placeholder: OmittedPicturePlaceholder
    image_element_id: str | None
    bbox: list[float] | None
    image_metadata: dict[str, Any]
    match_status: str
    ocr_text: str | None = None
    resolved_by: str | None = None


def extract_omitted_picture_placeholders(markdown: str) -> list[OmittedPicturePlaceholder]:
    placeholders: list[OmittedPicturePlaceholder] = []
    page_number = 1
    page_ordinals: dict[int, int] = {}

    for line_number, line in enumerate(markdown.splitlines(), start=1):
        heading_match = PAGE_HEADING_PATTERN.match(line.strip())
        if heading_match:
            page_number = int(heading_match.group(1))
            continue

        match = OMITTED_PICTURE_PATTERN.search(line)
        if match is None:
            continue

        page_ordinals[page_number] = page_ordinals.get(page_number, 0) + 1
        placeholders.append(
            OmittedPicturePlaceholder(
                page_number=page_number,
                page_ordinal=page_ordinals[page_number],
                width=int(match.group("width")),
                height=int(match.group("height")),
                raw_text=match.group(0),
                line_number=line_number,
            )
        )

    return placeholders


def match_omitted_pictures_to_images(markdown: str, pages: list[dict[str, Any]]) -> list[OmittedPictureMatch]:
    placeholders = extract_omitted_picture_placeholders(markdown)
    images_by_page = _build_sorted_image_lookup(pages)
    matches: list[OmittedPictureMatch] = []

    for placeholder in placeholders:
        page_images = images_by_page.get(placeholder.page_number, [])
        image_index = placeholder.page_ordinal - 1
        image = page_images[image_index] if image_index < len(page_images) else None
        if image is None:
            matches.append(
                OmittedPictureMatch(
                    page_number=placeholder.page_number,
                    page_ordinal=placeholder.page_ordinal,
                    placeholder=placeholder,
                    image_element_id=None,
                    bbox=None,
                    image_metadata={},
                    match_status="unmatched",
                )
            )
            continue

        matches.append(
            OmittedPictureMatch(
                page_number=placeholder.page_number,
                page_ordinal=placeholder.page_ordinal,
                placeholder=placeholder,
                image_element_id=str(image.get("element_id") or ""),
                bbox=_coerce_bbox(image.get("bbox")),
                image_metadata=dict(image.get("metadata") or {}),
                match_status="matched",
            )
        )

    for match in matches:
        if match.bbox is None:
            continue
        if _bbox_size_mismatch(match.placeholder, match.bbox):
            match.match_status = "unmatched"
            match.resolved_by = "rejected-initial-image-size-mismatch"
            match.bbox = None

    return matches


def run_targeted_ocr(
    pdf_path: Path,
    matches: list[OmittedPictureMatch],
    *,
    dpi: int = 216,
    ocr_backend: RapidOcrBackend | None = None,
) -> list[OmittedPictureMatch]:
    ocr_backend = ocr_backend or RapidOcrBackend()

    with fitz.open(pdf_path) as document:
        page_candidates = _build_page_graphic_candidates(document)
        used_candidates: set[tuple[int, int, int, int, int]] = set()

        for match in matches:
            if match.bbox:
                used_candidates.add(_bbox_key(match.page_number, match.bbox))

        for match in matches:
            if not match.bbox:
                candidate_page_number, candidate_bbox, candidate_source = _resolve_fallback_candidate(
                    match=match,
                    matches=matches,
                    page_candidates=page_candidates,
                    used_candidates=used_candidates,
                    document_page_count=document.page_count,
                )
                if candidate_bbox is not None and candidate_page_number is not None:
                    match.page_number = candidate_page_number
                    match.bbox = candidate_bbox
                    match.image_metadata = {
                        **match.image_metadata,
                        "fallback_candidate": True,
                        "fallback_source": candidate_source,
                    }
                    match.match_status = "matched"
                    match.resolved_by = candidate_source
                    used_candidates.add(_bbox_key(candidate_page_number, candidate_bbox))

            if match.match_status != "matched" or not match.bbox:
                continue
            if _should_skip_small_placeholder(match.placeholder, match.bbox):
                match.match_status = "skipped_small"
                match.resolved_by = "below-ocr-size-threshold"
                continue

            page = document[match.page_number - 1]
            pixmap = _render_bbox(page, match.bbox, dpi=dpi)
            match.ocr_text = ocr_backend.extract_text(pixmap)
            if match.ocr_text is None:
                match.match_status = "ocr_failed"
            elif not match.ocr_text.strip():
                match.match_status = "ocr_empty"
            else:
                match.match_status = "ocr_complete"
                match.resolved_by = match.resolved_by or "image-element"
    return matches


class RapidOcrBackend:
    def __init__(self) -> None:
        self.engine_name = "unknown"
        self._engine = self._build_engine()

    def extract_text(self, pixmap: fitz.Pixmap) -> str | None:
        try:
            numpy = self._load_optional_module("numpy")
            image = numpy.frombuffer(pixmap.samples, dtype=numpy.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            result = self._engine(image)
        except Exception:
            return None

        txts = getattr(result, "txts", None)
        if txts is not None:
            lines = [str(item).strip() for item in txts if str(item).strip()]
            return "\n".join(lines).strip()

        if isinstance(result, tuple):
            result = result[0]

        lines: list[str] = []
        for item in result or []:
            if not item or len(item) < 2:
                continue
            text_payload = item[1]
            text = text_payload[0] if isinstance(text_payload, (list, tuple)) and text_payload else text_payload
            normalized = str(text or "").strip()
            if normalized:
                lines.append(normalized)
        return "\n".join(lines).strip()

    def _build_engine(self) -> Any:
        try:
            rapidocr_module = self._load_optional_module("rapidocr")
            rapid_ocr_cls = getattr(rapidocr_module, "RapidOCR")
            engine_type = getattr(rapidocr_module, "EngineType")
            lang_rec = getattr(rapidocr_module, "LangRec")
            model_type = getattr(rapidocr_module, "ModelType")
            ocr_version = getattr(rapidocr_module, "OCRVersion")
            self.engine_name = "rapidocr[korean-ppocrv5]"
            return rapid_ocr_cls(
                params={
                    "Global.log_level": "error",
                    "Global.use_cls": True,
                    "Global.min_height": 8,
                    "Det.limit_side_len": 2048,
                    "Rec.engine_type": engine_type.ONNXRUNTIME,
                    "Rec.lang_type": lang_rec.KOREAN,
                    "Rec.model_type": model_type.MOBILE,
                    "Rec.ocr_version": ocr_version.PPOCRV5,
                }
            )
        except Exception:
            raise RuntimeError(
                "Korean RapidOCR backend is unavailable. Refusing to fall back to an unspecified OCR model."
            )

    def _load_optional_module(self, module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as primary_error:
            if OPTIONAL_DEPENDENCY_ROOT.exists():
                dependency_root = str(OPTIONAL_DEPENDENCY_ROOT)
                if dependency_root not in sys.path:
                    sys.path.insert(0, dependency_root)
                return importlib.import_module(module_name)
            raise primary_error


def _build_sorted_image_lookup(pages: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    images_by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        page_number = int(page.get("page_number") or 0)
        elements = page.get("elements") or []
        images = [element for element in elements if element.get("element_type") == "image" and _coerce_bbox(element.get("bbox"))]
        images.sort(key=_image_sort_key)
        images_by_page[page_number] = images
    return images_by_page


def _build_page_graphic_candidates(document: fitz.Document) -> dict[int, list[dict[str, Any]]]:
    candidates_by_page: dict[int, list[dict[str, Any]]] = {}
    for page_index in range(document.page_count):
        page = document[page_index]
        page_number = page_index + 1
        candidates: list[dict[str, Any]] = []

        for info in page.get_image_info(xrefs=True):
            bbox = _coerce_bbox(info.get("bbox"))
            if not _is_significant_bbox(bbox):
                continue
            candidates.append(
                {
                    "page_number": page_number,
                    "bbox": bbox,
                    "source": "page-image-info",
                    "width_hint": float(info.get("width") or 0),
                    "height_hint": float(info.get("height") or 0),
                }
            )

        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            bbox = _coerce_bbox([rect.x0, rect.y0, rect.x1, rect.y1] if rect else None)
            if not _is_significant_bbox(bbox):
                continue
            candidates.append(
                {
                    "page_number": page_number,
                    "bbox": bbox,
                    "source": "page-drawing-rect",
                    "width_hint": float(bbox[2] - bbox[0]),
                    "height_hint": float(bbox[3] - bbox[1]),
                }
            )

        candidates = _dedupe_candidates(candidates)
        candidates = _drop_container_candidates(candidates)
        candidates.sort(key=lambda item: _image_sort_key(item))
        candidates_by_page[page_number] = candidates
    return candidates_by_page


def _image_sort_key(element: dict[str, Any]) -> tuple[float, float, float, int]:
    bbox = _coerce_bbox(element.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    area = width * height
    order = int(element.get("order") or 0)
    return (round(float(bbox[1]), 2), round(float(bbox[0]), 2), -area, order)


def _resolve_fallback_candidate(
    *,
    match: OmittedPictureMatch,
    matches: list[OmittedPictureMatch],
    page_candidates: dict[int, list[dict[str, Any]]],
    used_candidates: set[tuple[int, int, int, int, int]],
    document_page_count: int,
) -> tuple[int | None, list[float] | None, str | None]:
    page_heading_count = len({item.placeholder.page_number for item in matches})
    if page_heading_count > 1:
        scoped_pages = [match.placeholder.page_number]
    else:
        scoped_pages = list(range(1, document_page_count + 1))

    best_choice: tuple[float, int, list[float], str] | None = None
    for page_number in scoped_pages:
        for candidate in page_candidates.get(page_number, []):
            bbox = candidate["bbox"]
            key = _bbox_key(page_number, bbox)
            if key in used_candidates:
                continue
            score = _candidate_score(match.placeholder, candidate)
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, page_number, bbox, str(candidate.get("source") or "fallback"))

    if best_choice is None:
        return None, None, None
    _, page_number, bbox, source = best_choice
    return page_number, bbox, source


def _candidate_score(placeholder: OmittedPicturePlaceholder, candidate: dict[str, Any]) -> float:
    bbox = candidate["bbox"]
    candidate_width = max(1.0, float(bbox[2] - bbox[0]))
    candidate_height = max(1.0, float(bbox[3] - bbox[1]))
    width_ratio = abs(candidate_width - placeholder.width) / max(float(placeholder.width), 1.0)
    height_ratio = abs(candidate_height - placeholder.height) / max(float(placeholder.height), 1.0)
    aspect_ratio = abs((candidate_width / candidate_height) - (placeholder.width / max(float(placeholder.height), 1.0)))
    return round((width_ratio * 0.35) + (height_ratio * 0.35) + (aspect_ratio * 0.3), 6)


def _bbox_size_mismatch(placeholder: OmittedPicturePlaceholder, bbox: list[float]) -> bool:
    candidate_width = max(1.0, float(bbox[2] - bbox[0]))
    candidate_height = max(1.0, float(bbox[3] - bbox[1]))
    width_ratio = candidate_width / max(float(placeholder.width), 1.0)
    height_ratio = candidate_height / max(float(placeholder.height), 1.0)
    return width_ratio < 0.45 or height_ratio < 0.45


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for candidate in candidates:
        bbox = candidate["bbox"]
        bbox_key = tuple(int(round(value)) for value in bbox)
        if bbox_key in seen:
            continue
        seen.add(bbox_key)
        deduped.append(candidate)
    return deduped


def _drop_container_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        bbox = fitz.Rect(candidate["bbox"])
        contains = 0
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_bbox = fitz.Rect(other["bbox"])
            if bbox.contains(other_bbox) and other_bbox.get_area() < bbox.get_area():
                contains += 1
        if contains >= 2:
            continue
        filtered.append(candidate)
    return filtered


def _is_significant_bbox(bbox: list[float] | None) -> bool:
    if bbox is None:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return width >= 40.0 and height >= 30.0


def _should_skip_small_placeholder(placeholder: OmittedPicturePlaceholder, bbox: list[float]) -> bool:
    placeholder_width = max(float(placeholder.width), 0.0)
    placeholder_height = max(float(placeholder.height), 0.0)
    placeholder_area = placeholder_width * placeholder_height
    bbox_width = max(0.0, float(bbox[2] - bbox[0]))
    bbox_height = max(0.0, float(bbox[3] - bbox[1]))
    bbox_area = bbox_width * bbox_height
    return (
        placeholder_width < MIN_OCR_PLACEHOLDER_WIDTH
        or placeholder_height < MIN_OCR_PLACEHOLDER_HEIGHT
        or placeholder_area < MIN_OCR_PLACEHOLDER_AREA
        or bbox_width < MIN_OCR_PLACEHOLDER_WIDTH
        or bbox_height < MIN_OCR_PLACEHOLDER_HEIGHT
        or bbox_area < MIN_OCR_PLACEHOLDER_AREA
    )


def _coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _bbox_key(page_number: int, bbox: list[float]) -> tuple[int, int, int, int, int]:
    return (page_number, *(int(round(value)) for value in bbox))


def _render_bbox(page: fitz.Page, bbox: list[float], *, dpi: int) -> fitz.Pixmap:
    scale = max(1.0, float(dpi) / 72.0)
    rect = fitz.Rect(bbox)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    if pixmap.n > 3:
        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
    return pixmap
