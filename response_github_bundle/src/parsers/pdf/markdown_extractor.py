from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import fitz
import pymupdf4llm

from src.parsers.common.models import DocumentClassification, DocumentIssue
from src.parsers.pdf.omitted_picture_ocr import RapidOcrBackend, match_omitted_pictures_to_images, run_targeted_ocr
from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor
from src.preprocess.markdown_cleaner import PageLayoutContext, PdfMarkdownPreprocessor, PreprocessTextBlock


OPTIONAL_DEPENDENCY_ROOT = Path(__file__).resolve().parents[3] / ".deps_parser_ext"

PDF_MARKDOWN_STRATEGIES = (
    "metadata-selected",
    "structtree-actualtext",
    "pymupdf4llm",
    "fitz-text",
    "pdftext",
    "pypdfium2",
    "camelot-hybrid",
)


@dataclass
class MarkdownExtractionResult:
    strategy_name: str
    applied_strategy: str
    markdown: str
    raw_markdown: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[DocumentIssue] = field(default_factory=list)
    elapsed_ms: float = 0.0


def select_pdf_markdown_strategy(classification: DocumentClassification) -> str:
    configured = classification.pdf_parser_strategy
    if configured:
        return configured

    producer = str(classification.metadata.get("producer") or "").lower()
    if "powerpoint" in producer or "ppt" in producer:
        return "structtree-actualtext"
    return "pymupdf4llm"


class PdfMarkdownExtractor:
    def __init__(self, *, enable_omitted_picture_ocr: bool = True) -> None:
        self.structtree_extractor = PowerPointStructTreeExtractor()
        self.preprocessor = PdfMarkdownPreprocessor()
        self.enable_omitted_picture_ocr = enable_omitted_picture_ocr

    def extract(
        self,
        path: Path,
        document: fitz.Document,
        classification: DocumentClassification,
        strategy_name: str = "metadata-selected",
    ) -> MarkdownExtractionResult:
        if strategy_name == "metadata-selected":
            return self._extract_selected(path, document, classification)
        if strategy_name not in PDF_MARKDOWN_STRATEGIES:
            raise ValueError(f"Unsupported PDF markdown strategy: {strategy_name}")
        return self._extract_raw(path, document, strategy_name)

    def _extract_selected(
        self,
        path: Path,
        document: fitz.Document,
        classification: DocumentClassification,
    ) -> MarkdownExtractionResult:
        selected_strategy = select_pdf_markdown_strategy(classification)
        attempts: list[str] = []
        issues: list[DocumentIssue] = []
        total_elapsed_ms = 0.0

        result = self._extract_raw(path, document, selected_strategy)
        attempts.append(selected_strategy)
        issues.extend(result.issues)
        total_elapsed_ms += result.elapsed_ms

        if self._is_usable_markdown(result.markdown):
            if selected_strategy == "structtree-actualtext":
                issues.append(
                    DocumentIssue(
                        code="mcid_markdown_used",
                        message="Producer metadata selected StructTree/MCID extraction for this PDF.",
                        severity="info",
                    )
                )
            selected_result = MarkdownExtractionResult(
                strategy_name="metadata-selected",
                applied_strategy=result.applied_strategy,
                markdown=result.markdown,
                raw_markdown=result.raw_markdown or result.markdown,
                metadata={
                    **result.metadata,
                    "selected_strategy": selected_strategy,
                    "attempted_strategies": attempts,
                    "fallback_used": False,
                },
                issues=issues,
                elapsed_ms=total_elapsed_ms,
            )
            return self._apply_preprocessing(selected_result, path, classification, document)

        fallback_order = ["pymupdf4llm", "fitz-text"] if selected_strategy == "structtree-actualtext" else ["fitz-text"]
        for fallback_strategy in fallback_order:
            issues.append(
                DocumentIssue(
                    code="markdown_strategy_fallback",
                    message=f"{selected_strategy} output was insufficient; falling back to {fallback_strategy}.",
                    severity="warning",
                )
            )
            fallback = self._extract_raw(path, document, fallback_strategy)
            attempts.append(fallback_strategy)
            issues.extend(fallback.issues)
            total_elapsed_ms += fallback.elapsed_ms
            if self._is_usable_markdown(fallback.markdown):
                selected_result = MarkdownExtractionResult(
                    strategy_name="metadata-selected",
                    applied_strategy=fallback.applied_strategy,
                    markdown=fallback.markdown,
                    raw_markdown=fallback.raw_markdown or fallback.markdown,
                    metadata={
                        **fallback.metadata,
                        "selected_strategy": selected_strategy,
                        "attempted_strategies": attempts,
                        "fallback_used": True,
                    },
                    issues=issues,
                    elapsed_ms=total_elapsed_ms,
                )
                return self._apply_preprocessing(selected_result, path, classification, document)
            result = fallback

        selected_result = MarkdownExtractionResult(
            strategy_name="metadata-selected",
            applied_strategy=result.applied_strategy,
            markdown=result.markdown,
            raw_markdown=result.raw_markdown or result.markdown,
            metadata={
                **result.metadata,
                "selected_strategy": selected_strategy,
                "attempted_strategies": attempts,
                "fallback_used": len(attempts) > 1,
            },
            issues=issues,
            elapsed_ms=total_elapsed_ms,
        )
        return self._apply_preprocessing(selected_result, path, classification, document)

    def _extract_raw(self, path: Path, document: fitz.Document, strategy_name: str) -> MarkdownExtractionResult:
        started = perf_counter()
        metadata: dict[str, Any]
        issues: list[DocumentIssue] = []
        markdown = ""

        if strategy_name == "structtree-actualtext":
            markdown, metadata = self.structtree_extractor.extract_markdown(document)
            if not metadata.get("used"):
                issues.append(
                    DocumentIssue(
                        code="mcid_markdown_unavailable",
                        message=f"StructTree/MCID extraction unavailable: {metadata.get('reason', 'unknown')}.",
                        severity="info",
                    )
                )
        elif strategy_name == "pymupdf4llm":
            try:
                markdown = pymupdf4llm.to_markdown(str(path))
                metadata = {"used": bool(markdown.strip()), "source": "pymupdf4llm"}
                if not markdown.strip():
                    issues.append(
                        DocumentIssue(
                            code="pymupdf4llm_empty",
                            message="pymupdf4llm returned empty markdown.",
                            severity="warning",
                        )
                    )
            except Exception as error:
                metadata = {"used": False, "source": "pymupdf4llm", "reason": str(error)}
                issues.append(
                    DocumentIssue(
                        code="pymupdf4llm_failed",
                        message=f"pymupdf4llm markdown extraction failed: {error}",
                        severity="warning",
                    )
                )
        elif strategy_name == "fitz-text":
            page_texts = [page.get_text("text", sort=True).strip() for page in document]
            markdown = self._page_markdown(page_texts)
            metadata = {"used": bool(markdown.strip()), "source": "fitz-text"}
        elif strategy_name == "pdftext":
            markdown, metadata, issues = self._extract_pdftext_markdown(path)
        elif strategy_name == "pypdfium2":
            markdown, metadata, issues = self._extract_pypdfium2_markdown(path)
        elif strategy_name == "camelot-hybrid":
            markdown, metadata, issues = self._extract_camelot_hybrid_markdown(path, document)
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unsupported PDF markdown strategy: {strategy_name}")

        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        return MarkdownExtractionResult(
            strategy_name=strategy_name,
            applied_strategy=strategy_name,
            markdown=markdown,
            raw_markdown=markdown,
            metadata=metadata,
            issues=issues,
            elapsed_ms=elapsed_ms,
        )

    def _apply_preprocessing(
        self,
        result: MarkdownExtractionResult,
        path: Path,
        classification: DocumentClassification,
        document: fitz.Document,
    ) -> MarkdownExtractionResult:
        markdown_with_ocr, ocr_metadata, ocr_issues = self._apply_omitted_picture_ocr(path, result.markdown, document)
        preprocess = self.preprocessor.preprocess(
            markdown_with_ocr,
            classification,
            page_contexts=self._build_page_contexts(document),
        )
        metadata = {**result.metadata, "omitted_picture_ocr": ocr_metadata, "preprocess": preprocess.metadata}
        return MarkdownExtractionResult(
            strategy_name=result.strategy_name,
            applied_strategy=result.applied_strategy,
            markdown=preprocess.markdown,
            raw_markdown=result.raw_markdown or result.markdown,
            metadata=metadata,
            issues=[*result.issues, *ocr_issues, *preprocess.issues],
            elapsed_ms=result.elapsed_ms,
        )

    def _apply_omitted_picture_ocr(
        self,
        path: Path,
        markdown: str,
        document: fitz.Document,
    ) -> tuple[str, dict[str, Any], list[DocumentIssue]]:
        if not self.enable_omitted_picture_ocr:
            return markdown, {"enabled": False, "placeholder_count": 0, "ocr_complete_count": 0, "changed": False}, []

        pages = self._build_ocr_page_elements(document)
        matches = match_omitted_pictures_to_images(markdown, pages)
        if not matches:
            return markdown, {"enabled": True, "placeholder_count": 0, "ocr_complete_count": 0, "changed": False}, []

        ocr_backend = RapidOcrBackend()
        results = run_targeted_ocr(path, matches, ocr_backend=ocr_backend)
        merged_markdown = self._merge_ocr_results_into_markdown(markdown, results)
        ocr_complete_count = sum(1 for item in results if item.match_status == "ocr_complete" and (item.ocr_text or "").strip())
        metadata = {
            "enabled": True,
            "placeholder_count": len(results),
            "matched_count": sum(1 for item in results if item.bbox),
            "ocr_complete_count": ocr_complete_count,
            "ocr_engine": ocr_backend.engine_name,
            "items": [
                {
                    "page_number": item.page_number,
                    "page_ordinal": item.page_ordinal,
                    "match_status": item.match_status,
                    "resolved_by": item.resolved_by,
                    "bbox": item.bbox,
                    "ocr_text": item.ocr_text,
                }
                for item in results
            ],
            "changed": merged_markdown != markdown,
        }
        issues: list[DocumentIssue] = []
        if ocr_complete_count:
            issues.append(
                DocumentIssue(
                    code="omitted_picture_ocr_merged",
                    message=f"Merged OCR text for {ocr_complete_count} omitted picture region(s) into markdown.",
                    severity="info",
                )
            )
        return merged_markdown, metadata, issues

    def _build_ocr_page_elements(self, document: fitz.Document) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        for page_index in range(document.page_count):
            page = document[page_index]
            elements: list[dict[str, Any]] = []
            try:
                image_infos = sorted(page.get_image_info(xrefs=True), key=lambda item: item.get("number", 0))
            except Exception:
                image_infos = []
            for order, info in enumerate(image_infos, start=1):
                bbox = info.get("bbox") or ()
                if len(bbox) != 4:
                    continue
                elements.append(
                    {
                        "element_id": f"p{page_index + 1}-ocrimg-{order}",
                        "element_type": "image",
                        "page_number": page_index + 1,
                        "order": order,
                        "bbox": [float(value) for value in bbox],
                        "metadata": {
                            "width": info.get("width"),
                            "height": info.get("height"),
                            "xref": info.get("xref"),
                        },
                    }
                )
            pages.append({"page_number": page_index + 1, "elements": elements})
        return pages

    def _merge_ocr_results_into_markdown(self, markdown: str, results: list[Any]) -> str:
        page_number = 1
        page_ordinal = 0
        merged_lines: list[str] = []
        result_lookup = {(item.placeholder.page_number, item.placeholder.page_ordinal): item for item in results}

        for line in markdown.splitlines():
            merged_lines.append(line)
            stripped = line.strip()
            heading = stripped.lower().startswith("# page ")
            if heading:
                try:
                    page_number = int(stripped.split()[-1])
                    page_ordinal = 0
                except ValueError:
                    pass
                continue

            if "intentionally omitted <==**" not in stripped:
                continue

            page_ordinal += 1
            result = result_lookup.get((page_number, page_ordinal))
            if result is None or result.match_status != "ocr_complete" or not (result.ocr_text or "").strip():
                continue

            merged_lines.append("")
            merged_lines.append("**----- Start of picture ocr -----**<br>")
            for ocr_line in str(result.ocr_text).splitlines():
                normalized = ocr_line.strip()
                if normalized:
                    merged_lines.append(f"{normalized}<br>")
            merged_lines.append("**----- End of picture ocr -----**<br>")

        return "\n".join(merged_lines)

    def _build_page_contexts(self, document: fitz.Document) -> list[PageLayoutContext]:
        contexts: list[PageLayoutContext] = []
        for page_index in range(document.page_count):
            page = document[page_index]
            text_blocks: list[PreprocessTextBlock] = []
            for block in page.get_text("blocks", sort=True):
                if len(block) < 5:
                    continue
                text = str(block[4] or "").replace("\r", "\n").strip()
                if not text:
                    continue
                text_blocks.append(
                    PreprocessTextBlock(
                        text=text,
                        bbox=[float(block[0]), float(block[1]), float(block[2]), float(block[3])],
                    )
                )

            image_bboxes: list[list[float]] = []
            try:
                for image_info in page.get_image_info(xrefs=True):
                    bbox = image_info.get("bbox") or ()
                    if len(bbox) == 4:
                        image_bboxes.append([float(value) for value in bbox])
            except Exception:
                image_bboxes = []

            table_bboxes: list[list[float]] = []
            try:
                for table in page.find_tables().tables:
                    bbox = tuple(float(value) for value in table.bbox)
                    if len(bbox) == 4:
                        table_bboxes.append(list(bbox))
            except Exception:
                table_bboxes = []

            contexts.append(
                PageLayoutContext(
                    page_number=page_index + 1,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    text_blocks=text_blocks,
                    image_bboxes=image_bboxes,
                    table_bboxes=table_bboxes,
                )
            )
        return contexts

    def _is_usable_markdown(self, markdown: str) -> bool:
        return len(markdown.strip()) >= 50

    def _extract_pdftext_markdown(self, path: Path) -> tuple[str, dict[str, Any], list[DocumentIssue]]:
        issues: list[DocumentIssue] = []
        try:
            extraction = self._load_optional_module("pdftext.extraction")
            page_texts = extraction.paginated_plain_text_output(str(path), sort=True, hyphens=False, workers=1)
            markdown = self._page_markdown(page_texts)
            metadata = {
                "used": bool(markdown.strip()),
                "source": "pdftext",
                "page_count": len(page_texts),
            }
            if not markdown.strip():
                issues.append(
                    DocumentIssue(
                        code="pdftext_empty",
                        message="pdftext returned empty markdown.",
                        severity="warning",
                    )
                )
            return markdown, metadata, issues
        except Exception as error:
            return "", {"used": False, "source": "pdftext", "reason": str(error)}, [
                DocumentIssue(
                    code="pdftext_failed",
                    message=f"pdftext markdown extraction failed: {error}",
                    severity="warning",
                )
            ]

    def _extract_pypdfium2_markdown(self, path: Path) -> tuple[str, dict[str, Any], list[DocumentIssue]]:
        issues: list[DocumentIssue] = []
        pdfium = self._load_optional_module("pypdfium2")
        document = pdfium.PdfDocument(str(path))
        page_texts: list[str] = []
        try:
            for page_index in range(len(document)):
                page = document[page_index]
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_bounded() or ""
                finally:
                    text_page.close()
                    page.close()
                page_texts.append(self._normalize_page_text(text))
        except Exception as error:
            return "", {"used": False, "source": "pypdfium2", "reason": str(error)}, [
                DocumentIssue(
                    code="pypdfium2_failed",
                    message=f"pypdfium2 markdown extraction failed: {error}",
                    severity="warning",
                )
            ]
        finally:
            document.close()

        markdown = self._page_markdown(page_texts)
        metadata = {
            "used": bool(markdown.strip()),
            "source": "pypdfium2",
            "page_count": len(page_texts),
        }
        if not markdown.strip():
            issues.append(
                DocumentIssue(
                    code="pypdfium2_empty",
                    message="pypdfium2 returned empty markdown.",
                    severity="warning",
                )
            )
        return markdown, metadata, issues

    def _extract_camelot_hybrid_markdown(
        self,
        path: Path,
        document: fitz.Document,
    ) -> tuple[str, dict[str, Any], list[DocumentIssue]]:
        issues: list[DocumentIssue] = []
        page_texts = [page.get_text("text", sort=True).strip() for page in document]
        tables_by_page: dict[int, list[str]] = {}
        flavor_used = "lattice"

        try:
            camelot = self._load_optional_module("camelot")
            page_spec = f"1-{document.page_count}"
            tables = camelot.read_pdf(str(path), pages=page_spec, suppress_stdout=True, flavor="lattice")
            if getattr(tables, "n", 0) == 0:
                flavor_used = "stream"
                tables = camelot.read_pdf(str(path), pages=page_spec, suppress_stdout=True, flavor="stream")

            for table in tables:
                page_number = int(str(getattr(table, "page", "1")).split(",")[0])
                rows = table.df.fillna("").values.tolist()
                markdown_table = self._rows_to_markdown(rows)
                if markdown_table:
                    tables_by_page.setdefault(page_number, []).append(markdown_table)
        except Exception as error:
            issues.append(
                DocumentIssue(
                    code="camelot_hybrid_failed",
                    message=f"Camelot table extraction failed: {error}",
                    severity="warning",
                )
            )

        parts: list[str] = []
        for page_number, page_text in enumerate(page_texts, start=1):
            parts.append(f"# Page {page_number}\n")
            if page_text:
                parts.append(page_text)
            for table_markdown in tables_by_page.get(page_number, []):
                if parts[-1] and not parts[-1].endswith("\n"):
                    parts.append("")
                parts.append(table_markdown)
            parts.append("")

        markdown = "\n".join(parts).strip()
        metadata = {
            "used": bool(markdown.strip()),
            "source": "camelot-hybrid",
            "table_count": sum(len(items) for items in tables_by_page.values()),
            "flavor": flavor_used,
        }
        if not markdown.strip():
            issues.append(
                DocumentIssue(
                    code="camelot_hybrid_empty",
                    message="Camelot hybrid returned empty markdown.",
                    severity="warning",
                )
            )
        return markdown, metadata, issues

    def _page_markdown(self, page_texts: list[str]) -> str:
        parts: list[str] = []
        for page_number, page_text in enumerate(page_texts, start=1):
            parts.append(f"# Page {page_number}\n")
            if page_text.strip():
                parts.append(page_text.strip())
            parts.append("")
        return "\n".join(parts).strip()

    def _rows_to_markdown(self, rows: list[list[Any]]) -> str:
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            return ""

        normalized: list[list[str]] = []
        for row in rows:
            normalized.append([(str(cell) if cell is not None else "").replace("\n", "<br>").strip() for cell in row] + [""] * (width - len(row)))

        header = normalized[0]
        separator = ["---"] * width
        body = normalized[1:] or [[""] * width]
        markdown_rows = [header, separator, *body]
        return "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)

    def _normalize_page_text(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

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
