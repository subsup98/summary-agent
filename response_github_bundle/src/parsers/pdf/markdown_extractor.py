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
                page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
                if isinstance(page_chunks, list) and page_chunks:
                    page_texts = [str(chunk.get("text") or "").strip() for chunk in page_chunks]
                    markdown = self._page_markdown(page_texts)
                    metadata = {
                        "used": bool(markdown.strip()),
                        "source": "pymupdf4llm",
                        "page_chunks": True,
                        "page_count": len(page_chunks),
                    }
                else:
                    markdown = pymupdf4llm.to_markdown(str(path))
                    metadata = {"used": bool(markdown.strip()), "source": "pymupdf4llm", "page_chunks": False}
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
        mcid_lookup = self.structtree_extractor.build_mcid_lookup(document)
        for page_index in range(document.page_count):
            page = document[page_index]
            elements: list[dict[str, Any]] = []
            resource_map = self._build_image_resource_map(page)
            draw_ops = self._extract_image_draw_operations(page, set(resource_map))
            try:
                image_infos = sorted(page.get_image_info(xrefs=True), key=lambda item: item.get("number", 0))
            except Exception:
                image_infos = []
            for order, info in enumerate(image_infos, start=1):
                bbox = info.get("bbox") or ()
                if len(bbox) != 4:
                    continue
                draw_op = draw_ops[order - 1] if order - 1 < len(draw_ops) else {}
                mcid = draw_op.get("mcid")
                mcid_matches = mcid_lookup.get(page_index + 1, {}).get(mcid, []) if isinstance(mcid, int) else []
                mcid_text = self._merge_mcid_text(mcid_matches)
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
                            "mcid": mcid,
                            "mcid_text": mcid_text or None,
                            "mcid_roles": self._role_labels(mcid_matches),
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
        seen_merge_keys: set[tuple[int, tuple[int, int, int, int] | None, str]] = set()

        for line in markdown.splitlines():
            stripped = line.strip()
            heading = stripped.lower().startswith("# page ")
            if heading:
                merged_lines.append(line)
                try:
                    page_number = int(stripped.split()[-1])
                    page_ordinal = 0
                except ValueError:
                    pass
                continue

            if "intentionally omitted <==**" not in stripped:
                merged_lines.append(line)
                continue

            page_ordinal += 1
            result = result_lookup.get((page_number, page_ordinal))
            if result is None:
                continue

            mcid_text = str((result.image_metadata or {}).get("mcid_text") or "").strip()
            ocr_text = str(result.ocr_text or "").strip()
            merged_text = mcid_text or ocr_text
            if not merged_text:
                continue

            bbox = result.bbox if isinstance(result.bbox, list) else None
            rounded_bbox = (
                tuple(int(round(float(value))) for value in bbox)
                if bbox and len(bbox) == 4
                else None
            )
            normalized_text = " ".join(merged_text.split())
            merge_key = (page_number, rounded_bbox, normalized_text)
            if merge_key in seen_merge_keys:
                continue
            seen_merge_keys.add(merge_key)

            merged_lines.append("")
            label = "picture mcid" if mcid_text else "picture ocr"
            merged_lines.append(f"**----- Start of {label} -----**<br>")
            for ocr_line in merged_text.splitlines():
                normalized = ocr_line.strip()
                if normalized:
                    merged_lines.append(f"{normalized}<br>")
            merged_lines.append(f"**----- End of {label} -----**<br>")

        return "\n".join(merged_lines)

    def _build_image_resource_map(self, page: fitz.Page) -> dict[str, dict[str, Any]]:
        resources: dict[str, dict[str, Any]] = {}
        for item in page.get_images(full=True):
            if len(item) < 8:
                continue
            name = str(item[7] or "")
            if not name:
                continue
            resources[name] = {
                "name": name,
                "xref": self._as_int(item[0]),
            }
        return resources

    def _extract_image_draw_operations(self, page: fitz.Page, image_names: set[str]) -> list[dict[str, Any]]:
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

    def _merge_mcid_text(self, mcid_matches: list[dict[str, Any]]) -> str:
        texts: list[str] = []
        seen: set[str] = set()
        for match in mcid_matches:
            text = str(match.get("text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
        return " ".join(texts[:8]).strip()

    def _role_labels(self, mcid_matches: list[dict[str, Any]]) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for match in mcid_matches:
            label = "{}/{}".format(match.get("block_role", "P"), match.get("leaf_role", "Span"))
            if label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    def _as_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

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

        total_rows = len(normalized)
        header = normalized[0]
        separator = ["---"] * width
        body = normalized[1:] or [[""] * width]
        markdown_rows = [header, separator, *body]
        table_meta = f"[표: {total_rows}행 × {width}열]"
        return table_meta + "\n" + "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)

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
