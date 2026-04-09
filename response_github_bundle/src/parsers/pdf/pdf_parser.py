from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz

from src.parsers.common.models import (
    DocumentClassification,
    DocumentElement,
    DocumentIssue,
    ParsedDocument,
    ParsedPage,
)
from src.parsers.common.serialization import build_document_summary, extract_markdown_sections
from src.parsers.pdf.markdown_extractor import PdfMarkdownExtractor
from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor
from src.shared.io import iso_now, make_artifact_stem
from src.structure.hwp_postprocessing import HwpPostprocessRules, build_blocks


class PdfParser:
    parser_name = "pdf-metadata-router"

    def __init__(self, *, enable_omitted_picture_ocr: bool = True) -> None:
        self.markdown_extractor = PdfMarkdownExtractor(enable_omitted_picture_ocr=enable_omitted_picture_ocr)
        self.structtree_extractor = PowerPointStructTreeExtractor()
        self.postprocess_rules = HwpPostprocessRules()

    def parse(self, path: Path, classification: DocumentClassification) -> ParsedDocument:
        document = fitz.open(path)
        issues: list[DocumentIssue] = []
        try:
            extraction = self.markdown_extractor.extract(path, document, classification, strategy_name="metadata-selected")
            issues.extend(extraction.issues)
            mcid_lookup = self.structtree_extractor.build_mcid_lookup(document)
            pages: list[ParsedPage] = []

            for page_index in range(document.page_count):
                page_number = page_index + 1
                page = document[page_index]
                page_dict = page.get_text("dict", sort=True)
                page_text = page.get_text("text", sort=True)
                elements: list[DocumentElement] = []
                order = 1

                for block in page_dict.get("blocks", []):
                    if block.get("type") == 0:
                        block_text = self._extract_block_text(block)
                        if not block_text:
                            continue
                        element_type = "caption" if self._is_caption(block_text) else "text"
                        elements.append(
                            DocumentElement(
                                element_id=f"p{page_number}-e{order}",
                                element_type=element_type,
                                page_number=page_number,
                                order=order,
                                bbox=[float(value) for value in block.get("bbox", [])],
                                text=block_text,
                                metadata={"source": "fitz-text-block"},
                            )
                        )
                        order += 1

                image_elements = self._extract_images(
                    page=page,
                    page_number=page_number,
                    start_order=order,
                    mcid_lookup=mcid_lookup.get(page_number, {}),
                )
                elements.extend(image_elements)
                order += len(image_elements)

                tables, table_issues = self._extract_tables(page, page_number, order)
                elements.extend(tables)
                issues.extend(table_issues)

                pages.append(
                    ParsedPage(
                        page_number=page_number,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        text_length=len(page_text.strip()),
                        elements=elements,
                        metadata={
                            "table_count": sum(1 for element in elements if element.element_type == "table"),
                            "image_count": sum(1 for element in elements if element.element_type == "image"),
                        },
                    )
                )

            selected_strategy = extraction.metadata.get("selected_strategy", extraction.applied_strategy)
            parser_name = self._build_parser_name(selected_strategy, extraction.applied_strategy)

            postprocess_result = build_blocks(extraction.markdown, self.postprocess_rules, source_format="pdf")

            parsed = ParsedDocument(
                document_id=make_artifact_stem(path),
                source_name=path.name,
                source_path=path.as_posix(),
                extension=path.suffix.lower(),
                status="parsed",
                parser_name=parser_name,
                classification=classification,
                metadata={
                    "page_count": document.page_count,
                    "metadata": document.metadata or {},
                    "markdown_source": extraction.applied_strategy,
                    "markdown_strategy": extraction.strategy_name,
                    "markdown_metadata": extraction.metadata,
                    "markdown_raw": extraction.raw_markdown,
                    "markdown_elapsed_ms": extraction.elapsed_ms,
                    "postprocess_logs": postprocess_result.logs,
                    "rule_version": self.postprocess_rules.version,
                },
                sections=extract_markdown_sections(extraction.markdown),
                pages=pages,
                markdown=extraction.markdown,
                blocks=postprocess_result.blocks,
                chunks=postprocess_result.chunks,
                issues=issues,
                created_at=iso_now(),
            )
            parsed.summary = build_document_summary(parsed)
            return parsed
        finally:
            document.close()

    def _build_parser_name(self, selected_strategy: str, applied_strategy: str) -> str:
        if selected_strategy == applied_strategy:
            return f"{self.parser_name}[{applied_strategy}]"
        return f"{self.parser_name}[{selected_strategy}->{applied_strategy}]"

    def _extract_block_text(self, block: dict) -> str:
        lines: list[str] = []
        for line in block.get("lines", []):
            fragments = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text:
                    fragments.append(text)
            line_text = "".join(fragments).strip()
            if line_text:
                lines.append(line_text)
        return "\n".join(lines).strip()

    def _extract_images(
        self,
        page: fitz.Page,
        page_number: int,
        start_order: int,
        mcid_lookup: dict[int, list[dict[str, Any]]],
    ) -> list[DocumentElement]:
        try:
            image_infos = sorted(page.get_image_info(xrefs=True), key=lambda item: item.get("number", 0))
        except Exception:
            image_infos = []

        if not image_infos:
            return []

        resource_map = self._build_image_resource_map(page)
        draw_ops = self._extract_image_draw_operations(page, set(resource_map))
        elements: list[DocumentElement] = []
        order = start_order

        for index, info in enumerate(image_infos):
            bbox = [float(value) for value in info.get("bbox", ())]
            if len(bbox) != 4:
                continue

            draw_op = draw_ops[index] if index < len(draw_ops) else {}
            xobject_name = str(draw_op.get("xobject_name") or "")
            resource = resource_map.get(xobject_name, {})
            mcid = draw_op.get("mcid")
            mcid_matches = mcid_lookup.get(mcid, []) if isinstance(mcid, int) else []
            mcid_text = self._merge_mcid_text(mcid_matches)
            role_labels = self._role_labels(mcid_matches)
            rendered_xref = self._as_int(info.get("xref"))
            resource_xref = self._as_int(resource.get("xref"))
            smask_xref = self._as_int(resource.get("smask_xref"))

            metadata = {
                "source": "fitz-image-xobject",
                "width": info.get("width"),
                "height": info.get("height"),
                "colorspace": info.get("cs-name"),
                "xobject_name": xobject_name or None,
                "resource_xref": resource_xref,
                "rendered_xref": rendered_xref,
                "smask_xref": smask_xref,
                "mcid": mcid,
                "content_stream_index": index,
                "mcid_text": mcid_text or None,
                "mcid_roles": role_labels,
                "mcid_match_count": len(mcid_matches),
            }
            if resource.get("name"):
                metadata["resource_name"] = resource["name"]

            elements.append(
                DocumentElement(
                    element_id=f"p{page_number}-e{order}",
                    element_type="image",
                    page_number=page_number,
                    order=order,
                    bbox=bbox,
                    text=mcid_text or None,
                    metadata=metadata,
                )
            )
            order += 1

        return elements

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
                "smask_xref": self._as_int(item[1]),
                "width": self._as_int(item[2]),
                "height": self._as_int(item[3]),
                "bpc": self._as_int(item[4]),
                "colorspace": item[5],
                "filter": item[8] if len(item) > 8 else None,
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
                operations.append(
                    {
                        "xobject_name": xobject_name,
                        "mcid": current_mcid,
                    }
                )

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

    def _extract_tables(
        self,
        page: fitz.Page,
        page_number: int,
        start_order: int,
    ) -> tuple[list[DocumentElement], list[DocumentIssue]]:
        try:
            tables = page.find_tables().tables
        except Exception as error:
            return [], [
                DocumentIssue(
                    code="table_detection_failed",
                    message=f"Table detection failed on page {page_number}: {error}",
                    severity="warning",
                )
            ]

        elements: list[DocumentElement] = []
        issues: list[DocumentIssue] = []
        order = start_order

        for index, table in enumerate(tables, start=1):
            try:
                bbox = tuple(float(value) for value in table.bbox)
                rows = table.extract()
            except Exception as error:
                issues.append(
                    DocumentIssue(
                        code="table_extraction_failed",
                        message=f"Skipped table {index} on page {page_number}: {error}",
                        severity="warning",
                    )
                )
                continue

            if not self._keep_table(page, bbox, rows):
                issues.append(
                    DocumentIssue(
                        code="table_skipped_low_confidence",
                        message=f"Skipped low-confidence table {index} on page {page_number}.",
                        severity="info",
                    )
                )
                continue

            elements.append(
                DocumentElement(
                    element_id=f"p{page_number}-e{order}",
                    element_type="table",
                    page_number=page_number,
                    order=order,
                    bbox=list(bbox),
                    markdown=self._rows_to_markdown(rows),
                    metadata={
                        "row_count": table.row_count,
                        "column_count": table.col_count,
                        "cells": rows,
                    },
                )
            )
            order += 1

        return elements, issues

    def _keep_table(self, page: fitz.Page, bbox: tuple[float, float, float, float], rows: list[list[str | None]]) -> bool:
        if len(rows) < 2:
            return False

        column_count = max((len(row) for row in rows), default=0)
        if column_count < 2:
            return False

        page_area = float(page.rect.width * page.rect.height)
        table_area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        area_ratio = table_area / page_area if page_area else 0.0

        total_cells = sum(len(row) for row in rows)
        non_empty_cells = sum(1 for row in rows for cell in row if (cell or "").strip())
        non_empty_ratio = non_empty_cells / total_cells if total_cells else 0.0

        if area_ratio > 0.6 and total_cells > 20:
            return False
        if non_empty_ratio < 0.15:
            return False
        return True

    def _rows_to_markdown(self, rows: list[list[str | None]]) -> str:
        width = max((len(row) for row in rows), default=0)
        normalized: list[list[str]] = []
        for row in rows:
            normalized.append([(cell or "").replace("\n", "<br>").strip() for cell in row] + [""] * (width - len(row)))

        if not normalized:
            return ""

        header = normalized[0]
        separator = ["---"] * width
        body = normalized[1:] or [[""] * width]
        markdown_rows = [header, separator, *body]
        return "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)

    def _is_caption(self, text: str) -> bool:
        first_line = text.splitlines()[0].strip()
        return bool(re.match(r"^(그림|Figure|Table)\s*\d*", first_line, re.IGNORECASE))
