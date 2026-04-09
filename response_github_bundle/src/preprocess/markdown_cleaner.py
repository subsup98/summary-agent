from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from src.parsers.common.models import DocumentClassification, DocumentIssue


PAGE_HEADING_PATTERN = re.compile(r"^#\s+Page\s+(\d+)\s*$", re.IGNORECASE)
SPACE_COLLAPSE_PATTERN = re.compile(r"\s+")
NUMBER_TOKEN_PATTERN = re.compile(r"\b\d+(?:[.,/-]\d+)*\b")
DATE_PATTERN = re.compile(
    r"(?:\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b)|"
    r"(?:\b\d{4}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?\b)"
)
VERSION_PATTERN = re.compile(r"\b(?:ver(?:sion)?|v)\.?\s*\d+(?:\.\d+){0,3}\b", re.IGNORECASE)
PAGE_TOKEN_PATTERN = re.compile(
    r"^(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$|^-\s*\d{1,4}\s*-$",
    re.IGNORECASE,
)
STANDALONE_NUMBER_PATTERN = re.compile(r"^\d{1,4}$")
COMMON_PERIOD_PATTERN = re.compile(r"^(?:[1-4]Q\d{2}|\d{4})$", re.IGNORECASE)
CAPTION_PROTECTION_PATTERN = re.compile(r"^(?:표|Table|Figure|그림)\s*\d*", re.IGNORECASE)
SOURCE_PROTECTION_PATTERN = re.compile(r"^(?:자료|출처)\s*[:：]", re.IGNORECASE)
COPYRIGHT_PATTERN = re.compile(r"(?:copyright|all rights reserved|저작권|무단전재|무단 복제)", re.IGNORECASE)
DISCLAIMER_PATTERN = re.compile(r"(?:disclaimer|면책|투자판단|참고자료|internal use only)", re.IGNORECASE)
WATERMARK_PATTERN = re.compile(r"(?:confidential|draft|internal|for discussion|watermark)", re.IGNORECASE)
ENHANCED_DISCLAIMER_PATTERN = re.compile(
    r"(?:forward-looking statement(?:s)?|important notice|not be relied upon)",
    re.IGNORECASE,
)
VALUE_TOKEN_PATTERN = re.compile(
    r"^[+-]?"
    r"(?:\d[\d,./]*)(?:\s*(?:%|%p|bp|억원|천원|만원|조원|원|주|배|x)?)"
    r"(?:\s+\d[\d,./]*(?:\s*(?:억원|천원|만원|조원|원|주|배|x)?)?)*$",
    re.IGNORECASE,
)
TOC_PRIORITY_HINTS: tuple[tuple[str, int], ...] = (
    ("key highlights", 20),
    ("appendix", 90),
    ("사업", 30),
    ("실적", 35),
    ("요약", 40),
    ("전망", 50),
    ("esg", 60),
)
@dataclass
class MarkdownPreprocessResult:
    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)
    issues: list[DocumentIssue] = field(default_factory=list)


@dataclass
class SeriesRow:
    label: str
    values: list[str]


@dataclass
class FactRow:
    row_path: list[str]
    facts: list[tuple[str, str]]


@dataclass
class PreprocessTextBlock:
    text: str
    bbox: list[float]


@dataclass
class PageLayoutContext:
    page_number: int
    width: float
    height: float
    text_blocks: list[PreprocessTextBlock] = field(default_factory=list)
    image_bboxes: list[list[float]] = field(default_factory=list)
    table_bboxes: list[list[float]] = field(default_factory=list)


class PdfMarkdownPreprocessor:
    profile_name = "pdf-safe-v2"
    edge_ratio = 0.12
    repeat_ratio_threshold = 0.5
    remove_threshold = 0.75
    repeated_noise_min_length = 24

    def preprocess(
        self,
        markdown: str,
        classification: DocumentClassification,
        page_contexts: list[PageLayoutContext] | None = None,
    ) -> MarkdownPreprocessResult:
        normalized_markdown = markdown.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized_markdown:
            return MarkdownPreprocessResult(
                markdown="",
                metadata={
                    "enabled": True,
                    "profile": self._profile_name(classification),
                    "changed": False,
                    "page_count": 0,
                    "applied_rules": [],
                    "pages": [],
                    "repeated_candidates": [],
                    "removal_log": [],
                    "preserved_candidates": [],
                },
            )

        pages = self._split_pages(normalized_markdown)
        context_by_page = {context.page_number: context for context in page_contexts or []}
        repeated_catalog = self._build_repeated_catalog(pages, context_by_page)

        cleaned_pages: list[tuple[int, list[str], dict[str, Any]]] = []
        removal_log: list[dict[str, Any]] = []
        preserved_candidates: list[dict[str, Any]] = []
        total_counts = {
            "removed_page_number_lines": 0,
            "removed_repeated_header_lines": 0,
            "removed_repeated_footer_lines": 0,
            "removed_noise_lines": 0,
            "spacing_normalized_lines": 0,
            "toc_entries_compacted": 0,
            "series_tables_structured": 0,
            "series_rows_structured": 0,
            "financial_tables_structured": 0,
            "financial_fact_rows_structured": 0,
            "chart_clusters_collapsed": 0,
            "chart_cluster_items_compacted": 0,
            "candidate_lines": 0,
            "preserved_candidate_lines": 0,
            "paragraph_merges": 0,
        }
        financial_rule_hits: dict[str, int] = {}

        for page_number, lines in pages:
            cleaned_lines, page_summary = self._clean_page(
                page_number=page_number,
                lines=lines,
                repeated_catalog=repeated_catalog,
                context=context_by_page.get(page_number),
            )
            cleaned_pages.append((page_number, cleaned_lines, page_summary))
            removal_log.extend(page_summary.pop("removals"))
            preserved_candidates.extend(page_summary.pop("preserved_candidates"))
            for key in total_counts:
                total_counts[key] += int(page_summary.get(key, 0) or 0)
            page_rule_hits = page_summary.get("financial_table_rule_hits") or {}
            if isinstance(page_rule_hits, dict):
                for rule_name, count in page_rule_hits.items():
                    financial_rule_hits[rule_name] = financial_rule_hits.get(rule_name, 0) + int(count or 0)

        cleaned_markdown = self._render_pages(cleaned_pages)
        changed = cleaned_markdown != normalized_markdown
        applied_rules = [
            name
            for name, count in (
                ("remove-page-number", total_counts["removed_page_number_lines"]),
                ("remove-repeated-header", total_counts["removed_repeated_header_lines"]),
                ("remove-repeated-footer", total_counts["removed_repeated_footer_lines"]),
                ("remove-noise-line", total_counts["removed_noise_lines"]),
                ("normalize-spacing", total_counts["spacing_normalized_lines"]),
                ("merge-paragraph-lines", total_counts["paragraph_merges"]),
                ("compact-toc-page", total_counts["toc_entries_compacted"]),
                ("structure-series-table", total_counts["series_tables_structured"]),
                ("structure-financial-table", total_counts["financial_tables_structured"]),
            )
            if count
        ]

        metadata = {
            "enabled": True,
            "profile": self._profile_name(classification),
            "changed": changed,
            "page_count": len(cleaned_pages),
            "edge_ratio": self.edge_ratio,
            "repeat_ratio_threshold": self.repeat_ratio_threshold,
            "remove_threshold": self.remove_threshold,
            "applied_rules": applied_rules,
            "repeated_candidates": self._serialize_repeated_catalog(repeated_catalog),
            "removal_log": removal_log,
            "preserved_candidates": preserved_candidates,
            "financial_table_rule_hits": financial_rule_hits,
            **total_counts,
            "pages": [summary for _, _, summary in cleaned_pages],
        }

        issues: list[DocumentIssue] = []
        if changed:
            issues.append(
                DocumentIssue(
                    code="markdown_preprocessed",
                    message=(
                        "Conservative markdown preprocessing applied: "
                        f"page numbers {total_counts['removed_page_number_lines']}, "
                        f"repeated headers {total_counts['removed_repeated_header_lines']}, "
                        f"repeated footers {total_counts['removed_repeated_footer_lines']}, "
                        f"other noise {total_counts['removed_noise_lines']}, "
                        f"spacing normalized {total_counts['spacing_normalized_lines']}, "
                        f"paragraph merges {total_counts['paragraph_merges']}, "
                        f"TOC entries compacted {total_counts['toc_entries_compacted']}, "
                        f"series tables {total_counts['series_tables_structured']}, "
                        f"series rows {total_counts['series_rows_structured']}, "
                        f"financial tables {total_counts['financial_tables_structured']}, "
                        f"financial fact rows {total_counts['financial_fact_rows_structured']}."
                    ),
                    severity="info",
                )
            )

        return MarkdownPreprocessResult(markdown=cleaned_markdown, metadata=metadata, issues=issues)

    def _profile_name(self, classification: DocumentClassification) -> str:
        producer = str(classification.metadata.get("producer") or "").lower()
        if "powerpoint" in producer or "ppt" in producer:
            return "pdf-powerpoint-safe-v2"
        return self.profile_name

    def _split_pages(self, markdown: str) -> list[tuple[int, list[str]]]:
        pages: list[tuple[int, list[str]]] = []
        current_page: int | None = None
        current_lines: list[str] = []

        for line in markdown.splitlines():
            match = PAGE_HEADING_PATTERN.match(line.strip())
            if match:
                if current_page is not None:
                    pages.append((current_page, current_lines))
                current_page = int(match.group(1))
                current_lines = []
                continue
            current_lines.append(line)

        if current_page is not None:
            pages.append((current_page, current_lines))
        else:
            pages.append((1, markdown.splitlines()))
        return pages

    def _build_repeated_catalog(
        self,
        pages: list[tuple[int, list[str]]],
        context_by_page: dict[int, PageLayoutContext],
    ) -> dict[str, dict[str, Any]]:
        page_count = len(pages)
        if page_count < 2:
            return {}

        counts: dict[str, dict[str, Any]] = {}
        minimum_repeat = max(2, math.ceil(page_count * self.repeat_ratio_threshold))
        candidates = self._collect_edge_candidates(pages, context_by_page)

        for candidate in candidates:
            normalized = candidate["normalized_text"]
            if not normalized:
                continue
            entry = counts.setdefault(
                normalized,
                {
                    "normalized_text": normalized,
                    "canonical_text": candidate["text"],
                    "pages": set(),
                    "edges": set(),
                    "regex_matches": set(),
                    "protected_pages": set(),
                },
            )
            entry["pages"].add(candidate["page_number"])
            entry["edges"].add(candidate["edge"])
            entry["regex_matches"].update(candidate["regex_matches"])
            if candidate["protected"]:
                entry["protected_pages"].add(candidate["page_number"])

        catalog: dict[str, dict[str, Any]] = {}
        for normalized, entry in counts.items():
            repeated_pages = len(entry["pages"])
            if repeated_pages < minimum_repeat:
                continue
            if repeated_pages == len(entry["protected_pages"]):
                continue
            repeat_ratio = repeated_pages / page_count
            catalog[normalized] = {
                "normalized_text": normalized,
                "canonical_text": entry["canonical_text"],
                "pages": sorted(entry["pages"]),
                "repeat_count": repeated_pages,
                "repeat_ratio": round(repeat_ratio, 4),
                "edges": sorted(entry["edges"]),
                "regex_matches": sorted(entry["regex_matches"]),
            }
        return catalog

    def _collect_edge_candidates(
        self,
        pages: list[tuple[int, list[str]]],
        context_by_page: dict[int, PageLayoutContext],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for page_number, lines in pages:
            context = context_by_page.get(page_number)
            if context and context.text_blocks:
                for block in context.text_blocks:
                    edge = self._detect_edge(block.bbox, context.height)
                    if edge is not None:
                        candidates.append(
                            self._candidate_record(
                                page_number=page_number,
                                text=block.text,
                                bbox=block.bbox,
                                edge=edge,
                                context=context,
                            )
                        )
                        continue
                    if self._looks_like_repeatable_noise_line(block.text):
                        candidates.append(
                            self._candidate_record(
                                page_number=page_number,
                                text=block.text,
                                bbox=block.bbox,
                                edge="body",
                                context=context,
                            )
                        )
                continue

            non_empty = [line.strip() for line in lines if line.strip()]
            for edge, window in (("top", non_empty[:4]), ("bottom", non_empty[-4:])):
                for line in window:
                    candidates.append(
                        {
                            "page_number": page_number,
                            "text": line,
                            "normalized_text": self._normalize_compare_key(line),
                            "bbox": None,
                            "edge": edge,
                            "regex_matches": self._regex_matches(line),
                            "protected": self._looks_like_protected_text(line),
                        }
                    )
            for line in non_empty[4:-4]:
                if not self._looks_like_repeatable_noise_line(line):
                    continue
                candidates.append(
                    {
                        "page_number": page_number,
                        "text": line,
                        "normalized_text": self._normalize_compare_key(line),
                        "bbox": None,
                        "edge": "body",
                        "regex_matches": self._regex_matches(line),
                        "protected": self._looks_like_protected_text(line),
                    }
                )
        return candidates

    def _candidate_record(
        self,
        *,
        page_number: int,
        text: str,
        bbox: list[float] | None,
        edge: str,
        context: PageLayoutContext,
    ) -> dict[str, Any]:
        return {
            "page_number": page_number,
            "text": text.strip(),
            "normalized_text": self._normalize_compare_key(text),
            "bbox": bbox,
            "edge": edge,
            "regex_matches": self._regex_matches(text),
            "protected": self._is_protected_block(text, bbox, context),
        }

    def _detect_edge(self, bbox: list[float] | None, page_height: float) -> str | None:
        if not bbox or len(bbox) != 4 or page_height <= 0:
            return None
        top_limit = page_height * self.edge_ratio
        bottom_limit = page_height * (1.0 - self.edge_ratio)
        if bbox[1] <= top_limit:
            return "top"
        if bbox[3] >= bottom_limit:
            return "bottom"
        return None

    def _clean_page(
        self,
        *,
        page_number: int,
        lines: list[str],
        repeated_catalog: dict[str, dict[str, Any]],
        context: PageLayoutContext | None,
    ) -> tuple[list[str], dict[str, Any]]:
        summary = {
            "page_number": page_number,
            "changed": False,
            "removed_page_number_lines": 0,
            "removed_repeated_header_lines": 0,
            "removed_repeated_footer_lines": 0,
            "removed_noise_lines": 0,
            "spacing_normalized_lines": 0,
            "toc_entries_compacted": 0,
            "series_tables_structured": 0,
            "series_rows_structured": 0,
            "financial_tables_structured": 0,
            "financial_fact_rows_structured": 0,
            "chart_clusters_collapsed": 0,
            "chart_cluster_items_compacted": 0,
            "candidate_lines": 0,
            "preserved_candidate_lines": 0,
            "paragraph_merges": 0,
            "financial_table_rule_hits": {},
            "removals": [],
            "preserved_candidates": [],
        }
        if not lines:
            return [], summary

        edge_lookup = self._build_page_edge_lookup(context)
        non_empty_indexes = [index for index, line in enumerate(lines) if line.strip()]
        first_window = set(non_empty_indexes[:4])
        last_window = set(non_empty_indexes[-4:])
        stripped_lines = [line.strip() for line in lines if line.strip()]
        toc_page = bool(stripped_lines and stripped_lines[0].lower() in {"contents", "appendix"})
        kept_lines: list[str] = []

        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue

            decision = self._evaluate_line(
                page_number=page_number,
                line=stripped,
                line_index=index,
                in_top_window=index in first_window,
                in_bottom_window=index in last_window,
                repeated_catalog=repeated_catalog,
                edge_lookup=edge_lookup,
                context=context,
            )
            if decision is not None:
                if toc_page and "page_number" in decision["regex_matches"] and not decision["repeated"]:
                    summary["preserved_candidate_lines"] += 1
                    summary["preserved_candidates"].append({**decision, "decision": "keep", "remove": False, "reasons": [*decision["reasons"], "toc-page-ref"]})
                    kept_lines.append(stripped)
                    continue
                summary["candidate_lines"] += 1
                if decision["remove"]:
                    self._apply_removal_counters(summary, decision)
                    summary["changed"] = True
                    summary["removals"].append(decision)
                    continue
                summary["preserved_candidate_lines"] += 1
                summary["preserved_candidates"].append(decision)

            normalized_line = self._normalize_spacing(stripped)
            if normalized_line != stripped:
                summary["spacing_normalized_lines"] += 1
                summary["changed"] = True
            kept_lines.append(normalized_line)

        rewritten = self._compact_blank_lines(kept_lines)
        rewritten = self._merge_paragraph_lines(rewritten, summary)
        rewritten = self._compact_toc_page(rewritten, repeated_catalog, summary)
        rewritten = self._structure_series_blocks(rewritten, summary)
        rewritten = self._structure_financial_table_blocks(rewritten, summary)
        rewritten = self._compact_blank_lines(rewritten)
        return rewritten, summary

    def _build_page_edge_lookup(self, context: PageLayoutContext | None) -> dict[str, list[dict[str, Any]]]:
        lookup: dict[str, list[dict[str, Any]]] = {}
        if context is None:
            return lookup
        for block in context.text_blocks:
            edge = self._detect_edge(block.bbox, context.height)
            if edge is None:
                continue
            normalized = self._normalize_compare_key(block.text)
            if not normalized:
                continue
            lookup.setdefault(normalized, []).append(
                self._candidate_record(
                    page_number=context.page_number,
                    text=block.text,
                    bbox=block.bbox,
                    edge=edge,
                    context=context,
                )
            )
        return lookup

    def _evaluate_line(
        self,
        *,
        page_number: int,
        line: str,
        line_index: int,
        in_top_window: bool,
        in_bottom_window: bool,
        repeated_catalog: dict[str, dict[str, Any]],
        edge_lookup: dict[str, list[dict[str, Any]]],
        context: PageLayoutContext | None,
    ) -> dict[str, Any] | None:
        normalized = self._normalize_compare_key(line)
        regex_matches = self._regex_matches(line)
        matched_blocks = edge_lookup.get(normalized, [])
        repeated = repeated_catalog.get(normalized)

        protected = self._looks_like_protected_text(line)
        protection_reasons: list[str] = []
        bbox: list[float] | None = None
        edge: str | None = None
        location_score = 0.0

        for block in matched_blocks:
            bbox = block["bbox"]
            edge = edge or block["edge"]
            location_score = max(location_score, 0.35)
            if block["protected"]:
                protected = True
                protection_reasons.append("protected-region")

        if not matched_blocks and in_top_window:
            edge = "top"
            location_score = max(location_score, 0.2)
        if not matched_blocks and in_bottom_window and edge is None:
            edge = "bottom"
            location_score = max(location_score, 0.2)

        if protected and self._looks_like_protected_text(line):
            protection_reasons.append("caption-or-source-pattern")

        repeat_score = 0.0
        repeat_ratio = 0.0
        repeated_flag = False
        if repeated is not None:
            repeated_flag = True
            repeat_ratio = float(repeated["repeat_ratio"])
            repeat_score = min(0.55, 0.25 + repeat_ratio * 0.4)
            edge = edge or (repeated["edges"][0] if repeated["edges"] else None)

        regex_score = self._regex_score(regex_matches)
        candidate = bool(regex_matches or repeated_flag or location_score > 0.0)
        if not candidate:
            return None

        reasons: list[str] = []
        if location_score:
            reasons.append(f"edge:{edge or 'unknown'}")
        if repeated_flag:
            reasons.append(f"repeat:{repeat_ratio:.2f}")
        for regex_match in regex_matches:
            reasons.append(f"regex:{regex_match}")
        reasons.extend(reason for reason in protection_reasons if reason not in reasons)

        if protected:
            return {
                "page_number": page_number,
                "line_index": line_index,
                "text": line,
                "normalized_text": normalized,
                "bbox": bbox,
                "edge": edge,
                "score": round(location_score + repeat_score + regex_score, 4),
                "decision": "keep",
                "remove": False,
                "repeated": repeated_flag,
                "repeat_ratio": round(repeat_ratio, 4),
                "regex_matches": regex_matches,
                "protected": True,
                "reasons": reasons,
            }

        if regex_matches and not (location_score > 0.0 or repeated_flag):
            return {
                "page_number": page_number,
                "line_index": line_index,
                "text": line,
                "normalized_text": normalized,
                "bbox": bbox,
                "edge": edge,
                "score": round(regex_score, 4),
                "decision": "keep",
                "remove": False,
                "repeated": repeated_flag,
                "repeat_ratio": round(repeat_ratio, 4),
                "regex_matches": regex_matches,
                "protected": False,
                "reasons": reasons,
            }

        score = location_score + repeat_score + regex_score
        remove = score >= self.remove_threshold
        if "page_number" in regex_matches and location_score >= 0.2:
            remove = score >= 0.45

        return {
            "page_number": page_number,
            "line_index": line_index,
            "text": line,
            "normalized_text": normalized,
            "bbox": bbox,
            "edge": edge,
            "score": round(score, 4),
            "decision": "remove" if remove else "keep",
            "remove": remove,
            "repeated": repeated_flag,
            "repeat_ratio": round(repeat_ratio, 4),
            "regex_matches": regex_matches,
            "protected": False,
            "reasons": reasons,
        }

    def _apply_removal_counters(self, summary: dict[str, Any], decision: dict[str, Any]) -> None:
        regex_matches = set(decision["regex_matches"])
        edge = decision.get("edge")
        if "page_number" in regex_matches:
            summary["removed_page_number_lines"] += 1
            return
        if decision.get("repeated") and edge == "top":
            summary["removed_repeated_header_lines"] += 1
            return
        if decision.get("repeated") and edge == "bottom":
            summary["removed_repeated_footer_lines"] += 1
            return
        summary["removed_noise_lines"] += 1

    def _regex_matches(self, line: str) -> list[str]:
        matches: list[str] = []
        stripped = line.strip()
        if PAGE_TOKEN_PATTERN.fullmatch(stripped):
            matches.append("page_number")
        if DATE_PATTERN.search(stripped):
            matches.append("date")
        if VERSION_PATTERN.search(stripped):
            matches.append("version")
        if COPYRIGHT_PATTERN.search(stripped):
            matches.append("copyright")
        if DISCLAIMER_PATTERN.search(stripped):
            matches.append("disclaimer")
        if ENHANCED_DISCLAIMER_PATTERN.search(stripped) and "disclaimer" not in matches:
            matches.append("disclaimer")
        if WATERMARK_PATTERN.search(stripped):
            matches.append("watermark")
        return matches

    def _regex_score(self, matches: list[str]) -> float:
        if not matches:
            return 0.0
        score = 0.0
        weights = {
            "page_number": 0.35,
            "copyright": 0.18,
            "disclaimer": 0.18,
            "watermark": 0.18,
            "date": 0.12,
            "version": 0.12,
        }
        for match in matches:
            score += weights.get(match, 0.1)
        return min(score, 0.4)

    def _is_protected_block(
        self,
        text: str,
        bbox: list[float] | None,
        context: PageLayoutContext,
    ) -> bool:
        if self._looks_like_protected_text(text):
            return True
        if not bbox or len(bbox) != 4:
            return False
        for protected_bbox in [*context.image_bboxes, *context.table_bboxes]:
            if self._bbox_within_margin(bbox, protected_bbox, margin=28.0):
                return True
        return False

    def _looks_like_protected_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return bool(CAPTION_PROTECTION_PATTERN.match(stripped) or SOURCE_PROTECTION_PATTERN.match(stripped))

    def _looks_like_repeatable_noise_line(self, text: str) -> bool:
        stripped = SPACE_COLLAPSE_PATTERN.sub(" ", text.strip())
        if len(stripped) < self.repeated_noise_min_length:
            return False
        regex_matches = set(self._regex_matches(stripped))
        if {"copyright", "disclaimer", "watermark"} & regex_matches:
            return True
        if stripped.lower().startswith(("forward-looking statements", "important notice", "notice:", "disclaimer:")):
            return True
        if len(re.findall(r"[A-Za-z가-힣]{2,}", stripped)) >= 8 and not self._has_structural_prefix(stripped):
            return True
        return False

    def _bbox_within_margin(self, a: list[float], b: list[float], margin: float) -> bool:
        if len(a) != 4 or len(b) != 4:
            return False
        return not (
            a[2] < b[0] - margin
            or a[0] > b[2] + margin
            or a[3] < b[1] - margin
            or a[1] > b[3] + margin
        )

    def _normalize_compare_key(self, line: str) -> str:
        normalized = SPACE_COLLAPSE_PATTERN.sub(" ", line.strip()).casefold()
        normalized = DATE_PATTERN.sub("<date>", normalized)
        normalized = VERSION_PATTERN.sub("<version>", normalized)
        normalized = NUMBER_TOKEN_PATTERN.sub("<num>", normalized)
        normalized = re.sub(r"\bpage\s+<num>\b", "<page>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _normalize_spacing(self, line: str) -> str:
        updated = line.strip()
        updated = re.sub(r"(?<!\d),(?=\S)", ", ", updated)
        updated = re.sub(r"\b(QoQ|YoY|MoM)(?=[+-])", r"\1 ", updated)
        updated = re.sub(r"\(\s+", "(", updated)
        updated = re.sub(r"\s+\)", ")", updated)
        updated = re.sub(r"\s{2,}", " ", updated)
        return updated

    def _compact_blank_lines(self, lines: list[str]) -> list[str]:
        compacted: list[str] = []
        blank_seen = False
        for line in lines:
            if line.strip():
                compacted.append(line.strip())
                blank_seen = False
                continue
            if not blank_seen and compacted:
                compacted.append("")
            blank_seen = True
        while compacted and not compacted[0].strip():
            compacted.pop(0)
        while compacted and not compacted[-1].strip():
            compacted.pop()
        return compacted

    def _merge_paragraph_lines(self, lines: list[str], summary: dict[str, Any]) -> list[str]:
        if not lines:
            return lines

        merged: list[str] = []
        for line in lines:
            if not line.strip():
                merged.append("")
                continue
            if not merged or not self._can_merge_paragraph_lines(merged[-1], line):
                merged.append(line)
                continue

            previous = merged.pop()
            if previous.endswith("-") and not previous.endswith("--"):
                combined = previous[:-1] + line.lstrip()
            else:
                combined = previous.rstrip() + " " + line.lstrip()
            merged.append(combined)
            summary["paragraph_merges"] += 1
            summary["changed"] = True
        return merged

    def _can_merge_paragraph_lines(self, previous: str, current: str) -> bool:
        if not previous.strip() or not current.strip():
            return False
        if self._has_structural_prefix(previous) or self._has_structural_prefix(current):
            return False
        if self._looks_like_protected_text(previous) or self._looks_like_protected_text(current):
            return False
        if self._looks_like_value(previous) or self._looks_like_value(current):
            return False
        if self._looks_like_title(previous) or self._looks_like_title(current):
            return False
        if previous.endswith((".", "!", "?")):
            return False
        return True

    def _looks_like_title(self, line: str) -> bool:
        tokens = re.findall(r"[A-Za-z]+|[가-힣]+|\d+", line)
        if not tokens or len(tokens) > 10:
            return False
        if len(line) > 80:
            return False
        if any(char in line for char in ".!?,;:"):
            return False
        return True

    def _compact_toc_page(
        self,
        lines: list[str],
        repeated_catalog: dict[str, dict[str, Any]],
        summary: dict[str, Any],
    ) -> list[str]:
        stripped = [line.strip() for line in lines if line.strip()]
        if not stripped:
            return lines

        marker = stripped[0]
        if marker.lower() not in {"contents", "appendix"}:
            return lines

        page_refs = [line for line in stripped[1:] if self._is_toc_page_ref(line)]
        if len(page_refs) < 4:
            return lines

        title_candidates = [line for line in stripped[1:] if not self._is_numericish(line)]
        if not title_candidates:
            return lines

        normalized_titles = self._prepare_toc_titles(marker, title_candidates, page_refs, repeated_catalog)
        if len(normalized_titles) < len(page_refs):
            return lines

        entries = list(zip(normalized_titles[: len(page_refs)], page_refs))
        if not entries:
            return lines

        rewritten = [marker, ""]
        for title, page_ref in entries:
            rewritten.append(f"- {title} ..... {page_ref}")
        summary["toc_entries_compacted"] += len(entries)
        summary["changed"] = True
        return rewritten

    def _prepare_toc_titles(
        self,
        marker: str,
        title_candidates: list[str],
        page_refs: list[str],
        repeated_catalog: dict[str, dict[str, Any]],
    ) -> list[str]:
        candidates = [candidate for candidate in title_candidates if candidate.lower() != "contents"]
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.casefold()
            if normalized in seen:
                continue
            deduped.append(candidate)
            seen.add(normalized)
        candidates = deduped

        while len(candidates) > len(page_refs):
            drop_index = self._find_documentlike_candidate(candidates, repeated_catalog)
            candidates.pop(drop_index)

        if len(candidates) > 1:
            candidates = [candidates[-1], *candidates[:-1]]

        if marker.lower() == "contents":
            appendix_titles = [candidate for candidate in candidates if candidate.casefold() == "appendix"]
            non_appendix = [candidate for candidate in candidates if candidate.casefold() != "appendix"]
            candidates = [*non_appendix, *appendix_titles]

        return self._sort_toc_candidates(candidates)

    def _find_documentlike_candidate(self, candidates: list[str], repeated_catalog: dict[str, dict[str, Any]]) -> int:
        best_index = 0
        best_score = -1.0
        repeated_texts = {entry["canonical_text"] for entry in repeated_catalog.values()}
        for index, candidate in enumerate(candidates):
            score = self._header_similarity(candidate, repeated_texts)
            if score > best_score:
                best_index = index
                best_score = score
        return best_index

    def _header_similarity(self, line: str, repeated_texts: set[str]) -> float:
        if not repeated_texts:
            return 0.0
        line_tokens = set(re.findall(r"[A-Za-z]+|[가-힣]+|\d+", self._normalize_compare_key(line)))
        if not line_tokens:
            return 0.0
        best = 0.0
        for header in repeated_texts:
            header_tokens = set(re.findall(r"[A-Za-z]+|[가-힣]+|\d+", self._normalize_compare_key(header)))
            if not header_tokens:
                continue
            score = len(line_tokens & header_tokens) / len(line_tokens)
            if score > best:
                best = score
        return best

    def _sort_toc_candidates(self, candidates: list[str]) -> list[str]:
        weighted: list[tuple[int, int, str]] = []
        matched = 0
        for index, candidate in enumerate(candidates):
            lowered = candidate.casefold()
            priority = 999 + index
            for hint, hint_priority in TOC_PRIORITY_HINTS:
                if hint.casefold() in lowered:
                    priority = hint_priority
                    matched += 1
                    break
            weighted.append((priority, index, candidate))
        if matched < max(2, len(candidates) // 2):
            return candidates
        weighted.sort(key=lambda item: (item[0], item[1]))
        return [candidate for _, _, candidate in weighted]

    def _is_toc_page_ref(self, line: str) -> bool:
        stripped = line.strip()
        if not STANDALONE_NUMBER_PATTERN.fullmatch(stripped):
            return False
        return stripped.startswith("0") or int(stripped) >= 10

    def _is_numericish(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return self._looks_like_value(stripped) or COMMON_PERIOD_PATTERN.fullmatch(stripped) is not None

    def _looks_like_value(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 32:
            return False
        if self._has_structural_prefix(stripped):
            return False
        if ":" in stripped and not stripped.startswith(("1Q", "2Q", "3Q", "4Q")):
            return False
        if COMMON_PERIOD_PATTERN.fullmatch(stripped):
            return True
        if re.search(r"[.!?]{2,}", stripped):
            return False
        return VALUE_TOKEN_PATTERN.fullmatch(stripped) is not None

    def _structure_series_blocks(self, lines: list[str], summary: dict[str, Any]) -> list[str]:
        tokens = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
        if not tokens:
            return lines

        rewritten: list[str] = []
        cursor_line = 0
        token_index = 0

        while token_index < len(tokens):
            period_end, periods = self._collect_period_header_tokens(tokens, token_index)
            if periods:
                series_end, notes, rows = self._collect_series_rows_tokens(tokens, period_end, len(periods))
                if len(rows) >= 3:
                    start_line = tokens[token_index][0]
                    end_line = tokens[series_end - 1][0] + 1
                    rewritten.extend(lines[cursor_line:start_line])
                    rewritten.extend(self._render_series_block(periods, notes, rows))
                    rewritten.append("")
                    summary["series_tables_structured"] += 1
                    summary["series_rows_structured"] += len(rows)
                    summary["changed"] = True
                    cursor_line = end_line
                    token_index = series_end
                    continue
            token_index += 1

        rewritten.extend(lines[cursor_line:])
        return self._compact_blank_lines(rewritten)

    def _collect_period_header_tokens(
        self,
        tokens: list[tuple[int, str]],
        start_index: int,
    ) -> tuple[int, list[str]]:
        periods: list[str] = []
        index = start_index
        while index < len(tokens):
            line = tokens[index][1]
            if not COMMON_PERIOD_PATTERN.fullmatch(line):
                break
            periods.append(line)
            index += 1
        if len(periods) < 4 or len(set(periods)) != len(periods):
            return start_index, []
        return index, periods

    def _collect_series_rows_tokens(
        self,
        tokens: list[tuple[int, str]],
        start_index: int,
        width: int,
    ) -> tuple[int, list[str], list[SeriesRow]]:
        notes: list[str] = []
        rows: list[SeriesRow] = []
        index = start_index
        failures = 0

        while index < len(tokens):
            line = tokens[index][1]
            if self._is_series_note(line):
                if rows:
                    break
                notes.append(line)
                index += 1
                continue
            if self._is_series_boundary(line) and rows:
                break

            row, next_index = self._parse_series_row(tokens, index, width)
            if row is not None:
                rows.append(row)
                index = next_index
                failures = 0
                continue

            if rows:
                failures += 1
                if failures >= 2:
                    break
            else:
                if self._looks_like_value(line) or self._is_series_label(line):
                    failures += 1
                    if failures >= 3:
                        break
            index += 1

        return index, notes, rows

    def _parse_series_row(
        self,
        tokens: list[tuple[int, str]],
        start_index: int,
        width: int,
    ) -> tuple[SeriesRow | None, int]:
        label_before = self._parse_label_before_values(tokens, start_index, width)
        if label_before is not None:
            return label_before
        return self._parse_values_before_label(tokens, start_index, width)

    def _parse_label_before_values(
        self,
        tokens: list[tuple[int, str]],
        start_index: int,
        width: int,
    ) -> tuple[SeriesRow | None, int]:
        if start_index + width >= len(tokens):
            return None, start_index
        label = tokens[start_index][1]
        if not self._is_series_label(label):
            return None, start_index
        values = [tokens[start_index + offset][1] for offset in range(1, width + 1)]
        if not all(self._looks_like_value(value) for value in values):
            return None, start_index
        return SeriesRow(label=label, values=values), start_index + width + 1

    def _parse_values_before_label(
        self,
        tokens: list[tuple[int, str]],
        start_index: int,
        width: int,
    ) -> tuple[SeriesRow | None, int]:
        if start_index + width >= len(tokens):
            return None, start_index
        values = [tokens[start_index + offset][1] for offset in range(width)]
        label = tokens[start_index + width][1]
        if not all(self._looks_like_value(value) for value in values):
            return None, start_index
        if not self._is_series_label(label):
            return None, start_index
        return SeriesRow(label=label, values=values), start_index + width + 1

    def _is_series_note(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or COMMON_PERIOD_PATTERN.fullmatch(stripped):
            return False
        if stripped.startswith(("*", "[")):
            return True
        if not stripped.startswith("("):
            return False
        return not self._strip_leading_parenthetical_groups(stripped)

    def _is_series_boundary(self, line: str) -> bool:
        return self._has_structural_prefix(line) or COMMON_PERIOD_PATTERN.fullmatch(line) is not None or self._looks_like_sentence(line)

    def _is_series_label(self, line: str) -> bool:
        if not line or len(line) > 48:
            return False
        if self._has_structural_prefix(line):
            return False
        if COMMON_PERIOD_PATTERN.fullmatch(line):
            return False
        if self._looks_like_value(line):
            return False
        return bool(re.search(r"[A-Za-z가-힣]", line))

    def _looks_like_sentence(self, line: str) -> bool:
        if len(line) < 20:
            return False
        if self._has_structural_prefix(line):
            return False
        if re.search(r"[.!?]", line):
            return True
        return len(re.findall(r"[A-Za-z가-힣]{2,}", line)) >= 4

    def _has_structural_prefix(self, line: str) -> bool:
        return (
            line.startswith("#")
            or line.startswith(">")
            or line.startswith("|")
            or line.startswith("- ")
            or line.startswith("* ")
        )

    def _strip_leading_parenthetical_groups(self, line: str) -> str:
        remainder = line.strip()
        while remainder.startswith("("):
            closing_index = self._find_parenthetical_prefix_end(remainder)
            if closing_index <= 0:
                break
            remainder = remainder[closing_index + 1 :].strip()
        return remainder

    def _find_parenthetical_prefix_end(self, line: str) -> int:
        depth = 0
        for index, char in enumerate(line):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
            elif depth == 0:
                return -1
        return -1

    def _render_series_block(self, periods: list[str], notes: list[str], rows: list[SeriesRow]) -> list[str]:
        rendered = [f"**[Series Table]** {' | '.join(periods)}"]
        rendered.extend(notes)
        for row in rows:
            rendered.append(f"- {row.label}: {' | '.join(row.values)}")
        return rendered

    def _structure_financial_table_blocks(self, lines: list[str], summary: dict[str, Any]) -> list[str]:
        if not lines:
            return lines

        rewritten: list[str] = []
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped.startswith("|"):
                rewritten.append(lines[index])
                index += 1
                continue

            start = index
            block_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block_lines.append(lines[index].strip())
                index += 1

            rendered = self._try_render_financial_table_block(block_lines, summary)
            if rendered is None:
                rewritten.extend(lines[start:index])
                continue

            rewritten.extend(rendered)
            rewritten.append("")

        return self._compact_blank_lines(rewritten)

    def _try_render_financial_table_block(self, block_lines: list[str], summary: dict[str, Any]) -> list[str] | None:
        rows = [self._parse_markdown_table_row(line) for line in block_lines]
        rows = [row for row in rows if row]
        if len(rows) < 3:
            return None

        non_separator_rows = [row for row in rows if not self._is_markdown_separator_row(row)]
        if len(non_separator_rows) < 3:
            return None

        title_row = non_separator_rows[0]
        header_row = non_separator_rows[1]
        data_rows = non_separator_rows[2:]
        if len(header_row) < 3 or len(data_rows) < 2:
            return None

        period_groups = [self._split_table_cell(cell) for cell in header_row[1:]]
        pure_period_columns = [group for group in period_groups if group and all(self._looks_like_financial_period_token(token) for token in group)]
        if len(pure_period_columns) < 4:
            return None

        title = self._normalize_inline_markdown(" ".join(cell for cell in title_row if cell.strip()))
        unit = self._extract_unit_from_title(title)
        rendered = [f"**[Financial Fact Table]** {title}" if title else "**[Financial Fact Table]**"]
        if unit:
            rendered.append(f"(Unit: {unit})")

        pending_labels: dict[int, list[str]] = {}
        fact_rows: list[FactRow] = []

        for row in data_rows:
            current_labels = self._split_table_cell(row[0]) if row else []
            if current_labels:
                current_labels = [label for label in current_labels if label not in {"구분", "회사잠정", "(십억원)"}]

            for column_index in range(1, len(row)):
                periods = period_groups[column_index - 1] if column_index - 1 < len(period_groups) else []
                values = self._split_table_cell(row[column_index])
                if not periods or not values:
                    continue

                labels_for_column = current_labels if current_labels else pending_labels.get(column_index, [])
                if not labels_for_column:
                    continue

                fact_group = self._build_fact_rows(labels_for_column, periods, values)
                if not fact_group:
                    continue

                fact_rows.extend(fact_group)
                covered = len(fact_group)
                if current_labels:
                    remainder = current_labels[covered:]
                else:
                    remainder = labels_for_column[covered:]

                if remainder:
                    pending_labels[column_index] = remainder
                else:
                    pending_labels.pop(column_index, None)

        merged_fact_rows: list[FactRow] = []
        fact_row_lookup: dict[str, FactRow] = {}
        for fact_row in fact_rows:
            existing = fact_row_lookup.get(fact_row.label)
            if existing is None:
                merged = FactRow(label=fact_row.label, facts=list(fact_row.facts))
                fact_row_lookup[fact_row.label] = merged
                merged_fact_rows.append(merged)
                continue
            existing.facts.extend(fact_row.facts)

        if len(merged_fact_rows) < 3:
            return None

        for fact_row in merged_fact_rows:
            rendered.append(f"- {fact_row.label}: " + " | ".join(f"{period}={value}" for period, value in fact_row.facts))

        summary["financial_tables_structured"] += 1
        summary["financial_fact_rows_structured"] += len(merged_fact_rows)
        rule_hits = summary.setdefault("financial_table_rule_hits", {})
        if isinstance(rule_hits, dict):
            rule_hits["hierarchical_periodic_table"] = int(rule_hits.get("hierarchical_periodic_table", 0) or 0) + 1
        summary["changed"] = True
        return rendered

    def _parse_markdown_table_row(self, line: str) -> list[str]:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            return []
        return [cell.strip() for cell in stripped[1:-1].split("|")]

    def _is_markdown_separator_row(self, row: list[str]) -> bool:
        if not row:
            return False
        return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) is not None for cell in row)

    def _split_table_cell(self, text: str) -> list[str]:
        cleaned = text.replace("&nbsp;", " ")
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        parts = [part.strip() for part in re.split(r"<br\s*/?>", cleaned, flags=re.IGNORECASE)]
        return [part for part in parts if part and part != "-"]

    def _normalize_inline_markdown(self, text: str) -> str:
        cleaned = text.replace("&nbsp;", " ")
        cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = SPACE_COLLAPSE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def _extract_unit_from_title(self, title: str) -> str | None:
        match = re.search(r"\((?:단위|unit)\s*:\s*([^)]+)\)", title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"\(([^()]*(?:%|원|million|billion)[^()]*)\)", title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _build_fact_rows(self, labels: list[str], periods: list[str], values: list[str]) -> list[FactRow]:
        width = len(periods)
        if not labels or not periods or len(values) < width:
            return []

        row_count = min(len(labels), len(values) // width)
        fact_rows: list[FactRow] = []
        for row_index in range(row_count):
            label = self._normalize_fact_label(labels[row_index])
            if not label:
                continue
            start = row_index * width
            row_values = values[start : start + width]
            if len(row_values) != width:
                continue
            fact_rows.append(FactRow(label=label, facts=list(zip(periods, row_values))))
        return fact_rows

    def _normalize_fact_label(self, label: str) -> str:
        cleaned = self._normalize_inline_markdown(label)
        return cleaned.strip(":| ")

    def _try_render_financial_table_block(self, block_lines: list[str], summary: dict[str, Any]) -> list[str] | None:
        rows = [self._parse_markdown_table_row(line) for line in block_lines]
        rows = [row for row in rows if row]
        if len(rows) < 3:
            return None

        non_separator_rows = [row for row in rows if not self._is_markdown_separator_row(row)]
        if len(non_separator_rows) < 3:
            return None

        title_row = non_separator_rows[0]
        header_row = non_separator_rows[1]
        data_rows = non_separator_rows[2:]
        if len(header_row) < 3 or len(data_rows) < 2:
            return None

        period_groups = [self._split_table_cell(cell) for cell in header_row[1:]]
        pure_period_columns = [group for group in period_groups if group and all(self._looks_like_financial_period_token(token) for token in group)]
        if len(pure_period_columns) < 4:
            return None

        title = self._normalize_inline_markdown(" ".join(cell for cell in title_row if cell.strip()))
        unit = self._extract_unit_from_title(title)
        rendered = [f"**[Financial Fact Table]** {title}" if title else "**[Financial Fact Table]**"]
        if unit:
            rendered.append(f"(Unit: {unit})")

        packed_rendered = self._try_render_packed_period_table(
            data_rows=data_rows,
            period_groups=period_groups,
            base_rendered=rendered,
            summary=summary,
        )
        if packed_rendered is not None:
            return packed_rendered

        current_root_label: str | None = None
        pending_paths: dict[int, list[list[str]]] = {}
        fact_rows: list[FactRow] = []

        for row in data_rows:
            current_labels = self._split_table_cell(row[0]) if row else []
            if current_labels:
                current_labels = [label for label in current_labels if label not in {"구분", "회사잠정", "(십억원)"}]
                current_paths, current_root_label = self._build_row_paths(current_labels, current_root_label)
            else:
                current_paths = []

            for column_index in range(1, len(row)):
                periods = period_groups[column_index - 1] if column_index - 1 < len(period_groups) else []
                values = self._split_table_cell(row[column_index])
                if not periods or not values:
                    continue

                paths_for_column = current_paths if current_paths else pending_paths.get(column_index, [])
                if not paths_for_column:
                    continue

                fact_group = self._build_fact_rows(paths_for_column, periods, values)
                if not fact_group:
                    continue

                fact_rows.extend(fact_group)
                covered = len(fact_group)
                remainder = current_paths[covered:] if current_paths else paths_for_column[covered:]
                if remainder:
                    pending_paths[column_index] = remainder
                else:
                    pending_paths.pop(column_index, None)

        merged_fact_rows: list[FactRow] = []
        fact_row_lookup: dict[str, FactRow] = {}
        for fact_row in fact_rows:
            lookup_key = " > ".join(fact_row.row_path)
            existing = fact_row_lookup.get(lookup_key)
            if existing is None:
                merged = FactRow(row_path=list(fact_row.row_path), facts=list(fact_row.facts))
                fact_row_lookup[lookup_key] = merged
                merged_fact_rows.append(merged)
                continue
            existing.facts.extend(fact_row.facts)

        if len(merged_fact_rows) < 3:
            return None

        for fact_row in merged_fact_rows:
            rendered.append(
                f"- [row_path] {' > '.join(fact_row.row_path)}: "
                + " | ".join(f"{period}={value}" for period, value in fact_row.facts)
            )

        summary["financial_tables_structured"] += 1
        summary["financial_fact_rows_structured"] += len(merged_fact_rows)
        rule_hits = summary.setdefault("financial_table_rule_hits", {})
        if isinstance(rule_hits, dict):
            rule_hits["hierarchical_periodic_table"] = int(rule_hits.get("hierarchical_periodic_table", 0) or 0) + 1
        summary["changed"] = True
        return rendered

    def _build_fact_rows(self, row_paths: list[list[str]], periods: list[str], values: list[str]) -> list[FactRow]:
        width = len(periods)
        if not row_paths or not periods or len(values) < width:
            return []

        row_count = min(len(row_paths), len(values) // width)
        fact_rows: list[FactRow] = []
        for row_index in range(row_count):
            row_path = [self._normalize_fact_label(part) for part in row_paths[row_index]]
            row_path = [part for part in row_path if part]
            if not row_path:
                continue
            start = row_index * width
            row_values = values[start : start + width]
            if len(row_values) != width:
                continue
            fact_rows.append(FactRow(row_path=row_path, facts=list(zip(periods, row_values))))
        return fact_rows

    def _try_render_packed_period_table(
        self,
        *,
        data_rows: list[list[str]],
        period_groups: list[list[str]],
        base_rendered: list[str],
        summary: dict[str, Any],
    ) -> list[str] | None:
        flat_periods: list[str] = []
        for group in period_groups:
            for token in group:
                if self._looks_like_financial_period_token(token):
                    flat_periods.append(self._normalize_inline_markdown(token))

        if len(flat_periods) < 4:
            return None

        packed_rows: list[FactRow] = []
        for row in data_rows:
            fact_row = self._parse_packed_period_row(row, flat_periods)
            if fact_row is None:
                continue
            packed_rows.append(fact_row)

        if len(packed_rows) < 4:
            return None

        rendered = list(base_rendered)
        for fact_row in packed_rows:
            rendered.append(
                f"- [row_path] {' > '.join(fact_row.row_path)}: "
                + " | ".join(f"{period}={value}" for period, value in fact_row.facts)
            )

        summary["financial_tables_structured"] += 1
        summary["financial_fact_rows_structured"] += len(packed_rows)
        rule_hits = summary.setdefault("financial_table_rule_hits", {})
        if isinstance(rule_hits, dict):
            rule_hits["packed_period_summary_table"] = int(rule_hits.get("packed_period_summary_table", 0) or 0) + 1
        summary["changed"] = True
        return rendered

    def _parse_packed_period_row(self, row: list[str], periods: list[str]) -> FactRow | None:
        if not row:
            return None
        first_tokens = self._split_table_cell(row[0])
        if not first_tokens:
            return None

        labels: list[str] = []
        values: list[str] = []
        seen_value = False
        for token in first_tokens:
            if self._looks_like_value(token):
                seen_value = True
                values.append(self._normalize_inline_markdown(token))
                continue
            if seen_value:
                return None
            labels.append(self._normalize_fact_label(token))

        if len(labels) != 1:
            return None

        for cell in row[1:]:
            for token in self._split_table_cell(cell):
                normalized = self._normalize_inline_markdown(token)
                if self._looks_like_value(normalized):
                    values.append(normalized)

        if len(values) < len(periods):
            return None

        facts = list(zip(periods, values[: len(periods)]))
        return FactRow(row_path=[labels[0]], facts=facts)

    def _build_row_paths(self, labels: list[str], current_root_label: str | None) -> tuple[list[list[str]], str | None]:
        normalized: list[str] = []
        for label in labels:
            cleaned = self._normalize_fact_label(label)
            if cleaned:
                normalized.append(cleaned)
        if not normalized:
            return [], current_root_label

        index = 0
        root_label = current_root_label
        include_root_row = False
        if root_label is None or self._is_section_like_label(normalized[0]):
            root_label = normalized[0]
            index = 1
            include_root_row = True

        if root_label is None:
            return [], current_root_label

        row_paths: list[list[str]] = []
        if include_root_row:
            row_paths.append([root_label])
        active_parent_path: list[str] | None = None
        remaining = normalized[index:]

        for position, label in enumerate(remaining):
            next_label = remaining[position + 1] if position + 1 < len(remaining) else None
            role = self._classify_row_label(label, next_label)

            if role == "section":
                path = [root_label, label]
                active_parent_path = path if next_label is not None and not self._is_section_like_label(next_label) else None
            elif role == "parent":
                if active_parent_path and self._is_section_like_label(active_parent_path[-1]):
                    path = [*active_parent_path, label]
                else:
                    path = [root_label, label]
                active_parent_path = path
            elif role == "child" and active_parent_path:
                path = [*active_parent_path, label]
            else:
                path = [root_label, label]
                active_parent_path = None

            row_paths.append(path)

        if not row_paths:
            row_paths.append([root_label])
        return row_paths, root_label

    def _classify_row_label(self, label: str, next_label: str | None) -> str:
        if self._is_section_like_label(label):
            return "section"
        if self._is_parent_like_label(label):
            return "parent"
        if self._is_child_like_label(label):
            return "child"
        if next_label and self._is_child_like_label(next_label):
            return "parent"
        return "leaf"

    def _is_section_like_label(self, label: str) -> bool:
        lowered = label.casefold()
        section_hints = ("매출", "이익", "성장률", "margin", "profit", "revenue", "재무", "비용")
        return any(hint in lowered for hint in section_hints)

    def _is_parent_like_label(self, label: str) -> bool:
        lowered = label.casefold()
        parent_hints = ("north america", "south america", "emea", "apac", "alao", "europe", "asia", "korea", "china", "japan")
        return any(hint in lowered for hint in parent_hints)

    def _is_child_like_label(self, label: str) -> bool:
        lowered = label.casefold()
        child_hints = ("bobcat", "industrial truck", "compact", "gme", "portable power", "mottrol")
        return any(hint in lowered for hint in child_hints)

    def _looks_like_financial_period_token(self, token: str) -> bool:
        normalized = self._normalize_inline_markdown(token)
        if not normalized:
            return False
        return COMMON_PERIOD_PATTERN.fullmatch(normalized) is not None or re.fullmatch(r"[1-4]Q\d{2}[A-Z]?", normalized, re.IGNORECASE) is not None

    def _serialize_repeated_catalog(self, repeated_catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        items = list(repeated_catalog.values())
        items.sort(key=lambda item: (-item["repeat_ratio"], item["canonical_text"]))
        return items

    def _render_pages(self, pages: list[tuple[int, list[str], dict[str, Any]]]) -> str:
        parts: list[str] = []
        for page_number, lines, _ in pages:
            parts.append(f"# Page {page_number}")
            parts.append("")
            parts.extend(lines)
            parts.append("")
        return "\n".join(parts).strip() + "\n"
