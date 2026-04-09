from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import fitz


BLOCK_ROLES = {
    "Document",
    "Part",
    "Sect",
    "Div",
    "P",
    "L",
    "LI",
    "Title",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "Textbox",
    "Caption",
    "Note",
    "Table",
    "TR",
    "TD",
    "TH",
}
HEADING_ROLES = {"Title", "H1", "H2", "H3", "H4", "H5", "H6"}
WORD_CHAR_PATTERN = re.compile(r"[0-9A-Za-z\u3131-\u318E\uAC00-\uD7A3]")


@dataclass
class StructTextRun:
    page_number: int
    block_id: int
    block_role: str
    leaf_role: str
    text: str
    mcids: list[int]


class PowerPointStructTreeExtractor:
    def extract_markdown(self, document: fitz.Document) -> tuple[str, dict[str, Any]]:
        runs = self.extract_runs(document)

        if not runs:
            return "", {"used": False, "reason": "no-actualtext-runs"}

        markdown_lines: list[str] = []
        current_page: int | None = None
        current_block_id: int | None = None
        current_block_role: str | None = None
        current_fragments: list[str] = []
        block_count = 0

        def flush_block() -> None:
            nonlocal current_fragments, block_count
            if current_page is None or not current_fragments:
                current_fragments = []
                return

            block_text = self._normalize_text(self._join_fragments(current_fragments))
            current_fragments = []
            if not block_text:
                return

            block_count += 1
            if current_block_role in HEADING_ROLES:
                heading_level = 1 if current_block_role == "Title" else int(current_block_role[-1])
                markdown_lines.append(f"{'#' * min(max(heading_level, 1), 6)} {block_text}")
            elif current_block_role == "LI":
                block_text = re.sub(r"^[\-\u2022\u25AA\u25CF\uf0a7+\s]+", "", block_text)
                markdown_lines.append(f"- {block_text}")
            else:
                markdown_lines.append(block_text)
            markdown_lines.append("")

        for run in runs:
            if run.page_number != current_page:
                flush_block()
                current_page = run.page_number
                markdown_lines.append(f"# Page {run.page_number}")
                markdown_lines.append("")
                current_block_id = None
                current_block_role = None

            if run.block_id != current_block_id:
                flush_block()
                current_block_id = run.block_id
                current_block_role = run.block_role

            current_fragments.append(run.text)

        flush_block()

        markdown = "\n".join(markdown_lines).strip()
        metadata = {
            "used": bool(markdown),
            "source": "structtree-actualtext",
            "run_count": len(runs),
            "block_count": block_count,
            "pages": sorted({run.page_number for run in runs}),
        }
        return markdown, metadata

    def extract_runs(self, document: fitz.Document) -> list[StructTextRun]:
        catalog_xref = document.pdf_catalog()
        struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
        if struct_root[0] != "xref":
            return []

        root_xref = int(struct_root[1].split()[0])
        page_map = {document.page_xref(index): index + 1 for index in range(document.page_count)}
        runs: list[StructTextRun] = []

        self._walk_struct_tree(
            document=document,
            xref=root_xref,
            page_map=page_map,
            runs=runs,
            inherited_page=None,
            block_xref=None,
            block_role=None,
        )
        return runs

    def build_mcid_lookup(self, document: fitz.Document) -> dict[int, dict[int, list[dict[str, Any]]]]:
        lookup: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for run in self.extract_runs(document):
            for mcid in run.mcids:
                lookup[run.page_number][mcid].append(
                    {
                        "text": self._normalize_text(run.text),
                        "block_role": run.block_role,
                        "leaf_role": run.leaf_role,
                        "block_id": run.block_id,
                    }
                )

        return {
            page_number: {mcid: items for mcid, items in mcid_map.items()}
            for page_number, mcid_map in lookup.items()
        }

    def _walk_struct_tree(
        self,
        document: fitz.Document,
        xref: int,
        page_map: dict[int, int],
        runs: list[StructTextRun],
        inherited_page: int | None,
        block_xref: int | None,
        block_role: str | None,
    ) -> None:
        role_name = self._get_name_value(document, xref, "S")
        page_number = inherited_page
        page_value = document.xref_get_key(xref, "Pg")
        if page_value[0] == "xref":
            page_xref = int(page_value[1].split()[0])
            page_number = page_map.get(page_xref, inherited_page)

        next_block_xref = block_xref
        next_block_role = block_role
        if role_name in BLOCK_ROLES:
            next_block_xref = xref
            next_block_role = role_name

        actual_text = self._get_string_value(document, xref, "ActualText") or self._get_string_value(document, xref, "Alt")
        child_items = self._parse_k_value(document.xref_get_key(xref, "K"))

        if actual_text and page_number is not None:
            mcids = [value for kind, value in child_items if kind == "int"]
            runs.append(
                StructTextRun(
                    page_number=page_number,
                    block_id=next_block_xref or xref,
                    block_role=next_block_role or role_name or "P",
                    leaf_role=role_name or "Span",
                    text=actual_text,
                    mcids=mcids,
                )
            )

        for item_kind, value in child_items:
            if item_kind == "xref":
                self._walk_struct_tree(
                    document=document,
                    xref=value,
                    page_map=page_map,
                    runs=runs,
                    inherited_page=page_number,
                    block_xref=next_block_xref,
                    block_role=next_block_role,
                )

    def _parse_k_value(self, k_value: tuple[str, str]) -> list[tuple[str, int]]:
        value_type, value = k_value
        if value_type == "xref":
            return [("xref", int(value.split()[0]))]
        if value_type != "array":
            return []

        refs = [("xref", int(match)) for match in re.findall(r"(\d+) 0 R", value)]
        if refs:
            return refs

        return [("int", int(match)) for match in re.findall(r"(?<!\d)(\d+)(?!\d)", value)]

    def _get_name_value(self, document: fitz.Document, xref: int, key: str) -> str | None:
        value_type, value = document.xref_get_key(xref, key)
        if value_type == "name":
            return value.lstrip("/")
        return None

    def _get_string_value(self, document: fitz.Document, xref: int, key: str) -> str | None:
        value_type, value = document.xref_get_key(xref, key)
        if value_type == "string":
            return value
        return None

    def _normalize_text(self, text: str) -> str:
        cleaned = text.replace("\ufeff", "")
        cleaned = cleaned.replace("\uf0a7", "•")
        cleaned = cleaned.replace("\u2028", "\n").replace("\u2029", "\n")
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    def _join_fragments(self, fragments: list[str]) -> str:
        if not fragments:
            return ""

        joined = fragments[0]
        for fragment in fragments[1:]:
            if self._needs_space(joined, fragment):
                joined += " "
            joined += fragment
        return joined

    def _needs_space(self, previous: str, current: str) -> bool:
        prev = previous.rstrip()
        curr = current.lstrip()
        if not prev or not curr:
            return False
        if curr[0] in ",.:;!?)]}%":
            return False
        if prev[-1] in "([{%":
            return False
        return bool(WORD_CHAR_PATTERN.match(prev[-1])) and bool(WORD_CHAR_PATTERN.match(curr[0]))
