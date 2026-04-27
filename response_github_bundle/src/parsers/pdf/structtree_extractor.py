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
        # StructTree 루트 찾기
        catalog_xref = document.pdf_catalog()
        struct_root = document.xref_get_key(catalog_xref, "StructTreeRoot")
        if struct_root[0] != "xref":
            return "", {"used": False, "reason": "no-struct-tree"}

        root_xref = int(struct_root[1].split()[0])
        page_map = {document.page_xref(i): i + 1 for i in range(document.page_count)}

        # 테이블 xref → 파싱된 행 데이터 수집
        table_xrefs: dict[int, list[list[dict[str, Any]]]] = {}
        self._find_tables(document, root_xref, page_map, None, table_xrefs)

        # 테이블에 속하는 모든 하위 xref 수집 (run에서 제외하기 위해)
        table_member_xrefs: set[int] = set()
        for table_xref in table_xrefs:
            self._collect_descendant_xrefs(document, table_xref, table_member_xrefs)

        # 테이블 xref → 페이지 매핑
        table_page_map: dict[int, int] = {}
        if table_xrefs:
            self._map_table_pages(document, root_xref, page_map, None, table_page_map, set(table_xrefs.keys()))

        runs = self.extract_runs(document)
        mcid_lookup = self._build_mcid_lookup_from_runs(runs)

        if not runs and not table_xrefs:
            return "", {"used": False, "reason": "no-actualtext-runs"}

        markdown_lines: list[str] = []
        current_page: int | None = None
        current_block_id: int | None = None
        current_block_role: str | None = None
        current_fragments: list[str] = []
        block_count = 0
        emitted_tables: set[int] = set()
        table_count = 0

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

        def emit_page_tables(page_number: int) -> None:
            nonlocal table_count
            for txref, rows in table_xrefs.items():
                if txref in emitted_tables:
                    continue
                if table_page_map.get(txref) != page_number:
                    continue
                emitted_tables.add(txref)
                md = self._render_table_block(
                    document=document,
                    page_number=page_number,
                    table_xref=txref,
                    rows=rows,
                    page_mcid_lookup=mcid_lookup.get(page_number, {}),
                )
                if md:
                    table_count += 1
                    markdown_lines.append(md)
                    markdown_lines.append("")

        for run in runs:
            # 테이블 소속 run은 건너뛰고, 해당 페이지의 테이블을 마크다운으로 삽입
            if run.block_id in table_member_xrefs:
                if run.page_number != current_page:
                    flush_block()
                    current_page = run.page_number
                    markdown_lines.append(f"# Page {run.page_number}")
                    markdown_lines.append("")
                    current_block_id = None
                    current_block_role = None
                emit_page_tables(run.page_number)
                continue

            if run.page_number != current_page:
                flush_block()
                if current_page is not None:
                    emit_page_tables(current_page)
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
        if current_page is not None:
            emit_page_tables(current_page)

        # 어떤 페이지에도 emit 안 된 테이블 처리
        for txref, rows in table_xrefs.items():
            if txref not in emitted_tables:
                page = table_page_map.get(txref)
                if page:
                    emitted_tables.add(txref)
                    md = self._render_table_block(
                        document=document,
                        page_number=page,
                        table_xref=txref,
                        rows=rows,
                        page_mcid_lookup=mcid_lookup.get(page, {}),
                    )
                    if md:
                        table_count += 1
                        markdown_lines.append(f"# Page {page}")
                        markdown_lines.append("")
                        markdown_lines.append(md)
                        markdown_lines.append("")

        markdown = "\n".join(markdown_lines).strip()
        metadata = {
            "used": bool(markdown),
            "source": "structtree-actualtext",
            "run_count": len(runs),
            "block_count": block_count,
            "table_count": table_count,
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

    # ------------------------------------------------------------------
    # Table 구조 파싱 (StructTree Table→TR→TD/TH 계층)
    # ------------------------------------------------------------------

    def _find_tables(
        self,
        document: fitz.Document,
        xref: int,
        page_map: dict[int, int],
        inherited_page: int | None,
        result: dict[int, list[list[dict[str, Any]]]],
    ) -> None:
        """StructTree를 재귀 탐색하며 Table 노드를 찾고 행/열 데이터를 파싱."""
        role = self._get_name_value(document, xref, "S")
        page_number = inherited_page
        page_value = document.xref_get_key(xref, "Pg")
        if page_value[0] == "xref":
            page_xref = int(page_value[1].split()[0])
            page_number = page_map.get(page_xref, inherited_page)

        if role == "Table":
            rows = self._parse_table_node(document, xref)
            if rows:
                result[xref] = rows
            return

        for kind, child_xref in self._parse_k_value(document.xref_get_key(xref, "K")):
            if kind == "xref":
                self._find_tables(document, child_xref, page_map, page_number, result)

    def _parse_table_node(self, document: fitz.Document, table_xref: int) -> list[list[dict[str, Any]]]:
        """Table xref에서 TR/TD/TH 계층을 파싱하여 행 리스트를 반환."""
        rows: list[list[dict[str, Any]]] = []
        for kind, child_xref in self._parse_k_value(document.xref_get_key(table_xref, "K")):
            if kind != "xref":
                continue
            role = self._get_name_value(document, child_xref, "S")
            if role == "TR":
                cells = self._parse_row_node(document, child_xref)
                if cells:
                    rows.append(cells)
            elif role in {"THead", "TBody", "TFoot"}:
                for gk, gx in self._parse_k_value(document.xref_get_key(child_xref, "K")):
                    if gk == "xref" and self._get_name_value(document, gx, "S") == "TR":
                        cells = self._parse_row_node(document, gx)
                        if cells:
                            rows.append(cells)
        return rows

    def _parse_row_node(self, document: fitz.Document, tr_xref: int) -> list[dict[str, Any]]:
        """TR 노드에서 TD/TH 셀들을 파싱."""
        cells: list[dict[str, Any]] = []
        for kind, child_xref in self._parse_k_value(document.xref_get_key(tr_xref, "K")):
            if kind != "xref":
                continue
            role = self._get_name_value(document, child_xref, "S")
            if role in {"TD", "TH"}:
                cell_text = self._collect_text_recursive(document, child_xref)
                cells.append({"role": role, "text": cell_text, "xref": child_xref})
        return cells

    def _collect_text_recursive(self, document: fitz.Document, xref: int) -> str:
        """노드와 하위 노드에서 ActualText를 재귀적으로 수집."""
        texts: list[str] = []
        actual_text = self._get_string_value(document, xref, "ActualText") or self._get_string_value(document, xref, "Alt")
        if actual_text:
            texts.append(actual_text.replace("\ufeff", "").strip())
        for kind, child_xref in self._parse_k_value(document.xref_get_key(xref, "K")):
            if kind == "xref":
                child_text = self._collect_text_recursive(document, child_xref)
                if child_text:
                    texts.append(child_text)
        return " ".join(texts).strip()

    def _table_rows_to_markdown(self, rows: list[list[dict[str, Any]]]) -> str:
        """파싱된 행 데이터를 마크다운 테이블 문자열로 변환."""
        if not rows:
            return ""
        col_count = max(len(row) for row in rows)
        lines: list[str] = []
        for row_idx, row in enumerate(rows):
            cells = [c["text"].replace("|", "\\|").replace("\n", " ") for c in row]
            while len(cells) < col_count:
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")
            if row_idx == 0:
                lines.append("| " + " | ".join(["---"] * col_count) + " |")
        return "\n".join(lines)

    def _table_rows_to_markdown(self, rows: list[list[dict[str, Any]]]) -> str:
        """Render StructTree table rows as markdown while preserving a left stub header when possible."""
        if not rows:
            return ""
        normalized_rows = self._normalize_table_rows(rows)
        col_count = max(len(row) for row in normalized_rows)
        lines: list[str] = []
        for row_idx, row in enumerate(normalized_rows):
            cells = [c["text"].replace("|", "\\|").replace("\n", " ") for c in row]
            while len(cells) < col_count:
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")
            if row_idx == 0:
                lines.append("| " + " | ".join(["---"] * col_count) + " |")
        return "\n".join(lines)

    def _normalize_table_rows(self, rows: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        if not rows:
            return []

        normalized_rows = [list(row) for row in rows]
        row_lengths = [len(row) for row in normalized_rows if row]
        if not row_lengths:
            return normalized_rows

        target_col_count = max(set(row_lengths), key=row_lengths.count)
        if target_col_count <= 1:
            return normalized_rows

        first_row = normalized_rows[0]
        if (
            len(first_row) == target_col_count - 1
            and len(normalized_rows) > 1
            and self._should_prepend_stub_header(first_row, normalized_rows[1:], target_col_count)
        ):
            first_row.insert(0, {"role": "TH", "text": "", "xref": None})

        return normalized_rows

    def _should_prepend_stub_header(
        self,
        header_row: list[dict[str, Any]],
        body_rows: list[list[dict[str, Any]]],
        target_col_count: int,
    ) -> bool:
        full_rows = [row for row in body_rows if len(row) == target_col_count]
        if not full_rows:
            return False

        sample_row = full_rows[0]
        if len(sample_row) < 2:
            return False

        stub_text = sample_row[0].get("text", "")
        if not stub_text or self._looks_numeric(stub_text):
            return False

        return (
            any(self._looks_value_header(cell.get("text", "")) for cell in header_row)
            and any(self._looks_numeric(cell.get("text", "")) for cell in sample_row[1:])
        )

    def _looks_numeric(self, text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        candidate = candidate.replace(",", "").replace("%", "").replace("’", "'")
        candidate = re.sub(r"\s+", "", candidate)
        candidate = candidate.replace("△", "-").replace("▲", "+")
        return bool(re.fullmatch(r"[+\-−]?\d+(?:\.\d+)?", candidate))

    def _looks_value_header(self, text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        if self._looks_numeric(candidate):
            return True

        normalized = candidate.upper().replace("’", "'").replace("‘", "'")
        normalized = re.sub(r"\s+", "", normalized)
        if re.fullmatch(r"\dQ\d{2,4}", normalized):
            return True
        if re.fullmatch(r"'\d{2,4}(YTD|Q\d|H\d|FY)", normalized):
            return True
        if re.fullmatch(r"\d{2,4}(YTD|Q\d|H\d|FY)", normalized):
            return True
        return normalized.endswith("YTD")

    def _map_table_pages(
        self,
        document: fitz.Document,
        xref: int,
        page_map: dict[int, int],
        inherited_page: int | None,
        result: dict[int, int],
        target_xrefs: set[int],
    ) -> None:
        """테이블 xref가 어느 페이지에 속하는지 매핑."""
        page_number = inherited_page
        page_value = document.xref_get_key(xref, "Pg")
        if page_value[0] == "xref":
            page_xref = int(page_value[1].split()[0])
            page_number = page_map.get(page_xref, inherited_page)

        if xref in target_xrefs and page_number is not None:
            result[xref] = page_number

        for kind, child_xref in self._parse_k_value(document.xref_get_key(xref, "K")):
            if kind == "xref":
                self._map_table_pages(document, child_xref, page_map, page_number, result, target_xrefs)

    def _collect_descendant_xrefs(self, document: fitz.Document, xref: int, result: set[int]) -> None:
        """특정 xref의 모든 하위 xref를 재귀 수집."""
        result.add(xref)
        for kind, child_xref in self._parse_k_value(document.xref_get_key(xref, "K")):
            if kind == "xref":
                self._collect_descendant_xrefs(document, child_xref, result)

    def _build_mcid_lookup_from_runs(self, runs: list[StructTextRun]) -> dict[int, dict[int, list[dict[str, Any]]]]:
        lookup: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for run in runs:
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

    def _render_table_block(
        self,
        document: fitz.Document,
        page_number: int,
        table_xref: int,
        rows: list[list[dict[str, Any]]],
        page_mcid_lookup: dict[int, list[dict[str, Any]]],
    ) -> str:
        hybrid = self._render_hybrid_irregular_table(
            document=document,
            page_number=page_number,
            table_xref=table_xref,
            rows=rows,
            page_mcid_lookup=page_mcid_lookup,
        )
        if hybrid:
            return hybrid
        return self._table_rows_to_markdown(rows)

    def _render_hybrid_irregular_table(
        self,
        document: fitz.Document,
        page_number: int,
        table_xref: int,
        rows: list[list[dict[str, Any]]],
        page_mcid_lookup: dict[int, list[dict[str, Any]]],
    ) -> str:
        del table_xref
        row_lengths = [len(row) for row in rows]
        if row_lengths != [3, 5, 4, 5]:
            return ""

        page = document[page_number - 1]
        image_elements = self._build_page_image_elements(page, page_mcid_lookup)
        if not image_elements:
            return ""

        matched_boxes = self._match_table_cells_to_image_boxes(rows, image_elements)
        if sum(1 for box in matched_boxes if box is not None) < 8:
            return ""

        normalized_rows = self._normalize_irregular_goal_table_rows(rows)
        if len(normalized_rows) != 4 or any(len(row) != 5 for row in normalized_rows[1:]):
            return ""

        header_labels = [self._escape_html(cell.get("text", "")) for cell in normalized_rows[0]]
        body_rows = [
            [self._escape_html(cell.get("text", "")) for cell in row]
            for row in normalized_rows[1:]
        ]

        lines = [
            "<table>",
            "  <thead>",
            "    <tr>",
            f"      <th colspan=\"2\">{header_labels[0]}</th>",
            f"      <th colspan=\"2\">{header_labels[1]}</th>",
            f"      <th>{header_labels[2]}</th>",
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
        for row in body_rows:
            lines.append("    <tr>")
            for value in row:
                lines.append(f"      <td>{value}</td>")
            lines.append("    </tr>")
        lines.extend(["  </tbody>", "</table>"])
        return "\n".join(lines)

    def _normalize_irregular_goal_table_rows(self, rows: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        normalized = [list(row) for row in rows]
        if len(normalized) != 4:
            return normalized
        if len(normalized[1]) == 5 and len(normalized[2]) == 4:
            repeated_stub = dict(normalized[1][0])
            normalized[2].insert(0, repeated_stub)
        return normalized

    def _build_page_image_elements(
        self,
        page: fitz.Page,
        page_mcid_lookup: dict[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        try:
            image_infos = sorted(page.get_image_info(xrefs=True), key=lambda item: item.get("number", 0))
        except Exception:
            return []

        if not image_infos:
            return []

        resource_map = self._build_image_resource_map(page)
        draw_ops = self._extract_image_draw_operations(page, set(resource_map))
        elements: list[dict[str, Any]] = []
        for index, info in enumerate(image_infos):
            bbox = [float(value) for value in info.get("bbox", ())]
            if len(bbox) != 4:
                continue
            draw_op = draw_ops[index] if index < len(draw_ops) else {}
            mcid = draw_op.get("mcid")
            mcid_matches = page_mcid_lookup.get(mcid, []) if isinstance(mcid, int) else []
            text = self._merge_mcid_text(mcid_matches)
            if not text:
                continue
            elements.append({"text": text, "bbox": bbox})
        elements.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return elements

    def _build_image_resource_map(self, page: fitz.Page) -> dict[str, dict[str, Any]]:
        resources: dict[str, dict[str, Any]] = {}
        for item in page.get_images(full=True):
            if len(item) < 8:
                continue
            name = str(item[7] or "")
            if not name:
                continue
            resources[name] = {"name": name}
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

    def _match_table_cells_to_image_boxes(
        self,
        rows: list[list[dict[str, Any]]],
        image_elements: list[dict[str, Any]],
    ) -> list[list[float] | None]:
        normalized_tokens = [self._normalize_match_text(element["text"]) for element in image_elements]
        matches: list[list[float] | None] = []
        for row in rows:
            for cell in row:
                target = self._normalize_match_text(cell.get("text", ""))
                if not target:
                    matches.append(None)
                    continue
                best_index: int | None = None
                best_score = 0.0
                for index, candidate in enumerate(normalized_tokens):
                    score = self._score_match_text(candidate, target)
                    if score > best_score:
                        best_score = score
                        best_index = index
                matches.append(image_elements[best_index]["bbox"] if best_index is not None and best_score > 0 else None)
        return matches

    def _normalize_match_text(self, text: str) -> str:
        normalized = (text or "")
        normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
        normalized = normalized.replace("\u201c", '\"').replace("\u201d", '\"')
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip().lower()

    def _score_match_text(self, candidate: str, target: str) -> float:
        if not candidate or not target:
            return 0.0
        if candidate == target:
            return 1000.0
        if candidate in target:
            return 700.0 - abs(len(target) - len(candidate))
        if target in candidate:
            return 650.0 - abs(len(candidate) - len(target))

        candidate_tokens = set(candidate.split())
        target_tokens = [token for token in target.split() if token]
        if not candidate_tokens or not target_tokens:
            return 0.0

        overlap = 0.0
        for token in target_tokens:
            if token in candidate_tokens:
                overlap += 10.0 if len(token) > 1 else 3.0
        return overlap

    def _escape_html(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('\"', "&quot;")
        )
