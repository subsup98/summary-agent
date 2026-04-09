from __future__ import annotations
import difflib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

from src.shared.io import ensure_directory, iso_now, read_text_with_fallback, write_json, write_text


class ParsingReviewSiteBuilder:
    def __init__(
        self,
        parsed_root: Path,
        site_root: Path,
        overlay_site_root: Path | None = None,
    ) -> None:
        self.parsed_root = parsed_root
        self.site_root = site_root
        self.overlay_site_root = overlay_site_root

    def build(self, run_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        ensure_directory(self.site_root)
        documents: list[dict[str, Any]] = []

        for json_path in sorted(self.parsed_root.glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("status") != "parsed":
                continue
            documents.append(self._build_document(payload))

        total_pages = sum(int(document.get("page_count", 0)) for document in documents)
        total_issues = sum(int(document.get("issue_count", 0)) for document in documents)
        total_text_characters = sum(int(document.get("text_characters", 0)) for document in documents)
        total_elements = sum(int(document.get("element_count", 0)) for document in documents)

        manifest = {
            "generated_at": iso_now(),
            "site_root": self.site_root.as_posix(),
            "index_path": (self.site_root / "index.html").as_posix(),
            "document_count": len(documents),
            "total_pages": total_pages,
            "total_issues": total_issues,
            "total_text_characters": total_text_characters,
            "total_elements": total_elements,
            "overlay_index_path": self._overlay_index_path(),
            "run_summary": self._build_run_summary(run_summary),
            "documents": documents,
        }
        write_text(self.site_root / "index.html", self._render_index_html(manifest))
        write_json(self.site_root / "manifest.json", manifest)
        return manifest

    def _normalize_match_key(self, value: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u3131-\u318E\uAC00-\uD7A3]+", "_", value.strip().lower()).strip("_")

    def _infer_project_root(self) -> Path | None:
        for base in (self.parsed_root, self.site_root):
            current = base.resolve()
            for parent in [current, *current.parents]:
                if (parent / "data").exists() and (parent / "outputs").exists():
                    return parent
        return None

    def _resolve_comparison_root(self) -> Path:
        return self.parsed_root.parent.parent / "comparisons"

    def _read_optional_text(self, path: Path | None) -> str:
        if not path or not path.exists() or path.is_dir():
            return ""
        text, _ = read_text_with_fallback(path)
        return text

    def _find_matching_directory(self, root: Path, candidate_keys: set[str]) -> Path | None:
        if not root.exists():
            return None
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            child_key = self._normalize_match_key(child.name)
            if child_key and child_key in candidate_keys:
                return child
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            child_key = self._normalize_match_key(child.name)
            if child_key and any(key and (key in child_key or child_key in key) for key in candidate_keys):
                return child
        return None

    def _load_postprocess_comparison(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        source_name = str(payload.get("source_name", "")).strip()
        source_path = str(payload.get("source_path", "")).strip()
        document_id = str(payload.get("document_id", "")).strip()
        extension = str(payload.get("extension", "")).strip().lower().lstrip(".")
        document_type = str((payload.get("classification") or {}).get("document_type") or extension).strip().lower()
        candidate_keys = {
            self._normalize_match_key(value)
            for value in (
                source_name,
                Path(source_name).stem if source_name else "",
                source_path,
                Path(source_path).stem if source_path else "",
                document_id,
                document_id.split("--", 1)[0] if document_id else "",
            )
            if value
        }
        candidate_keys.discard("")

        comparison_root = self._resolve_comparison_root()

        if document_type in {"pdf", "doc"}:
            format_root = comparison_root / document_type
            match_dir = self._find_matching_directory(format_root, candidate_keys)
            if not match_dir:
                return None
            metadata = {}
            metadata_path = match_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stages = metadata.get("stages") or {}
            before_path = match_dir / str((stages.get("before") or {}).get("file") or "")
            after_path = match_dir / str((stages.get("after") or {}).get("file") or "")
            raw_path = match_dir / str((stages.get("raw") or {}).get("file") or "")
            return {
                "format": document_type,
                "label": "Postprocess Comparison",
                "root": match_dir.as_posix(),
                "metadata": metadata,
                "raw_text": self._read_optional_text(raw_path),
                "before_text": self._read_optional_text(before_path),
                "after_text": self._read_optional_text(after_path),
                "before_path": before_path.as_posix() if before_path.exists() else None,
                "after_path": after_path.as_posix() if after_path.exists() else None,
                "raw_path": raw_path.as_posix() if raw_path.exists() else None,
            }

        if document_type == "hwp":
            project_root = self._infer_project_root()
            if not project_root:
                return None
            before_root = project_root / "data" / "before" / "hwp"
            after_root = project_root / "data" / "after" / "hwp"
            before_dir = self._find_matching_directory(before_root, candidate_keys)
            after_dir = self._find_matching_directory(after_root, candidate_keys)
            if not before_dir or not after_dir:
                return None
            before_path = before_dir / "raw_extraction.md"
            after_path = after_dir / "postprocessed.md"
            before_metadata = {}
            after_metadata = {}
            if (before_dir / "metadata.json").exists():
                before_metadata = json.loads((before_dir / "metadata.json").read_text(encoding="utf-8"))
            if (after_dir / "metadata.json").exists():
                after_metadata = json.loads((after_dir / "metadata.json").read_text(encoding="utf-8"))
            return {
                "format": "hwp",
                "label": "Postprocess Comparison",
                "root": before_dir.as_posix(),
                "metadata": {
                    "stages": {
                        "before": {
                            "file": before_path.name,
                            "description": before_metadata.get("description", "Raw extraction before postprocessing"),
                            "line_count": before_metadata.get("line_count"),
                        },
                        "after": {
                            "file": after_path.name,
                            "description": after_metadata.get("description", "Structured output after postprocessing"),
                            "line_count": after_metadata.get("line_count"),
                            "rule_version": after_metadata.get("rule_version"),
                            "block_count": after_metadata.get("block_count"),
                            "chunk_count": after_metadata.get("chunk_count"),
                            "kv_table_count": after_metadata.get("kv_table_count"),
                        },
                    }
                },
                "before_text": self._read_optional_text(before_path),
                "after_text": self._read_optional_text(after_path),
                "before_path": before_path.as_posix() if before_path.exists() else None,
                "after_path": after_path.as_posix() if after_path.exists() else None,
            }
        return None

    def _tokenize_for_diff(self, value: str) -> list[str]:
        return re.findall(r"\s+|[^\s]+", value)

    def _render_diff_tokens(self, tokens: list[str], css_class: str | None = None) -> str:
        escaped = html.escape("".join(tokens))
        if not css_class:
            return escaped
        return '<mark class="{css_class}">{content}</mark>'.format(
            css_class=css_class,
            content=escaped,
        )

    def _render_word_level_diff(self, before_text: str, after_text: str) -> tuple[str, str]:
        before_tokens = self._tokenize_for_diff(before_text)
        after_tokens = self._tokenize_for_diff(after_text)
        matcher = difflib.SequenceMatcher(a=before_tokens, b=after_tokens)
        before_parts: list[str] = []
        after_parts: list[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            before_chunk = before_tokens[i1:i2]
            after_chunk = after_tokens[j1:j2]
            if tag == "equal":
                before_parts.append(self._render_diff_tokens(before_chunk))
                after_parts.append(self._render_diff_tokens(after_chunk))
            elif tag == "delete":
                before_parts.append(self._render_diff_tokens(before_chunk, "diff-del"))
            elif tag == "insert":
                after_parts.append(self._render_diff_tokens(after_chunk, "diff-add"))
            else:
                before_parts.append(self._render_diff_tokens(before_chunk, "diff-del"))
                after_parts.append(self._render_diff_tokens(after_chunk, "diff-add"))
        return "".join(before_parts), "".join(after_parts)

    def _build_diff_blocks(self, before_text: str, after_text: str) -> list[dict[str, Any]]:
        before_lines = before_text.splitlines()
        after_lines = after_text.splitlines()
        matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
        blocks: list[dict[str, Any]] = []
        change_index = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            change_index += 1
            before_chunk = "\n".join(before_lines[i1:i2]).strip()
            after_chunk = "\n".join(after_lines[j1:j2]).strip()
            before_html, after_html = self._render_word_level_diff(before_chunk, after_chunk)
            blocks.append(
                {
                    "anchor": f"change-{change_index}",
                    "tag": tag,
                    "title": {
                        "replace": "텍스트가 바뀐 구간",
                        "delete": "후처리에서 제거된 구간",
                        "insert": "후처리에서 추가된 구간",
                    }.get(tag, "변경 구간"),
                    "before_range": f"{i1 + 1}-{max(i2, i1 + 1)}",
                    "after_range": f"{j1 + 1}-{max(j2, j1 + 1)}",
                    "before_html": before_html or '<span class="muted">(empty)</span>',
                    "after_html": after_html or '<span class="muted">(empty)</span>',
                    "preview": (after_chunk or before_chunk or "").splitlines()[0][:120],
                }
            )
        return blocks

    def _render_stage_summary_rows(self, comparison: dict[str, Any]) -> str:
        stages = (comparison.get("metadata") or {}).get("stages") or {}
        rows: list[str] = []
        for stage_name in ("raw", "before", "after"):
            stage = stages.get(stage_name) or {}
            if not stage:
                continue
            summary_parts = []
            for key in (
                "line_count",
                "candidate_count",
                "profile",
                "rule_version",
                "block_count",
                "chunk_count",
                "kv_table_count",
            ):
                value = stage.get(key)
                if value not in (None, "", []):
                    summary_parts.append(f"{key}={value}")
            rows.append(
                "<tr><th>{name}</th><td>{description}</td><td><code>{summary}</code></td></tr>".format(
                    name=html.escape(stage_name),
                    description=html.escape(str(stage.get("description") or "n/a")),
                    summary=html.escape(", ".join(summary_parts) or "n/a"),
                )
            )
        if not rows:
            rows.append('<tr><td colspan="3">No comparison metadata is available.</td></tr>')
        return "\n".join(rows)

    def _render_change_navigation(self, blocks: list[dict[str, Any]]) -> str:
        if not blocks:
            return '<p class="muted">후처리 전후를 비교할 변경 구간이 없습니다.</p>'
        items = []
        for index, block in enumerate(blocks, start=1):
            preview = html.escape(block.get("preview") or block.get("title") or "")
            items.append(
                '<a class="change-chip" href="#{anchor}">#{index} {preview}</a>'.format(
                    anchor=html.escape(block["anchor"]),
                    index=index,
                    preview=preview or "변경 구간",
                )
            )
        return "\n".join(items)

    def _render_change_blocks(self, blocks: list[dict[str, Any]]) -> str:
        if not blocks:
            return '<div class="change-card"><p class="muted">후처리 비교 데이터는 찾았지만 실제 변경 구간은 감지되지 않았습니다.</p></div>'
        cards = []
        for block in blocks:
            cards.append(
                """
<article class="change-card" id="{anchor}">
  <div class="change-card-header">
    <h3>{title}</h3>
    <p><code>before {before_range}</code> <code>after {after_range}</code></p>
  </div>
  <div class="change-columns">
    <section>
      <strong>Before</strong>
      <div class="diff-block diff-before">{before_html}</div>
    </section>
    <section>
      <strong>After</strong>
      <div class="diff-block diff-after">{after_html}</div>
    </section>
  </div>
</article>
""".strip().format(
                    anchor=html.escape(block["anchor"]),
                    title=html.escape(str(block["title"])),
                    before_range=html.escape(str(block["before_range"])),
                    after_range=html.escape(str(block["after_range"])),
                    before_html=block["before_html"],
                    after_html=block["after_html"],
                )
            )
        return "\n".join(cards)

    def _render_postprocess_comparison(self, comparison: dict[str, Any] | None) -> str:
        if not comparison:
            return ""
        before_text = str(comparison.get("before_text") or "")
        after_text = str(comparison.get("after_text") or "")
        if not before_text and not after_text:
            return ""
        diff_blocks = self._build_diff_blocks(before_text, after_text)
        return """
<section class="panel" id="postprocess-diff" style="margin-bottom:1rem;">
  <header class="panel-header">
    <h2>Postprocess Diff</h2>
    <p>문서별 후처리 전후를 바로 비교하고, 실제 바뀐 구간만 먼저 스크롤하며 볼 수 있게 정리했습니다.</p>
  </header>
  <div class="panel-body scroll">
    <div class="comparison-summary">
      <article class="comparison-kpi">
        <strong>Format</strong>
        <span>{format_name}</span>
      </article>
      <article class="comparison-kpi">
        <strong>Changed Blocks</strong>
        <span>{change_count}</span>
      </article>
      <article class="comparison-kpi">
        <strong>Before Lines</strong>
        <span>{before_lines}</span>
      </article>
      <article class="comparison-kpi">
        <strong>After Lines</strong>
        <span>{after_lines}</span>
      </article>
    </div>

    <div class="comparison-nav">
      {change_navigation}
    </div>

    <div class="comparison-stage-table">
      <table>
        <thead>
          <tr>
            <th>Stage</th>
            <th>Description</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {stage_rows}
        </tbody>
      </table>
    </div>

    <div class="change-list">
      {change_blocks}
    </div>

    <details>
      <summary>Open Full Before / After Text</summary>
      <div class="change-columns" style="margin-top:0.9rem;">
        <section>
          <strong>Before</strong>
          <pre>{before_text}</pre>
        </section>
        <section>
          <strong>After</strong>
          <pre>{after_text}</pre>
        </section>
      </div>
    </details>
  </div>
</section>
""".strip().format(
            format_name=html.escape(str(comparison.get("format") or "unknown").upper()),
            change_count=len(diff_blocks),
            before_lines=len(before_text.splitlines()),
            after_lines=len(after_text.splitlines()),
            change_navigation=self._render_change_navigation(diff_blocks),
            stage_rows=self._render_stage_summary_rows(comparison),
            change_blocks=self._render_change_blocks(diff_blocks),
            before_text=html.escape(before_text or "(No before text available.)"),
            after_text=html.escape(after_text or "(No after text available.)"),
        )

    def _build_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        document_id = str(payload.get("document_id", "unknown"))
        summary = self._compute_document_summary(payload)
        overlay_href = self._overlay_document_href(document_id)
        markdown_path = self.parsed_root.parent / "markdown" / f"{document_id}.md"
        markdown_href = self._relative_href(markdown_path) if markdown_path.exists() else None
        page_assets = self._collect_page_assets(document_id, summary["page_count"])
        document_type = str(
            payload.get("classification", {}).get("document_type")
            or payload.get("extension", "").lstrip(".")
            or "unknown"
        )
        html_path = self.site_root / f"{document_id}.html"

        document_summary = {
            "document_id": document_id,
            "source_name": payload.get("source_name", ""),
            "source_path": payload.get("source_path", ""),
            "html_path": html_path.as_posix(),
            "page_count": summary["page_count"],
            "section_count": summary["section_count"],
            "issue_count": summary["issue_count"],
            "text_characters": summary["text_characters"],
            "element_count": summary["element_count"],
            "document_type": document_type,
            "extension": payload.get("extension", ""),
            "parser_name": payload.get("parser_name", ""),
            "selected_strategy": payload.get("metadata", {}).get("markdown_metadata", {}).get("selected_strategy", ""),
            "applied_strategy": payload.get("metadata", {}).get("markdown_source", ""),
            "overlay_href": overlay_href,
            "markdown_href": markdown_href,
            "markdown_path": markdown_path.as_posix() if markdown_path.exists() else None,
            "original_page_count": sum(1 for href in page_assets.values() if href),
        }
        write_text(html_path, self._render_comparison_document_html(payload, document_summary, page_assets))
        return document_summary

    def _compute_document_summary(self, payload: dict[str, Any]) -> dict[str, int]:
        summary = payload.get("summary") or {}
        page_count = int(summary.get("page_count", len(payload.get("pages", []))))
        section_count = int(summary.get("section_count", len(payload.get("sections", []))))
        issue_count = int(summary.get("issue_count", len(payload.get("issues", []))))
        text_characters = int(
            summary.get(
                "text_characters",
                sum(int(page.get("text_length", 0) or 0) for page in payload.get("pages", [])),
            )
        )
        element_counts = summary.get("element_counts") or {}
        if element_counts:
            element_count = sum(int(value or 0) for value in element_counts.values())
        else:
            element_count = sum(len(page.get("elements", [])) for page in payload.get("pages", []))
        return {
            "page_count": page_count,
            "section_count": section_count,
            "issue_count": issue_count,
            "text_characters": text_characters,
            "element_count": int(element_count),
        }

    def _build_run_summary(self, run_summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if not run_summary:
            return None
        return {
            "version": run_summary.get("version"),
            "scope": run_summary.get("scope"),
            "started_at": run_summary.get("started_at"),
            "finished_at": run_summary.get("finished_at"),
            "source_root": run_summary.get("source_root"),
            "total_documents": run_summary.get("total_documents"),
            "parsed_documents": run_summary.get("parsed_documents"),
            "fallback_documents": run_summary.get("fallback_documents"),
            "failed_documents": run_summary.get("failed_documents"),
        }

    def _overlay_index_path(self) -> str | None:
        if not self.overlay_site_root:
            return None
        target = self.overlay_site_root / "index.html"
        if not target.exists():
            return None
        return target.as_posix()

    def _overlay_document_href(self, document_id: str) -> str | None:
        if not self.overlay_site_root:
            return None
        target = self.overlay_site_root / f"{document_id}.html"
        if not target.exists():
            return None
        return self._relative_href(target)

    def _relative_href(self, target: Path) -> str:
        return Path(os.path.relpath(target, start=self.site_root)).as_posix()

    def _collect_page_assets(self, document_id: str, page_count: int) -> dict[int, str | None]:
        assets: dict[int, str | None] = {}
        if not self.overlay_site_root:
            return {page_number: None for page_number in range(1, page_count + 1)}

        asset_root = self.overlay_site_root / "assets" / document_id
        for page_number in range(1, page_count + 1):
            candidate = asset_root / f"page-{page_number:03d}.png"
            assets[page_number] = self._relative_href(candidate) if candidate.exists() else None
        return assets

    def _render_index_html(self, manifest: dict[str, Any]) -> str:
        run_summary = manifest.get("run_summary") or {}
        overlay_index_href = (
            self._relative_href(Path(manifest["overlay_index_path"]))
            if manifest.get("overlay_index_path")
            else None
        )
        rows = []

        for document in manifest["documents"]:
            search_blob = " ".join(
                [
                    str(document.get("source_name", "")),
                    str(document.get("document_type", "")),
                    str(document.get("parser_name", "")),
                    str(document.get("selected_strategy", "")),
                    str(document.get("applied_strategy", "")),
                ]
            ).lower()
            overlay_cell = (
                '<a href="{href}">Overlay</a>'.format(href=html.escape(document["overlay_href"]))
                if document.get("overlay_href")
                else '<span class="muted">n/a</span>'
            )
            rows.append(
                """
<tr data-search="{search}" data-type="{doc_type_value}" data-parser="{parser_value}">
  <td><a href="{href}">{name}</a></td>
  <td>{doc_type}</td>
  <td>{parser}</td>
  <td>{strategy}</td>
  <td>{pages}</td>
  <td>{issues}</td>
  <td>{overlay}</td>
</tr>
""".strip().format(
                    search=html.escape(search_blob),
                    doc_type_value=html.escape(str(document.get("document_type", "")).lower()),
                    parser_value=html.escape(str(document.get("parser_name", "")).lower()),
                    href=html.escape(f"{document['document_id']}.html"),
                    name=html.escape(str(document.get("source_name", ""))),
                    doc_type=html.escape(str(document.get("document_type", ""))),
                    parser=html.escape(str(document.get("parser_name", ""))),
                    strategy=html.escape(
                        str(document.get("selected_strategy") or document.get("applied_strategy") or "n/a")
                    ),
                    pages=document.get("page_count", 0),
                    issues=document.get("issue_count", 0),
                    overlay=overlay_cell,
                )
            )

        if not rows:
            rows.append('<tr><td colspan="7">No parsed documents are available.</td></tr>')

        parser_options = sorted(
            {
                str(document.get("parser_name", "")).strip()
                for document in manifest["documents"]
                if str(document.get("parser_name", "")).strip()
            }
        )
        parser_option_html = ['<option value="">All parsers</option>']
        parser_option_html.extend(
            '<option value="{value}">{label}</option>'.format(
                value=html.escape(parser.lower()),
                label=html.escape(parser),
            )
            for parser in parser_options
        )

        type_options = sorted(
            {
                str(document.get("document_type", "")).strip()
                for document in manifest["documents"]
                if str(document.get("document_type", "")).strip()
            }
        )
        type_option_html = ['<option value="">All types</option>']
        type_option_html.extend(
            '<option value="{value}">{label}</option>'.format(
                value=html.escape(doc_type.lower()),
                label=html.escape(doc_type),
            )
            for doc_type in type_options
        )

        run_lines = []
        if run_summary:
            run_lines.extend(
                [
                    "<li><strong>Version</strong> <code>{}</code></li>".format(
                        html.escape(str(run_summary.get("version") or "n/a"))
                    ),
                    "<li><strong>Source Root</strong> <code>{}</code></li>".format(
                        html.escape(str(run_summary.get("source_root") or "n/a"))
                    ),
                    "<li><strong>Run Status</strong> parsed <code>{}</code>, fallback <code>{}</code>, failed <code>{}</code></li>".format(
                        run_summary.get("parsed_documents", 0),
                        run_summary.get("fallback_documents", 0),
                        run_summary.get("failed_documents", 0),
                    ),
                    "<li><strong>Finished</strong> <code>{}</code></li>".format(
                        html.escape(str(run_summary.get("finished_at") or run_summary.get("started_at") or "n/a"))
                    ),
                ]
            )
        else:
            run_lines.append("<li>No run summary was provided for this review build.</li>")

        overlay_button = (
            '<a class="button secondary" href="{href}">Open Overlay Review</a>'.format(
                href=html.escape(overlay_index_href)
            )
            if overlay_index_href
            else ""
        )

        return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Parsing Results Review</title>
  <style>
    :root {{
      --bg: #f0e9de;
      --panel: rgba(255, 251, 246, 0.88);
      --line: rgba(52, 39, 26, 0.14);
      --ink: #1f1a15;
      --muted: #675d53;
      --accent: #9f4728;
      --accent-2: #23547d;
      --shadow: 0 22px 58px rgba(52, 38, 25, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 112, 74, 0.18), transparent 24rem),
        radial-gradient(circle at bottom right, rgba(35, 84, 125, 0.12), transparent 20rem),
        linear-gradient(180deg, #faf5ed 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", "Malgun Gothic", serif;
    }}
    a {{
      color: inherit;
      text-decoration-thickness: 0.08em;
      text-underline-offset: 0.15em;
    }}
    .wrap {{
      max-width: 96rem;
      margin: 0 auto;
      padding: 2.35rem 1.3rem 3rem;
    }}
    .hero {{
      display: grid;
      gap: 0.85rem;
    }}
    .eyebrow {{
      margin: 0;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 4rem);
      line-height: 0.98;
    }}
    .hero p {{
      margin: 0;
      max-width: 60rem;
      color: var(--muted);
      line-height: 1.55;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(18rem, 0.8fr);
      gap: 1rem;
      margin-top: 1rem;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 1.3rem;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .panel-inner {{
      padding: 1rem 1.05rem;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.8rem;
      margin-top: 1rem;
    }}
    .stat {{
      padding: 0.95rem 1rem;
      border-radius: 1.1rem;
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid rgba(52, 39, 26, 0.08);
    }}
    .stat strong {{
      display: block;
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.45rem;
    }}
    .stat span {{
      font-size: clamp(1.45rem, 3vw, 2.25rem);
      line-height: 1;
    }}
    h2 {{
      margin: 0 0 0.7rem;
      font-size: 1.04rem;
    }}
    .muted {{
      color: var(--muted);
    }}
    .meta-list {{
      margin: 0;
      padding-left: 1.1rem;
      line-height: 1.55;
      color: #3e372f;
    }}
    .meta-list code {{
      overflow-wrap: anywhere;
      color: #2b2620;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 12rem 13rem auto;
      gap: 0.75rem;
      margin-top: 0.9rem;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 0.3rem;
      font-size: 0.9rem;
    }}
    input, select {{
      min-height: 2.7rem;
      padding: 0.65rem 0.8rem;
      border-radius: 0.9rem;
      border: 1px solid rgba(52, 39, 26, 0.14);
      background: rgba(255, 255, 255, 0.72);
      color: var(--ink);
      font: inherit;
    }}
    .actions {{
      display: flex;
      gap: 0.7rem;
      flex-wrap: wrap;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 2.75rem;
      padding: 0.7rem 1rem;
      border-radius: 999px;
      color: #fff;
      text-decoration: none;
      background: linear-gradient(135deg, #9d4527 0%, var(--accent) 100%);
    }}
    .button.secondary {{
      background: linear-gradient(135deg, #37638f 0%, var(--accent-2) 100%);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.75rem;
      font-size: 0.93rem;
    }}
    th, td {{
      padding: 0.64rem 0.45rem;
      border-bottom: 1px solid rgba(52, 39, 26, 0.1);
      text-align: left;
      vertical-align: top;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
    }}
    @media (max-width: 980px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .stats {{
        grid-template-columns: 1fr 1fr;
      }}
      .toolbar {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <p class="eyebrow">Parsing Review</p>
      <h1>Document Parsing Results</h1>
      <p>Structured parsing artifacts are grouped in one place so you can inspect the extracted markdown, metadata, issue list, and page-level coverage without leaving the browser.</p>
    </header>

    <section class="stats">
      <article class="stat"><strong>Documents</strong><span>{document_count}</span></article>
      <article class="stat"><strong>Pages</strong><span>{total_pages}</span></article>
      <article class="stat"><strong>Elements</strong><span>{total_elements}</span></article>
      <article class="stat"><strong>Issues</strong><span>{total_issues}</span></article>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-inner">
          <h2>Browse Parsed Documents</h2>
          <p class="muted">Filter by document type or parser, then open a document to inspect its markdown, section outline, raw metadata, and raw JSON payload.</p>
          <div class="toolbar">
            <label>
              Search
              <input id="search-input" type="search" placeholder="Document name, parser, strategy">
            </label>
            <label>
              Type
              <select id="type-filter">{type_options}</select>
            </label>
            <label>
              Parser
              <select id="parser-filter">{parser_options}</select>
            </label>
            <div class="actions">
              {overlay_button}
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Document</th>
                <th>Type</th>
                <th>Parser</th>
                <th>Strategy</th>
                <th>Pages</th>
                <th>Issues</th>
                <th>Overlay</th>
              </tr>
            </thead>
            <tbody id="document-rows">
              {rows}
            </tbody>
          </table>
        </div>
      </section>

      <aside class="panel">
        <div class="panel-inner">
          <h2>Run Summary</h2>
          <ul class="meta-list">
            {run_lines}
          </ul>
          <p class="muted" style="margin-top:0.8rem;">Generated at <code>{generated_at}</code>.</p>
        </div>
      </aside>
    </div>
  </main>
  <script>
    const searchInput = document.getElementById('search-input');
    const typeFilter = document.getElementById('type-filter');
    const parserFilter = document.getElementById('parser-filter');
    const rows = Array.from(document.querySelectorAll('#document-rows tr'));

    function applyFilters() {{
      const searchValue = (searchInput.value || '').trim().toLowerCase();
      const typeValue = (typeFilter.value || '').trim().toLowerCase();
      const parserValue = (parserFilter.value || '').trim().toLowerCase();

      for (const row of rows) {{
        const matchesSearch = !searchValue || (row.dataset.search || '').includes(searchValue);
        const matchesType = !typeValue || (row.dataset.type || '') === typeValue;
        const matchesParser = !parserValue || (row.dataset.parser || '') === parserValue;
        row.hidden = !(matchesSearch && matchesType && matchesParser);
      }}
    }}

    searchInput.addEventListener('input', applyFilters);
    typeFilter.addEventListener('change', applyFilters);
    parserFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
""".format(
            document_count=manifest.get("document_count", 0),
            total_pages=manifest.get("total_pages", 0),
            total_elements=manifest.get("total_elements", 0),
            total_issues=manifest.get("total_issues", 0),
            type_options="\n".join(type_option_html),
            parser_options="\n".join(parser_option_html),
            overlay_button=overlay_button,
            rows="\n".join(rows),
            run_lines="\n".join(run_lines),
            generated_at=html.escape(str(manifest.get("generated_at") or "n/a")),
        )

    def _build_markdown_page_map(
        self,
        pages: list[dict[str, Any]],
        sections: list[dict[str, Any]],
        markdown: str,
    ) -> dict[int, str]:
        lines = markdown.splitlines()
        page_start_lines: dict[int, int] = {}

        for line_number, line in enumerate(lines, start=1):
            match = re.match(r"^#\s+Page\s+(\d+)\b", line.strip(), re.IGNORECASE)
            if match:
                page_start_lines[int(match.group(1))] = line_number

        if not page_start_lines:
            for section in sections:
                title = str(section.get("title", "")).strip()
                match = re.match(r"^Page\s+(\d+)\b", title, re.IGNORECASE)
                if match:
                    page_start_lines[int(match.group(1))] = int(section.get("line_number", 1))

        if not page_start_lines:
            if pages:
                return {int(pages[0].get("page_number", 1)): markdown}
            return {1: markdown}

        ordered_pages = sorted(page_start_lines)
        page_map: dict[int, str] = {}
        for index, page_number in enumerate(ordered_pages):
            start_line = max(page_start_lines[page_number], 1)
            next_start = page_start_lines[ordered_pages[index + 1]] if index + 1 < len(ordered_pages) else len(lines) + 1
            excerpt = "\n".join(lines[start_line - 1 : next_start - 1]).strip()
            page_map[page_number] = excerpt

        for page in pages:
            page_number = int(page.get("page_number", 1))
            page_map.setdefault(page_number, markdown if page_number == 1 else "")
        return page_map

    def _build_preprocess_page_map(self, preprocess: dict[str, Any]) -> dict[int, dict[str, Any]]:
        page_map: dict[int, dict[str, Any]] = {}
        for page_summary in preprocess.get("pages") or []:
            page_number = int(page_summary.get("page_number", 0) or 0)
            if page_number:
                page_map[page_number] = page_summary
        return page_map

    def _build_removal_log_by_page(self, preprocess: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        page_map: dict[int, list[dict[str, Any]]] = {}
        for entry in preprocess.get("removal_log") or []:
            page_number = int(entry.get("page_number", 0) or 0)
            if not page_number:
                continue
            page_map.setdefault(page_number, []).append(entry)
        return page_map

    def _format_preprocess_status(self, preprocess: dict[str, Any]) -> str:
        if not preprocess.get("enabled"):
            return "n/a"
        return "changed" if preprocess.get("changed") else "checked"

    def _format_preprocess_page_summary(self, page_summary: dict[str, Any]) -> str:
        if not page_summary:
            return "no page summary"
        parts = []
        for label, key in (
            ("page-number", "removed_page_number_lines"),
            ("header", "removed_repeated_header_lines"),
            ("footer", "removed_repeated_footer_lines"),
            ("noise", "removed_noise_lines"),
            ("spacing", "spacing_normalized_lines"),
            ("toc", "toc_entries_compacted"),
            ("series", "series_tables_structured"),
            ("chart", "chart_clusters_collapsed"),
        ):
            value = int(page_summary.get(key, 0) or 0)
            if value:
                parts.append(f"{label}:{value}")
        return ", ".join(parts) if parts else "no changes"

    def _render_preprocess_rows(self, preprocess: dict[str, Any]) -> str:
        if not preprocess.get("enabled"):
            return '<tr><td colspan="2">No preprocessing metadata is available.</td></tr>'

        applied_rules = ", ".join(str(rule) for rule in preprocess.get("applied_rules") or []) or "none"
        rows = [
            ("Profile", preprocess.get("profile", "n/a")),
            ("Status", self._format_preprocess_status(preprocess)),
            ("Applied Rules", applied_rules),
            ("Page Numbers Removed", preprocess.get("removed_page_number_lines", 0)),
            ("Repeated Headers Removed", preprocess.get("removed_repeated_header_lines", 0)),
            ("Repeated Footers Removed", preprocess.get("removed_repeated_footer_lines", 0)),
            ("Noise Lines Removed", preprocess.get("removed_noise_lines", 0)),
            ("Spacing Normalized", preprocess.get("spacing_normalized_lines", 0)),
            ("Paragraph Merges", preprocess.get("paragraph_merges", 0)),
            ("Candidate Lines Logged", preprocess.get("candidate_lines", 0)),
            ("Preserved Candidates", preprocess.get("preserved_candidate_lines", 0)),
            ("Removal Log Entries", len(preprocess.get("removal_log") or [])),
            ("TOC Entries Compacted", preprocess.get("toc_entries_compacted", 0)),
            ("Series Tables Structured", preprocess.get("series_tables_structured", 0)),
            ("Series Rows Structured", preprocess.get("series_rows_structured", 0)),
            ("Chart Clusters Collapsed", preprocess.get("chart_clusters_collapsed", 0)),
            ("Chart Items Compacted", preprocess.get("chart_cluster_items_compacted", 0)),
        ]
        return "\n".join(
            "<tr><th>{label}</th><td>{value}</td></tr>".format(
                label=html.escape(str(label)),
                value=html.escape(str(value)),
            )
            for label, value in rows
        )

    def _render_removal_log(self, removals: list[dict[str, Any]], limit: int = 20) -> str:
        if not removals:
            return '<li class="muted">No removal log entries were recorded.</li>'
        items = []
        for entry in removals[:limit]:
            reason_text = ", ".join(str(reason) for reason in entry.get("reasons") or []) or "n/a"
            items.append(
                "<li><strong>Page {page}</strong> <code>{decision}</code> {text}<br><span class=\"muted\">{reasons}</span></li>".format(
                    page=entry.get("page_number", "?"),
                    decision=html.escape(str(entry.get("decision", "remove"))),
                    text=html.escape(str(entry.get("text", ""))[:180]),
                    reasons=html.escape(reason_text),
                )
            )
        if len(removals) > limit:
            items.append('<li class="muted">{count} more removal entries omitted.</li>'.format(count=len(removals) - limit))
        return "\n".join(items)

    def _render_element_rows(self, elements: list[dict[str, Any]], limit: int = 18) -> str:
        rows = []
        for element in elements[:limit]:
            preview = " ".join(str(element.get("text") or element.get("markdown") or "").split())
            rows.append(
                """
<tr>
  <td>{order}</td>
  <td>{element_type}</td>
  <td><code>{bbox}</code></td>
  <td>{preview}</td>
</tr>
""".strip().format(
                    order=element.get("order", ""),
                    element_type=html.escape(str(element.get("element_type", ""))),
                    bbox=html.escape(self._format_bbox(element.get("bbox"))),
                    preview=html.escape(preview[:140] + ("..." if len(preview) > 140 else "")) or "&nbsp;",
                )
            )

        if len(elements) > limit:
            rows.append(
                '<tr><td colspan="4" class="muted">{extra} more elements omitted from this preview.</td></tr>'.format(
                    extra=len(elements) - limit
                )
            )

        if not rows:
            rows.append('<tr><td colspan="4">No elements were captured for this page.</td></tr>')
        return "\n".join(rows)

    def _format_bbox(self, bbox: Any) -> str:
        if not bbox:
            return "n/a"
        if isinstance(bbox, list):
            return "[{}]".format(", ".join(f"{float(value):.1f}" for value in bbox))
        return str(bbox)

    def _render_comparison_document_html(
        self,
        payload: dict[str, Any],
        document_summary: dict[str, Any],
        page_assets: dict[int, str | None],
    ) -> str:
        classification = payload.get("classification") or {}
        metadata = payload.get("metadata") or {}
        pages = payload.get("pages") or []
        sections = payload.get("sections") or []
        issues = payload.get("issues") or []
        markdown = payload.get("markdown", "")
        raw_markdown = str(metadata.get("markdown_raw") or "")
        preprocess = metadata.get("markdown_metadata", {}).get("preprocess", {}) or {}
        postprocess_comparison = self._load_postprocess_comparison(payload)
        page_markdown_map = self._build_markdown_page_map(pages, sections, markdown)
        raw_page_markdown_map = self._build_markdown_page_map(pages, sections, raw_markdown)
        preprocess_page_map = self._build_preprocess_page_map(preprocess)
        removal_log_by_page = self._build_removal_log_by_page(preprocess)

        if pages:
            default_page = int(pages[0].get("page_number", 1))
        else:
            default_page = 1

        overview_cards = [
            ("Document Type", document_summary.get("document_type", "unknown")),
            ("Parser", document_summary.get("parser_name", "unknown")),
            ("Pages", str(document_summary.get("page_count", 0))),
            ("Sections", str(document_summary.get("section_count", 0))),
            ("Elements", str(document_summary.get("element_count", 0))),
            ("Preprocess", self._format_preprocess_status(preprocess)),
        ]
        overview_html = "\n".join(
            """
<article class="metric">
  <strong>{label}</strong>
  <span>{value}</span>
</article>
""".strip().format(
                label=html.escape(label),
                value=html.escape(str(value)),
            )
            for label, value in overview_cards
        )

        page_options = []
        original_views = []
        markdown_views = []
        parsed_views = []
        for page in pages or [{"page_number": 1, "text_length": 0, "elements": []}]:
            page_number = int(page.get("page_number", 1))
            hidden = "" if page_number == default_page else " hidden"
            page_preprocess = preprocess_page_map.get(page_number, {})
            page_options.append(
                '<option value="{value}">Page {label}</option>'.format(value=page_number, label=page_number)
            )

            image_href = page_assets.get(page_number)
            original_views.append(
                """
<div class="page-view"{hidden} data-page="{page_number}">
  <div class="panel-caption">Page {page_number}</div>
  {content}
</div>
""".strip().format(
                    hidden=hidden,
                    page_number=page_number,
                    content=(
                        '<img src="{src}" alt="Original page {page_number}">'.format(
                            src=html.escape(str(image_href)),
                            page_number=page_number,
                        )
                        if image_href
                        else '<p class="muted">Original page preview is unavailable for this document.</p>'
                    ),
                )
            )

            markdown_views.append(
                """
<div class="page-view"{hidden} data-page="{page_number}">
  <div class="panel-caption">Page {page_number}</div>
  <ul class="page-meta">
    <li><strong>Preprocess</strong> <code>{preprocess_summary}</code></li>
  </ul>
  <pre>{markdown_excerpt}</pre>
  <details>
    <summary>Open Removed Lines</summary>
    <ul class="issues">
      {removed_lines}
    </ul>
  </details>
  <details>
    <summary>Open Raw Markdown</summary>
    <pre class="json-block">{raw_markdown_excerpt}</pre>
  </details>
</div>
""".strip().format(
                    hidden=hidden,
                    page_number=page_number,
                    preprocess_summary=html.escape(self._format_preprocess_page_summary(page_preprocess)),
                    markdown_excerpt=html.escape(page_markdown_map.get(page_number, "") or "(No markdown excerpt for this page.)"),
                    removed_lines=self._render_removal_log(removal_log_by_page.get(page_number, []), limit=8),
                    raw_markdown_excerpt=html.escape(
                        raw_page_markdown_map.get(page_number, "") or "(No raw markdown excerpt for this page.)"
                    ),
                )
            )

            parsed_views.append(
                """
<div class="page-view"{hidden} data-page="{page_number}">
  <div class="panel-caption">Page {page_number}</div>
  <ul class="page-meta">
    <li><strong>Text Length</strong> <code>{text_length}</code></li>
    <li><strong>Elements</strong> <code>{element_count}</code></li>
    <li><strong>Element Types</strong> <code>{element_types}</code></li>
  </ul>
  <table class="element-table">
    <thead>
      <tr>
        <th>Order</th>
        <th>Type</th>
        <th>BBox</th>
        <th>Preview</th>
      </tr>
    </thead>
    <tbody>
      {element_rows}
    </tbody>
  </table>
  <details>
    <summary>Open Page JSON</summary>
    <pre class="json-block">{page_json}</pre>
  </details>
</div>
""".strip().format(
                    hidden=hidden,
                    page_number=page_number,
                    text_length=page.get("text_length", 0),
                    element_count=len(page.get("elements", [])),
                    element_types=html.escape(self._format_element_types(page.get("elements", []))),
                    element_rows=self._render_element_rows(page.get("elements", [])),
                    page_json=html.escape(json.dumps(page, ensure_ascii=False, indent=2)),
                )
            )

        issue_html = []
        for issue in issues:
            issue_html.append(
                "<li><strong>{severity}</strong> {message}</li>".format(
                    severity=html.escape(str(issue.get("severity", "info")).upper()),
                    message=html.escape(str(issue.get("message", ""))),
                )
            )
        if not issue_html:
            issue_html.append('<li class="muted">No issues were recorded for this document.</li>')

        section_rows = []
        for section in sections:
            section_rows.append(
                """
<tr>
  <td>{line}</td>
  <td>H{level}</td>
  <td>{title}</td>
</tr>
""".strip().format(
                    line=section.get("line_number", 0),
                    level=section.get("level", 0),
                    title=html.escape(str(section.get("title", ""))),
                )
            )
        if not section_rows:
            section_rows.append('<tr><td colspan="3">No markdown headings were detected.</td></tr>')

        overlay_button = (
            '<a class="button secondary" href="{href}">Open Overlay Review</a>'.format(
                href=html.escape(str(document_summary["overlay_href"]))
            )
            if document_summary.get("overlay_href")
            else ""
        )
        markdown_file_button = (
            '<a class="button secondary" href="{href}" target="_blank" rel="noopener">Open Markdown File</a>'.format(
                href=html.escape(str(document_summary["markdown_href"]))
            )
            if document_summary.get("markdown_href")
            else ""
        )
        action_buttons = "\n".join(
            button
            for button in [
                '<a class="button" href="#full-markdown">Jump To Full Markdown</a>',
                '<a class="button secondary" href="#postprocess-diff">Jump To Postprocess Diff</a>' if postprocess_comparison else "",
                markdown_file_button,
                overlay_button,
            ]
            if button
        )

        return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f3ede2;
      --panel: rgba(255, 252, 247, 0.9);
      --line: rgba(52, 39, 26, 0.14);
      --ink: #1f1a15;
      --muted: #665d54;
      --accent: #9d4527;
      --accent-2: #29577e;
      --shadow: 0 22px 58px rgba(52, 38, 25, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 112, 74, 0.17), transparent 24rem),
        linear-gradient(180deg, #fbf6ef 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", "Malgun Gothic", serif;
    }}
    a {{
      color: inherit;
      text-decoration-thickness: 0.08em;
      text-underline-offset: 0.15em;
    }}
    .wrap {{
      max-width: 110rem;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    .hero {{
      display: grid;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }}
    .hero a {{
      width: fit-content;
      color: var(--muted);
      text-decoration: none;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 0.82rem;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.85rem, 4vw, 3.2rem);
      line-height: 0.98;
    }}
    .hero p {{
      margin: 0;
      max-width: 70rem;
      color: var(--muted);
      line-height: 1.55;
    }}
    .actions {{
      display: flex;
      gap: 0.7rem;
      flex-wrap: wrap;
      margin-top: 0.25rem;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 2.7rem;
      padding: 0.7rem 1rem;
      border-radius: 999px;
      color: #fff;
      text-decoration: none;
      background: linear-gradient(135deg, #9d4527 0%, var(--accent) 100%);
    }}
    .button.secondary {{
      background: linear-gradient(135deg, #37638f 0%, var(--accent-2) 100%);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 0.8rem;
      margin-bottom: 1rem;
    }}
    .metric {{
      padding: 0.95rem 1rem;
      border-radius: 1.1rem;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(52, 39, 26, 0.08);
      box-shadow: var(--shadow);
    }}
    .metric strong {{
      display: block;
      margin-bottom: 0.45rem;
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .metric span {{
      font-size: clamp(1.2rem, 2.7vw, 1.85rem);
      line-height: 1.05;
    }}
    .compare-controls {{
      display: flex;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }}
    .compare-controls select {{
      min-height: 2.7rem;
      padding: 0.65rem 0.8rem;
      border-radius: 0.9rem;
      border: 1px solid rgba(52, 39, 26, 0.14);
      background: rgba(255, 255, 255, 0.75);
      color: var(--ink);
      font: inherit;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: minmax(18rem, 1fr) minmax(18rem, 1fr) minmax(20rem, 1.05fr);
      gap: 1rem;
      align-items: start;
      margin-bottom: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 1.3rem;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-header {{
      padding: 0.95rem 1rem 0.55rem;
      border-bottom: 1px solid rgba(52, 39, 26, 0.08);
    }}
    .panel-header h2 {{
      margin: 0;
      font-size: 1.02rem;
    }}
    .panel-header p {{
      margin: 0.3rem 0 0;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    .panel-body {{
      padding: 1rem;
      min-height: 34rem;
    }}
    .panel-caption {{
      margin-bottom: 0.7rem;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .page-view[hidden] {{
      display: none;
    }}
    .original-panel img {{
      width: 100%;
      display: block;
      border-radius: 1rem;
      border: 1px solid rgba(52, 39, 26, 0.1);
      background: white;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.8rem;
      line-height: 1.5;
      color: #29231d;
    }}
    .scroll {{
      max-height: 32rem;
      overflow: auto;
      padding-right: 0.15rem;
    }}
    .page-meta {{
      margin: 0 0 0.85rem;
      padding-left: 1.1rem;
      line-height: 1.5;
      color: #3f372f;
    }}
    .page-meta code {{
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 0.58rem 0.42rem;
      border-bottom: 1px solid rgba(52, 39, 26, 0.1);
      text-align: left;
      vertical-align: top;
    }}
    .element-table {{
      margin-bottom: 0.85rem;
    }}
    details {{
      border-top: 1px solid rgba(52, 39, 26, 0.1);
      padding-top: 0.8rem;
      margin-top: 0.8rem;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    .json-block {{
      margin-top: 0.8rem;
      max-height: 18rem;
      overflow: auto;
      padding-right: 0.15rem;
    }}
    .details-grid {{
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
      gap: 1rem;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 1rem;
    }}
    .panel-inner {{
      padding: 1rem 1.05rem;
    }}
    h3 {{
      margin: 0 0 0.7rem;
      font-size: 1rem;
    }}
    .issues {{
      margin: 0;
      padding-left: 1.15rem;
      line-height: 1.55;
    }}
    .muted {{
      color: var(--muted);
    }}
    .comparison-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 0.95rem;
    }}
    .comparison-kpi {{
      padding: 0.9rem 0.95rem;
      border-radius: 1rem;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(52, 39, 26, 0.08);
    }}
    .comparison-kpi strong {{
      display: block;
      margin-bottom: 0.4rem;
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .comparison-kpi span {{
      font-size: 1.35rem;
      line-height: 1.1;
    }}
    .comparison-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-bottom: 1rem;
    }}
    .change-chip {{
      display: inline-flex;
      align-items: center;
      min-height: 2.1rem;
      max-width: 100%;
      padding: 0.45rem 0.8rem;
      border-radius: 999px;
      background: rgba(41, 87, 126, 0.1);
      border: 1px solid rgba(41, 87, 126, 0.18);
      text-decoration: none;
      font-size: 0.9rem;
    }}
    .comparison-stage-table {{
      margin-bottom: 1rem;
    }}
    .change-list {{
      display: grid;
      gap: 0.9rem;
    }}
    .change-card {{
      padding: 1rem;
      border: 1px solid rgba(52, 39, 26, 0.1);
      border-radius: 1.1rem;
      background: rgba(255, 255, 255, 0.62);
      scroll-margin-top: 1rem;
    }}
    .change-card-header {{
      display: flex;
      justify-content: space-between;
      gap: 0.8rem;
      align-items: start;
      margin-bottom: 0.75rem;
    }}
    .change-card-header h3 {{
      margin: 0;
    }}
    .change-card-header p {{
      margin: 0;
      color: var(--muted);
      white-space: nowrap;
    }}
    .change-columns {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 0.9rem;
    }}
    .diff-block {{
      padding: 0.9rem;
      border-radius: 0.95rem;
      min-height: 5rem;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.58;
      background: rgba(250, 245, 237, 0.9);
      border: 1px solid rgba(52, 39, 26, 0.08);
    }}
    .diff-before {{
      background: rgba(168, 63, 40, 0.06);
    }}
    .diff-after {{
      background: rgba(35, 84, 125, 0.06);
    }}
    .diff-add {{
      background: rgba(83, 168, 117, 0.34);
      color: inherit;
      padding: 0.05em 0.1em;
      border-radius: 0.2rem;
    }}
    .diff-del {{
      background: rgba(214, 87, 69, 0.26);
      color: inherit;
      padding: 0.05em 0.1em;
      border-radius: 0.2rem;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 1280px) {{
      .compare-grid {{
        grid-template-columns: 1fr;
      }}
      .panel-body {{
        min-height: auto;
      }}
    }}
    @media (max-width: 980px) {{
      .metrics {{
        grid-template-columns: 1fr 1fr 1fr;
      }}
      .details-grid {{
        grid-template-columns: 1fr;
      }}
      .comparison-summary {{
        grid-template-columns: 1fr 1fr;
      }}
      .change-columns {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .metrics {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <a href="index.html">Back To Parsing Review</a>
      <h1>{title}</h1>
      <p>Original page, markdown excerpt, and parsed page payload are shown side by side. Change the page selector to compare extraction quality page by page on a single screen.</p>
      <div class="actions">
        {action_buttons}
      </div>
    </header>

    <section class="metrics">
      {overview_html}
    </section>

    <section class="compare-controls">
      <label for="page-select"><strong>Compare Page</strong></label>
      <select id="page-select">{page_options}</select>
    </section>

    <section class="compare-grid">
      <article class="panel original-panel">
        <header class="panel-header">
          <h2>Original</h2>
          <p>Rendered source page preview.</p>
        </header>
        <div class="panel-body scroll">
          {original_views}
        </div>
      </article>

      <article class="panel">
        <header class="panel-header">
          <h2>Markdown</h2>
          <p>Cleaned markdown is shown by default. Open the raw excerpt inside each page card when you need the unprocessed output.</p>
        </header>
        <div class="panel-body scroll">
          {markdown_views}
        </div>
      </article>

      <article class="panel">
        <header class="panel-header">
          <h2>Parsed Result</h2>
          <p>Selected page summary, element preview, and raw page JSON.</p>
        </header>
        <div class="panel-body scroll">
          {parsed_views}
        </div>
      </article>
    </section>

    {postprocess_comparison_section}

    <section class="panel" id="full-markdown" style="margin-bottom:1rem;">
      <header class="panel-header">
        <h2>Full Markdown</h2>
        <p>The entire cleaned markdown output for this document is shown here. Expand the raw block when you need the unprocessed markdown captured before cleanup.</p>
      </header>
      <div class="panel-body scroll">
        <pre>{full_markdown}</pre>
        <details>
          <summary>Open Full Raw Markdown</summary>
          <pre class="json-block">{full_raw_markdown}</pre>
        </details>
      </div>
    </section>

    <section class="details-grid">
      <div class="stack">
        <article class="panel">
          <div class="panel-inner">
            <h3>Issues</h3>
            <ul class="issues">
              {issue_html}
            </ul>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h3>Sections</h3>
            <table>
              <thead>
                <tr>
                  <th>Line</th>
                  <th>Level</th>
                  <th>Title</th>
                </tr>
              </thead>
              <tbody>
                {section_rows}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <div class="stack">
        <article class="panel">
          <div class="panel-inner">
            <h3>Preprocessing</h3>
            <table>
              <tbody>
                {preprocess_rows}
              </tbody>
            </table>
            <details>
              <summary>Open Removal Log</summary>
              <ul class="issues" style="margin-top:0.8rem;">
                {removal_log_html}
              </ul>
            </details>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h3>Classification</h3>
            <div class="scroll">
              <pre>{classification_json}</pre>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h3>Metadata</h3>
            <div class="scroll">
              <pre>{metadata_json}</pre>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h3>Raw Payload</h3>
            <details>
              <summary>Open Full JSON</summary>
              <pre class="json-block">{raw_json}</pre>
            </details>
          </div>
        </article>
      </div>
    </section>
  </main>
  <script>
    const pageSelect = document.getElementById('page-select');
    const pageViews = Array.from(document.querySelectorAll('.page-view'));

    function syncPageViews() {{
      const selected = pageSelect.value;
      for (const view of pageViews) {{
        view.hidden = view.dataset.page !== selected;
      }}
    }}

    pageSelect.addEventListener('change', syncPageViews);
    syncPageViews();
  </script>
</body>
</html>
""".format(
            title=html.escape(str(payload.get("source_name", document_summary.get("document_id", "")))),
            action_buttons=action_buttons,
            overview_html=overview_html,
            page_options="\n".join(page_options),
            original_views="\n".join(original_views),
            markdown_views="\n".join(markdown_views),
            parsed_views="\n".join(parsed_views),
            postprocess_comparison_section=self._render_postprocess_comparison(postprocess_comparison),
            full_markdown=html.escape(markdown or "(No cleaned markdown was produced for this document.)"),
            full_raw_markdown=html.escape(raw_markdown or "(No raw markdown was captured for this document.)"),
            issue_html="\n".join(issue_html),
            section_rows="\n".join(section_rows),
            preprocess_rows=self._render_preprocess_rows(preprocess),
            removal_log_html=self._render_removal_log(preprocess.get("removal_log") or []),
            classification_json=html.escape(json.dumps(classification, ensure_ascii=False, indent=2)),
            metadata_json=html.escape(json.dumps(metadata, ensure_ascii=False, indent=2)),
            raw_json=html.escape(json.dumps(payload, ensure_ascii=False, indent=2)),
        )

    def _render_document_html(self, payload: dict[str, Any], document_summary: dict[str, Any]) -> str:
        classification = payload.get("classification") or {}
        metadata = payload.get("metadata") or {}
        pages = payload.get("pages") or []
        sections = payload.get("sections") or []
        issues = payload.get("issues") or []
        markdown = payload.get("markdown", "")

        overview_cards = [
            ("Document Type", document_summary.get("document_type", "unknown")),
            ("Parser", document_summary.get("parser_name", "unknown")),
            ("Pages", str(document_summary.get("page_count", 0))),
            ("Sections", str(document_summary.get("section_count", 0))),
            ("Elements", str(document_summary.get("element_count", 0))),
            ("Issues", str(document_summary.get("issue_count", 0))),
        ]
        overview_html = "\n".join(
            """
<article class="metric">
  <strong>{label}</strong>
  <span>{value}</span>
</article>
""".strip().format(
                label=html.escape(label),
                value=html.escape(str(value)),
            )
            for label, value in overview_cards
        )

        issue_html = []
        for issue in issues:
            issue_html.append(
                """
<li>
  <strong>{severity}</strong>
  <span>{message}</span>
</li>
""".strip().format(
                    severity=html.escape(str(issue.get("severity", "info")).upper()),
                    message=html.escape(str(issue.get("message", ""))),
                )
            )
        if not issue_html:
            issue_html.append('<li class="muted">No issues were recorded for this document.</li>')

        section_rows = []
        for section in sections:
            section_rows.append(
                """
<tr>
  <td>{line}</td>
  <td>H{level}</td>
  <td>{title}</td>
</tr>
""".strip().format(
                    line=section.get("line_number", 0),
                    level=section.get("level", 0),
                    title=html.escape(str(section.get("title", ""))),
                )
            )
        if not section_rows:
            section_rows.append('<tr><td colspan="3">No markdown headings were detected.</td></tr>')

        page_rows = []
        for page in pages:
            page_rows.append(
                """
<tr>
  <td>{page_number}</td>
  <td>{text_length}</td>
  <td>{element_count}</td>
  <td>{element_types}</td>
  <td>{sample}</td>
</tr>
""".strip().format(
                    page_number=page.get("page_number", 0),
                    text_length=page.get("text_length", 0),
                    element_count=len(page.get("elements", [])),
                    element_types=html.escape(self._format_element_types(page.get("elements", []))),
                    sample=html.escape(self._page_sample(page.get("elements", []))),
                )
            )
        if not page_rows:
            page_rows.append('<tr><td colspan="5">No page payload was captured.</td></tr>')

        overlay_button = (
            '<a class="button secondary" href="{href}">Open Overlay Review</a>'.format(
                href=html.escape(str(document_summary["overlay_href"]))
            )
            if document_summary.get("overlay_href")
            else ""
        )

        return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f3ede2;
      --panel: rgba(255, 252, 247, 0.9);
      --line: rgba(52, 39, 26, 0.14);
      --ink: #1f1a15;
      --muted: #665d54;
      --accent: #9d4527;
      --accent-2: #29577e;
      --shadow: 0 22px 58px rgba(52, 38, 25, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 112, 74, 0.17), transparent 24rem),
        linear-gradient(180deg, #fbf6ef 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", "Malgun Gothic", serif;
    }}
    a {{
      color: inherit;
      text-decoration-thickness: 0.08em;
      text-underline-offset: 0.15em;
    }}
    .wrap {{
      max-width: 98rem;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    .hero {{
      display: grid;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }}
    .hero a {{
      width: fit-content;
      color: var(--muted);
      text-decoration: none;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 0.82rem;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.85rem, 4vw, 3.2rem);
      line-height: 0.98;
    }}
    .hero p {{
      margin: 0;
      max-width: 62rem;
      color: var(--muted);
      line-height: 1.55;
    }}
    .actions {{
      display: flex;
      gap: 0.7rem;
      flex-wrap: wrap;
      margin-top: 0.2rem;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 2.7rem;
      padding: 0.7rem 1rem;
      border-radius: 999px;
      color: #fff;
      text-decoration: none;
      background: linear-gradient(135deg, #9d4527 0%, var(--accent) 100%);
    }}
    .button.secondary {{
      background: linear-gradient(135deg, #37638f 0%, var(--accent-2) 100%);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 0.8rem;
      margin-bottom: 1rem;
    }}
    .metric {{
      padding: 0.95rem 1rem;
      border-radius: 1.1rem;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(52, 39, 26, 0.08);
      box-shadow: var(--shadow);
    }}
    .metric strong {{
      display: block;
      margin-bottom: 0.45rem;
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .metric span {{
      font-size: clamp(1.2rem, 2.7vw, 1.85rem);
      line-height: 1.05;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(22rem, 0.85fr);
      gap: 1rem;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 1.3rem;
      box-shadow: var(--shadow);
    }}
    .panel-inner {{
      padding: 1rem 1.05rem;
    }}
    h2 {{
      margin: 0 0 0.7rem;
      font-size: 1.04rem;
    }}
    .muted {{
      color: var(--muted);
    }}
    .issues {{
      margin: 0;
      padding-left: 1.15rem;
      line-height: 1.55;
    }}
    .issues li {{
      margin-bottom: 0.42rem;
    }}
    .issues strong {{
      margin-right: 0.45rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 0.62rem 0.42rem;
      border-bottom: 1px solid rgba(52, 39, 26, 0.1);
      text-align: left;
      vertical-align: top;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.8rem;
      line-height: 1.5;
      color: #29231d;
    }}
    .scroll {{
      max-height: 32rem;
      overflow: auto;
      padding-right: 0.2rem;
    }}
    details {{
      border-top: 1px solid rgba(52, 39, 26, 0.1);
      padding-top: 0.8rem;
      margin-top: 0.8rem;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 1080px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .metrics {{
        grid-template-columns: 1fr 1fr 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .metrics {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <a href="index.html">Back To Parsing Review</a>
      <h1>{title}</h1>
      <p>Inspect the extracted markdown, the page coverage summary, and the raw structured payload for this document. Use the overlay review when you need original-page positioning for PDF output.</p>
      <div class="actions">
        <a class="button" href="#markdown">Jump To Markdown</a>
        {overlay_button}
      </div>
    </header>

    <section class="metrics">
      {overview_html}
    </section>

    <div class="layout">
      <section class="stack">
        <article class="panel">
          <div class="panel-inner">
            <h2>Page Coverage</h2>
            <table>
              <thead>
                <tr>
                  <th>Page</th>
                  <th>Text Length</th>
                  <th>Elements</th>
                  <th>Element Types</th>
                  <th>Sample</th>
                </tr>
              </thead>
              <tbody>
                {page_rows}
              </tbody>
            </table>
          </div>
        </article>

        <article class="panel" id="markdown">
          <div class="panel-inner">
            <h2>Markdown</h2>
            <div class="scroll">
              <pre>{markdown}</pre>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h2>Raw Payload</h2>
            <details>
              <summary>Open Raw JSON</summary>
              <div class="scroll" style="margin-top:0.8rem;">
                <pre>{raw_json}</pre>
              </div>
            </details>
          </div>
        </article>
      </section>

      <aside class="stack">
        <article class="panel">
          <div class="panel-inner">
            <h2>Document Identity</h2>
            <table>
              <tbody>
                <tr><th>Source Name</th><td>{source_name}</td></tr>
                <tr><th>Source Path</th><td><code>{source_path}</code></td></tr>
                <tr><th>Document ID</th><td><code>{document_id}</code></td></tr>
                <tr><th>Extension</th><td><code>{extension}</code></td></tr>
              </tbody>
            </table>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h2>Issues</h2>
            <ul class="issues">
              {issue_html}
            </ul>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h2>Sections</h2>
            <table>
              <thead>
                <tr>
                  <th>Line</th>
                  <th>Level</th>
                  <th>Title</th>
                </tr>
              </thead>
              <tbody>
                {section_rows}
              </tbody>
            </table>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h2>Classification</h2>
            <div class="scroll">
              <pre>{classification_json}</pre>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-inner">
            <h2>Metadata</h2>
            <div class="scroll">
              <pre>{metadata_json}</pre>
            </div>
          </div>
        </article>
      </aside>
    </div>
  </main>
</body>
</html>
""".format(
            title=html.escape(str(payload.get("source_name", document_summary.get("document_id", "")))),
            overlay_button=overlay_button,
            overview_html=overview_html,
            page_rows="\n".join(page_rows),
            markdown=html.escape(markdown),
            raw_json=html.escape(json.dumps(payload, ensure_ascii=False, indent=2)),
            source_name=html.escape(str(payload.get("source_name", ""))),
            source_path=html.escape(str(payload.get("source_path", ""))),
            document_id=html.escape(str(payload.get("document_id", ""))),
            extension=html.escape(str(payload.get("extension", ""))),
            issue_html="\n".join(issue_html),
            section_rows="\n".join(section_rows),
            classification_json=html.escape(json.dumps(classification, ensure_ascii=False, indent=2)),
            metadata_json=html.escape(json.dumps(metadata, ensure_ascii=False, indent=2)),
        )

    def _page_sample(self, elements: list[dict[str, Any]]) -> str:
        for element in elements:
            text = (element.get("text") or element.get("markdown") or "").strip()
            if text:
                collapsed = " ".join(text.split())
                return collapsed[:180] + ("..." if len(collapsed) > 180 else "")
        return "n/a"

    def _format_element_types(self, elements: list[dict[str, Any]]) -> str:
        counts: dict[str, int] = {}
        for element in elements:
            key = str(element.get("element_type") or "unknown")
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())) or "n/a"
