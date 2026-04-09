from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import fitz

from src.shared.io import ensure_directory, iso_now, write_bytes, write_json, write_text


class DocumentReviewSiteBuilder:
    def __init__(self, parsed_root: Path, site_root: Path) -> None:
        self.parsed_root = parsed_root
        self.site_root = site_root

    def build(self, comparison_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        ensure_directory(self.site_root)
        comparison_by_source = {
            document["source_path"]: document for document in (comparison_summary or {}).get("documents", [])
        }
        documents: list[dict[str, Any]] = []

        for json_path in sorted(self.parsed_root.glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("status") != "parsed":
                continue

            source_path = Path(payload["source_path"])
            comparison = comparison_by_source.get(source_path.as_posix())
            documents.append(self._build_document(payload, source_path, comparison))

        manifest = {
            "generated_at": iso_now(),
            "site_root": self.site_root.as_posix(),
            "index_path": (self.site_root / "index.html").as_posix(),
            "document_count": len(documents),
            "documents": documents,
        }
        write_text(self.site_root / "index.html", self._render_index_html(manifest))
        write_json(self.site_root / "manifest.json", manifest)
        return manifest

    def _build_document(self, payload: dict[str, Any], source_path: Path, comparison: dict[str, Any] | None) -> dict[str, Any]:
        document_id = payload["document_id"]
        extension = payload.get("extension", "")
        is_pdf = extension == ".pdf" and source_path.exists()
        asset_root = self.site_root / "assets" / document_id
        page_assets = self._render_page_assets(source_path, asset_root) if is_pdf else []
        page_asset_map = {page["page_number"]: page for page in page_assets}
        document_type = payload.get("classification", {}).get("document_type", extension.lstrip(".") or "unknown")

        document_summary = {
            "document_id": document_id,
            "source_name": payload["source_name"],
            "source_path": payload["source_path"],
            "html_path": (self.site_root / f"{document_id}.html").as_posix(),
            "page_count": len(payload.get("pages", [])),
            "issue_count": len(payload.get("issues", [])),
            "document_type": document_type,
            "extension": extension,
            "review_mode": "pdf-overlay" if is_pdf else "parsed-content",
            "selected_strategy": payload.get("metadata", {}).get("markdown_metadata", {}).get("selected_strategy", ""),
            "applied_strategy": payload.get("metadata", {}).get("markdown_source", ""),
            "producer": payload.get("classification", {}).get("metadata", {}).get("producer", ""),
        }

        write_text(
            self.site_root / f"{document_id}.html",
            self._render_document_html(
                payload=payload,
                comparison=comparison,
                page_asset_map=page_asset_map,
                is_pdf=is_pdf,
            ),
        )
        return document_summary

    def _render_page_assets(self, source_path: Path, asset_root: Path) -> list[dict[str, Any]]:
        ensure_directory(asset_root)
        document = fitz.open(source_path)
        try:
            pages: list[dict[str, Any]] = []
            matrix = fitz.Matrix(1.5, 1.5)
            for page_number, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                filename = f"page-{page_number:03d}.png"
                output_path = asset_root / filename
                write_bytes(output_path, pixmap.tobytes("png"))
                pages.append(
                    {
                        "page_number": page_number,
                        "relative_path": output_path.relative_to(self.site_root).as_posix(),
                    }
                )
            return pages
        finally:
            document.close()

    def _render_document_html(
        self,
        payload: dict[str, Any],
        comparison: dict[str, Any] | None,
        page_asset_map: dict[int, dict[str, Any]],
        is_pdf: bool,
    ) -> str:
        source_name = html.escape(payload["source_name"])
        classification = payload.get("classification", {})
        metadata = payload.get("metadata", {})
        document_type = html.escape(classification.get("document_type", payload.get("extension", "").lstrip(".") or "unknown"))
        parser_name = html.escape(payload.get("parser_name", "unknown"))
        issue_count = len(payload.get("issues", []))
        page_count = len(payload.get("pages", []))
        producer = html.escape(classification.get("metadata", {}).get("producer", "") or "n/a")
        selected_strategy = html.escape(metadata.get("markdown_metadata", {}).get("selected_strategy", "") or "n/a")
        applied_strategy = html.escape(metadata.get("markdown_source", "") or "n/a")
        extraction_method = html.escape(metadata.get("extraction_method", "") or "n/a")
        source_path = html.escape(payload.get("source_path", ""))
        extension = html.escape(payload.get("extension", ""))

        element_types = sorted(
            {
                str(element.get("element_type", "unknown"))
                for page in payload.get("pages", [])
                for element in page.get("elements", [])
                if element.get("element_type")
            }
        )
        filter_controls = "\n".join(
            '<label><input type="checkbox" data-filter="{value}" checked> {label}</label>'.format(
                value=html.escape(element_type),
                label=html.escape(element_type.title()),
            )
            for element_type in element_types
        ) or '<p class="muted">No element filters available.</p>'

        page_sections = []
        for page in payload.get("pages", []):
            if is_pdf:
                page_sections.append(self._render_pdf_page(page, page_asset_map))
            else:
                page_sections.append(self._render_text_page(page))

        issue_lines = [
            "<li>{severity}: {message}</li>".format(
                severity=html.escape(issue.get("severity", "info")),
                message=html.escape(issue.get("message", "")),
            )
            for issue in payload.get("issues", [])
        ]
        issues_html = "<ul class=\"issues\">{}</ul>".format("".join(issue_lines)) if issue_lines else "<p class=\"muted\">No issues.</p>"

        comparison_section = self._render_comparison_section(comparison)

        page_intro = (
            "Rendered original PDF pages with parser overlays. Hover for a short summary and click for full detail."
            if is_pdf
            else "Synthetic review view for parsed elements. Original document layout is not available, so parsed blocks are shown as interactive cards."
        )

        return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --paper: #f4efe6;
      --ink: #1f1b16;
      --muted: #6b6258;
      --panel: rgba(255, 251, 245, 0.92);
      --line: rgba(46, 38, 29, 0.16);
      --text: rgba(199, 78, 42, 0.35);
      --table: rgba(38, 93, 139, 0.35);
      --image: rgba(27, 128, 95, 0.35);
      --caption: rgba(123, 74, 170, 0.35);
      --unknown: rgba(112, 96, 78, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(222, 122, 74, 0.18), transparent 28rem),
        linear-gradient(180deg, #fcf8f1 0%, var(--paper) 100%);
      font-family: Georgia, "Times New Roman", "Malgun Gothic", serif;
    }}
    a {{ color: inherit; }}
    .hero {{
      padding: 2rem 1.5rem 1rem;
      border-bottom: 1px solid var(--line);
    }}
    .hero a {{
      text-decoration: none;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .hero h1 {{
      margin: 0.75rem 0 0.45rem;
      font-size: clamp(1.6rem, 3.4vw, 2.7rem);
      line-height: 1.02;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 56rem;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(18rem, 20.75rem) minmax(0, 1fr);
      gap: 1rem;
      padding: 1rem 1.25rem 1.5rem;
      align-items: start;
    }}
    .sidebar {{
      position: sticky;
      top: 0.75rem;
      align-self: start;
    }}
    .sidebar-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.7rem;
    }}
    .panel {{
      min-height: 10.75rem;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 0.75rem 0.8rem;
      box-shadow: 0 18px 40px rgba(66, 41, 22, 0.06);
      backdrop-filter: blur(10px);
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0 0 0.55rem;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .panel-scroll {{
      max-height: 10rem;
      overflow: auto;
      padding-right: 0.15rem;
    }}
    .meta-grid {{
      display: grid;
      gap: 0.42rem;
      font-size: 0.84rem;
    }}
    .meta-row {{
      display: grid;
      gap: 0.12rem;
    }}
    .meta-row strong {{
      font-size: 0.66rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .meta-row code {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 0.72rem;
      color: #3d352d;
    }}
    .filter-grid {{
      display: grid;
      gap: 0.35rem;
      font-size: 0.86rem;
    }}
    .filter-grid label {{
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }}
    .hover-title,
    .detail-title {{
      margin: 0 0 0.55rem;
      font-size: 0.94rem;
      line-height: 1.25;
    }}
    .hover-body,
    .detail-body,
    .detail-json {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #3c342d;
      font-size: 0.82rem;
      line-height: 1.4;
    }}
    .detail-grid {{
      display: grid;
      gap: 0.42rem;
    }}
    .detail-json {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.71rem;
      padding-top: 0.3rem;
      border-top: 1px solid var(--line);
    }}
    .muted {{
      margin: 0;
      color: var(--muted);
    }}
    .main {{
      display: grid;
      gap: 1rem;
    }}
    .section-card {{
      background: rgba(255, 252, 248, 0.84);
      border: 1px solid var(--line);
      border-radius: 1.4rem;
      padding: 0.9rem;
      box-shadow: 0 22px 50px rgba(53, 38, 26, 0.07);
    }}
    .section-card h2 {{
      margin: 0 0 0.8rem;
      font-size: 1.05rem;
    }}
    .pages {{
      display: grid;
      gap: 1rem;
    }}
    .page-card {{
      background: rgba(255, 252, 248, 0.84);
      border: 1px solid var(--line);
      border-radius: 1.4rem;
      padding: 0.9rem;
      box-shadow: 0 22px 50px rgba(53, 38, 26, 0.07);
    }}
    .page-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
      margin-bottom: 0.8rem;
    }}
    .page-header h3 {{
      margin: 0;
      font-size: 1.05rem;
    }}
    .page-header span {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .page-stage {{
      position: relative;
      border-radius: 1rem;
      overflow: hidden;
      border: 1px solid rgba(58, 46, 35, 0.18);
      background: white;
    }}
    .page-stage img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .overlay {{
      position: absolute;
      display: block;
      border-radius: 0.35rem;
      border: 1px solid rgba(28, 24, 20, 0.28);
      cursor: pointer;
      padding: 0;
      background: transparent;
      box-shadow: inset 0 0 0 999px transparent;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    .overlay[data-badge]:hover::after,
    .overlay[data-badge]:focus::after {{
      content: attr(data-badge);
      position: absolute;
      left: 0;
      top: 0;
      transform: translate(0.15rem, 0.15rem);
      padding: 0.12rem 0.32rem;
      border-radius: 999px;
      background: rgba(28, 24, 20, 0.86);
      color: #fff;
      font-size: 0.64rem;
      letter-spacing: 0.04em;
      white-space: nowrap;
      pointer-events: none;
    }}
    .overlay:hover,
    .overlay:focus,
    .list-element:hover,
    .list-element:focus {{
      transform: scale(1.01);
      outline: none;
      box-shadow: inset 0 0 0 999px rgba(255, 255, 255, 0.16), 0 0 0 2px rgba(28, 24, 20, 0.18);
    }}
    .overlay-text, .list-element-text {{ background: var(--text); }}
    .overlay-table, .list-element-table {{ background: var(--table); }}
    .overlay-image, .list-element-image {{ background: var(--image); }}
    .overlay-caption, .list-element-caption {{ background: var(--caption); }}
    .overlay-unknown, .list-element-unknown {{ background: var(--unknown); }}
    .overlay.hidden,
    .list-element.hidden {{ display: none; }}
    .synthetic-stage {{
      display: grid;
      gap: 0.7rem;
    }}
    .list-element {{
      width: 100%;
      text-align: left;
      border: 1px solid rgba(28, 24, 20, 0.16);
      border-radius: 0.9rem;
      padding: 0.85rem 0.9rem;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    .list-label {{
      display: flex;
      justify-content: space-between;
      gap: 0.8rem;
      margin-bottom: 0.35rem;
      font-size: 0.82rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #3d352d;
    }}
    .list-preview {{
      margin: 0;
      color: #2d2924;
      font-size: 0.95rem;
      line-height: 1.45;
      display: -webkit-box;
      -webkit-line-clamp: 4;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .comparison-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.87rem;
    }}
    .comparison-table th,
    .comparison-table td {{
      padding: 0.45rem 0.35rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .issues {{
      margin: 0;
      padding-left: 1.1rem;
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .shell {{ grid-template-columns: 1fr; padding: 1.25rem; }}
      .sidebar {{ position: static; }}
      .hero {{ padding: 2rem 1.25rem 1rem; }}
    }}
    @media (max-width: 760px) {{
      .sidebar-grid {{ grid-template-columns: 1fr; }}
      .panel {{ min-height: auto; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <a href="index.html">Back to index</a>
    <h1>{source_name}</h1>
    <p>{page_intro}</p>
  </header>
  <div class="shell">
    <aside class="sidebar">
      <div class="sidebar-grid">
        <section class="panel">
          <h2>Overview</h2>
          <div class="panel-scroll">
            <div class="meta-grid">
              <div class="meta-row"><strong>Type</strong><span>{document_type}</span></div>
              <div class="meta-row"><strong>Parser</strong><span>{parser_name}</span></div>
              <div class="meta-row"><strong>Extension</strong><span>{extension}</span></div>
              <div class="meta-row"><strong>Pages</strong><span>{page_count}</span></div>
              <div class="meta-row"><strong>Issues</strong><span>{issue_count}</span></div>
              <div class="meta-row"><strong>Producer</strong><span>{producer}</span></div>
              <div class="meta-row"><strong>Strategy</strong><span>{selected_strategy}</span></div>
              <div class="meta-row"><strong>Applied</strong><span>{applied_strategy}</span></div>
              <div class="meta-row"><strong>Extraction</strong><span>{extraction_method}</span></div>
              <div class="meta-row"><strong>Source</strong><code>{source_path}</code></div>
            </div>
          </div>
        </section>
        <section class="panel">
          <h2>Controls</h2>
          <div class="panel-scroll">
            <div class="filter-grid">
              {filter_controls}
            </div>
          </div>
        </section>
        <section class="panel">
          <h2>Hover Summary</h2>
          <div class="panel-scroll">
            <h3 class="hover-title" id="hover-title">Move over an element</h3>
            <p class="hover-body" id="hover-body">A short explanation for the current overlay or parsed block appears here.</p>
          </div>
        </section>
        <section class="panel">
          <h2>Selected Element</h2>
          <div class="panel-scroll">
            <div class="detail-grid">
              <h3 class="detail-title" id="detail-title">Click an element</h3>
              <p class="detail-body" id="detail-body">Detailed text, markdown, source, and metadata will be shown here.</p>
              <pre class="detail-json" id="detail-json">No element selected.</pre>
            </div>
          </div>
        </section>
      </div>
    </aside>
    <main class="main">
      <section class="section-card">
        <h2>Pages</h2>
        <p class="muted">{page_intro}</p>
      </section>
      <div class="pages">
        {page_sections}
      </div>
      <section class="section-card">
        <h2>Comparison</h2>
        {comparison_section}
      </section>
      <section class="section-card">
        <h2>Pipeline Issues</h2>
        {issues_html}
      </section>
    </main>
  </div>
  <script>
    const hoverTitle = document.getElementById('hover-title');
    const hoverBody = document.getElementById('hover-body');
    const detailTitle = document.getElementById('detail-title');
    const detailBody = document.getElementById('detail-body');
    const detailJson = document.getElementById('detail-json');
    const defaultHoverTitle = hoverTitle.textContent;
    const defaultHoverBody = hoverBody.textContent;

    const setHoverSummary = (title, body) => {{
      hoverTitle.textContent = title || defaultHoverTitle;
      hoverBody.textContent = body || defaultHoverBody;
    }};

    const resetHoverSummary = () => {{
      hoverTitle.textContent = defaultHoverTitle;
      hoverBody.textContent = defaultHoverBody;
    }};

    const syncFilters = () => {{
      document.querySelectorAll('[data-filter]').forEach((checkbox) => {{
        const type = checkbox.dataset.filter;
        document.querySelectorAll(`[data-element-type="${{type}}"]`).forEach((element) => {{
          element.classList.toggle('hidden', !checkbox.checked);
        }});
      }});
    }};

    document.querySelectorAll('[data-filter]').forEach((checkbox) => {{
      checkbox.addEventListener('change', syncFilters);
    }});
    syncFilters();

    document.querySelectorAll('[data-element]').forEach((button) => {{
      button.addEventListener('mouseenter', () => {{
        setHoverSummary(button.dataset.summaryTitle, button.dataset.summaryBody);
      }});
      button.addEventListener('focus', () => {{
        setHoverSummary(button.dataset.summaryTitle, button.dataset.summaryBody);
      }});
      button.addEventListener('mouseleave', resetHoverSummary);
      button.addEventListener('blur', resetHoverSummary);
      button.addEventListener('click', () => {{
        try {{
          const payload = JSON.parse(button.dataset.element);
          const metadata = payload.metadata || {{}};
          const titleParts = [`${{payload.element_type}}`, `page ${{payload.page_number}}`, `${{payload.element_id}}`];
          if (metadata.mcid !== undefined && metadata.mcid !== null) {{
            titleParts.push(`MCID ${{metadata.mcid}}`);
          }}
          detailTitle.textContent = titleParts.join(' · ');

          const bodyParts = [];
          if (payload.text) {{
            bodyParts.push(payload.text);
          }} else if (payload.markdown) {{
            bodyParts.push(payload.markdown);
          }}
          const metaParts = [];
          if (metadata.mcid !== undefined && metadata.mcid !== null) {{
            metaParts.push(`MCID: ${{metadata.mcid}}`);
          }}
          if (metadata.xobject_name) {{
            metaParts.push(`XObject: ${{metadata.xobject_name}}`);
          }}
          if (metadata.resource_xref !== undefined && metadata.resource_xref !== null) {{
            metaParts.push(`Resource xref: ${{metadata.resource_xref}}`);
          }}
          if (metadata.rendered_xref !== undefined && metadata.rendered_xref !== null && metadata.rendered_xref !== metadata.resource_xref) {{
            metaParts.push(`Rendered xref: ${{metadata.rendered_xref}}`);
          }}
          if (metadata.smask_xref !== undefined && metadata.smask_xref !== null) {{
            metaParts.push(`SMask xref: ${{metadata.smask_xref}}`);
          }}
          if (metadata.mcid_roles && metadata.mcid_roles.length) {{
            metaParts.push(`Roles: ${{metadata.mcid_roles.join(', ')}}`);
          }}
          if (metaParts.length) {{
            bodyParts.push(metaParts.join('\\n'));
          }}
          detailBody.textContent = bodyParts.join('\\n\\n') || 'No text payload available.';
          detailJson.textContent = JSON.stringify(payload, null, 2);
        }} catch (error) {{
          detailTitle.textContent = 'Parse error';
          detailBody.textContent = String(error);
          detailJson.textContent = String(error);
        }}
      }});
    }});
  </script>
</body>
</html>
""".format(
            title=source_name,
            source_name=source_name,
            page_intro=html.escape(page_intro),
            document_type=document_type,
            parser_name=parser_name,
            extension=extension,
            page_count=page_count,
            issue_count=issue_count,
            producer=producer,
            selected_strategy=selected_strategy,
            applied_strategy=applied_strategy,
            extraction_method=extraction_method,
            source_path=source_path,
            filter_controls=filter_controls,
            page_sections="\n".join(page_sections) or '<section class="page-card"><p class="muted">No parsed pages.</p></section>',
            comparison_section=comparison_section,
            issues_html=issues_html,
        )

    def _render_pdf_page(self, page: dict[str, Any], page_asset_map: dict[int, dict[str, Any]]) -> str:
        page_number = page["page_number"]
        asset = page_asset_map.get(page_number)
        if not asset:
            return """
<section class="page-card">
  <div class="page-header"><h3>Page {page_number}</h3><span>asset missing</span></div>
  <p class="muted">Rendered PDF asset not available for this page.</p>
</section>
""".strip().format(page_number=page_number)

        overlays = []
        for element in page.get("elements", []):
            bbox = element.get("bbox") or []
            if len(bbox) != 4:
                continue

            left = max(0.0, min(100.0, bbox[0] / max(page["width"], 1.0) * 100))
            top = max(0.0, min(100.0, bbox[1] / max(page["height"], 1.0) * 100))
            width = max(0.25, min(100.0, (bbox[2] - bbox[0]) / max(page["width"], 1.0) * 100))
            height = max(0.25, min(100.0, (bbox[3] - bbox[1]) / max(page["height"], 1.0) * 100))
            element_type = self._safe_element_type(element.get("element_type"))
            payload_json = self._element_payload_json(element)
            summary_title, summary_body = self._element_summary(element)
            badge = self._element_badge(element)
            title_text = summary_title if not summary_body else f"{summary_title} | {summary_body}"

            overlays.append(
                """
<button class="overlay overlay-{element_type}" data-element-type="{element_type}" data-element="{payload}" data-summary-title="{summary_title}" data-summary-body="{summary_body}"{badge_attr} title="{title_text}" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%"></button>
""".strip().format(
                    element_type=element_type,
                    payload=html.escape(payload_json),
                    summary_title=html.escape(summary_title),
                    summary_body=html.escape(summary_body),
                    badge_attr=f' data-badge="{html.escape(badge)}"' if badge else "",
                    title_text=html.escape(title_text),
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                )
            )

        return """
<section class="page-card">
  <div class="page-header">
    <h3>Page {page_number}</h3>
    <span>{element_count} elements</span>
  </div>
  <div class="page-stage">
    <img src="{image_path}" alt="Page {page_number}">
    {overlays}
  </div>
</section>
""".strip().format(
            page_number=page_number,
            element_count=len(page.get("elements", [])),
            image_path=html.escape(asset["relative_path"]),
            overlays="\n".join(overlays),
        )

    def _render_text_page(self, page: dict[str, Any]) -> str:
        items = []
        for element in page.get("elements", []):
            element_type = self._safe_element_type(element.get("element_type"))
            payload_json = self._element_payload_json(element)
            summary_title, summary_body = self._element_summary(element)
            preview_text = html.escape((element.get("text") or element.get("markdown") or "").strip() or "No text payload.")
            label = "{element_type} · {element_id}".format(
                element_type=html.escape(element.get("element_type", "unknown")),
                element_id=html.escape(element.get("element_id", "")),
            )
            source = html.escape(str((element.get("metadata") or {}).get("source", "parsed")))
            items.append(
                """
<button class="list-element list-element-{element_type}" data-element-type="{element_type}" data-element="{payload}" data-summary-title="{summary_title}" data-summary-body="{summary_body}">
  <div class="list-label"><span>{label}</span><span>{source}</span></div>
  <p class="list-preview">{preview}</p>
</button>
""".strip().format(
                    element_type=element_type,
                    payload=html.escape(payload_json),
                    summary_title=html.escape(summary_title),
                    summary_body=html.escape(summary_body),
                    label=label,
                    source=source,
                    preview=preview_text,
                )
            )

        return """
<section class="page-card">
  <div class="page-header">
    <h3>Page {page_number}</h3>
    <span>{element_count} elements</span>
  </div>
  <div class="synthetic-stage">
    {items}
  </div>
</section>
""".strip().format(
            page_number=page["page_number"],
            element_count=len(page.get("elements", [])),
            items="\n".join(items) or '<p class="muted">No parsed elements on this page.</p>',
        )

    def _render_comparison_section(self, comparison: dict[str, Any] | None) -> str:
        if not comparison:
            return (
                "<p class=\"muted\">This review uses the main parsing pipeline output directly. "
                "The overlay reflects the DOC/HWP/PDF routing pipeline and, for PDFs, the producer-based strategy selected by that pipeline.</p>"
            )

        rows = []
        for result in comparison.get("results", []):
            rows.append(
                """
<tr>
  <td>{strategy}</td>
  <td>{applied}</td>
  <td>{status}</td>
  <td>{elapsed}</td>
  <td>{score}</td>
  <td>{f1}</td>
  <td>{issues}</td>
</tr>
""".strip().format(
                    strategy=html.escape(result["strategy_name"]),
                    applied=html.escape(result["applied_strategy"]),
                    status=html.escape(result["status"]),
                    elapsed=result["elapsed_ms"],
                    score=result["quality_score"],
                    f1=result["token_f1"],
                    issues=result["issue_count"],
                )
            )

        return """
<table class="comparison-table">
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Applied</th>
      <th>Status</th>
      <th>Time (ms)</th>
      <th>Score</th>
      <th>Token F1</th>
      <th>Issues</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip().format(rows="\n".join(rows))

    def _element_payload_json(self, element: dict[str, Any]) -> str:
        payload = {
            "element_id": element.get("element_id"),
            "element_type": element.get("element_type"),
            "page_number": element.get("page_number"),
            "bbox": element.get("bbox"),
            "text": (element.get("text") or "")[:1600],
            "markdown": (element.get("markdown") or "")[:1600],
            "metadata": element.get("metadata") or {},
        }
        return json.dumps(payload, ensure_ascii=False)

    def _element_summary(self, element: dict[str, Any]) -> tuple[str, str]:
        element_type = str(element.get("element_type", "unknown"))
        page_number = element.get("page_number", "?")
        metadata = element.get("metadata") or {}
        source = str(metadata.get("source", "parsed"))
        text = (element.get("text") or element.get("markdown") or "").strip().replace("\n", " ")
        snippet = text[:160] + ("..." if len(text) > 160 else "")
        mcid = metadata.get("mcid")
        xobject_name = metadata.get("xobject_name")
        resource_xref = metadata.get("resource_xref")
        rendered_xref = metadata.get("rendered_xref")
        roles = metadata.get("mcid_roles") or []

        summary_title = f"Page {page_number} · {element_type} · {source}"
        if mcid is not None:
            summary_title += f" · MCID {mcid}"

        detail_parts = []
        if snippet:
            detail_parts.append(snippet)

        metadata_parts = []
        if mcid is not None:
            metadata_parts.append(f"MCID {mcid}")
        if xobject_name:
            metadata_parts.append(f"XObject {xobject_name}")
        if resource_xref is not None:
            metadata_parts.append(f"xref {resource_xref}")
        if rendered_xref is not None and rendered_xref != resource_xref:
            metadata_parts.append(f"rendered {rendered_xref}")
        if roles:
            metadata_parts.append("roles " + ", ".join(str(role) for role in roles[:2]))
        if metadata_parts:
            detail_parts.append(" · ".join(metadata_parts))

        summary_body = "\n".join(detail_parts) if detail_parts else "No text preview available."
        return summary_title, summary_body

    def _element_badge(self, element: dict[str, Any]) -> str:
        metadata = element.get("metadata") or {}
        mcid = metadata.get("mcid")
        if mcid is None:
            return ""
        return f"MCID {mcid}"

    def _safe_element_type(self, element_type: Any) -> str:
        value = str(element_type or "unknown").strip().lower()
        return value if value in {"text", "table", "image", "caption"} else "unknown"

    def _render_index_html(self, manifest: dict[str, Any]) -> str:
        rows = []
        for document in manifest["documents"]:
            rows.append(
                """
<tr>
  <td><a href="{href}">{name}</a></td>
  <td>{doc_type}</td>
  <td>{mode}</td>
  <td>{producer}</td>
  <td>{selected}</td>
  <td>{pages}</td>
  <td>{issues}</td>
</tr>
""".strip().format(
                    href=html.escape(Path(document["html_path"]).name),
                    name=html.escape(document["source_name"]),
                    doc_type=html.escape(document["document_type"]),
                    mode=html.escape(document["review_mode"]),
                    producer=html.escape(document["producer"] or "n/a"),
                    selected=html.escape(document["selected_strategy"] or "n/a"),
                    pages=document["page_count"],
                    issues=document["issue_count"],
                )
            )

        return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Review UI</title>
  <style>
    body {{
      margin: 0;
      color: #181511;
      background:
        linear-gradient(135deg, rgba(213, 112, 74, 0.18), transparent 28rem),
        linear-gradient(180deg, #f7f1e7 0%, #efe7d9 100%);
      font-family: Georgia, "Times New Roman", "Malgun Gothic", serif;
    }}
    .wrap {{
      max-width: 86rem;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 3rem;
    }}
    h1 {{
      margin: 0 0 0.75rem;
      font-size: clamp(2rem, 4vw, 3.6rem);
      line-height: 1;
    }}
    p {{
      margin: 0;
      max-width: 58rem;
      color: #5f574e;
    }}
    .panel {{
      margin-top: 1.6rem;
      padding: 1rem 1.1rem;
      border-radius: 1.2rem;
      background: rgba(255, 251, 245, 0.84);
      border: 1px solid rgba(42, 31, 21, 0.16);
      box-shadow: 0 22px 60px rgba(51, 36, 25, 0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.75rem;
    }}
    th, td {{
      padding: 0.6rem 0.45rem;
      border-bottom: 1px solid rgba(42, 31, 21, 0.12);
      text-align: left;
      vertical-align: top;
    }}
    a {{
      color: inherit;
      text-decoration-thickness: 0.08em;
      text-underline-offset: 0.16em;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Document Review UI</h1>
    <p>Generated at {generated_at}. PDF documents keep original-page overlays, while DOC, HWP, DOCX, and TXT documents are shown as parsed-content review pages.</p>
    <section class="panel">
      <strong>Total documents:</strong> {document_count}
      <table>
        <thead>
          <tr>
            <th>Document</th>
            <th>Type</th>
            <th>Mode</th>
            <th>Producer</th>
            <th>Strategy</th>
            <th>Pages</th>
            <th>Issues</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
""".format(
            generated_at=html.escape(manifest["generated_at"]),
            document_count=manifest["document_count"],
            rows="\n".join(rows),
        )


PdfOverlaySiteBuilder = DocumentReviewSiteBuilder
