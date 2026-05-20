from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "drawing_guided_pages_04_16"
PDF_NAME = "\ubbf8\ub798\uc5d0\uc14b\uc99d\uad8c 3\ubd84\uae30 \uc2e4\uc801\ubcf4\uace0\uc11c.pdf"
PDF_PATH = Path.home() / "Desktop" / "all_docs" / PDF_NAME
PAGE_JSON_PATHS = [
    OUTPUT_DIR / "page_04_drawing_guided_structure.json",
    OUTPUT_DIR / "page_16_drawing_guided_structure.json",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = [load_page_payload(path) for path in PAGE_JSON_PATHS]
    render_page_images(pages)
    html_path = OUTPUT_DIR / "structure_region_viewer.html"
    html_path.write_text(build_html(pages), encoding="utf-8")
    print(html_path.as_posix())


def load_page_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    page_number = int(payload["page_number"])
    payload["image_file"] = f"page_{page_number:02d}_render.png"
    payload["json_file"] = path.name
    return payload


def render_page_images(pages: list[dict[str, Any]]) -> None:
    with fitz.open(PDF_PATH) as document:
        for payload in pages:
            page_number = int(payload["page_number"])
            out = OUTPUT_DIR / payload["image_file"]
            page = document[page_number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(out)


def build_html(pages: list[dict[str, Any]]) -> str:
    data_json = json.dumps(pages, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mirae Asset Q3 Structure Viewer</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #f57c17;
      --blue: #1769e0;
      --green: #07844f;
      --purple: #7c3ed9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 "Segoe UI", Arial, sans-serif;
    }}
    header {{
      height: 54px;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{
      font-size: 16px;
      margin: 0;
      font-weight: 700;
    }}
    .tabs {{
      display: flex;
      gap: 6px;
      margin-left: auto;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
    }}
    button.active {{
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(620px, 1fr) 420px;
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 54px);
    }}
    .canvasPanel,
    .detailPanel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .canvasToolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .canvasToolbar strong {{ font-size: 15px; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .stageWrap {{
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 112px);
    }}
    .stage {{
      position: relative;
      width: min(100%, 1180px);
      margin: 0 auto;
      background: #fff;
      box-shadow: 0 1px 10px rgba(16, 24, 40, 0.08);
    }}
    .stage img {{
      display: block;
      width: 100%;
      height: auto;
      user-select: none;
    }}
    .regionBox {{
      position: absolute;
      border: 2px solid var(--blue);
      background: rgba(23, 105, 224, 0.07);
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}
    .regionBox:nth-of-type(3n + 1) {{ border-color: #ef3b2d; background: rgba(239, 59, 45, 0.07); }}
    .regionBox:nth-of-type(3n + 2) {{ border-color: var(--blue); background: rgba(23, 105, 224, 0.07); }}
    .regionBox:nth-of-type(3n) {{ border-color: var(--green); background: rgba(7, 132, 79, 0.07); }}
    .regionBox.active {{
      border-color: var(--accent);
      background: rgba(245, 124, 23, 0.16);
      box-shadow: 0 0 0 3px rgba(245, 124, 23, 0.28);
      z-index: 3;
    }}
    .regionLabel {{
      position: absolute;
      left: 4px;
      top: -22px;
      padding: 2px 5px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.95);
      color: var(--ink);
      border: 1px solid var(--line);
      font-size: 11px;
      white-space: nowrap;
      pointer-events: none;
    }}
    .detailPanel {{
      display: flex;
      flex-direction: column;
      min-width: 0;
      max-height: calc(100vh - 82px);
    }}
    .detailHeader {{
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }}
    .detailHeader h2 {{
      margin: 0 0 4px;
      font-size: 16px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 76px 1fr;
      gap: 4px 8px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}
    .regionList {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 6px;
      max-height: 240px;
      overflow: auto;
    }}
    .regionItem {{
      display: grid;
      grid-template-columns: 72px 1fr;
      gap: 6px;
      text-align: left;
      padding: 8px;
      border-radius: 6px;
    }}
    .regionItem.active {{
      border-color: var(--accent);
      background: #fff8f2;
    }}
    .regionId {{ font-weight: 700; color: var(--accent); }}
    .regionType {{ color: var(--ink); overflow-wrap: anywhere; }}
    .content {{
      padding: 12px;
      overflow: auto;
      flex: 1;
    }}
    .sectionTitle {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0 0 14px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      color: #26323f;
      font: 12px/1.45 Consolas, "Courier New", monospace;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      .detailPanel {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Mirae Asset Q3 Region Viewer</h1>
    <span class="hint">Click a box to inspect the parser text assigned to that region.</span>
    <div class="tabs" id="tabs"></div>
  </header>
  <main>
    <section class="canvasPanel">
      <div class="canvasToolbar">
        <strong id="pageTitle"></strong>
        <span class="hint" id="strategy"></span>
      </div>
      <div class="stageWrap">
        <div class="stage" id="stage"></div>
      </div>
    </section>
    <aside class="detailPanel">
      <div class="detailHeader">
        <h2 id="regionTitle">Region</h2>
        <div class="meta" id="regionMeta"></div>
      </div>
      <div class="regionList" id="regionList"></div>
      <div class="content">
        <h3 class="sectionTitle">Assigned Text / Markdown</h3>
        <pre id="regionText"></pre>
        <h3 class="sectionTitle">X Coordinate Structure</h3>
        <pre id="regionXStructure"></pre>
        <h3 class="sectionTitle">Evidence</h3>
        <pre id="regionEvidence"></pre>
      </div>
    </aside>
  </main>
  <script>
    const DATA = {data_json};
    let pageIndex = 0;
    let regionIndex = 0;

    const tabs = document.getElementById("tabs");
    const stage = document.getElementById("stage");
    const pageTitle = document.getElementById("pageTitle");
    const strategy = document.getElementById("strategy");
    const regionList = document.getElementById("regionList");
    const regionTitle = document.getElementById("regionTitle");
    const regionMeta = document.getElementById("regionMeta");
    const regionText = document.getElementById("regionText");
    const regionXStructure = document.getElementById("regionXStructure");
    const regionEvidence = document.getElementById("regionEvidence");

    function pct(value, total) {{
      return (value / total * 100).toFixed(4) + "%";
    }}

    function initTabs() {{
      tabs.innerHTML = "";
      DATA.forEach((page, index) => {{
        const btn = document.createElement("button");
        btn.textContent = `Page ${{page.page_number}}`;
        btn.onclick = () => {{
          pageIndex = index;
          regionIndex = 0;
          render();
        }};
        tabs.appendChild(btn);
      }});
    }}

    function render() {{
      const page = DATA[pageIndex];
      const [pageW, pageH] = page.page_size;
      pageTitle.textContent = `Page ${{page.page_number}}`;
      strategy.textContent = page.strategy;
      [...tabs.children].forEach((btn, index) => btn.classList.toggle("active", index === pageIndex));

      stage.innerHTML = "";
      const img = document.createElement("img");
      img.src = page.image_file;
      img.alt = `Page ${{page.page_number}}`;
      stage.appendChild(img);

      page.regions.forEach((region, index) => {{
        const [x0, y0, x1, y1] = region.bbox;
        const box = document.createElement("div");
        box.className = "regionBox" + (index === regionIndex ? " active" : "");
        box.style.left = pct(x0, pageW);
        box.style.top = pct(y0, pageH);
        box.style.width = pct(x1 - x0, pageW);
        box.style.height = pct(y1 - y0, pageH);
        box.onclick = () => {{
          regionIndex = index;
          renderRegion();
          renderBoxesOnly();
        }};

        const label = document.createElement("div");
        label.className = "regionLabel";
        const predictedType = region.type_classification?.predicted_type;
        label.textContent = `${{region.id}} - ${{predictedType || region.type}}`;
        box.appendChild(label);
        stage.appendChild(box);
      }});

      renderRegionList();
      renderRegion();
    }}

    function renderBoxesOnly() {{
      [...stage.querySelectorAll(".regionBox")].forEach((box, index) => {{
        box.classList.toggle("active", index === regionIndex);
      }});
      [...regionList.children].forEach((item, index) => {{
        item.classList.toggle("active", index === regionIndex);
      }});
    }}

    function renderRegionList() {{
      const page = DATA[pageIndex];
      regionList.innerHTML = "";
      page.regions.forEach((region, index) => {{
        const item = document.createElement("button");
        item.className = "regionItem" + (index === regionIndex ? " active" : "");
        item.onclick = () => {{
          regionIndex = index;
          renderRegion();
          renderBoxesOnly();
        }};
        const predictedType = region.type_classification?.predicted_type || "";
        const typeLabel = predictedType ? `${{predictedType}} / ${{region.type}}` : region.type;
        item.innerHTML = `<span class="regionId">${{escapeHtml(region.id)}}</span><span class="regionType">${{escapeHtml(typeLabel)}}</span>`;
        regionList.appendChild(item);
      }});
    }}

    function renderRegion() {{
      const page = DATA[pageIndex];
      const region = page.regions[regionIndex];
      const bbox = region.bbox.map(v => Number(v).toFixed(2)).join(", ");
      const classification = region.type_classification || {{}};
      const predictedType = classification.predicted_type || "";
      regionTitle.textContent = predictedType
        ? `${{region.id}} - ${{predictedType}}`
        : `${{region.id}} - ${{region.type}}`;
      regionMeta.innerHTML = `
        <span>bbox</span><span>[${{bbox}}]</span>
        <span>predicted</span><span>${{escapeHtml(predictedType || "n/a")}}${{classification.confidence ? " (" + classification.confidence + ")" : ""}}</span>
        <span>previous</span><span>${{escapeHtml(region.type || "")}}</span>
        <span>source</span><span>${{escapeHtml(region.source || "")}}</span>
        <span>tokens</span><span>${{region.token_count ?? 0}}</span>
      `;
      regionText.textContent = region.markdown || "_No content extracted._";
      regionXStructure.textContent = JSON.stringify(region.x_coordinate_structure || null, null, 2);
      regionEvidence.textContent = JSON.stringify({{
        type_classification: region.type_classification || null,
        evidence: region.evidence || {{}}
      }}, null, 2);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    initTabs();
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
