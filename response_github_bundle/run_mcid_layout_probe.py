from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import fitz

from src.parsers.pdf.structtree_extractor import PowerPointStructTreeExtractor, StructTextRun


def normalize_text(text: str) -> str:
    cleaned = (text or "").replace("\ufeff", "")
    cleaned = cleaned.replace("\u2028", "\n").replace("\u2029", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def text_key(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def sample_text(text: str) -> str:
    return normalize_text(text).replace("\n", " / ")


def bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return (x0, y0, x1, y1)


@dataclass
class VisibleFragment:
    page_number: int
    fragment_index: int
    text: str
    text_key: str
    bbox: tuple[float, float, float, float]


@dataclass
class MatchedFragment:
    page_number: int
    fragment_index: int
    text: str
    bbox: tuple[float, float, float, float]
    mcid: int | None
    struct_block_id: int
    block_role: str
    leaf_role: str
    ambiguous: bool


def parse_pages(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            for page_number in range(start, end + step, step):
                pages.add(page_number)
            continue
        pages.add(int(token))
    return pages


def extract_visible_fragments(page: fitz.Page, page_number: int) -> list[VisibleFragment]:
    fragments: list[VisibleFragment] = []
    page_dict = page.get_text("dict", sort=False)
    index = 1
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = tuple(float(value) for value in block.get("bbox", ()))
        if len(bbox) != 4:
            continue
        lines: list[str] = []
        for line in block.get("lines", []):
            line_text = "".join(str(span.get("text") or "") for span in line.get("spans", []))
            line_text = line_text.rstrip()
            if line_text:
                lines.append(line_text)
        text = normalize_text("\n".join(lines))
        key = text_key(text)
        if not key:
            continue
        fragments.append(
            VisibleFragment(
                page_number=page_number,
                fragment_index=index,
                text=text,
                text_key=key,
                bbox=bbox,
            )
        )
        index += 1
    return fragments


def build_run_buckets(
    runs: list[StructTextRun],
    *,
    page_filter: set[int] | None = None,
) -> tuple[dict[int, dict[str, deque[StructTextRun]]], dict[int, int]]:
    buckets: dict[int, dict[str, deque[StructTextRun]]] = defaultdict(lambda: defaultdict(deque))
    counts: dict[int, int] = defaultdict(int)
    for run in runs:
        if page_filter and run.page_number not in page_filter:
            continue
        key = text_key(run.text)
        if not key:
            continue
        buckets[run.page_number][key].append(run)
        counts[run.page_number] += 1
    return buckets, counts


def match_page_fragments(
    page_number: int,
    fragments: list[VisibleFragment],
    run_buckets: dict[str, deque[StructTextRun]],
) -> tuple[list[MatchedFragment], list[VisibleFragment], int]:
    initial_bucket_sizes = {key: len(value) for key, value in run_buckets.items()}
    matched: list[MatchedFragment] = []
    unmatched: list[VisibleFragment] = []

    for fragment in fragments:
        bucket = run_buckets.get(fragment.text_key)
        if not bucket:
            unmatched.append(fragment)
            continue
        run = bucket.popleft()
        matched.append(
            MatchedFragment(
                page_number=page_number,
                fragment_index=fragment.fragment_index,
                text=fragment.text,
                bbox=fragment.bbox,
                mcid=run.mcids[0] if run.mcids else None,
                struct_block_id=run.block_id,
                block_role=run.block_role,
                leaf_role=run.leaf_role,
                ambiguous=initial_bucket_sizes.get(fragment.text_key, 0) > 1,
            )
        )

    ambiguous_count = sum(1 for item in matched if item.ambiguous)
    return matched, unmatched, ambiguous_count


def render_page_layout(
    page: fitz.Page,
    matched_fragments: list[MatchedFragment],
    *,
    columns: int,
) -> str:
    if not matched_fragments:
        return "[no matched fragments]"

    items = sorted(matched_fragments, key=lambda item: (item.bbox[1], item.bbox[0]))
    heights = [max(1.0, item.bbox[3] - item.bbox[1]) for item in items]
    tolerance = max(1.2, min(2.4, median(heights) * 0.18))

    rows: list[dict[str, Any]] = []
    for item in items:
        top = item.bbox[1]
        if rows and abs(top - rows[-1]["top"]) <= tolerance:
            rows[-1]["items"].append(item)
            rows[-1]["tops"].append(top)
            rows[-1]["top"] = median(rows[-1]["tops"])
        else:
            rows.append({"top": top, "tops": [top], "items": [item]})

    positive_gaps = [
        rows[index]["top"] - rows[index - 1]["top"]
        for index in range(1, len(rows))
        if rows[index]["top"] - rows[index - 1]["top"] > tolerance
    ]
    base_gap = median(positive_gaps) if positive_gaps else max(12.0, median(heights) * 1.5)

    rendered_lines: list[str] = []
    previous_top: float | None = None
    page_width = max(float(page.rect.width), 1.0)

    for row in rows:
        if previous_top is not None:
            gap = row["top"] - previous_top
            if gap > base_gap * 1.8:
                blank_count = max(1, min(6, int(round(gap / base_gap)) - 1))
                rendered_lines.extend([""] * blank_count)
        previous_top = row["top"]

        line_chars = [" "] * columns
        cursor = 0
        for item in sorted(row["items"], key=lambda fragment: fragment.bbox[0]):
            text = normalize_text(item.text).replace("\n", " / ")
            if not text:
                continue
            target_col = int(round((item.bbox[0] / page_width) * (columns - 1)))
            start_col = max(cursor, min(columns - 1, target_col))
            for offset, character in enumerate(text):
                column = start_col + offset
                if column >= columns:
                    break
                line_chars[column] = character
            cursor = min(columns, start_col + len(text) + 1)
        rendered_lines.append("".join(line_chars).rstrip())

    return "\n".join(rendered_lines).rstrip()


def build_layout_markdown(
    pdf_path: Path,
    page_results: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> str:
    lines = [
        "# MCID Layout Probe",
        "",
        f"- Source PDF: `{pdf_path.as_posix()}`",
        "- Structure source: `StructTreeRoot / MCID / ActualText`",
        "- Position source: `page.get_text(\"dict\", sort=False)`",
        "- Match rule: exact text-key match between visible fragments and StructTree runs on the same page",
        "",
        "## Summary",
        "",
        "| Page | Struct Runs | Visible Fragments | Matched | Visible Coverage | Run Coverage | Ambiguous | Remaining Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in page_results:
        lines.append(
            "| {page_number} | {struct_run_count} | {visible_fragment_count} | {matched_fragment_count} | {coverage_pct:.1f}% | {run_coverage_pct:.1f}% | {ambiguous_match_count} | {remaining_run_count} |".format(
                **result
            )
        )

    for result in page_results:
        lines.extend(
            [
                "",
                f"## Page {result['page_number']}",
                "",
                "- Coverage: `{matched_fragment_count}/{visible_fragment_count}` visible fragments matched".format(**result),
                "- Run coverage: `{matched_run_count}/{struct_run_count}` StructTree runs linked to a positioned fragment".format(
                    **result
                ),
                "- Ambiguous matches: `{}`".format(result["ambiguous_match_count"]),
                "- Remaining unmatched MCID runs: `{}`".format(result["remaining_run_count"]),
            ]
        )
        if result["unmatched_fragment_samples"]:
            lines.append(
                "- Unmatched visible samples: `{}`".format(" | ".join(result["unmatched_fragment_samples"]))
            )
        if result["remaining_run_samples"]:
            lines.append(
                "- Remaining MCID samples: `{}`".format(" | ".join(result["remaining_run_samples"]))
            )
        lines.extend(["", "```text", result["layout"], "```"])

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Layout markdown: `{(output_dir / 'layout.md').as_posix()}`",
            f"- Match diagnostics: `{(output_dir / 'matches.json').as_posix()}`",
            f"- HTML overlay: `{(output_dir / 'index.html').as_posix()}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_page_assets(
    document: fitz.Document,
    page_numbers: list[int],
    *,
    output_dir: Path,
    matrix: fitz.Matrix | None = None,
) -> dict[int, str]:
    asset_root = output_dir / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    matrix = matrix or fitz.Matrix(1.5, 1.5)
    relative_paths: dict[int, str] = {}
    for page_number in page_numbers:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        filename = f"page-{page_number:03d}.png"
        output_path = asset_root / filename
        output_path.write_bytes(pixmap.tobytes("png"))
        relative_paths[page_number] = output_path.relative_to(output_dir).as_posix()
    return relative_paths


def overlay_kind(fragment: dict[str, Any]) -> str:
    if fragment.get("ambiguous"):
        return "ambiguous"
    block_role = str(fragment.get("block_role") or "")
    if block_role in {"Title", "H1", "H2", "H3", "H4", "H5", "H6"}:
        return "heading"
    if block_role in {"Table", "TR", "TD", "TH"}:
        return "table"
    if block_role in {"L", "LI"}:
        return "list"
    if block_role in {"Caption", "Note"}:
        return "caption"
    return "text"


def trunc(text: str, limit: int = 88) -> str:
    text = sample_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_overlay_html(
    pdf_path: Path,
    page_results: list[dict[str, Any]],
    asset_paths: dict[int, str],
    *,
    output_dir: Path,
) -> str:
    total_visible = sum(int(page["visible_fragment_count"]) for page in page_results)
    total_matched = sum(int(page["matched_fragment_count"]) for page in page_results)
    total_runs = sum(int(page["struct_run_count"]) for page in page_results)
    total_remaining = sum(int(page["remaining_run_count"]) for page in page_results)
    total_ambiguous = sum(int(page["ambiguous_match_count"]) for page in page_results)

    page_sections: list[str] = []
    for page in page_results:
        page_number = int(page["page_number"])
        image_path = asset_paths.get(page_number, "")
        if not image_path:
            continue
        page_width = max(float(page["page_size"]["width"]), 1.0)
        page_height = max(float(page["page_size"]["height"]), 1.0)
        overlays: list[str] = []

        for fragment in page.get("matched_fragments", []):
            bbox = fragment.get("bbox") or []
            if len(bbox) != 4:
                continue
            left = max(0.0, min(100.0, float(bbox[0]) / page_width * 100.0))
            top = max(0.0, min(100.0, float(bbox[1]) / page_height * 100.0))
            width = max(0.2, min(100.0, (float(bbox[2]) - float(bbox[0])) / page_width * 100.0))
            height = max(0.2, min(100.0, (float(bbox[3]) - float(bbox[1])) / page_height * 100.0))
            kind = overlay_kind(fragment)
            payload = {
                "page_number": fragment.get("page_number"),
                "fragment_index": fragment.get("fragment_index"),
                "mcid": fragment.get("mcid"),
                "struct_block_id": fragment.get("struct_block_id"),
                "block_role": fragment.get("block_role"),
                "leaf_role": fragment.get("leaf_role"),
                "bbox": fragment.get("bbox"),
                "text": sample_text(str(fragment.get("text") or "")),
                "ambiguous": bool(fragment.get("ambiguous")),
            }
            badge = f"MCID {fragment.get('mcid')}" if fragment.get("mcid") is not None else "MCID ?"
            label = trunc(str(fragment.get("text") or ""), limit=36)
            title_text = f"{badge} | {payload['block_role']}/{payload['leaf_role']} | {payload['text']}"
            overlays.append(
                """
<button class="overlay overlay-{kind}" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%"
  data-kind="{kind}" data-payload="{payload}" title="{title}">
  <span>{label}</span>
</button>
""".strip().format(
                    kind=html.escape(kind),
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    payload=html.escape(json.dumps(payload, ensure_ascii=False)),
                    title=html.escape(title_text),
                    label=html.escape(label),
                )
            )

        for fragment in page.get("unmatched_fragments", []):
            bbox = fragment.get("bbox") or []
            if len(bbox) != 4:
                continue
            left = max(0.0, min(100.0, float(bbox[0]) / page_width * 100.0))
            top = max(0.0, min(100.0, float(bbox[1]) / page_height * 100.0))
            width = max(0.2, min(100.0, (float(bbox[2]) - float(bbox[0])) / page_width * 100.0))
            height = max(0.2, min(100.0, (float(bbox[3]) - float(bbox[1])) / page_height * 100.0))
            payload = {
                "page_number": fragment.get("page_number"),
                "fragment_index": fragment.get("fragment_index"),
                "bbox": fragment.get("bbox"),
                "text": sample_text(str(fragment.get("text") or "")),
                "note": "Visible fragment with no exact MCID-text match.",
            }
            overlays.append(
                """
<button class="overlay overlay-unmatched" style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%"
  data-kind="unmatched" data-payload="{payload}" title="{title}">
</button>
""".strip().format(
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    payload=html.escape(json.dumps(payload, ensure_ascii=False)),
                    title=html.escape(f"Unmatched visible fragment | {payload['text']}"),
                )
            )

        page_sections.append(
            """
<section class="page-card">
  <div class="page-header">
    <h2>Page {page_number}</h2>
    <div class="page-stats">
      <span>visible {visible_fragment_count}</span>
      <span>matched {matched_fragment_count}</span>
      <span>runs {struct_run_count}</span>
      <span>remaining {remaining_run_count}</span>
    </div>
  </div>
  <div class="page-stage">
    <img src="{image_path}" alt="Page {page_number}">
    {overlays}
  </div>
</section>
""".strip().format(
                page_number=page_number,
                visible_fragment_count=page["visible_fragment_count"],
                matched_fragment_count=page["matched_fragment_count"],
                struct_run_count=page["struct_run_count"],
                remaining_run_count=page["remaining_run_count"],
                image_path=html.escape(image_path),
                overlays="\n".join(overlays),
            )
        )

    first_payload = {}
    for page in page_results:
        matched_fragments = page.get("matched_fragments", [])
        if matched_fragments:
            first = matched_fragments[0]
            first_payload = {
                "page_number": first.get("page_number"),
                "fragment_index": first.get("fragment_index"),
                "mcid": first.get("mcid"),
                "struct_block_id": first.get("struct_block_id"),
                "block_role": first.get("block_role"),
                "leaf_role": first.get("leaf_role"),
                "bbox": first.get("bbox"),
                "text": sample_text(str(first.get("text") or "")),
                "ambiguous": bool(first.get("ambiguous")),
            }
            break

    first_payload_json = html.escape(json.dumps(first_payload, ensure_ascii=False, indent=2)) if first_payload else "{}"

    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MCID Overlay Probe</title>
  <style>
    :root {{
      --paper: #f7f3ea;
      --ink: #1f1a14;
      --muted: #6b645b;
      --line: rgba(38, 30, 22, 0.15);
      --panel: rgba(255, 252, 246, 0.94);
      --matched: rgba(32, 124, 202, 0.24);
      --heading: rgba(163, 87, 32, 0.28);
      --table: rgba(30, 133, 100, 0.24);
      --list: rgba(115, 84, 184, 0.24);
      --caption: rgba(185, 83, 116, 0.24);
      --ambiguous: rgba(230, 137, 35, 0.34);
      --unmatched: rgba(148, 84, 72, 0.22);
      --shadow: 0 18px 40px rgba(58, 35, 16, 0.09);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(208, 119, 75, 0.16), transparent 30rem),
        linear-gradient(180deg, #fffdf8 0%, var(--paper) 100%);
      font-family: Georgia, "Times New Roman", "Malgun Gothic", serif;
    }}
    .hero {{
      padding: 1.6rem 1.4rem 1rem;
      border-bottom: 1px solid var(--line);
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(1.7rem, 3vw, 2.5rem);
      line-height: 1.03;
    }}
    .hero p {{
      margin: 0.55rem 0 0;
      color: var(--muted);
      max-width: 72rem;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-top: 0.9rem;
    }}
    .chip {{
      padding: 0.42rem 0.65rem;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      font-size: 0.85rem;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(18rem, 22rem) minmax(0, 1fr);
      gap: 1rem;
      padding: 1rem 1.1rem 1.5rem;
      align-items: start;
    }}
    .sidebar {{
      position: sticky;
      top: 0.8rem;
      display: grid;
      gap: 0.85rem;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 0.9rem 0.95rem;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .panel h2 {{
      margin: 0 0 0.55rem;
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .legend {{
      display: grid;
      gap: 0.45rem;
      font-size: 0.9rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 0.55rem;
    }}
    .swatch {{
      width: 1rem;
      height: 1rem;
      border-radius: 0.25rem;
      border: 1px solid rgba(29, 24, 18, 0.16);
      flex: none;
    }}
    .swatch-text {{ background: var(--matched); }}
    .swatch-heading {{ background: var(--heading); }}
    .swatch-table {{ background: var(--table); }}
    .swatch-list {{ background: var(--list); }}
    .swatch-caption {{ background: var(--caption); }}
    .swatch-ambiguous {{ background: var(--ambiguous); }}
    .swatch-unmatched {{
      background: rgba(255,255,255,0.7);
      border-style: dashed;
      border-color: rgba(148, 84, 72, 0.65);
    }}
    .meta {{
      display: grid;
      gap: 0.45rem;
      font-size: 0.9rem;
    }}
    .meta strong {{
      display: block;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.12rem;
    }}
    .detail-pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.82rem;
      line-height: 1.4;
      max-height: 20rem;
      overflow: auto;
    }}
    .pages {{
      display: grid;
      gap: 1rem;
    }}
    .page-card {{
      border: 1px solid var(--line);
      border-radius: 1rem;
      background: rgba(255,255,255,0.75);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .page-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.8rem;
      padding: 0.9rem 1rem 0.75rem;
      border-bottom: 1px solid var(--line);
    }}
    .page-header h2 {{
      margin: 0;
      font-size: 1.05rem;
    }}
    .page-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .page-stage {{
      position: relative;
      padding: 0.9rem;
      background:
        linear-gradient(135deg, rgba(132, 98, 62, 0.05), transparent 40%),
        #fbf8f2;
    }}
    .page-stage img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 0.75rem;
      border: 1px solid var(--line);
      box-shadow: 0 10px 30px rgba(42, 28, 17, 0.06);
    }}
    .overlay {{
      position: absolute;
      border: 1px solid rgba(24, 24, 24, 0.08);
      border-radius: 0.3rem;
      background: var(--matched);
      color: #0f1622;
      cursor: pointer;
      overflow: hidden;
      transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
      padding: 0;
    }}
    .overlay:hover,
    .overlay:focus {{
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(35, 24, 15, 0.18);
      outline: none;
      z-index: 3;
    }}
    .overlay span {{
      display: block;
      width: 100%;
      height: 100%;
      padding: 0.08rem 0.14rem;
      font-size: 0.58rem;
      line-height: 1.2;
      text-align: left;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      opacity: 0.86;
    }}
    .overlay-text {{ background: var(--matched); }}
    .overlay-heading {{ background: var(--heading); }}
    .overlay-table {{ background: var(--table); }}
    .overlay-list {{ background: var(--list); }}
    .overlay-caption {{ background: var(--caption); }}
    .overlay-ambiguous {{ background: var(--ambiguous); }}
    .overlay-unmatched {{
      background: transparent;
      border: 1.5px dashed rgba(148, 84, 72, 0.72);
      z-index: 1;
    }}
    .overlay-unmatched span {{ display: none; }}
    .footer {{
      padding: 0 1.2rem 1.2rem;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    @media (max-width: 980px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>MCID Overlay Probe</h1>
    <p><strong>Source PDF</strong>: {source_pdf}<br>
    StructTree/MCID/ActualText로 텍스트 조각을 읽고, <code>page.get_text("dict", sort=False)</code>의 visible fragment bbox와 exact text-key 매칭해 오버레이를 그렸습니다.</p>
    <div class="stats">
      <span class="chip">Pages {page_count}</span>
      <span class="chip">Visible fragments {total_visible}</span>
      <span class="chip">Matched {total_matched}</span>
      <span class="chip">Struct runs {total_runs}</span>
      <span class="chip">Remaining runs {total_remaining}</span>
      <span class="chip">Ambiguous matches {total_ambiguous}</span>
    </div>
  </header>
  <main class="shell">
    <aside class="sidebar">
      <section class="panel">
        <h2>Legend</h2>
        <div class="legend">
          <div class="legend-item"><span class="swatch swatch-text"></span><span>Matched text fragment</span></div>
          <div class="legend-item"><span class="swatch swatch-heading"></span><span>Heading role</span></div>
          <div class="legend-item"><span class="swatch swatch-table"></span><span>Table role</span></div>
          <div class="legend-item"><span class="swatch swatch-list"></span><span>List role</span></div>
          <div class="legend-item"><span class="swatch swatch-caption"></span><span>Caption or note role</span></div>
          <div class="legend-item"><span class="swatch swatch-ambiguous"></span><span>Matched but duplicate text key</span></div>
          <div class="legend-item"><span class="swatch swatch-unmatched"></span><span>Visible fragment without MCID match</span></div>
        </div>
      </section>
      <section class="panel">
        <h2>Selection</h2>
        <div class="meta">
          <div><strong>Page</strong><span id="detail-page">-</span></div>
          <div><strong>MCID</strong><span id="detail-mcid">-</span></div>
          <div><strong>Role</strong><span id="detail-role">-</span></div>
          <div><strong>BBox</strong><span id="detail-bbox">-</span></div>
          <div><strong>Text</strong><span id="detail-text">Select an overlay.</span></div>
        </div>
      </section>
      <section class="panel">
        <h2>Payload</h2>
        <pre class="detail-pre" id="detail-json">{first_payload_json}</pre>
      </section>
    </aside>
    <section class="pages">
      {page_sections}
    </section>
  </main>
  <div class="footer">
    <p>Generated in <code>{output_dir}</code>. Hover overlays for a quick tooltip and click to inspect the matched payload.</p>
  </div>
  <script>
    const detailPage = document.getElementById('detail-page');
    const detailMcid = document.getElementById('detail-mcid');
    const detailRole = document.getElementById('detail-role');
    const detailBbox = document.getElementById('detail-bbox');
    const detailText = document.getElementById('detail-text');
    const detailJson = document.getElementById('detail-json');

    function updateDetail(button) {{
      const payload = JSON.parse(button.dataset.payload || '{{}}');
      detailPage.textContent = payload.page_number ?? '-';
      detailMcid.textContent = payload.mcid ?? '-';
      const blockRole = payload.block_role || payload.note || '-';
      const leafRole = payload.leaf_role ? ` / ${{payload.leaf_role}}` : '';
      detailRole.textContent = `${{blockRole}}${{leafRole}}`;
      detailBbox.textContent = Array.isArray(payload.bbox) ? payload.bbox.map(v => Number(v).toFixed(2)).join(', ') : '-';
      detailText.textContent = payload.text || '(empty)';
      detailJson.textContent = JSON.stringify(payload, null, 2);
      document.querySelectorAll('.overlay.active').forEach(node => node.classList.remove('active'));
      button.classList.add('active');
    }}

    document.querySelectorAll('.overlay').forEach((button, index) => {{
      button.addEventListener('click', () => updateDetail(button));
      if (index === 0) {{
        updateDetail(button);
      }}
    }});
  </script>
</body>
</html>
""".format(
        source_pdf=html.escape(pdf_path.as_posix()),
        page_count=len(page_results),
        total_visible=total_visible,
        total_matched=total_matched,
        total_runs=total_runs,
        total_remaining=total_remaining,
        total_ambiguous=total_ambiguous,
        first_payload_json=first_payload_json,
        page_sections="\n".join(page_sections),
        output_dir=html.escape(output_dir.as_posix()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a MCID-backed markdown layout probe for a tagged PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF.")
    parser.add_argument("--pages", type=str, default="", help="Comma-separated pages or ranges, e.g. 1,2,4-6")
    parser.add_argument("--columns", type=int, default=140, help="Character width for the rendered layout.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/mcid_layout_probe/<pdf-stem>/",
    )
    args = parser.parse_args()

    pdf_path = args.pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"No such PDF: {pdf_path}")

    page_filter = parse_pages(args.pages)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path.cwd() / "outputs" / "mcid_layout_probe" / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PowerPointStructTreeExtractor()
    document = fitz.open(pdf_path)
    try:
        runs = extractor.extract_runs(document)
        if not runs:
            raise RuntimeError("No StructTree / ActualText runs were found in this PDF.")

        run_buckets_by_page, run_counts_by_page = build_run_buckets(runs, page_filter=page_filter)
        page_numbers = sorted(page_filter or {run.page_number for run in runs})
        page_results: list[dict[str, Any]] = []

        for page_number in page_numbers:
            if page_number < 1 or page_number > document.page_count:
                continue
            page = document[page_number - 1]
            fragments = extract_visible_fragments(page, page_number)
            run_buckets = run_buckets_by_page.get(page_number, {})
            matched, unmatched, ambiguous_count = match_page_fragments(page_number, fragments, run_buckets)
            remaining_runs = [run for bucket in run_buckets.values() for run in bucket]
            matched_boxes = [fragment.bbox for fragment in matched]
            coverage = (len(matched) / len(fragments) * 100.0) if fragments else 0.0

            page_results.append(
                {
                    "page_number": page_number,
                    "page_size": {"width": float(page.rect.width), "height": float(page.rect.height)},
                    "visible_fragment_count": len(fragments),
                    "matched_fragment_count": len(matched),
                    "matched_run_count": len(matched),
                    "coverage_pct": round(coverage, 1),
                    "run_coverage_pct": round(
                        (len(matched) / run_counts_by_page.get(page_number, 1) * 100.0)
                        if run_counts_by_page.get(page_number)
                        else 0.0,
                        1,
                    ),
                    "ambiguous_match_count": ambiguous_count,
                    "remaining_run_count": len(remaining_runs),
                    "matched_bbox_union": list(bbox_union(matched_boxes)) if matched_boxes else None,
                    "unmatched_fragment_samples": [sample_text(fragment.text) for fragment in unmatched[:8]],
                    "remaining_run_samples": [sample_text(run.text) for run in remaining_runs[:8]],
                    "layout": render_page_layout(page, matched, columns=max(40, args.columns)),
                    "matched_fragments": [asdict(fragment) for fragment in matched],
                    "unmatched_fragments": [asdict(fragment) for fragment in unmatched],
                    "remaining_runs": [
                        {
                            "page_number": run.page_number,
                            "block_id": run.block_id,
                            "block_role": run.block_role,
                            "leaf_role": run.leaf_role,
                            "text": normalize_text(run.text),
                            "mcids": list(run.mcids),
                        }
                        for run in remaining_runs
                    ],
                    "struct_run_count": run_counts_by_page.get(page_number, 0),
                }
            )
        asset_paths = render_page_assets(document, page_numbers, output_dir=output_dir)
    finally:
        document.close()

    layout_markdown = build_layout_markdown(pdf_path, page_results, output_dir=output_dir)
    overlay_html = build_overlay_html(pdf_path, page_results, asset_paths, output_dir=output_dir)
    layout_path = output_dir / "layout.md"
    matches_path = output_dir / "matches.json"
    html_path = output_dir / "index.html"
    layout_path.write_text(layout_markdown, encoding="utf-8")
    html_path.write_text(overlay_html, encoding="utf-8")
    matches_path.write_text(
        json.dumps(
            {
                "source_pdf": pdf_path.as_posix(),
                "columns": max(40, args.columns),
                "html_overlay_path": html_path.as_posix(),
                "pages": page_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"layout_path={layout_path}")
    print(f"matches_path={matches_path}")
    print(f"html_path={html_path}")
    for page in page_results:
        print(
            "page={page_number} matched={matched_fragment_count}/{visible_fragment_count} coverage={coverage_pct:.1f}% remaining_runs={remaining_run_count}".format(
                **page
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
