from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.llm_ready import _extract_markdown_table_contexts, _render_table_fact_rows  # noqa: E402


TARGET_DIR = Path(r"C:\Users\yongseop.im\Desktop\미래에셋3분기 관련 문서")
BASE_PATH = TARGET_DIR / "미레에셋3분기 전체.md"
DEST_PATH = TARGET_DIR / "LLM_입력용_미래에셋3분기_전체.md"
LEGACY_DEST_PATH = TARGET_DIR / "llm용 미래에셋3분기 전체.md"
WORK_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "miraeasset_q3_llm_ready" / "LLM_입력용_미래에셋3분기_전체.with_metadata.md"
SOURCE_NAME = "미래에셋증권 3분기 실적보고서.pdf"


def main() -> None:
    base = BASE_PATH.read_text(encoding="utf-8-sig")
    base = re.sub(r"^# .*$", "# LLM용 미래에셋3분기 전체", base, count=1, flags=re.M)
    base = _add_original_source_to_region_blocks(base)

    fact_section, fact_count = _build_kpi_fact_section(base)
    text = base.rstrip() + "\n\n" + fact_section.rstrip() + "\n"

    WORK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORK_OUTPUT_PATH.write_text(text, encoding="utf-8-sig")
    shutil.copyfile(WORK_OUTPUT_PATH, DEST_PATH)
    shutil.copyfile(WORK_OUTPUT_PATH, LEGACY_DEST_PATH)

    checks = [
        f"Original source: {SOURCE_NAME}",
        "Region type: kpi_pair_panel",
        "BBox: [32.0, 92.0, 263.5, 183.5]",
        "Source kind: parser_markdown_table",
        "요약 손익계산서 / 순영업수익 / 3Q24 = 572.8",
        "(연결) 당기순이익 / 3Q25 = 343.8",
    ]
    print(
        json.dumps(
            {
                "output": str(DEST_PATH),
                "legacy_output": str(LEGACY_DEST_PATH),
                "work_output": str(WORK_OUTPUT_PATH),
                "chars": len(text),
                "fact_count": fact_count,
                "checks": {check: check in text for check in checks},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _add_original_source_to_region_blocks(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_source = False
    source_has_original = False

    for line in lines:
        stripped = line.strip()
        if stripped == "Source:":
            in_source = True
            source_has_original = False
            output.append(line)
            continue

        if in_source and stripped.startswith("- Original source:"):
            source_has_original = True
            output.append(line)
            continue

        if in_source and stripped.startswith("- Page:") and not source_has_original:
            output.append(f"- Original source: {SOURCE_NAME}")
            source_has_original = True

        if in_source and not stripped.startswith("- ") and stripped:
            in_source = False

        output.append(line)

    return "\n".join(output).rstrip() + "\n"


def _build_kpi_fact_section(base: str) -> tuple[str, int]:
    page_matches = list(re.finditer(r"^## Page (\d+)\s*$", base, flags=re.M))
    lines = [
        "## QA용 KPI facts",
        "",
        "표/기간형 값은 QA 검색 안정성을 위해 `표 제목 / 항목 / 기간 = 값` 형태로 다시 기록했습니다.",
        "각 fact에는 원본 소스, source kind, page, region id/type, bbox를 함께 둡니다.",
        "",
    ]
    fact_count = 0

    for index, match in enumerate(page_matches):
        page_number = int(match.group(1))
        start = match.end()
        end = page_matches[index + 1].start() if index + 1 < len(page_matches) else len(base)
        page_text = base[start:end]
        contexts = _extract_markdown_table_contexts(page_text)

        for table_index, context in enumerate(contexts, start=1):
            table_title = context.get("title") or f"Page {page_number} Table {table_index}"
            unit = context.get("unit") or ""
            facts = _render_table_fact_rows(
                page_number=page_number,
                table_index=table_index,
                table_title=table_title,
                unit=unit,
                table_text=context.get("table_text") or "",
            )
            if not facts:
                continue

            meta = _infer_table_source_metadata(page_number, table_index, table_title, page_text)
            lines.extend(
                [
                    f"### Page {page_number} Table {table_index}",
                    "",
                    f"Original source: {SOURCE_NAME}",
                    f"Source kind: {meta['source_kind']}",
                    f"Page: {page_number}",
                    f"Region id: {meta['region_id']}",
                    f"Region type: {meta['region_type']}",
                    f"BBox: {meta['bbox']}",
                    f"Table title: {table_title}",
                ]
            )
            if unit:
                lines.append(f"Unit: {unit}")
            lines.extend(["", facts, ""])
            fact_count += facts.count("\n- ") + 1

    return "\n".join(lines), fact_count


def _infer_table_source_metadata(page_number: int, table_index: int, table_title: str, page_text: str) -> dict[str, str]:
    if _looks_visual_region_table(table_title, page_text):
        region = _nearest_region_metadata_for_table(table_index, page_text)
        if region:
            return {
                "source_kind": "visual_semantic_region",
                "region_id": region.get("region_id", ""),
                "region_type": region.get("region_type", ""),
                "bbox": region.get("bbox", ""),
            }
    return {
        "source_kind": "parser_markdown_table",
        "region_id": "",
        "region_type": "markdown_table",
        "bbox": "",
    }


def _looks_visual_region_table(table_title: str, page_text: str) -> bool:
    return bool(table_title.startswith("Page ") or re.search(r"#### Page \d+ / P\d+-R\d+ /", page_text))


def _nearest_region_metadata_for_table(table_index: int, page_text: str) -> dict[str, str] | None:
    # This is intentionally conservative. Parser markdown tables do not have visual bboxes,
    # while visual sections already carry Source metadata in their own blocks.
    regions = []
    pattern = re.compile(
        r"#### Page \d+ / (?P<region_id>P\d+-R\d+) / (?P<region_type>[^\n]+).*?"
        r"Source:\s*\n(?:- Original source: [^\n]+\n)?- Page: \d+\s*\n- Region: (?P=region_id)\s*\n"
        r"- Type: (?P=region_type)\s*\n- BBox: (?P<bbox>\[[^\n]+\])",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(page_text):
        regions.append(match.groupdict())
    if 1 <= table_index <= len(regions):
        return regions[table_index - 1]
    return None


if __name__ == "__main__":
    main()
