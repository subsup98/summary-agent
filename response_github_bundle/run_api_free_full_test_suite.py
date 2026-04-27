from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "requested_md_bundle" / "api_free_full_test_suite.md"

NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:[.,/-]\d+)*\b")
TEMPORAL_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b[1-4]q\d{2}\b|\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b", re.IGNORECASE)


@dataclass
class DocBundle:
    document_id: str
    source_name: str
    path: Path
    payload: dict


@dataclass
class Candidate:
    doc_id: str
    source_name: str
    strategy: str
    chunk_index: int
    section_hint: str
    text: str
    overlap: int
    numeric_overlap: int
    temporal_overlap: int
    table_bonus: float
    score: float
    distance: float
    matching_tokens: list[str]


def old_tokens(text: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower()))


def new_tokens(text: str) -> set[str]:
    normalized = str(text or "").lower()
    basic_tokens = [token for token in re.findall(r"[0-9A-Za-z가-힣]+", normalized) if token]
    if not basic_tokens:
        return set()

    tokens = set(basic_tokens)
    compact = re.sub(r"[^0-9a-zA-Z가-힣]+", "", normalized)
    if compact and 2 <= len(compact) <= 24:
        tokens.add(compact)

    for left, right in zip(basic_tokens, basic_tokens[1:]):
        if len(left) < 2 or len(right) < 2:
            continue
        combined = f"{left}{right}"
        if len(combined) <= 24:
            tokens.add(combined)

    return tokens


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"<br\s*/?>", "\n", str(text or ""), flags=re.IGNORECASE)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    parts = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", normalized) if item.strip()]
    return parts


def extract_numeric_tokens(text: str) -> set[str]:
    return set(NUMERIC_TOKEN_RE.findall(str(text or "")))


def extract_temporal_tokens(text: str) -> set[str]:
    return set(token.lower() for token in TEMPORAL_TOKEN_RE.findall(str(text or "")))


def text_has_table_structure(text: str) -> bool:
    sample = str(text or "")
    return sample.count("|") >= 4 or "[row_path]" in sample.lower() or "[financial fact table]" in sample.lower()


def query_prefers_tabular_asset(query: str) -> bool:
    normalized = str(query or "").lower()
    hints = ("표", "테이블", "비율", "수익률", "per", "pbr", "bps", "eps")
    return any(hint in normalized for hint in hints)


def table_query_bonus(query: str, text: str, section_hint: str, tokenizer: Callable[[str], set[str]]) -> float:
    if not query_prefers_tabular_asset(query):
        return 0.0
    score = 0.0
    if text_has_table_structure(text):
        score += 12.0
    header_like_tokens = {"성장률", "비율", "증감", "달성률", "개최일자", "참석인원", "항목", "실적", "계획", "수익률", "per"}
    query_tokens = tokenizer(query)
    section_tokens = tokenizer(section_hint)
    text_tokens = tokenizer(text)
    score += len((query_tokens & header_like_tokens) & section_tokens) * 6.0
    score += len((query_tokens & header_like_tokens) & text_tokens) * 8.0
    return score


def score_candidate(
    *,
    query: str,
    text: str,
    source_name: str,
    section_hint: str,
    tokenizer: Callable[[str], set[str]],
) -> tuple[float, int, int, int, float, list[str]]:
    query_tokens = tokenizer(query)
    text_tokens = tokenizer(text)
    section_tokens = tokenizer(section_hint)
    source_tokens = tokenizer(source_name)
    matching = sorted(query_tokens & text_tokens)
    overlap = len(query_tokens & text_tokens) + len(query_tokens & section_tokens) * 3 + len(query_tokens & source_tokens) * 6
    numeric_overlap = len(extract_numeric_tokens(query) & extract_numeric_tokens(text))
    temporal_overlap = len(extract_temporal_tokens(query) & extract_temporal_tokens(text))
    bonus = table_query_bonus(query, text, section_hint, tokenizer)
    score = overlap + numeric_overlap * 8 + temporal_overlap * 6 + bonus
    return score, len(matching), numeric_overlap, temporal_overlap, bonus, matching


def load_doc_bundles() -> list[DocBundle]:
    doc_paths = sorted(PROJECT_ROOT.glob("outputs/ui_runs/*/structured/documents/*.json"))
    bundles: list[DocBundle] = []
    for path in doc_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundles.append(
            DocBundle(
                document_id=str(payload.get("document_id") or path.stem),
                source_name=str(payload.get("source_name") or path.name),
                path=path,
                payload=payload,
            )
        )
    return bundles


def gather_candidates(
    bundles: list[DocBundle],
    *,
    query: str,
    tokenizer: Callable[[str], set[str]],
    use_hard_filter: bool,
    chunk_strategy: str,
) -> list[Candidate]:
    results: list[Candidate] = []
    query_token_set = tokenizer(query)
    for bundle in bundles:
        raw_chunks = bundle.payload.get("semantic_chunks") if chunk_strategy == "semantic" else bundle.payload.get("chunks")
        raw_chunks = raw_chunks or []
        for index, chunk in enumerate(raw_chunks):
            text = str(chunk.get("text") or chunk.get("serialized_text") or "").strip()
            if not text:
                continue
            section_hint = str(chunk.get("section_hint") or chunk.get("section") or "")
            score, token_overlap, numeric_overlap, temporal_overlap, bonus, matching = score_candidate(
                query=query,
                text=text,
                source_name=bundle.source_name,
                section_hint=section_hint,
                tokenizer=tokenizer,
            )
            if use_hard_filter and query_token_set and token_overlap == 0:
                continue
            results.append(
                Candidate(
                    doc_id=bundle.document_id,
                    source_name=bundle.source_name,
                    strategy=chunk_strategy,
                    chunk_index=int(chunk.get("chunk_index", index) or index),
                    section_hint=section_hint,
                    text=text,
                    overlap=token_overlap,
                    numeric_overlap=numeric_overlap,
                    temporal_overlap=temporal_overlap,
                    table_bonus=bonus,
                    score=score,
                    distance=float(max(0.0, 100.0 - token_overlap)),
                    matching_tokens=matching,
                )
            )
    results.sort(key=lambda item: (item.score, -item.distance, len(item.text)), reverse=True)
    return results


def evidence_sentences(query: str, candidate: Candidate, tokenizer: Callable[[str], set[str]], use_hard_filter: bool) -> list[tuple[float, str]]:
    results: list[tuple[float, str]] = []
    query_tokens = tokenizer(query)
    section_tokens = tokenizer(candidate.section_hint)
    for sentence in split_sentences(candidate.text):
        sentence_tokens = tokenizer(sentence)
        overlap = len(query_tokens & sentence_tokens)
        if use_hard_filter and overlap == 0:
            continue
        score = overlap * 10.0
        score += len(query_tokens & section_tokens) * 6.0
        score += len(extract_numeric_tokens(query) & extract_numeric_tokens(sentence)) * 12.0
        score += len(extract_temporal_tokens(query) & extract_temporal_tokens(sentence)) * 10.0
        score += table_query_bonus(query, sentence, candidate.section_hint, tokenizer)
        score -= candidate.distance
        results.append((score, sentence))
    results.sort(key=lambda item: item[0], reverse=True)
    return results


def pick_bundles_for_focus(bundles: list[DocBundle]) -> tuple[DocBundle, DocBundle]:
    nh = next(bundle for bundle in bundles if "9d1b6d54" in bundle.path.name)
    doosan = next(bundle for bundle in bundles if "0a582719" in bundle.path.name)
    return nh, doosan


def render_top_candidates(title: str, items: list[Candidate], limit: int = 5) -> list[str]:
    lines = [f"### {title}", "", "| rank | source | strategy | chunk | score | overlap | numeric | temporal | table_bonus | matching_tokens | preview |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rank, item in enumerate(items[:limit], start=1):
        preview = re.sub(r"\s+", " ", item.text.replace("\n", " ")).strip()
        preview = preview[:110] + ("..." if len(preview) > 110 else "")
        preview_safe = preview.replace("|", "\\|")
        lines.append(
            f"| {rank} | {item.source_name[:28]} | {item.strategy} | {item.chunk_index} | {item.score:.1f} | {item.overlap} | {item.numeric_overlap} | {item.temporal_overlap} | {item.table_bonus:.1f} | `{item.matching_tokens}` | {preview_safe} |"
        )
    lines.append("")
    return lines


def render_suite(bundles: list[DocBundle]) -> str:
    nh, doosan = pick_bundles_for_focus(bundles)
    lines: list[str] = [
        "# API-Free Full Test Suite",
        "",
        "- scope: tokenization, overlap, structured fallback ranking, evidence scoring, table preference, numeric/temporal bonus, regression and debug outputs",
        f"- document_count: {len(bundles)}",
        f"- focus_docs: `{nh.source_name}`, `{doosan.source_name}`",
        "",
    ]

    token_queries = [
        "주가수익률 궁금해",
        "주가 수익률 궁금해",
        "주가수익율 궁금해",
        "PER 궁금해",
        "2025 PER 알려줘",
    ]

    lines.extend(["## 1. Tokenization Compare", ""])
    lines.append("| query | old_tokens | new_tokens |")
    lines.append("| --- | --- | --- |")
    for query in token_queries:
        lines.append(f"| {query} | `{sorted(old_tokens(query))}` | `{sorted(new_tokens(query))}` |")
    lines.append("")

    lines.extend(["## 2. Overlap Compare", ""])
    probe_text = "주가수익률 PER 멀티플과 목표주가 추이"
    lines.append("| query | old_overlap | new_overlap | old_matching | new_matching |")
    lines.append("| --- | --- | --- | --- | --- |")
    for query in token_queries:
        old_matching = sorted(old_tokens(query) & old_tokens(probe_text))
        new_matching = sorted(new_tokens(query) & new_tokens(probe_text))
        lines.append(f"| {query} | {len(old_matching)} | {len(new_matching)} | `{old_matching}` | `{new_matching}` |")
    lines.append("")

    lines.extend(["## 3. Structured Fallback Chunk Search", ""])
    lines.extend(render_top_candidates("before / semantic / 주가 수익률 궁금해", gather_candidates(bundles, query="주가 수익률 궁금해", tokenizer=old_tokens, use_hard_filter=True, chunk_strategy="semantic")))
    lines.extend(render_top_candidates("after / semantic / 주가 수익률 궁금해", gather_candidates(bundles, query="주가 수익률 궁금해", tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")))

    lines.extend(["## 4. Evidence Sentence Extraction", ""])
    before_candidates = gather_candidates(bundles, query="PER 궁금해", tokenizer=old_tokens, use_hard_filter=True, chunk_strategy="semantic")
    after_candidates = gather_candidates(bundles, query="PER 궁금해", tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")
    for label, items, tokenizer, hard_filter in (
        ("before", before_candidates, old_tokens, True),
        ("after", after_candidates, new_tokens, False),
    ):
        lines.append(f"### {label}")
        lines.append("")
        if not items:
            lines.append("no candidates\n")
            continue
        evidence = evidence_sentences("PER 궁금해", items[0], tokenizer, hard_filter)
        lines.append("| rank | score | sentence |")
        lines.append("| --- | --- | --- |")
        for rank, (score, sentence) in enumerate(evidence[:5], start=1):
            preview = re.sub(r"\s+", " ", sentence).strip()
            preview_safe = preview.replace("|", "\\|")
            lines.append(f"| {rank} | {score:.1f} | {preview_safe} |")
        lines.append("")

    lines.extend(["## 5. Chunk Sort Key Reordering", ""])
    lines.extend(render_top_candidates("before / rule / PER 궁금해", gather_candidates(bundles, query="PER 궁금해", tokenizer=old_tokens, use_hard_filter=True, chunk_strategy="rule")))
    lines.extend(render_top_candidates("after / rule / PER 궁금해", gather_candidates(bundles, query="PER 궁금해", tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="rule")))

    lines.extend(["## 6. Sentence Relevance Comparison", ""])
    lines.append("| mode | top_sentence_score | top_sentence |")
    lines.append("| --- | --- | --- |")
    for label, items, tokenizer, hard_filter in (
        ("before", before_candidates, old_tokens, True),
        ("after", after_candidates, new_tokens, False),
    ):
        if not items:
            lines.append(f"| {label} | - | no evidence |")
            continue
        top = evidence_sentences("PER 궁금해", items[0], tokenizer, hard_filter)
        if not top:
            lines.append(f"| {label} | - | no evidence |")
            continue
        top_sentence = re.sub(r"\s+", " ", top[0][1]).replace("|", "\\|")
        lines.append(f"| {label} | {top[0][0]:.1f} | {top_sentence} |")
    lines.append("")

    lines.extend(["## 7. Default Strategy Check", ""])
    chroma_text = (PROJECT_ROOT / "src" / "retrieval" / "chroma_retriever.py").read_text(encoding="utf-8")
    qa_text = (PROJECT_ROOT / "src" / "retreival_lanchain" / "retrieval_qa.py").read_text(encoding="utf-8")
    chroma_defaults = re.findall(r'strategy: str = "[^"]+"', chroma_text)[:3]
    qa_defaults = re.findall(r'strategy: str = "[^"]+"', qa_text)[:3]
    lines.append("| file | default strategy signature found |")
    lines.append("| --- | --- |")
    lines.append(f"| chroma_retriever.py | `{chroma_defaults}` |")
    lines.append(f"| retrieval_qa.py | `{qa_defaults}` |")
    lines.append("")

    lines.extend(["## 8. Query Variants Smoke Test", ""])
    smoke_queries = ["주가수익률 궁금해", "주가 수익률 궁금해", "주가수익율 궁금해", "PER 궁금해"]
    lines.append("| query | before top1 overlap | after top1 overlap | before top1 source | after top1 source |")
    lines.append("| --- | --- | --- | --- | --- |")
    for query in smoke_queries:
        before = gather_candidates(bundles, query=query, tokenizer=old_tokens, use_hard_filter=True, chunk_strategy="semantic")
        after = gather_candidates(bundles, query=query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")
        before_top = before[0] if before else None
        after_top = after[0] if after else None
        lines.append(
            f"| {query} | {before_top.overlap if before_top else '-'} | {after_top.overlap if after_top else '-'} | {before_top.source_name[:24] if before_top else '-'} | {after_top.source_name[:24] if after_top else '-'} |"
        )
    lines.append("")

    lines.extend(["## 9. Table Query Bonus Impact", ""])
    for query in ("PER 알려줘", "PER 표로 알려줘"):
        candidates = gather_candidates(bundles, query=query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="rule")
        lines.extend(render_top_candidates(f"after / rule / {query}", candidates))

    lines.extend(["## 10. Numeric and Temporal Bonus", ""])
    lines.append("| query | top score | top numeric overlap | top temporal overlap | top source |")
    lines.append("| --- | --- | --- | --- | --- |")
    for query in ("PER 알려줘", "2025 PER 알려줘", "2026년 목표주가 알려줘"):
        items = gather_candidates(bundles, query=query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")
        top = items[0] if items else None
        lines.append(f"| {query} | {top.score if top else '-'} | {top.numeric_overlap if top else '-'} | {top.temporal_overlap if top else '-'} | {top.source_name[:24] if top else '-'} |")
    lines.append("")

    lines.extend(["## 11. Fallback Answer Path Preview", ""])
    query = "PER 궁금해"
    items = gather_candidates(bundles, query=query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")
    lines.append(f"- query: `{query}`")
    if items:
        evidence = evidence_sentences(query, items[0], new_tokens, False)
        summary = evidence[0][1] if evidence else items[0].text
        summary = re.sub(r"\s+", " ", summary).strip()
        lines.append(f"- fallback answer preview: `{summary[:220]}`")
        lines.append(f"- source: `{items[0].source_name}` chunk `{items[0].chunk_index}`")
    lines.append("")

    lines.extend(["## 12. Regression Test", ""])
    regression_queries = [
        ("PER 궁금해", doosan.document_id),
        ("개최일자 알려줘", nh.document_id),
    ]
    lines.append("| query | expected doc id fragment | before top doc | after top doc |")
    lines.append("| --- | --- | --- | --- |")
    for query, expected in regression_queries:
        before = gather_candidates(bundles, query=query, tokenizer=old_tokens, use_hard_filter=True, chunk_strategy="rule")
        after = gather_candidates(bundles, query=query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="rule")
        lines.append(f"| {query} | {expected[-8:]} | {(before[0].doc_id[-8:] if before else '-')} | {(after[0].doc_id[-8:] if after else '-')} |")
    lines.append("")

    lines.extend(["## 13. Record-Set and Table Structure Search", ""])
    recordset_md = PROJECT_ROOT / "outputs" / "requested_md_bundle" / "hwp_existing_pipeline_recordset_preview.md"
    if recordset_md.exists():
        md_text = recordset_md.read_text(encoding="utf-8")
        table_lines = [line for line in md_text.splitlines() if line.strip().startswith("|")]
        hits = [line for line in table_lines if "개최일자" in line or "참석인원" in line]
        lines.append(f"- recordset preview file: `{recordset_md.as_posix()}`")
        lines.append(f"- markdown_table_line_count: {len(table_lines)}")
        lines.append(f"- agenda_table_header_hits: {len(hits)}")
    lines.append("")

    lines.extend(["## 14. Document-wise Top-k Distribution", ""])
    dist_query = "PER 궁금해"
    dist_candidates = gather_candidates(bundles, query=dist_query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")[:10]
    distribution: dict[str, int] = {}
    for item in dist_candidates:
        distribution[item.source_name] = distribution.get(item.source_name, 0) + 1
    lines.append(f"- query: `{dist_query}`")
    lines.append("| source | top10_count |")
    lines.append("| --- | --- |")
    for source_name, count in sorted(distribution.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {source_name[:48]} | {count} |")
    lines.append("")

    lines.extend(["## 15. Debug Output", ""])
    debug_query = "주가 수익률 궁금해"
    debug_items = gather_candidates(bundles, query=debug_query, tokenizer=new_tokens, use_hard_filter=False, chunk_strategy="semantic")[:5]
    lines.append(f"- query: `{debug_query}`")
    for rank, item in enumerate(debug_items, start=1):
        debug_preview = re.sub(r"\s+", " ", item.text).strip()[:240]
        lines.extend(
            [
                f"### Candidate {rank}",
                f"- source: `{item.source_name}`",
                f"- chunk_index: `{item.chunk_index}`",
                f"- section_hint: `{item.section_hint[:120]}`",
                f"- overlap: `{item.overlap}`",
                f"- numeric_overlap: `{item.numeric_overlap}`",
                f"- temporal_overlap: `{item.temporal_overlap}`",
                f"- table_bonus: `{item.table_bonus}`",
                f"- matching_tokens: `{item.matching_tokens}`",
                f"- preview: `{debug_preview}`",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full API-free retrieval diagnostics suite.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundles = load_doc_bundles()
    report = render_suite(bundles)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
