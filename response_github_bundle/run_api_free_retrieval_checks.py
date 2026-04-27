from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOC_GLOB = "*0a582719.json"
DEFAULT_QUERIES = [
    "주가수익률 궁금해",
    "주가 수익률 궁금해",
    "주가수익율 궁금해",
    "PER 궁금해",
]


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


@dataclass
class Candidate:
    strategy: str
    chunk_index: int
    section_hint: str
    text: str
    token_overlap: int
    matching_tokens: list[str]


def load_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_candidates(
    payload: dict,
    query: str,
    *,
    tokenizer: Callable[[str], set[str]],
    chunk_strategy: str,
    top_k: int,
) -> list[Candidate]:
    query_tokens = tokenizer(query)
    raw_chunks = payload.get("semantic_chunks") if chunk_strategy == "semantic" else payload.get("chunks")
    raw_chunks = raw_chunks or []
    candidates: list[Candidate] = []

    for index, chunk in enumerate(raw_chunks):
        text = str(chunk.get("text") or chunk.get("serialized_text") or "").strip()
        if not text:
            continue
        section_hint = str(chunk.get("section_hint") or chunk.get("section") or "")
        chunk_tokens = tokenizer(text)
        matching = sorted(query_tokens & chunk_tokens)
        candidates.append(
            Candidate(
                strategy=chunk_strategy,
                chunk_index=int(chunk.get("chunk_index", index) or index),
                section_hint=section_hint,
                text=text,
                token_overlap=len(matching),
                matching_tokens=matching,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.token_overlap,
            len(item.matching_tokens),
            len(item.text),
        ),
        reverse=True,
    )
    return candidates[:top_k]


def render_report(
    *,
    document_path: Path,
    payload: dict,
    queries: list[str],
    top_k: int,
) -> str:
    lines: list[str] = [
        f"# API-Free Retrieval Checks",
        "",
        f"- document: `{document_path.as_posix()}`",
        f"- source_name: `{payload.get('source_name')}`",
        f"- semantic_chunk_count: {len(payload.get('semantic_chunks') or [])}",
        f"- rule_chunk_count: {len(payload.get('chunks') or [])}",
        "",
    ]

    for query in queries:
        old_query_tokens = sorted(old_tokens(query))
        new_query_tokens = sorted(new_tokens(query))
        lines.extend(
            [
                f"## Query: {query}",
                "",
                f"- old_tokens: `{old_query_tokens}`",
                f"- new_tokens: `{new_query_tokens}`",
                "",
            ]
        )

        for chunk_strategy in ("semantic", "rule"):
            label = "semantic_chunks" if chunk_strategy == "semantic" else "rule_chunks"
            lines.append(f"### {label}")
            lines.append("")

            before = build_candidates(
                payload,
                query,
                tokenizer=old_tokens,
                chunk_strategy=chunk_strategy,
                top_k=top_k,
            )
            after = build_candidates(
                payload,
                query,
                tokenizer=new_tokens,
                chunk_strategy=chunk_strategy,
                top_k=top_k,
            )

            lines.append("| mode | rank | chunk_index | overlap | matching_tokens | section_hint | preview |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for mode, items in (("before", before), ("after", after)):
                if not items:
                    lines.append(f"| {mode} | - | - | 0 | [] |  | no matches |")
                    continue
                for rank, item in enumerate(items, start=1):
                    preview = item.text.replace("\n", " ")
                    preview = re.sub(r"\s+", " ", preview).strip()
                    preview = preview[:120] + ("..." if len(preview) > 120 else "")
                    section_hint = re.sub(r"\s+", " ", item.section_hint).strip()[:60]
                    lines.append(
                        "| {mode} | {rank} | {chunk_index} | {overlap} | `{matching}` | {section_hint} | {preview} |".format(
                            mode=mode,
                            rank=rank,
                            chunk_index=item.chunk_index,
                            overlap=item.token_overlap,
                            matching=item.matching_tokens,
                            section_hint=section_hint,
                            preview=preview.replace("|", "\\|"),
                        )
                    )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API-free retrieval diagnostics against structured JSON documents.")
    parser.add_argument(
        "--doc-glob",
        default=DEFAULT_DOC_GLOB,
        help="Glob under outputs/ui_runs/*/structured/documents to pick the target JSON.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=None,
        help="Query to test. Can be supplied multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "requested_md_bundle" / "api_free_retrieval_checks.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc_candidates = sorted(PROJECT_ROOT.glob(f"outputs/ui_runs/*/structured/documents/{args.doc_glob}"))
    if not doc_candidates:
        raise FileNotFoundError(f"No document matched glob: {args.doc_glob}")

    document_path = doc_candidates[-1]
    payload = load_document(document_path)
    queries = args.queries or list(DEFAULT_QUERIES)
    report = render_report(document_path=document_path, payload=payload, queries=queries, top_k=args.top_k)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
