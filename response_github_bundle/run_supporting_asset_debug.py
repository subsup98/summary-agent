from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.retrieval.chroma_retriever import ChromaRetriever


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENT_ID = "농협_2022년_9월말_기준_사업보고서--f5c705b1"
DEFAULT_QUERIES = [
    "주요 경영비율 표에서 BIS 비율은 몇 %인가요?",
    "주요사업 추진 실적 표에서 경제사업 성장률은 얼마인가요?",
    "총회(대의원회) 표에서 개최일자와 참석인원을 알려주세요.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug supporting-asset table selection for QA citations.")
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID, help="Target document_id")
    parser.add_argument(
        "--strategy",
        default="semantic",
        choices=("semantic", "rule_based"),
        help="Retrieval strategy used for answer/citation generation",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Question to ask. Can be repeated. Defaults to a small built-in set.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to outputs/manual_parse_check/supporting_asset_debug/<timestamp>.json",
    )
    return parser.parse_args()


def build_output_path(raw_output: str | None) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        return output_path if output_path.is_absolute() else (PROJECT_ROOT / output_path).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        PROJECT_ROOT
        / "outputs"
        / "manual_parse_check"
        / "supporting_asset_debug"
        / f"supporting_asset_debug_{timestamp}.json"
    )


def sanitize_answer_payload(answer: dict[str, Any]) -> dict[str, Any]:
    citations = answer.get("citations") if isinstance(answer.get("citations"), list) else []
    return {
        "query": answer.get("query"),
        "strategy": answer.get("strategy"),
        "answer": answer.get("answer"),
        "warning": answer.get("warning"),
        "citations": citations,
        "match_count": len(answer.get("matches") or []),
    }


def main() -> None:
    args = parse_args()
    queries = args.queries or list(DEFAULT_QUERIES)
    output_path = build_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = ChromaRetriever(project_root=PROJECT_ROOT)
    results: list[dict[str, Any]] = []
    try:
        for query in queries:
            answer = retriever.answer_question(
                query=query,
                strategy=args.strategy,
                document_id=args.document_id,
            )
            debug_citations: list[dict[str, Any]] = []
            for citation in answer.get("citations") or []:
                if not isinstance(citation, dict):
                    continue
                debug_payload = retriever.debug_supporting_asset_candidates(query, citation)
                debug_citations.append(
                    {
                        "citation": citation,
                        "supporting_asset_debug": debug_payload,
                    }
                )
            results.append(
                {
                    "query": query,
                    "answer_payload": sanitize_answer_payload(answer),
                    "citation_debug": debug_citations,
                }
            )
    finally:
        retriever.close()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "document_id": args.document_id,
        "strategy": args.strategy,
        "query_count": len(queries),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
