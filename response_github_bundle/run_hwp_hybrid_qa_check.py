from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.chroma_retriever import ChromaRetriever


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENT_ROOT = PROJECT_ROOT / "outputs" / "manual_parse_check" / "hwp_hybrid_20260421" / "structured" / "documents"
DOCUMENT_ID = "농협_2022년_9월말_기준_사업보고서--20072d3d"
QUESTIONS = [
    "주요 경영비율 표에서 BIS 비율은 몇 %인가요?",
    "주요 경영비율 표에서 예대비율과 총자본 비율을 알려주세요.",
    "손익계산서 표에서 2022년 9월말 당기순손익은 얼마인가요?",
    "재무상태표에서 자산 총계와 부채와 자본 총계는 얼마인가요?",
    "주요사업 추진 실적 표에서 경제사업의 22년 9월말 실적과 달성률은 얼마인가요?",
]


class ManualCheckRetriever(ChromaRetriever):
    def _structured_document_roots(self) -> list[Path]:
        roots = super()._structured_document_roots()
        if DOCUMENT_ROOT.exists():
            roots.insert(0, DOCUMENT_ROOT)
        return roots


def main() -> None:
    retriever = ManualCheckRetriever(project_root=PROJECT_ROOT)
    results: list[dict[str, object]] = []
    try:
        for query in QUESTIONS:
            answer = retriever.answer_question(
                query=query,
                strategy="rule_based",
                top_k=5,
                document_id=DOCUMENT_ID,
            )
            results.append(
                {
                    "query": query,
                    "answer": answer.get("answer"),
                    "match_count": len(answer.get("matches", [])),
                    "citations": answer.get("citations", []),
                    "evidence": answer.get("evidence", []),
                    "warning": answer.get("warning"),
                    "openai_enabled": answer.get("openai_enabled"),
                }
            )
    finally:
        retriever.close()

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
