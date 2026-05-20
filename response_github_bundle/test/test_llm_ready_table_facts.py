from __future__ import annotations

from src.indexing.llm_ready import build_llm_ready_artifacts


def test_llm_ready_table_relation_keeps_trailing_title_unit_and_period_facts() -> None:
    payload = {
        "markdown": "\n".join(
            [
                "# Page 5",
                "",
                "2025 년 3 분기 실적보고서",
                "",
                "2025 년 3 분기 재무실적 요약",
                "",
                "| | 3Q24 | 3Q25 |",
                "| --- | --- | --- |",
                "| 순영업수익 | 572.8 | 594.9 |",
                "| 영업이익 | 289.6 | 205.0 |",
                "",
                "요약 손익계산서",
                "",
                "(단위: 십억 원)",
            ]
        ),
        "page_summaries": [],
    }

    artifacts = build_llm_ready_artifacts(payload, [])
    relation_chunks = [
        chunk
        for chunk in artifacts["llm_ready_chunks"]
        if chunk.get("chunk_type") == "structured_relation"
    ]

    assert len(relation_chunks) == 1
    relation_text = relation_chunks[0]["text"]
    assert relation_chunks[0]["section_hint"] == "요약 손익계산서"
    assert "Table title: 요약 손익계산서" in relation_text
    assert "Unit: 단위: 십억 원" in relation_text
    assert "요약 손익계산서 / 순영업수익 / 3Q24 = 572.8 (단위: 십억 원)" in relation_text
    assert "요약 손익계산서 / 영업이익 / 3Q25 = 205.0 (단위: 십억 원)" in relation_text
