from rag.fusion import reciprocal_rank_fusion


def _chunk(chunk_id, page, text, document_id="paper_a", score=0.5):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page": page,
        "text": text,
        "score": score,
    }


def test_rrf_overlapping_result_sets():
    vector_results = [
        _chunk("paper_a_7_0", 7, "experimental method measurements", score=0.91),
        _chunk("paper_a_5_0", 5, "experiment temperature reaction", score=0.88),
        _chunk("paper_a_2_0", 2, "history experimental chemistry", score=0.80),
    ]

    bm25_results = [
        _chunk("paper_a_5_0", 5, "experiment temperature reaction", score=4.2),
        _chunk("paper_a_2_0", 2, "history experimental chemistry", score=3.1),
        _chunk("paper_a_7_0", 7, "experimental method measurements", score=2.7),
    ]

    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    chunk_ids = [result["chunk_id"] for result in fused]
    assert chunk_ids == ["paper_a_5_0", "paper_a_7_0", "paper_a_2_0"]
    assert len(chunk_ids) == len(set(chunk_ids))

    top = fused[0]
    assert top["page"] == 5
    assert top["document_id"] == "paper_a"
    assert "experiment" in top["text"]
    assert top["score"] == (1 / 62) + (1 / 61)


def test_rrf_disjoint_result_sets():
    vector_results = [
        _chunk("paper_a_2_0", 2, "history experimental chemistry", score=0.85),
        _chunk("paper_a_5_0", 5, "experiment temperature reaction", score=0.82),
    ]

    bm25_results = [
        _chunk("paper_a_9_0", 9, "sample ID 7781 calibration", score=5.0),
        _chunk("paper_a_7_0", 7, "experimental method measurements", score=3.4),
    ]

    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    chunk_ids = [result["chunk_id"] for result in fused]
    assert chunk_ids == [
        "paper_a_2_0",
        "paper_a_9_0",
        "paper_a_5_0",
        "paper_a_7_0",
    ]

    for result in fused:
        assert result["document_id"] == "paper_a"
        assert "page" in result
        assert "text" in result
        assert result["score"] > 0


def test_rrf_multi_document_result_sets():
    vector_results = [
        _chunk("paper_a_5_0", 5, "paper a experiment", document_id="paper_a"),
        _chunk("paper_b_1_0", 1, "paper b artificial intelligence", document_id="paper_b"),
    ]

    bm25_results = [
        _chunk("paper_b_1_0", 1, "paper b artificial intelligence", document_id="paper_b"),
        _chunk("paper_a_2_0", 2, "paper a chemistry history", document_id="paper_a"),
    ]

    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    chunk_ids = [result["chunk_id"] for result in fused]
    assert chunk_ids == ["paper_b_1_0", "paper_a_5_0", "paper_a_2_0"]

    documents = {result["document_id"] for result in fused}
    assert documents == {"paper_a", "paper_b"}


def test_rrf_is_deterministic_with_ties():
    first_list = [
        _chunk("doc_1_1_0", 1, "alpha", score=0.9),
        _chunk("doc_1_2_0", 2, "beta", score=0.8),
    ]

    second_list = [
        _chunk("doc_1_2_0", 2, "beta", score=0.7),
        _chunk("doc_1_1_0", 1, "alpha", score=0.6),
    ]

    first_run = reciprocal_rank_fusion([first_list, second_list])
    second_run = reciprocal_rank_fusion([first_list, second_list])

    assert first_run == second_run
    assert [result["chunk_id"] for result in first_run] == [
        "doc_1_1_0",
        "doc_1_2_0",
    ]
