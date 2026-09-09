def reciprocal_rank_fusion(ranked_lists, k=60):
    """Fuse multiple ranked retrieval lists with Reciprocal Rank Fusion.

    Each ranked list contains chunk dictionaries with at least ``chunk_id``.
    Metadata fields such as ``page``, ``text``, and ``document_id`` are
    preserved from the highest-ranked occurrence.

    Fusion is deterministic: ties are broken by chunk_id ascending order.
    """
    fused_scores = {}
    chunk_metadata = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (k + rank)
            )

            if chunk_id not in chunk_metadata:
                chunk_metadata[chunk_id] = dict(chunk)

    fused_results = []

    for chunk_id in sorted(
        fused_scores,
        key=lambda item: (-fused_scores[item], item)
    ):
        result = dict(chunk_metadata[chunk_id])
        result["score"] = fused_scores[chunk_id]
        fused_results.append(result)

    return fused_results
