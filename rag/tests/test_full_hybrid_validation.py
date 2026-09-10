from .conftest import TEST_ARTIFACTS
from unittest.mock import patch

import shutil

from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.rag_pipeline import ask_question, process_document
from rag.retrieval import bm25_directory_for, retrieve_chunks
from rag.sources import build_sources
from rag.vector_store import VectorStore


def test_bm25_index_is_cached_between_queries():
    build_calls = {"count": 0}
    original_build = BM25Index._build_index

    def counted_build(self, records):
        build_calls["count"] += 1
        return original_build(self, records)

    test_dir = f"{TEST_ARTIFACTS}/test_bm25_cache_dir"
    shutil.rmtree(test_dir, ignore_errors=True)

    index = BM25Index(persist_directory=test_dir)
    chunks = [
        {
            "chunk_id": "page_1_chunk_0",
            "page": 1,
            "text": "Cached BM25 index should not rebuild on every query.",
        }
    ]

    with patch.object(BM25Index, "_build_index", counted_build):
        index.add_chunks(chunks, document_id="doc_a")
        index.query("doc_a", "cached query", n_results=1)
        index.query("doc_a", "second cached query", n_results=1)

    assert build_calls["count"] == 1


def test_full_hybrid_validation_without_llm():
    chroma_dir = f"{TEST_ARTIFACTS}/test_full_validation_chroma"
    bm25_dir = bm25_directory_for(chroma_dir)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    chunks = [
        {
            "chunk_id": "page_1_chunk_0",
            "page": 1,
            "text": "Document A discusses sample ID 9001 for paper A.",
        },
        {
            "chunk_id": "page_2_chunk_0",
            "page": 2,
            "text": "Document A also mentions experiment controls and calibration.",
        },
    ]
    other_chunks = [
        {
            "chunk_id": "page_1_chunk_0",
            "page": 1,
            "text": "Document B discusses unrelated machine learning topics.",
        }
    ]

    for document_id, document_chunks in (
        ("doc_a", chunks),
        ("doc_b", other_chunks),
    ):
        embedded_chunks = []
        for chunk in document_chunks:
            indexed_chunk = dict(chunk)
            indexed_chunk["embedding"] = generate_embedding(chunk["text"])
            embedded_chunks.append(indexed_chunk)

        VectorStore(persist_directory=chroma_dir).add_chunks(
            embedded_chunks,
            document_id=document_id,
        )
        BM25Index(persist_directory=bm25_dir).add_chunks(
            document_chunks,
            document_id=document_id,
        )

    hybrid_results = retrieve_chunks(
        document_id="doc_a",
        question="sample ID 9001 calibration",
        n_results=2,
        persist_directory=chroma_dir,
        retrieval_mode="hybrid",
    )
    sources = build_sources(hybrid_results)

    assert hybrid_results
    assert sources
    assert all(result["page"] in [1, 2] for result in hybrid_results)

    isolated = retrieve_chunks(
        document_id="doc_a",
        question="machine learning",
        n_results=2,
        persist_directory=chroma_dir,
        retrieval_mode="hybrid",
    )

    for result in isolated:
        assert result["page"] in [1, 2]

    with patch("rag.rag_pipeline.generate_answer", return_value="mock answer"):
        response = ask_question(
            document_id="doc_a",
            question="sample ID 9001",
            n_results=2,
            persist_directory=chroma_dir,
        )

    assert response["answer"] == "mock answer"
    assert response["sources"]
    assert process_document.__name__ == "process_document"
