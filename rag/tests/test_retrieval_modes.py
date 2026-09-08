import shutil

from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.retrieval import bm25_directory_for, retrieve_chunks
from rag.vector_store import VectorStore
import pytest


def _index_both(chroma_dir, document_id="paper_a"):
    bm25_dir = bm25_directory_for(chroma_dir)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    chunks = [
        {
            "chunk_id": "page_5_chunk_0",
            "page": 5,
            "text": (
                "The experiment showed that increasing temperature "
                "improved the reaction rate by 42 percent."
            ),
        },
        {
            "chunk_id": "page_7_chunk_0",
            "page": 7,
            "text": (
                "The results showed that the experimental method "
                "produced more accurate measurements."
            ),
        },
        {
            "chunk_id": "page_2_chunk_0",
            "page": 2,
            "text": (
                "The paper introduces the history of "
                "experimental chemistry."
            ),
        },
    ]

    embedded_chunks = []
    for chunk in chunks:
        indexed_chunk = dict(chunk)
        indexed_chunk["embedding"] = generate_embedding(chunk["text"])
        embedded_chunks.append(indexed_chunk)

    VectorStore(persist_directory=chroma_dir).add_chunks(
        embedded_chunks,
        document_id=document_id,
    )
    BM25Index(persist_directory=bm25_dir).add_chunks(
        chunks,
        document_id=document_id,
    )


def test_retrieval_mode_vector():
    chroma_dir = "test_mode_vector_chroma"
    _index_both(chroma_dir)

    results = retrieve_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
        retrieval_mode="vector",
    )

    assert [result["page"] for result in results] == [7, 2, 5]


def test_retrieval_mode_bm25():
    chroma_dir = "test_mode_bm25_chroma"
    _index_both(chroma_dir)

    results = retrieve_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
        retrieval_mode="bm25",
    )

    assert [result["page"] for result in results] == [5, 2, 7]


def test_retrieval_mode_hybrid():
    chroma_dir = "test_mode_hybrid_chroma"
    _index_both(chroma_dir)

    results = retrieve_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
        retrieval_mode="hybrid",
    )

    assert [result["page"] for result in results] == [5, 7, 2]


def test_retrieval_mode_invalid():
    with pytest.raises(ValueError, match="Unsupported retrieval_mode"):
        retrieve_chunks(
            document_id="paper_a",
            question="test",
            retrieval_mode="semantic",
        )
