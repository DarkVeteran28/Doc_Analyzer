import shutil

from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.retrieval import (
    bm25_directory_for,
    retrieve_bm25_chunks,
    retrieve_chunks,
    retrieve_hybrid_chunks,
    retrieve_vector_chunks,
)
from rag.vector_store import VectorStore


def _shared_chunks():
    return [
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


def _index_both(chroma_dir, document_id="paper_a"):
    bm25_dir = bm25_directory_for(chroma_dir)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    chunks = []
    for chunk in _shared_chunks():
        indexed_chunk = dict(chunk)
        indexed_chunk["embedding"] = generate_embedding(chunk["text"])
        chunks.append(indexed_chunk)

    VectorStore(persist_directory=chroma_dir).add_chunks(
        chunks,
        document_id=document_id,
    )

    BM25Index(persist_directory=bm25_dir).add_chunks(
        _shared_chunks(),
        document_id=document_id,
    )

    return bm25_dir


def test_retrieve_chunks_defaults_to_hybrid():
    chroma_dir = "test_hybrid_default_chroma"
    bm25_dir = _index_both(chroma_dir)

    hybrid_results = retrieve_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
    )

    explicit_hybrid = retrieve_hybrid_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
        bm25_directory=bm25_dir,
    )

    assert hybrid_results == explicit_hybrid
    assert len(hybrid_results) == 3
    assert hybrid_results[0]["page"] == 5


def test_hybrid_fuses_vector_and_bm25_rankings():
    chroma_dir = "test_hybrid_fusion_chroma"
    bm25_dir = _index_both(chroma_dir)
    question = "What did the experiment show?"

    vector_results = retrieve_vector_chunks(
        document_id="paper_a",
        question=question,
        n_results=3,
        persist_directory=chroma_dir,
    )
    bm25_results = retrieve_bm25_chunks(
        document_id="paper_a",
        question=question,
        n_results=3,
        bm25_directory=bm25_dir,
    )
    hybrid_results = retrieve_hybrid_chunks(
        document_id="paper_a",
        question=question,
        n_results=3,
        persist_directory=chroma_dir,
        bm25_directory=bm25_dir,
    )

    vector_pages = [result["page"] for result in vector_results]
    bm25_pages = [result["page"] for result in bm25_results]
    hybrid_pages = [result["page"] for result in hybrid_results]

    assert vector_pages != bm25_pages
    assert hybrid_pages == [5, 7, 2]

    for result in hybrid_results:
        assert set(result.keys()) == {"page", "text", "score"}


def test_hybrid_falls_back_to_vector_without_bm25_index():
    chroma_dir = "test_hybrid_vector_only_chroma"
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_directory_for(chroma_dir), ignore_errors=True)

    chunks = []
    for chunk in _shared_chunks():
        indexed_chunk = dict(chunk)
        indexed_chunk["embedding"] = generate_embedding(chunk["text"])
        chunks.append(indexed_chunk)

    VectorStore(persist_directory=chroma_dir).add_chunks(
        chunks,
        document_id="paper_a",
    )

    hybrid_results = retrieve_hybrid_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
    )

    vector_results = retrieve_vector_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
    )

    assert [result["page"] for result in hybrid_results] == [
        result["page"] for result in vector_results
    ]
