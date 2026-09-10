import shutil

from .conftest import TEST_ARTIFACTS
from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.retrieval import bm25_directory_for, retrieve_chunks
from rag.sources import build_sources
from rag.vector_store import VectorStore


def _index_hybrid_corpus(chroma_dir, document_id="paper_a", include_duplicate_page=False):
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

    if include_duplicate_page:
        chunks.append(
            {
                "chunk_id": "page_7_chunk_1",
                "page": 7,
                "text": (
                    "The researchers repeated the experiment on page seven."
                ),
            }
        )

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

    return chunks


def test_hybrid_retrieval_preserves_source_pages():
    chroma_dir = f"{TEST_ARTIFACTS}/test_source_pages_chroma"
    _index_hybrid_corpus(chroma_dir, include_duplicate_page=True)

    retrieved_chunks = retrieve_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
    )

    sources = build_sources(retrieved_chunks)
    expected_pages = []
    seen_pages = set()

    for chunk in retrieved_chunks:
        if chunk["page"] not in seen_pages:
            seen_pages.add(chunk["page"])
            expected_pages.append(chunk["page"])

    assert [source["page"] for source in sources] == expected_pages

    for source in sources:
        assert source["text"]
        first_chunk = next(
            chunk for chunk in retrieved_chunks if chunk["page"] == source["page"]
        )
        assert source["text"] == first_chunk["text"]


def test_hybrid_sources_deduplicate_pages():
    chroma_dir = f"{TEST_ARTIFACTS}/test_source_dedup_chroma"
    _index_hybrid_corpus(chroma_dir, include_duplicate_page=True)

    retrieved_chunks = retrieve_chunks(
        document_id="paper_a",
        question="experiment measurements history",
        n_results=3,
        persist_directory=chroma_dir,
    )

    sources = build_sources(retrieved_chunks)
    pages = [source["page"] for source in sources]

    assert len(pages) == len(set(pages))
    assert all("text" in source for source in sources)


def test_hybrid_sources_use_fused_top_chunk_text():
    chroma_dir = f"{TEST_ARTIFACTS}/test_source_fused_text_chroma"
    _index_hybrid_corpus(chroma_dir)

    retrieved_chunks = retrieve_chunks(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
        persist_directory=chroma_dir,
    )

    sources = build_sources(retrieved_chunks)

    assert retrieved_chunks[0]["page"] == 5
    assert sources[0]["page"] == 5
    assert "experiment" in sources[0]["text"].lower()
