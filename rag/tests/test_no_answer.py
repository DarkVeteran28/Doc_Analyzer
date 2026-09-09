import shutil

from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.generation import build_rag_prompt
from rag.retrieval import bm25_directory_for, retrieve_chunks
from rag.sources import build_sources
from rag.vector_store import VectorStore


def test_empty_document_returns_no_chunks():
    chroma_dir = "test_no_answer_empty_chroma"
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_directory_for(chroma_dir), ignore_errors=True)

    results = retrieve_chunks(
        document_id="missing_doc",
        question="What is the capital of Mars?",
        n_results=3,
        persist_directory=chroma_dir,
    )

    assert results == []
    assert build_sources(results) == []


def test_no_answer_prompt_with_empty_context():
    prompt = build_rag_prompt(
        question="What is the capital of Mars?",
        retrieved_chunks=[],
    )

    assert "could not find enough information" in prompt.lower()
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "capital of Mars" in prompt


def test_unrelated_query_still_returns_retrieved_context_without_confidence_gate():
    chroma_dir = "test_no_answer_unrelated_chroma"
    bm25_dir = bm25_directory_for(chroma_dir)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    chunks = [
        {
            "chunk_id": "page_4_chunk_0",
            "page": 4,
            "text": "The experiment measured reaction rate under standard conditions.",
        }
    ]

    embedded = dict(chunks[0])
    embedded["embedding"] = generate_embedding(chunks[0]["text"])

    VectorStore(persist_directory=chroma_dir).add_chunks(
        [embedded],
        document_id="paper_a",
    )
    BM25Index(persist_directory=bm25_dir).add_chunks(
        chunks,
        document_id="paper_a",
    )

    results = retrieve_chunks(
        document_id="paper_a",
        question="xyzzy quantum florp",
        n_results=3,
        persist_directory=chroma_dir,
    )

    assert len(results) == 1
    prompt = build_rag_prompt("xyzzy quantum florp", results)
    assert "could not find enough information" in prompt.lower()
