import shutil

from conftest import TEST_ARTIFACTS
from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.retrieval import retrieve_chunks
from rag.vector_store import VectorStore


COMPARISON_CHUNKS = [
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
    {
        "chunk_id": "page_9_chunk_0",
        "page": 9,
        "text": (
            "Table 4 lists sample ID 7781 and control value 0.0031 "
            "for the calibration run."
        ),
    },
]


def _index_both_stores(chroma_dir, bm25_dir, document_id="paper_a"):
    chunks = []

    for chunk in COMPARISON_CHUNKS:
        indexed_chunk = dict(chunk)
        indexed_chunk["embedding"] = generate_embedding(chunk["text"])
        chunks.append(indexed_chunk)

    store = VectorStore(persist_directory=chroma_dir)
    store.add_chunks(chunks, document_id=document_id)

    bm25 = BM25Index(persist_directory=bm25_dir)
    bm25.add_chunks(COMPARISON_CHUNKS, document_id=document_id)

    return chunks


def _compare_retrieval(question, chroma_dir, bm25_dir, document_id="paper_a"):
    vector_results = retrieve_chunks(
        document_id=document_id,
        question=question,
        n_results=3,
        persist_directory=chroma_dir,
    )

    bm25_results = BM25Index(persist_directory=bm25_dir).query(
        document_id=document_id,
        question=question,
        n_results=3,
    )

    vector_pages = [result["page"] for result in vector_results]
    bm25_pages = [result["page"] for result in bm25_results]

    return {
        "question": question,
        "vector_pages": vector_pages,
        "bm25_pages": bm25_pages,
        "vector_top_page": vector_pages[0] if vector_pages else None,
        "bm25_top_page": bm25_pages[0] if bm25_pages else None,
        "same_top_result": (
            vector_pages[:1] == bm25_pages[:1]
            if vector_pages and bm25_pages
            else False
        ),
        "same_ranking": vector_pages == bm25_pages,
        "vector_results": vector_results,
        "bm25_results": bm25_results,
    }


def test_retrieval_comparison_exact_term_query():
    chroma_dir = f"{TEST_ARTIFACTS}/test_compare_chroma_exact"
    bm25_dir = f"{TEST_ARTIFACTS}/test_compare_bm25_exact"
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    _index_both_stores(chroma_dir, bm25_dir)

    comparison = _compare_retrieval(
        question="What did the experiment show?",
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
    )

    assert len(comparison["vector_pages"]) == 3
    assert len(comparison["bm25_pages"]) >= 1
    assert comparison["bm25_top_page"] == 5


def test_retrieval_comparison_numeric_query():
    chroma_dir = f"{TEST_ARTIFACTS}/test_compare_chroma_numeric"
    bm25_dir = f"{TEST_ARTIFACTS}/test_compare_bm25_numeric"
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    _index_both_stores(chroma_dir, bm25_dir)

    comparison = _compare_retrieval(
        question="sample ID 7781 calibration 0.0031",
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
    )

    assert comparison["bm25_top_page"] == 9
    assert len(comparison["vector_pages"]) == 3


def test_retrieval_comparison_records_differences():
    chroma_dir = f"{TEST_ARTIFACTS}/test_compare_chroma_diff"
    bm25_dir = f"{TEST_ARTIFACTS}/test_compare_bm25_diff"
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    _index_both_stores(chroma_dir, bm25_dir)

    queries = [
        "What did the experiment show?",
        "sample ID 7781 calibration 0.0031",
        "history of experimental chemistry",
    ]

    comparisons = [
        _compare_retrieval(
            question=question,
            chroma_dir=chroma_dir,
            bm25_dir=bm25_dir,
        )
        for question in queries
    ]

    for comparison in comparisons:
        assert comparison["vector_pages"]
        assert comparison["bm25_pages"]

    difference_count = sum(
        1 for comparison in comparisons if not comparison["same_ranking"]
    )

    assert difference_count >= 1
