import shutil

from conftest import TEST_ARTIFACTS
from rag.bm25 import BM25Index
from rag.chunking import chunk_pages


def _sample_chunks():
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


def test_bm25_exact_term_match():
    test_dir = f"{TEST_ARTIFACTS}/test_bm25_exact"
    shutil.rmtree(test_dir, ignore_errors=True)

    index = BM25Index(persist_directory=test_dir)
    chunks = _sample_chunks()
    index.add_chunks(chunks, document_id="paper_a")

    results = index.query(
        document_id="paper_a",
        question="What did the experiment show?",
        n_results=3,
    )

    assert len(results) >= 1
    assert results[0]["page"] == 5
    assert "experiment" in results[0]["text"].lower()


def test_bm25_number_match():
    test_dir = f"{TEST_ARTIFACTS}/test_bm25_numbers"
    shutil.rmtree(test_dir, ignore_errors=True)

    index = BM25Index(persist_directory=test_dir)
    chunks = _sample_chunks()
    index.add_chunks(chunks, document_id="paper_a")

    results = index.query(
        document_id="paper_a",
        question="42 percent reaction rate",
        n_results=3,
    )

    assert len(results) >= 1
    assert results[0]["page"] == 5
    assert "42" in results[0]["text"]


def test_bm25_metadata_preservation():
    test_dir = f"{TEST_ARTIFACTS}/test_bm25_metadata"
    shutil.rmtree(test_dir, ignore_errors=True)

    index = BM25Index(persist_directory=test_dir)
    chunks = _sample_chunks()
    index.add_chunks(chunks, document_id="paper_a")

    results = index.query(
        document_id="paper_a",
        question="experimental chemistry history",
        n_results=1,
    )

    assert len(results) == 1

    result = results[0]
    assert result["document_id"] == "paper_a"
    assert result["chunk_id"] == "paper_a_2_0"
    assert result["page"] == 2
    assert result["text"] == chunks[2]["text"]
    assert isinstance(result["score"], float)
    assert result["score"] > 0


def test_bm25_document_isolation():
    test_dir = f"{TEST_ARTIFACTS}/test_bm25_isolation"
    shutil.rmtree(test_dir, ignore_errors=True)

    index = BM25Index(persist_directory=test_dir)

    paper_a_chunks = _sample_chunks()
    index.add_chunks(paper_a_chunks, document_id="paper_a")

    paper_b_chunks = [
        {
            "chunk_id": "page_1_chunk_0",
            "page": 1,
            "text": (
                "This other paper discusses experiments "
                "in artificial intelligence."
            ),
        }
    ]
    index.add_chunks(paper_b_chunks, document_id="paper_b")

    results = index.query(
        document_id="paper_a",
        question="experiment artificial intelligence",
        n_results=3,
    )

    assert len(results) >= 1

    for result in results:
        assert result["document_id"] == "paper_a"
        assert result["chunk_id"].startswith("paper_a_")


def test_bm25_indexes_chunk_pages_output():
    test_dir = f"{TEST_ARTIFACTS}/test_bm25_chunk_pages"
    shutil.rmtree(test_dir, ignore_errors=True)

    pages = [
        {
            "page": 3,
            "text": (
                "Section 3.1 describes catalyst loading of 15 mg "
                "and baseline pressure 101325 pascals."
            ),
        }
    ]

    chunks = chunk_pages(pages, chunk_size=600, overlap=100)
    index = BM25Index(persist_directory=test_dir)
    indexed_count = index.add_chunks(chunks, document_id="report_x")

    assert indexed_count == len(chunks)

    results = index.query(
        document_id="report_x",
        question="101325 pascals catalyst",
        n_results=1,
    )

    assert len(results) == 1
    assert results[0]["page"] == 3
    assert "101325" in results[0]["text"]
