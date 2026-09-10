import shutil

from rag.embeddings import generate_embedding
from rag.vector_store import VectorStore
from rag.retrieval import retrieve_chunks
from .conftest import TEST_ARTIFACTS


def test_retrieval_filters_document():
    test_db = f"{TEST_ARTIFACTS}/test_retrieval_db"

    shutil.rmtree(test_db, ignore_errors=True)

    chunks = [
        {
            "chunk_id": "page_5_chunk_0",
            "page": 5,
            "text": (
                "The experiment showed that increasing "
                "temperature improved the reaction rate."
            )
        },
        {
            "chunk_id": "page_7_chunk_0",
            "page": 7,
            "text": (
                "The results showed that the experimental "
                "method produced more accurate measurements."
            )
        },
        {
            "chunk_id": "page_2_chunk_0",
            "page": 2,
            "text": (
                "The paper introduces the history of "
                "experimental chemistry."
            )
        }
    ]

    # Generate embeddings
    for chunk in chunks:
        chunk["embedding"] = generate_embedding(chunk["text"])

    store = VectorStore(
        persist_directory=test_db
    )

    # Store these chunks as Paper A
    store.add_chunks(
        chunks,
        document_id="paper_a"
    )

    # Add another document
    other_chunks = [
        {
            "chunk_id": "page_1_chunk_0",
            "page": 1,
            "text": (
                "This other paper discusses experiments "
                "in artificial intelligence."
            )
        }
    ]

    for chunk in other_chunks:
        chunk["embedding"] = generate_embedding(chunk["text"])

    store.add_chunks(
        other_chunks,
        document_id="paper_b"
    )

    # Ask a question about Paper A
    question = "What did the experiment show?"

    results = retrieve_chunks(
        document_id="paper_a",
        question=question,
        n_results=3,
        persist_directory=test_db
    )

    print("\nQuestion:")
    print(question)

    print("\nRetrieved chunks:")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}")

    # We requested 3 results
    assert len(results) == 3

    # Every result must belong to Paper A.
    # We verify this through the expected pages.
    pages = [result["page"] for result in results]

    assert all(page in [2, 5, 7] for page in pages)
