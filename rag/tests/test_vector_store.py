import shutil

from rag.embeddings import generate_embedding
from rag.vector_store import VectorStore
from conftest import TEST_ARTIFACTS


def test_vector_search():
    test_db = f"{TEST_ARTIFACTS}/test_chroma_db"

    # Remove old test database so every test starts fresh
    shutil.rmtree(test_db, ignore_errors=True)

    chunks = [
    {
        "chunk_id": "page_5_chunk_0",
        "page": 5,
        "text": (
            "The experiment showed that increasing the "
            "temperature significantly improved the reaction rate. "
            "The results demonstrated a clear relationship between "
            "temperature and reaction speed."
        )
    },
    {
        "chunk_id": "page_7_chunk_0",
        "page": 7,
        "text": (
            "The experiment showed that the new experimental "
            "method produced more accurate measurements. "
            "The results confirmed the effectiveness of the method."
        )
    },
    {
        "chunk_id": "page_3_chunk_0",
        "page": 3,
        "text": (
            "Before starting the laboratory work, researchers "
            "cleaned and organized all of the equipment."
        )
    },
    {
        "chunk_id": "page_10_chunk_0",
        "page": 10,
        "text": (
            "The researchers discussed possible improvements "
            "to the procedure that could be investigated in "
            "future studies."
        )
    }
    ]

    # Generate real embeddings
    for chunk in chunks:
        chunk["embedding"] = generate_embedding(chunk["text"])

    store = VectorStore(
        persist_directory=test_db
    )

    store.add_chunks(
        chunks,
        document_id="abc123"
    )

    # Embed the user's question
    question = "What did the experiment show?"
    question_embedding = generate_embedding(question)

    # Search ChromaDB
    results = store.query(
        question_embedding,
        n_results=3
    )

    print("\nQuestion:")
    print(question)

    print("\nSearch Results:")

    for i in range(3):
        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        similarity = 1 - distance

        print(f"\nResult {i + 1}")
        print(f"Page: {metadata['page']}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Text: {document}")

    assert len(results["documents"][0]) == 3

    # The first result should be semantically relevant
    assert results["metadatas"][0][0]["page"] in [5, 7]
