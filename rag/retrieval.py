from rag.embeddings import generate_embedding
from rag.vector_store import VectorStore


def retrieve_chunks(
    document_id,
    question,
    n_results=3,
    persist_directory="chroma_db"
):
    question_embedding = generate_embedding(question)

    store = VectorStore(
        persist_directory=persist_directory
    )

    results = store.query(
        query_embedding=question_embedding,
        n_results=n_results,
        document_id=document_id
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        similarity = 1 - distance

        retrieved_chunks.append({
            "page": metadata["page"],
            "text": document,
            "score": similarity
        })

    return retrieved_chunks