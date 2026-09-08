from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.fusion import reciprocal_rank_fusion
from rag.vector_store import VectorStore


def bm25_directory_for(persist_directory):
    if persist_directory == "chroma_db":
        return "bm25_index"
    return f"{persist_directory}_bm25"


def retrieve_vector_chunks(
    document_id,
    question,
    n_results=3,
    persist_directory="chroma_db",
):
    question_embedding = generate_embedding(question)

    store = VectorStore(
        persist_directory=persist_directory
    )

    results = store.query(
        query_embedding=question_embedding,
        n_results=n_results,
        document_id=document_id,
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        similarity = 1 - distance

        retrieved_chunks.append({
            "chunk_id": metadata["chunk_id"],
            "document_id": metadata["document_id"],
            "page": metadata["page"],
            "text": document,
            "score": similarity,
        })

    return retrieved_chunks


def retrieve_bm25_chunks(
    document_id,
    question,
    n_results=3,
    bm25_directory="bm25_index",
):
    index = BM25Index(persist_directory=bm25_directory)

    return index.query(
        document_id=document_id,
        question=question,
        n_results=n_results,
    )


def retrieve_hybrid_chunks(
    document_id,
    question,
    n_results=3,
    persist_directory="chroma_db",
    bm25_directory=None,
):
    if bm25_directory is None:
        bm25_directory = bm25_directory_for(persist_directory)

    vector_results = retrieve_vector_chunks(
        document_id=document_id,
        question=question,
        n_results=n_results,
        persist_directory=persist_directory,
    )

    bm25_results = retrieve_bm25_chunks(
        document_id=document_id,
        question=question,
        n_results=n_results,
        bm25_directory=bm25_directory,
    )

    ranked_lists = [
        result_list
        for result_list in [vector_results, bm25_results]
        if result_list
    ]

    if not ranked_lists:
        return []

    if len(ranked_lists) == 1:
        fused_results = ranked_lists[0]
    else:
        fused_results = reciprocal_rank_fusion(ranked_lists)

    return [
        {
            "page": chunk["page"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in fused_results[:n_results]
    ]


def _public_chunks(chunks):
    return [
        {
            "page": chunk["page"],
            "text": chunk["text"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]


def retrieve_chunks(
    document_id,
    question,
    n_results=3,
    persist_directory="chroma_db",
    retrieval_mode="hybrid",
):
    if retrieval_mode == "hybrid":
        return retrieve_hybrid_chunks(
            document_id=document_id,
            question=question,
            n_results=n_results,
            persist_directory=persist_directory,
        )

    if retrieval_mode == "vector":
        return _public_chunks(
            retrieve_vector_chunks(
                document_id=document_id,
                question=question,
                n_results=n_results,
                persist_directory=persist_directory,
            )
        )

    if retrieval_mode == "bm25":
        return _public_chunks(
            retrieve_bm25_chunks(
                document_id=document_id,
                question=question,
                n_results=n_results,
                bm25_directory=bm25_directory_for(persist_directory),
            )
        )

    raise ValueError(
        f"Unsupported retrieval_mode: {retrieval_mode!r}. "
        "Expected 'vector', 'bm25', or 'hybrid'."
    )
