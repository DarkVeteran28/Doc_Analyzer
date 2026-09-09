from rag.ingestion import extract_text_from_pdf
from rag.chunking import chunk_pages
from rag.embeddings import generate_embedding
from rag.vector_store import VectorStore
from rag.bm25 import BM25Index
from rag.retrieval import retrieve_chunks, bm25_directory_for
from rag.generation import generate_answer
from rag.sources import build_sources

def process_document(
    pdf_path,
    document_id,
    persist_directory="chroma_db"
):
    # Step 1: Extract pages from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Split pages into chunks
    chunks = chunk_pages(
        pages,
        chunk_size=600,
        overlap=100
    )

    # Step 3: Generate an embedding for every chunk
    for chunk in chunks:
        chunk["embedding"] = generate_embedding(
            chunk["text"]
        )

    # Step 4: Store chunks and embeddings in ChromaDB
    store = VectorStore(
        persist_directory=persist_directory
    )

    store.add_chunks(
        chunks,
        document_id=document_id
    )

    # Step 5: Index the same chunks for BM25 retrieval
    bm25_index = BM25Index(
        persist_directory=bm25_directory_for(persist_directory)
    )

    bm25_index.add_chunks(
        chunks,
        document_id=document_id
    )

    return {
        "document_id": document_id,
        "pages": len(pages),
        "chunks": len(chunks)
    }
def ask_question(
    document_id,
    question,
    n_results=3,
    persist_directory="chroma_db",
    retrieval_mode="hybrid",
):
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        document_id=document_id,
        question=question,
        n_results=n_results,
        persist_directory=persist_directory,
        retrieval_mode=retrieval_mode,
    )

    # Step 2: Generate answer using retrieved context
    answer = generate_answer(
        question,
        retrieved_chunks
    )

    # Step 3: Build source references
    sources = build_sources(
        retrieved_chunks
    )

    return {
        "answer": answer,
        "sources": sources
    }