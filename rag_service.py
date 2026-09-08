from rag.rag_pipeline import (
    process_document as _process_document,
    ask_question as _ask_question
)


def process_document(
    pdf_path,
    document_id,
    persist_directory="chroma_db"
):
    return _process_document(
        pdf_path=pdf_path,
        document_id=document_id,
        persist_directory=persist_directory
    )


def ask_question(
    document_id,
    question,
    n_results=3,
    persist_directory="chroma_db",
    retrieval_mode="hybrid",
):
    return _ask_question(
        document_id=document_id,
        question=question,
        n_results=n_results,
        persist_directory=persist_directory,
        retrieval_mode=retrieval_mode,
    )