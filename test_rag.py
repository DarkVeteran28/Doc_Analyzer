from rag.rag_pipeline import (
    process_document,
    ask_question
)


PDF_PATH = "rag/tests/sample.pdf"
DOCUMENT_ID = "sample_document"
DATABASE_PATH = "rag/test_rag_db"


def main():

    print("\n==============================")
    print("       RAG DOCUMENT TEST")
    print("==============================\n")

    # --------------------------------
    # 1. Process document
    # --------------------------------

    print("Loading PDF...\n")

    result = process_document(
        pdf_path=PDF_PATH,
        document_id=DOCUMENT_ID,
        persist_directory=DATABASE_PATH
    )

    print(f"Extracted {result['pages']} pages.")
    print(f"Created {result['chunks']} chunks.")
    print("\nGenerating embeddings...")
    print(f"Stored {result['chunks']} vectors.")

    print("\nDocument ready.")

    # --------------------------------
    # 2. Ask question
    # --------------------------------

    question = input("\nQuestion:\n")

    print("\nRetrieving relevant chunks...")

    response = ask_question(
        document_id=DOCUMENT_ID,
        question=question,
        n_results=3,
        persist_directory=DATABASE_PATH
    )

    # --------------------------------
    # 3. Show sources
    # --------------------------------

    print("\nTop sources:")

    for source in response["sources"]:
        print(f"Page {source['page']}")

    # --------------------------------
    # 4. Show answer
    # --------------------------------

    print("\nGenerating answer...\n")

    print("Answer:")
    print(response["answer"])

    # --------------------------------
    # 5. Done
    # --------------------------------

    print("\n==============================")
    print("        RAG TEST COMPLETE")
    print("==============================\n")


if __name__ == "__main__":
    main()