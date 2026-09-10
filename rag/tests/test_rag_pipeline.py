import shutil

from rag.rag_pipeline import (
    process_document,
    ask_question
)
from conftest import TEST_ARTIFACTS


def test_full_rag_pipeline():

    test_db = f"{TEST_ARTIFACTS}/test_pipeline_db"

    shutil.rmtree(
        test_db,
        ignore_errors=True
    )

    # --------------------------------
    # STEP 1: Process the PDF
    # --------------------------------

    result = process_document(
        pdf_path="rag/tests/sample.pdf",
        document_id="sample_document",
        persist_directory=test_db
    )

    print("\nDocument processing:")
    print(result)

    assert result["document_id"] == "sample_document"
    assert result["pages"] > 0
    assert result["chunks"] > 0

    # --------------------------------
    # STEP 2: Ask a question
    # --------------------------------

    question = "Who is the favorite Formula 1 driver?"

    response = ask_question(
        document_id="sample_document",
        question=question,
        n_results=3,
        persist_directory=test_db
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")

    for source in response["sources"]:
        print(
            f"Page {source['page']}: "
            f"{source['text']}"
        )

    # --------------------------------
    # STEP 3: Validate response
    # --------------------------------

    assert "answer" in response
    assert "sources" in response

    assert isinstance(
        response["answer"],
        str
    )

    assert len(response["answer"]) > 0

    assert isinstance(
        response["sources"],
        list
    )

    assert len(response["sources"]) > 0
