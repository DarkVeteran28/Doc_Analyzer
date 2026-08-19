from rag.ingestion import extract_text_from_pdf


def test_pdf_ingestion():
    pdf_path = "rag/tests/sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    assert len(pages) > 0

    for page in pages:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:500])

    print(f"\nPDF successfully read")
    print(f"Total pages: {len(pages)}")