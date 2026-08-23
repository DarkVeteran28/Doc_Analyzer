from rag.chunking import chunk_pages


def test_chunking_long_document():
    pages = [
        {
            "page": 1,
            "text": " ".join(["word"] * 8600)
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=600,
        overlap=100
    )

    assert len(chunks) == 18

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "page" in chunk
        assert "text" in chunk

        assert chunk["chunk_id"]
        assert chunk["page"] == 1
        assert chunk["text"]

    print(f"\nInput: 1 document")
    print(f"Output: {len(chunks)} chunks")

    for chunk in chunks:
        print(
            f"{chunk['chunk_id']} | "
            f"Page {chunk['page']} | "
            f"{len(chunk['text'].split())} words"
        )