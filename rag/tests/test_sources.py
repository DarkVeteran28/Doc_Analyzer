from rag.sources import build_sources


def test_build_sources_removes_duplicate_pages():

    retrieved_chunks = [
        {
            "page": 4,
            "text": "The experiment increased reaction speed.",
            "score": 0.91
        },
        {
            "page": 7,
            "text": "The results confirmed the hypothesis.",
            "score": 0.86
        },
        {
            "page": 7,
            "text": "The researchers repeated the experiment.",
            "score": 0.72
        }
    ]

    sources = build_sources(retrieved_chunks)

    print("\nSources:")

    for source in sources:
        print(f"Page {source['page']}: {source['text']}")

    assert len(sources) == 2

    assert sources[0]["page"] == 4
    assert sources[1]["page"] == 7

    assert "text" in sources[0]
    assert "text" in sources[1]