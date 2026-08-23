from rag.generation import build_rag_prompt


def test_build_rag_prompt():
    question = "What did the experiment show?"

    retrieved_chunks = [
        {
            "page": 5,
            "text": (
                "The experiment showed that increasing "
                "temperature improved the reaction rate."
            ),
            "score": 0.82
        },
        {
            "page": 7,
            "text": (
                "The results confirmed the effectiveness "
                "of the experimental method."
            ),
            "score": 0.79
        }
    ]

    prompt = build_rag_prompt(
        question,
        retrieved_chunks
    )

    print("\nGenerated RAG prompt:\n")
    print(prompt)

    assert "ONLY the provided context" in prompt
    assert "What did the experiment show?" in prompt

    assert "Page 5" in prompt
    assert "Page 7" in prompt

    assert "increasing temperature improved" in prompt
    assert "effectiveness of the experimental method" in prompt