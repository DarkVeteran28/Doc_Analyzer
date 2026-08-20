from rag.generation import generate_answer


def test_generate_answer():

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

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("\nLLM Answer:")
    print(answer)

    assert isinstance(answer, str)
    assert len(answer) > 0