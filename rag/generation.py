import ollama

MODEL_NAME = "qwen3:8b"


def build_rag_prompt(question, retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are an AI assistant answering questions about a document.

Answer the question using ONLY the provided context.

If the context does not contain enough information to answer,
say that you could not find enough information in the document.

Context:

{context}

Question:

{question}
"""

    return prompt


def generate_answer(question, retrieved_chunks):
    prompt = build_rag_prompt(
        question,
        retrieved_chunks
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
