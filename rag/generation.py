import os

import ollama

OLLAMA_MODEL = "qwen3:8b"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


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


def _generate_ollama(prompt):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return response["message"]["content"]


def _generate_gemini(prompt):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Gemini provider selected but GEMINI_API_KEY is not configured. "
            "Set the GEMINI_API_KEY environment variable before running."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai package is required for the Gemini provider. "
            "Install it with: pip install google-genai"
        ) from exc

    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL).strip()

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as exc:
        raise RuntimeError(
            f"Gemini API request failed ({type(exc).__name__}): {exc}"
        ) from exc


def generate_answer(question, retrieved_chunks):
    prompt = build_rag_prompt(question, retrieved_chunks)

    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        return _generate_ollama(prompt)

    if provider == "gemini":
        return _generate_gemini(prompt)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        "Supported values are 'ollama' and 'gemini'."
    )
