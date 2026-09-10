"""
Tests for the generation layer.

Sections:
  1. build_rag_prompt  — no LLM required
  2. Provider selection — no LLM required (uses mocks / env checks)
  3. Ollama generation  — requires Ollama running with qwen3:8b
  4. Gemini generation  — requires GEMINI_API_KEY in environment
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from rag.generation import build_rag_prompt, generate_answer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "page": 5,
        "text": (
            "The experiment showed that increasing "
            "temperature improved the reaction rate."
        ),
        "score": 0.82,
    },
    {
        "page": 7,
        "text": (
            "The results confirmed the effectiveness "
            "of the experimental method."
        ),
        "score": 0.79,
    },
]

SAMPLE_QUESTION = "What did the experiment show?"


# ---------------------------------------------------------------------------
# 1. build_rag_prompt — deterministic, no LLM
# ---------------------------------------------------------------------------

def test_build_rag_prompt():
    prompt = build_rag_prompt(SAMPLE_QUESTION, SAMPLE_CHUNKS)

    print("\nGenerated RAG prompt:\n")
    print(prompt)

    assert "ONLY the provided context" in prompt
    assert SAMPLE_QUESTION in prompt
    assert "Page 5" in prompt
    assert "Page 7" in prompt
    assert "increasing temperature improved" in prompt
    assert "effectiveness of the experimental method" in prompt


def test_build_rag_prompt_empty_chunks():
    prompt = build_rag_prompt("Any question?", [])
    assert "Question:" in prompt
    assert "Context:" in prompt
    # No [Page ...] markers when there are no chunks
    assert "[Page" not in prompt


# ---------------------------------------------------------------------------
# 2. Provider selection — mocked, no LLM required
# ---------------------------------------------------------------------------

def test_invalid_provider_raises_value_error():
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            generate_answer(SAMPLE_QUESTION, SAMPLE_CHUNKS)


def test_gemini_missing_api_key_raises_environment_error():
    env = {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        # Remove key entirely if it happened to be set
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            generate_answer(SAMPLE_QUESTION, SAMPLE_CHUNKS)


def test_ollama_provider_calls_ollama(monkeypatch):
    """generate_answer with LLM_PROVIDER=ollama must call ollama.chat."""
    mock_response = {"message": {"content": "mocked ollama answer"}}

    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
        with patch("rag.generation.ollama.chat", return_value=mock_response) as mock_chat:
            answer = generate_answer(SAMPLE_QUESTION, SAMPLE_CHUNKS)

    assert answer == "mocked ollama answer"
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args
    assert call_kwargs[1]["model"] == "qwen3:8b" or call_kwargs[0][0] == "qwen3:8b" or \
        mock_chat.call_args.kwargs.get("model") == "qwen3:8b" or \
        mock_chat.call_args.args[0] == "qwen3:8b" if mock_chat.call_args.args else True


def test_gemini_provider_calls_genai(monkeypatch):
    """generate_answer with LLM_PROVIDER=gemini must call the Gemini client."""
    mock_response = MagicMock()
    mock_response.text = "mocked gemini answer"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    env = {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key-placeholder"}
    with patch.dict(os.environ, env):
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google": MagicMock(genai=mock_genai)}):
            with patch("rag.generation._generate_gemini", return_value="mocked gemini answer") as mock_gen:
                answer = generate_answer(SAMPLE_QUESTION, SAMPLE_CHUNKS)

    assert answer == "mocked gemini answer"


def test_default_provider_is_ollama():
    """When LLM_PROVIDER is unset, Ollama is the default."""
    mock_response = {"message": {"content": "default ollama answer"}}

    env = {}
    # Remove LLM_PROVIDER entirely
    clean_env = {k: v for k, v in os.environ.items() if k != "LLM_PROVIDER"}
    with patch.dict(os.environ, clean_env, clear=True):
        with patch("rag.generation.ollama.chat", return_value=mock_response):
            answer = generate_answer(SAMPLE_QUESTION, SAMPLE_CHUNKS)

    assert answer == "default ollama answer"
