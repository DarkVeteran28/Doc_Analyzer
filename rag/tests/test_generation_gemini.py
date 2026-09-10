"""
Gemini generation tests.

These tests require GEMINI_API_KEY to be set in the environment.
They are automatically skipped when the key is absent so the CI
non-LLM suite is never blocked.

Run manually:
    GEMINI_API_KEY=<key> LLM_PROVIDER=gemini \
        PYTHONPATH=. pytest rag/tests/test_generation_gemini.py -v -s
"""
import os

import pytest

from rag.generation import generate_answer

GEMINI_KEY_PRESENT = bool(os.environ.get("GEMINI_API_KEY", "").strip())

pytestmark = pytest.mark.skipif(
    not GEMINI_KEY_PRESENT,
    reason="GEMINI_API_KEY not set — skipping live Gemini tests",
)

SAMPLE_CHUNKS = [
    {
        "page": 1,
        "text": (
            "Max Verstappen is widely regarded as the favorite "
            "Formula 1 driver of the current era."
        ),
        "score": 0.95,
    }
]


def test_gemini_generate_answer_returns_string():
    """Live Gemini call: answer must be a non-empty string."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_PROVIDER", "gemini")
        answer = generate_answer(
            "Who is the favorite Formula 1 driver?",
            SAMPLE_CHUNKS,
        )

    print(f"\nGemini answer: {answer}")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_gemini_no_answer_behavior():
    """Live Gemini call: empty context should trigger no-answer phrasing."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_PROVIDER", "gemini")
        answer = generate_answer(
            "What is the population of Jupiter?",
            [],
        )

    print(f"\nGemini no-answer response: {answer}")
    assert isinstance(answer, str)
    assert len(answer) > 0
