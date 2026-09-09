from pathlib import Path


def test_evaluation_analysis_document_exists():
    analysis_path = Path(__file__).resolve().parents[1] / "evaluation" / "ANALYSIS.md"
    content = analysis_path.read_text(encoding="utf-8")

    assert "Vector" in content
    assert "BM25" in content
    assert "Hybrid" in content
    assert "Limitations" in content
    assert "Does not exceed BM25" in content or "does not exceed BM25" in content
