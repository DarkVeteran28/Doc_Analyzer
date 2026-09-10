import json
import os

from rag.evaluation.metrics import load_evaluation_dataset
from rag.evaluation.run_evaluation import run_evaluation
from .conftest import TEST_ARTIFACTS


def test_expanded_evaluation_dataset_has_expected_queries():
    dataset = load_evaluation_dataset()

    assert len(dataset["chunks"]) == 10
    assert len(dataset["queries"]) == 25

    query_types = {query["query_type"] for query in dataset["queries"]}
    assert query_types == {
        "exact_keyword",
        "semantic",
        "mixed",
        "multi_page",
        "irrelevant",
    }


def test_expanded_evaluation_metrics():
    chroma_dir = f"{TEST_ARTIFACTS}/test_eval_expanded_chroma"
    report_path = f"{TEST_ARTIFACTS}/test_eval_expanded_chroma_report.json"
    os.makedirs(TEST_ARTIFACTS, exist_ok=True)
    report = run_evaluation(chroma_dir=chroma_dir, report_path=report_path)

    assert report["query_count"] == 25

    for mode in ("vector", "bm25", "hybrid"):
        scores = report["modes"][mode]
        assert scores["scored_query_count"] == 21
        assert scores["irrelevant_query_count"] == 4

        for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr"):
            assert 0.0 <= scores[metric] <= 1.0

    with open(report_path, encoding="utf-8") as handle:
        saved_report = json.load(handle)

    assert saved_report == report

    print("\nExpanded evaluation report:")
    print(json.dumps(report, indent=2))
