import json
from pathlib import Path


def load_evaluation_dataset(path=None):
    if path is None:
        path = Path(__file__).with_name("dataset.json")

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def recall_at_k(retrieved_pages, relevant_pages, k):
    if not relevant_pages:
        return 0.0

    top_pages = retrieved_pages[:k]
    hits = sum(1 for page in relevant_pages if page in top_pages)
    return hits / len(relevant_pages)


def reciprocal_rank(retrieved_pages, relevant_pages):
    relevant_set = set(relevant_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval_mode(
    retrieve_fn,
    queries,
    k=3,
):
    recall_total = 0.0
    rr_total = 0.0

    for query in queries:
        results = retrieve_fn(query["question"], k)
        retrieved_pages = [result["page"] for result in results]
        relevant_pages = query["relevant_pages"]

        recall_total += recall_at_k(
            retrieved_pages,
            relevant_pages,
            k,
        )
        rr_total += reciprocal_rank(
            retrieved_pages,
            relevant_pages,
        )

    query_count = len(queries)

    return {
        "recall_at_k": recall_total / query_count,
        "mrr": rr_total / query_count,
    }
