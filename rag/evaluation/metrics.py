import json
from pathlib import Path


def load_evaluation_dataset(path=None):
    if path is None:
        path = Path(__file__).with_name("dataset.json")

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def recall_at_k(retrieved_pages, relevant_pages, k):
    if not relevant_pages:
        return None

    top_pages = retrieved_pages[:k]
    hits = sum(1 for page in relevant_pages if page in top_pages)
    return hits / len(relevant_pages)


def reciprocal_rank(retrieved_pages, relevant_pages):
    if not relevant_pages:
        return None

    relevant_set = set(relevant_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval_mode(
    retrieve_fn,
    queries,
    k_values=(1, 3, 5),
):
    scored_queries = [
        query for query in queries if query.get("relevant_pages")
    ]
    irrelevant_queries = [
        query for query in queries if not query.get("relevant_pages")
    ]

    max_k = max(k_values)
    recall_totals = {k: 0.0 for k in k_values}
    mrr_total = 0.0

    for query in scored_queries:
        results = retrieve_fn(query["question"], max_k)
        retrieved_pages = [result["page"] for result in results]
        relevant_pages = query["relevant_pages"]

        for k in k_values:
            recall_totals[k] += recall_at_k(
                retrieved_pages,
                relevant_pages,
                k,
            )

        mrr_total += reciprocal_rank(
            retrieved_pages,
            relevant_pages,
        )

    scored_count = len(scored_queries)
    metrics = {
        "scored_query_count": scored_count,
        "irrelevant_query_count": len(irrelevant_queries),
    }

    if scored_count == 0:
        for k in k_values:
            metrics[f"recall_at_{k}"] = 0.0
        metrics["mrr"] = 0.0
        return metrics

    for k in k_values:
        metrics[f"recall_at_{k}"] = recall_totals[k] / scored_count

    metrics["mrr"] = mrr_total / scored_count
    return metrics


def evaluate_all_modes(
    retrieve_by_mode,
    queries,
    k_values=(1, 3, 5),
):
    return {
        mode: evaluate_retrieval_mode(
            retrieve_by_mode[mode],
            queries,
            k_values=k_values,
        )
        for mode in retrieve_by_mode
    }


def save_evaluation_report(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return path
