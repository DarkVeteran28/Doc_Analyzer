import json
import shutil

from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.evaluation.metrics import evaluate_retrieval_mode, load_evaluation_dataset
from rag.retrieval import bm25_directory_for, retrieve_chunks
from rag.vector_store import VectorStore


def _index_evaluation_corpus(chroma_dir, dataset):
    bm25_dir = bm25_directory_for(chroma_dir)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)

    document_id = dataset["document_id"]
    embedded_chunks = []

    for chunk in dataset["chunks"]:
        indexed_chunk = dict(chunk)
        indexed_chunk["embedding"] = generate_embedding(chunk["text"])
        embedded_chunks.append(indexed_chunk)

    VectorStore(persist_directory=chroma_dir).add_chunks(
        embedded_chunks,
        document_id=document_id,
    )
    BM25Index(persist_directory=bm25_dir).add_chunks(
        dataset["chunks"],
        document_id=document_id,
    )


def test_evaluation_metrics_on_small_dataset():
    chroma_dir = "test_eval_chroma"
    dataset = load_evaluation_dataset()
    _index_evaluation_corpus(chroma_dir, dataset)

    document_id = dataset["document_id"]
    k = 3

    def make_retriever(mode):
        def retrieve(question, n_results):
            return retrieve_chunks(
                document_id=document_id,
                question=question,
                n_results=n_results,
                persist_directory=chroma_dir,
                retrieval_mode=mode,
            )

        return retrieve

    vector_scores = evaluate_retrieval_mode(
        make_retriever("vector"),
        dataset["queries"],
        k=k,
    )
    bm25_scores = evaluate_retrieval_mode(
        make_retriever("bm25"),
        dataset["queries"],
        k=k,
    )
    hybrid_scores = evaluate_retrieval_mode(
        make_retriever("hybrid"),
        dataset["queries"],
        k=k,
    )

    for scores in (vector_scores, bm25_scores, hybrid_scores):
        assert 0.0 <= scores["recall_at_k"] <= 1.0
        assert 0.0 <= scores["mrr"] <= 1.0

    assert hybrid_scores["recall_at_k"] >= vector_scores["recall_at_k"]
    assert hybrid_scores["mrr"] >= bm25_scores["mrr"]

    report_path = chroma_dir + "_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "k": k,
                "vector": vector_scores,
                "bm25": bm25_scores,
                "hybrid": hybrid_scores,
            },
            handle,
            indent=2,
        )

    print("\nEvaluation report:")
    print(json.dumps(json.load(open(report_path)), indent=2))
