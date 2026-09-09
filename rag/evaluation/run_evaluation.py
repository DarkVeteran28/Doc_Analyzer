import json
import shutil

from rag.bm25 import BM25Index
from rag.embeddings import generate_embedding
from rag.evaluation.metrics import (
    evaluate_all_modes,
    load_evaluation_dataset,
    save_evaluation_report,
)
from rag.retrieval import bm25_directory_for, retrieve_chunks
from rag.vector_store import VectorStore


def index_evaluation_corpus(chroma_dir, dataset):
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


def build_retrievers(chroma_dir, document_id):
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

    return {
        "vector": make_retriever("vector"),
        "bm25": make_retriever("bm25"),
        "hybrid": make_retriever("hybrid"),
    }


def run_evaluation(chroma_dir="test_eval_expanded_chroma", report_path=None):
    dataset = load_evaluation_dataset()
    index_evaluation_corpus(chroma_dir, dataset)

    if report_path is None:
        report_path = f"{chroma_dir}_report.json"

    report = {
        "k_values": [1, 3, 5],
        "query_count": len(dataset["queries"]),
        "modes": evaluate_all_modes(
            build_retrievers(chroma_dir, dataset["document_id"]),
            dataset["queries"],
            k_values=(1, 3, 5),
        ),
    }

    save_evaluation_report(report, report_path)
    return report
