import json
import os
import re

from rank_bm25 import BM25Okapi


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _stable_chunk_id(document_id, chunk):
    chunk_number = chunk["chunk_id"].split("_")[-1]
    return f"{document_id}_{chunk['page']}_{chunk_number}"


class BM25Index:
    """Document-scoped BM25 index with persistence separate from ChromaDB."""

    def __init__(self, persist_directory="bm25_index"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self._cache = {}

    def _document_path(self, document_id):
        safe_id = re.sub(r"[^\w\-]", "_", document_id)
        return os.path.join(self.persist_directory, f"{safe_id}.json")

    def add_chunks(self, chunks, document_id):
        records = []

        for chunk in chunks:
            chunk_id = _stable_chunk_id(document_id, chunk)
            records.append({
                "document_id": document_id,
                "chunk_id": chunk_id,
                "page": chunk["page"],
                "text": chunk["text"],
                "tokens": _tokenize(chunk["text"]),
            })

        with open(self._document_path(document_id), "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)

        self._cache.pop(document_id, None)

        return len(records)

    def _load_records(self, document_id):
        path = self._document_path(document_id)

        if not os.path.exists(path):
            return []

        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _build_index(self, records):
        corpus = [record["tokens"] for record in records]
        return BM25Okapi(corpus)

    def _get_cached_index(self, document_id):
        path = self._document_path(document_id)

        if not os.path.exists(path):
            self._cache.pop(document_id, None)
            return [], None

        mtime = os.path.getmtime(path)
        cached = self._cache.get(document_id)

        if cached and cached["mtime"] == mtime:
            return cached["records"], cached["bm25"]

        records = self._load_records(document_id)
        bm25 = self._build_index(records) if records else None
        self._cache[document_id] = {
            "mtime": mtime,
            "records": records,
            "bm25": bm25,
        }

        return records, bm25

    def query(self, document_id, question, n_results=3):
        """Return ranked chunks for a document-scoped BM25 query."""
        records, bm25 = self._get_cached_index(document_id)

        if not records or bm25 is None:
            return []

        query_tokens = _tokenize(question)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(records)),
            key=lambda index: (-scores[index], records[index]["chunk_id"])
        )

        results = []

        for index in ranked_indices[:n_results]:
            record = records[index]
            results.append({
                "chunk_id": record["chunk_id"],
                "document_id": record["document_id"],
                "page": record["page"],
                "text": record["text"],
                "score": float(scores[index]),
            })

        return results

    def list_documents(self):
        documents = []

        for filename in os.listdir(self.persist_directory):
            if filename.endswith(".json"):
                documents.append(filename[:-5])

        return sorted(documents)
