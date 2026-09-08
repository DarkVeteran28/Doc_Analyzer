import json
import os
import re


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

        return len(records)

    def _load_records(self, document_id):
        path = self._document_path(document_id)

        if not os.path.exists(path):
            return []

        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def query(self, document_id, question, n_results=3):
        """Return ranked chunks for a document-scoped BM25 query.

        Full BM25 scoring is implemented in Checkpoint 2 after rank-bm25
        is added. This method defines the retrieval interface and metadata
        contract used by hybrid fusion.
        """
        records = self._load_records(document_id)

        if not records:
            return []

        raise NotImplementedError(
            "BM25 scoring is added in Checkpoint 2 with rank-bm25."
        )

    def list_documents(self):
        documents = []

        for filename in os.listdir(self.persist_directory):
            if filename.endswith(".json"):
                documents.append(filename[:-5])

        return sorted(documents)
