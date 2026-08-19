import chromadb


class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="documents",
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    def add_chunks(self, chunks, document_id):
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            chunk_number = chunk["chunk_id"].split("_")[-1]

            chunk_id = (
                f"{document_id}_{chunk['page']}_{chunk_number}"
            )

            ids.append(chunk_id)
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])

            metadatas.append({
                "document_id": document_id,
                "page": chunk["page"],
                "chunk_id": chunk_id
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding, n_results=3):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results

    def count(self):
        return self.collection.count()