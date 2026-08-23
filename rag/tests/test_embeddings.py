from rag.embeddings import generate_embedding
import numpy as np


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_embeddings():
    sentence_1 = "Machine learning is a branch of artificial intelligence."
    sentence_2 = "Machine learning is a branch of AI."
    sentence_3 = "Football players train every day."

    embedding_1 = generate_embedding(sentence_1)
    embedding_2 = generate_embedding(sentence_2)
    embedding_3 = generate_embedding(sentence_3)

    print(f"\nEmbedding dimensions: {len(embedding_1)}")

    similarity_related = cosine_similarity(
        embedding_1,
        embedding_2
    )

    similarity_unrelated = cosine_similarity(
        embedding_1,
        embedding_3
    )

    print(f"Related sentence similarity: {similarity_related:.4f}")
    print(f"Unrelated sentence similarity: {similarity_unrelated:.4f}")

    assert len(embedding_1) == 384
    assert len(embedding_2) == 384
    assert len(embedding_3) == 384

    assert similarity_related > similarity_unrelated