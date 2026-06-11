import faiss
import numpy as np


def create_faiss_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    if embeddings.shape[0] == 0:
        raise ValueError("Cannot create FAISS index with zero embeddings.")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_index(query_embedding, index, chunks, top_k=3):
    query_embedding = np.asarray(query_embedding, dtype="float32")

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"]
        })

    return results
