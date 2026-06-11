from pathlib import Path
import json
import sys

import faiss

from embedder import generate_embeddings
from retriever import search_index
from answer_generator import generate_extractive_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"

FAISS_INDEX_PATH = INDEX_DIR / "manuals.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks metadata not found: {CHUNKS_PATH}. Run build_index.py first."
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}. Run build_index.py first."
        )

    return faiss.read_index(str(FAISS_INDEX_PATH))


def parse_args():
    args = sys.argv[1:]
    debug = False

    if "--debug" in args:
        debug = True
        args.remove("--debug")

    query = " ".join(args).strip()

    return query, debug


def main():
    query, debug = parse_args()

    if not query:
        query = input("Ask a transformer maintenance question: ").strip()

    if not query:
        print("No query provided.")
        return

    print("\n=== Loading Saved Index ===")
    index = load_index()
    chunks = load_chunks()

    print(f"FAISS vectors loaded: {index.ntotal}")
    print(f"Chunks loaded: {len(chunks)}")

    print("\n=== Query Embedding ===")
    query_embedding = generate_embeddings([query])

    print(f"Query embedding shape: {query_embedding.shape}")

    print("\n=== Retrieval ===")
    results = search_index(query_embedding, index, chunks, top_k=3)
    response = generate_extractive_answer(query, results)

    print("Retrieval status: success")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}")

    print("\n=== Technician Answer ===")
    print(response["answer"])

    print("\n=== Sources ===")
    for source in response["sources"]:
        print(
            f"- {source['source']} | chunk {source['chunk_id']} | score {source['score']:.4f}"
        )

    if debug:
        print("\n=== Retrieved Chunks ===")
        for i, result in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result['source']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Score: {result['score']:.4f}")
            print(result["text"][:900])


if __name__ == "__main__":
    main()
