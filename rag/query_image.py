"""
query_image.py
End-to-end pipeline: image -> BLIP caption -> RAG query -> technician answer
Usage: python3 rag/query_image.py <image_path> [--debug]
"""

import sys
import os
import faiss
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.image_analyzer import analyze_equipment_image
from rag.embedder import generate_embeddings
from rag.retriever import search_index
from rag.answer_generator import generate_extractive_answer

INDEX_PATH  = "data/index/manuals.faiss"
CHUNKS_PATH = "data/index/chunks.json"


def run_image_query(image_path: str, debug: bool = False):

    # ── Step 1: BLIP caption ──────────────────────────────────────────────────
    print(f"\n=== Image Analysis ===")
    result    = analyze_equipment_image(image_path)
    caption   = result["caption"]
    rag_query = result["rag_query"]
    print(f"Image  : {image_path}")
    print(f"Caption: {caption}")
    print(f"Query  : {rag_query}")

    # ── Step 2: Load index ────────────────────────────────────────────────────
    print(f"\n=== Loading Index ===")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Vectors: {index.ntotal} | Chunks: {len(chunks)}")

    # ── Step 3: Embed query ───────────────────────────────────────────────────
    print(f"\n=== Embedding Query ===")
    query_vec = generate_embeddings([rag_query])
    print(f"Shape: {query_vec.shape}")

    # ── Step 4: Retrieve ──────────────────────────────────────────────────────
    print(f"\n=== Retrieval ===")
    hits = search_index(query_vec, index, chunks, top_k=3)
    print(f"Results: {len(hits)}")

    if debug:
        print("\n--- Retrieved Chunks ---")
        for i, h in enumerate(hits):
            print(f"\nChunk {i+1} | {h['source']} | score {h['score']:.4f}")
            print(h["text"])

    # ── Step 5: Answer ────────────────────────────────────────────────────────
    print(f"\n=== Technician Answer ===")
    print(f"Image shows: {caption}\n")
    output = generate_extractive_answer(rag_query, hits)
    print(output["answer"])

    print(f"\n=== Sources ===")
    for s in output["sources"]:
        print(f"- {s['source']} | chunk {s['chunk_id']} | score {s['score']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rag/query_image.py <image_path> [--debug]")
        sys.exit(1)

    image_path = sys.argv[1]
    debug      = "--debug" in sys.argv

    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    run_image_query(image_path, debug=debug)
