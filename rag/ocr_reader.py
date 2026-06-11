"""
ocr_reader.py
Reads text from equipment nameplates/warning labels using EasyOCR,
then feeds extracted text into the RAG pipeline.
Usage: python3 rag/ocr_reader.py <image_path> [--debug]
"""

import sys
import os
import faiss
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import generate_embeddings
from rag.retriever import search_index
from rag.answer_generator import generate_extractive_answer

INDEX_PATH  = "data/index/manuals.faiss"
CHUNKS_PATH = "data/index/chunks.json"

_ocr_reader = None

def load_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("Loading EasyOCR model (first run, will cache)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader

def extract_text_from_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    reader = load_ocr()
    print(f"Running OCR on: {image_path}")
    results = reader.readtext(image_path)
    # Extract just the text parts, filter low confidence
    texts = [text for (_, text, confidence) in results if confidence > 0.3]
    extracted = " ".join(texts).strip()
    return extracted

def build_rag_query(ocr_text: str) -> str:
    text_lower = ocr_text.lower()
    if any(w in text_lower for w in ["danger", "warning", "high voltage", "kv", "hazard"]):
        return f"Equipment label reads: {ocr_text}. What are the safety procedures and hazards?"
    elif any(w in text_lower for w in ["transformer", "mva", "kva", "voltage", "ampere"]):
        return f"Equipment nameplate reads: {ocr_text}. What inspection and maintenance steps apply?"
    elif any(w in text_lower for w in ["lockout", "tagout", "do not operate"]):
        return f"Equipment label reads: {ocr_text}. What are the lockout tagout procedures?"
    else:
        return f"Field technician reads label: {ocr_text}. What safety procedures apply?"

def run_ocr_query(image_path: str, debug: bool = False):

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    print(f"\n=== OCR Label Reading ===")
    ocr_text = extract_text_from_image(image_path)
    rag_query = build_rag_query(ocr_text)
    print(f"Image    : {image_path}")
    print(f"OCR Text : {ocr_text}")
    print(f"RAG Query: {rag_query}")

    if not ocr_text:
        print("Warning: no text extracted from image.")
        return

    # ── Step 2: Load index ────────────────────────────────────────────────────
    print(f"\n=== Loading Index ===")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Vectors: {index.ntotal} | Chunks: {len(chunks)}")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
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
    print(f"Label reads: {ocr_text}\n")
    output = generate_extractive_answer(rag_query, hits)
    print(output["answer"])

    print(f"\n=== Sources ===")
    for s in output["sources"]:
        print(f"- {s['source']} | chunk {s['chunk_id']} | score {s['score']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rag/ocr_reader.py <image_path> [--debug]")
        sys.exit(1)

    image_path = sys.argv[1]
    debug      = "--debug" in sys.argv

    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    run_ocr_query(image_path, debug=debug)
