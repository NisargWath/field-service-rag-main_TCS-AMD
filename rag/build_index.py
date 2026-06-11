from pathlib import Path
import json

import faiss

from document_loader import load_documents
from chunker import chunk_documents
from embedder import generate_embeddings
from retriever import create_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = PROJECT_ROOT / "data" / "manuals"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

FAISS_INDEX_PATH = INDEX_DIR / "manuals.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Document Loading ===")
    documents, diagnostics = load_documents(MANUALS_DIR)

    print(f"PDF files found: {diagnostics['pdf_files_found']}")
    print(f"Documents loaded: {diagnostics['documents_loaded']}")

    for issue in diagnostics["issues"]:
        print(f"WARNING: {issue}")

    if not documents:
        print("Index build stopped: no usable documents were loaded.")
        return

    print("\n=== Chunking ===")
    chunks = chunk_documents(documents)

    print(f"Chunks generated: {len(chunks)}")

    if not chunks:
        print("Index build stopped: no chunks were generated.")
        return

    chunk_texts = [chunk["text"] for chunk in chunks]

    print("\n=== Embedding ===")
    embeddings = generate_embeddings(chunk_texts)

    print(f"Embedding shape: {embeddings.shape}")

    print("\n=== FAISS Index ===")
    index = create_faiss_index(embeddings)

    print(f"FAISS vectors indexed: {index.ntotal}")

    print("\n=== Saving ===")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    print(f"Saved chunks metadata: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
