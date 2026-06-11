from pathlib import Path

from document_loader import load_documents
from chunker import chunk_documents
from embedder import generate_embeddings
from retriever import create_faiss_index, search_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = PROJECT_ROOT / "data" / "manuals"


def main():
    print("\n=== Document Loading ===")
    documents, diagnostics = load_documents(MANUALS_DIR)

    print(f"PDF files found: {diagnostics['pdf_files_found']}")
    print(f"Documents loaded: {diagnostics['documents_loaded']}")

    for issue in diagnostics["issues"]:
        print(f"WARNING: {issue}")

    if not documents:
        print("Pipeline stopped: no usable documents were loaded.")
        return

    print("\n=== Chunking ===")
    chunks = chunk_documents(documents)

    print(f"Chunks generated: {len(chunks)}")

    if not chunks:
        print("Pipeline stopped: no chunks were generated.")
        return

    chunk_texts = [chunk["text"] for chunk in chunks]

    print("\n=== Embedding ===")
    embeddings = generate_embeddings(chunk_texts)

    print(f"Embedding shape: {embeddings.shape}")

    print("\n=== FAISS Index ===")
    index = create_faiss_index(embeddings)

    print(f"FAISS vectors indexed: {index.ntotal}")

    print("\n=== Retrieval ===")
    query = "How should a field technician inspect a transformer?"

    query_embedding = generate_embeddings([query])
    results = search_index(query_embedding, index, chunks, top_k=3)

    print("Retrieval status: success")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"][:700])


if __name__ == "__main__":
    main()
