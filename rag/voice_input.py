"""
voice_input.py
Transcribes an audio file using Whisper, then feeds the transcript into RAG.
Usage: python3 rag/voice_input.py <audio_file> [--debug]
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

_whisper_model = None

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = load_whisper()
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()
    return transcript

def run_voice_query(audio_path: str, debug: bool = False):

    # ── Step 1: Transcribe ────────────────────────────────────────────────────
    print(f"\n=== Voice Transcription ===")
    transcript = transcribe_audio(audio_path)
    print(f"Audio     : {audio_path}")
    print(f"Transcript: {transcript}")

    if not transcript:
        print("Error: empty transcript, nothing to query.")
        return

    # ── Step 2: Load index ────────────────────────────────────────────────────
    print(f"\n=== Loading Index ===")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Vectors: {index.ntotal} | Chunks: {len(chunks)}")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print(f"\n=== Embedding Query ===")
    query_vec = generate_embeddings([transcript])
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
    print(f"Voice query: {transcript}\n")
    output = generate_extractive_answer(transcript, hits)
    print(output["answer"])

    print(f"\n=== Sources ===")
    for s in output["sources"]:
        print(f"- {s['source']} | chunk {s['chunk_id']} | score {s['score']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rag/voice_input.py <audio_file> [--debug]")
        sys.exit(1)

    audio_path = sys.argv[1]
    debug      = "--debug" in sys.argv

    if not os.path.exists(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)

    run_voice_query(audio_path, debug=debug)
