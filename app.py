"""
app.py
Flask API for the Multi-Modal Field Service RAG Assistant.
Uses Qwen 2.5 7B via vLLM for answer generation.
"""

import os
import json
import faiss
import tempfile
from flask import Flask, request, jsonify, send_from_directory

from rag.embedder import generate_embeddings
from rag.retriever import search_index
from rag.answer_generator import generate_extractive_answer
from rag.llm_generator import generate_llm_answer, is_vllm_available
from rag.image_analyzer import analyze_equipment_image
from rag.voice_input import transcribe_audio
from rag.ocr_reader import extract_text_from_image, build_rag_query

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_PATH  = "data/index/manuals.faiss"
CHUNKS_PATH = "data/index/chunks.json"
UPLOAD_DIR  = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static")

# ── Load index once at startup ────────────────────────────────────────────────
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)
print(f"Index loaded: {index.ntotal} vectors")

# ── Check vLLM at startup ─────────────────────────────────────────────────────
vllm_online = is_vllm_available()
print(f"vLLM status: {'ONLINE — using Qwen 2.5 7B' if vllm_online else 'OFFLINE — using extractive fallback'}")


# ── Shared RAG function ───────────────────────────────────────────────────────
def compute_confidence(hits):
    if not hits:
        return {"level": "Low", "top_score": 0.0,
                "message": "No manual matches found. Escalate to a supervisor or subject matter expert."}
    top = hits[0]["score"]
    if top >= 0.75:
        return {"level": "High", "top_score": round(top, 4),
                "message": "High confidence based on retrieved manual matches."}
    elif top >= 0.60:
        return {"level": "Medium", "top_score": round(top, 4),
                "message": "Medium confidence. Verify with the relevant manual section."}
    else:
        return {"level": "Low", "top_score": round(top, 4),
                "message": "Low confidence. Manuals may not contain enough information. Escalate to a supervisor or subject matter expert."}

def rag_answer(query: str) -> dict:
    query_vec = generate_embeddings([query])
    hits      = search_index(query_vec, index, chunks, top_k=5)

    if is_vllm_available():
        answer_text = generate_llm_answer(query, hits)
        mode = "llm"
    else:
        output      = generate_extractive_answer(query, hits)
        answer_text = output["answer"]
        mode = "extractive"

    return {
        "query":      query,
        "answer":     answer_text,
        "mode":       mode,
        "confidence": compute_confidence(hits),
        "sources": [
            {
                "source":   h["source"],
                "chunk_id": h["chunk_id"],
                "score":    h["score"],
                "snippet":  h["text"][:700].strip() if h.get("text") else ""
            }
            for h in hits[:3]
        ]
    }

# ── Frontend ──────────────────────────────────────────────────────────────────
@app.route("/")
def serve_frontend():
    return send_from_directory("static", "index.html")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "vectors": index.ntotal,
        "chunks":  len(chunks),
        "llm":     "Qwen2.5-7B via vLLM" if is_vllm_available() else "extractive fallback"
    })


@app.route("/query", methods=["POST"])
def query_text():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    return jsonify(rag_answer(query))


@app.route("/query/image", methods=["POST"])
def query_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file   = request.files["file"]
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        result    = analyze_equipment_image(tmp_path)
        caption   = result["caption"]
        rag_query = result["rag_query"]
        answer    = rag_answer(rag_query)
        return jsonify({
            "caption": caption,
            "query":   rag_query,
            "answer":  answer["answer"],
            "mode":    answer["mode"],
            "sources": answer["sources"]
        })
    finally:
        os.remove(tmp_path)


@app.route("/query/voice", methods=["POST"])
def query_voice():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file   = request.files["file"]
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        transcript = transcribe_audio(tmp_path)
        if not transcript:
            return jsonify({"error": "Could not transcribe audio"}), 422
        answer = rag_answer(transcript)
        return jsonify({
            "transcript": transcript,
            "query":      transcript,
            "answer":     answer["answer"],
            "mode":       answer["mode"],
            "sources":    answer["sources"]
        })
    finally:
        os.remove(tmp_path)


@app.route("/query/ocr", methods=["POST"])
def query_ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file   = request.files["file"]
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        ocr_text  = extract_text_from_image(tmp_path)
        if not ocr_text:
            return jsonify({"error": "No text found in image"}), 422
        rag_query = build_rag_query(ocr_text)
        answer    = rag_answer(rag_query)
        return jsonify({
            "ocr_text": ocr_text,
            "query":    rag_query,
            "answer":   answer["answer"],
            "mode":     answer["mode"],
            "sources":  answer["sources"]
        })
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
