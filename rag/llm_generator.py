"""
llm_generator.py
Uses Qwen 2.5 7B via vLLM OpenAI-compatible API to generate
grounded answers from retrieved RAG chunks.
"""

import requests
import json

VLLM_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """You are an expert field service assistant for energy and utility technicians.
You answer questions using ONLY the provided manual excerpts.
Give clear, numbered, actionable steps a field technician can follow on-site.
If the excerpts do not contain enough information, say so honestly.
Never make up information. Always be safety-focused and precise.
Keep answers concise — maximum 5 steps."""


def generate_llm_answer(query: str, chunks: list) -> str:
    """
    query  : the technician's question
    chunks : list of dicts with keys: text, source, chunk_id, score
    returns: LLM-generated answer string
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} from {chunk['source']}]\n{chunk['text'].strip()}"
        )
    context = "\n\n".join(context_parts)

    user_message = f"""Manual excerpts:
{context}

Technician's question: {query}

Provide a clear, numbered answer based only on the excerpts above."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            VLLM_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "LLM timeout — falling back to manual excerpts.\n" + "\n".join(
            f"{i+1}. {c['text'][:200]}" for i, c in enumerate(chunks)
        )
    except Exception as e:
        return f"LLM error: {e}"


def is_vllm_available() -> bool:
    """Quick health check for vLLM server."""
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except:
        return False
