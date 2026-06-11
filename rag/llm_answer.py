from transformers import pipeline


MODEL_NAME = "google/flan-t5-base"
_generator = None


def get_generator():
    global _generator

    if _generator is None:
        print(f"Loading answer model: {MODEL_NAME}")
        _generator = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            max_new_tokens=180,
            truncation=True
        )

    return _generator


def compact_text(text, max_chars=650):
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0]


def build_prompt(query, results):
    context_blocks = []

    for i, result in enumerate(results[:2], start=1):
        context_blocks.append(
            f"Source {i}: {result['source']} chunk {result['chunk_id']}\n"
            f"{compact_text(result['text'])}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are an AI assistant for transformer field technicians.

Using only the manual context, answer in 4 short bullet points.
Focus on inspection checks, safety, and maintenance actions.
Do not mention information that is not in the context.

Question:
{query}

Manual context:
{context}

Answer:
""".strip()


def generate_llm_answer(query, results):
    if not results:
        return "I could not find relevant information in the manuals."

    prompt = build_prompt(query, results)
    generator = get_generator()

    output = generator(
        prompt,
        max_new_tokens=180,
        truncation=True
    )[0]["generated_text"]

    return output.strip()
