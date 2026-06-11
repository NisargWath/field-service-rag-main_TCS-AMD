JUNK_PHRASES = [
    "osha archive document",
    "historical content",
    "research and review purposes only",
    "this document is presented here",
    "notice: this is an osha",
]

def is_junk(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in JUNK_PHRASES)

def is_dotline(sentence: str) -> bool:
    # Filter TOC lines like "Chapter 1 ........ 4" or "REVISION HISTORY ....."
    dot_count = sentence.count(".")
    non_dot = len(sentence.replace(".", "").replace(" ", "").strip())
    # If more than 30% of chars are dots, or fewer than 15 real chars — skip it
    return dot_count > 5 or non_dot < 15

def generate_extractive_answer(query, results):
    if not results:
        return {
            "answer": "I could not find relevant information in the manuals.",
            "sources": []
        }

    top_sources = [
        f"{result['source']} chunk {result['chunk_id']}"
        for result in results
    ]

    guidance_points = []
    for result in results:
        if is_junk(result["text"]):
            continue
        text = result["text"].strip()
        sentences = text.replace("\n", " ").split(". ")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            if is_dotline(sentence):
                continue
            if sentence not in guidance_points:
                guidance_points.append(sentence)
            if len(guidance_points) >= 5:
                break
        if len(guidance_points) >= 5:
            break

    if not guidance_points:
        guidance_points = ["Consult the relevant manual section for detailed procedures."]

    answer_lines = []
    answer_lines.append(f"Question: {query}")
    answer_lines.append("")
    answer_lines.append("Field technician answer:")
    for i, point in enumerate(guidance_points, start=1):
        answer_lines.append(f"{i}. {point}.")
    answer_lines.append("")
    answer_lines.append("Sources:")
    for source in top_sources:
        answer_lines.append(f"- {source}")

    return {
        "answer": "\n".join(answer_lines),
        "sources": [
            {
                "source": result["source"],
                "chunk_id": result["chunk_id"],
                "score": result["score"]
            }
            for result in results
        ]
    }
