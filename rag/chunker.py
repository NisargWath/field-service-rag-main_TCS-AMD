import re


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")

    # Remove repeated manual headers/footers and page-number artifacts.
    text = re.sub(r"Transformers:\s*Basics,\s*Maintenance,\s*and\s*Diagnostics", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Fix excessive whitespace while keeping sentence spacing readable.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join PDF-broken lines when they are likely part of the same sentence.
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []

    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue

        if cleaned_lines and cleaned_lines[-1]:
            previous = cleaned_lines[-1]

            if not previous.endswith((".", ":", ";", "?", "!", "•")):
                cleaned_lines[-1] = previous + " " + line
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text, chunk_size=1000, overlap=180):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_documents(documents, chunk_size=1000, overlap=180):
    chunks = []

    for document in documents:
        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(text_chunks):
            chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": chunk
            })

    return chunks
