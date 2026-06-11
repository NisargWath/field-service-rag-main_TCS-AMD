from pathlib import Path
import fitz


def extract_text_from_pdf(pdf_path):
    text_parts = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {pdf_path}: {e}")

    return "\n".join(text_parts).strip()


def load_documents(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Document folder does not exist: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        return [], {
            "pdf_files_found": 0,
            "documents_loaded": 0,
            "issues": ["No PDF files found in data/manuals"]
        }

    documents = []
    issues = []

    for pdf_file in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_file)

            if not text.strip():
                issues.append(f"{pdf_file.name}: no extractable text found")
                continue

            documents.append({
                "source": pdf_file.name,
                "path": str(pdf_file),
                "text": text
            })

        except Exception as e:
            issues.append(f"{pdf_file.name}: extraction failed: {e}")

    diagnostics = {
        "pdf_files_found": len(pdf_files),
        "documents_loaded": len(documents),
        "issues": issues
    }

    return documents, diagnostics
