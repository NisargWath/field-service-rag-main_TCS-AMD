# Current State

Project: Multi-Modal AI Field Service Assistant for Energy & Utility Field Technicians

Working RAG pipeline:
- PDF loading works
- Chunking works
- BGE embeddings work
- FAISS index saving/loading works
- Query retrieval works
- Extractive technician answer with sources works

Main commands:
python3 rag/build_index.py
python3 rag/query_rag.py "What should I check during transformer inspection?"
python3 rag/query_rag.py "What should I check during transformer inspection?" --debug

Current manual:
data/manuals/transformer_manual.pdf

Current saved index:
data/index/manuals.faiss
data/index/chunks.json

Next step:
Add more manuals, rebuild index, test more technician questions.
