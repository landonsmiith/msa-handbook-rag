import faiss
import fitz  # PyMuPDF
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - chunk_overlap
    return chunks

def store_embeddings(chunks, model_name="all-MiniLM-L6-v2", index_path="faiss_index"):
    embedder = SentenceTransformer(model_name)
    embeddings = np.array(embedder.encode(chunks))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

# Example usage
text = extract_text_from_pdf("MSA 2025 Handbook.pdf")
chunks = chunk_text(text)
index = store_embeddings(chunks)

with open("chunks.json", "w") as f:
    json.dump(chunks, f)

print(f"Stored {len(chunks)} chunks.")
