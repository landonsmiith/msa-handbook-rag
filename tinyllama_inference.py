from openai import OpenAI
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

index = faiss.read_index("faiss_index")
with open("chunks.json", "r") as f:
    chunks = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    _, indices = index.search(query_embedding, k)
    return "\n\n".join([chunks[i] for i in indices[0] if i < len(chunks)])

def generate_response(query):
    context = retrieve_context(query)
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context from the MSA 2025 Handbook."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
