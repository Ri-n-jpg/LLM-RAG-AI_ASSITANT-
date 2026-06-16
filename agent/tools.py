from groq import Groq
from sentence_transformers import SentenceTransformer
import os
import numpy as np

from .models import DocumentChunk


# -------------------------
# Load models
# -------------------------

api_key = os.getenv("GROQ_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if not api_key:
    raise Exception("❌ GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)


# -------------------------
# LLM CALL
# -------------------------

def ask_llm(messages):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


# -------------------------
# EMBEDDINGS
# -------------------------

def get_embedding(text):
    return embedding_model.encode(text).tolist()


# -------------------------
# COSINE SIMILARITY
# -------------------------

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


# -------------------------
# RAG SEARCH
# -------------------------

def search_chunks(question, top_k=3):
    question_embedding = get_embedding(question)

    results = []

    chunks = DocumentChunk.objects.all()

    for chunk in chunks:

        if not chunk.embedding:
            continue

        score = cosine_similarity(
            question_embedding,
            chunk.embedding
        )

        results.append((score, chunk.text))

    # IMPORTANT: explicit sorting
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]