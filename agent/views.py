import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from .models import ChatMessage, Document, DocumentChunk
from .tools import ask_llm, get_embedding

from pypdf import PdfReader


# =========================
# HOME
# =========================
def home(request):
    return render(request, "index.html")


# =========================
# CHUNKING
# =========================
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


# =========================
# CHAT API (RAG + GENERAL CHAT)
# =========================
@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        doc_id = data.get("doc_id")

        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        ChatMessage.objects.create(role="user", message=user_message)

        # -------------------------
        # GET DOCUMENT (SAFE)
        # -------------------------
        document = None

        if doc_id:
            document = Document.objects.filter(id=doc_id).first()
        else:
            document = Document.objects.order_by("-id").first()

        # -------------------------
        # RETRIEVE CHUNKS
        # -------------------------
        context = ""

        if document:
            chunks_qs = DocumentChunk.objects.filter(document=document)

            # semantic search
            question_embedding = get_embedding(user_message)

            scored = []
            for chunk in chunks_qs:
                score = cosine_similarity(question_embedding, chunk.embedding)
                scored.append((score, chunk.text))

            scored.sort(key=lambda x: x[0], reverse=True)

            top_chunks = scored[:4]

            context = "\n\n".join([t[1] for t in top_chunks])

        # -------------------------
        # INTENT DETECTION
        # -------------------------
        msg_lower = user_message.lower()

        is_summary = "summarize" in msg_lower or "summary" in msg_lower
        is_resume = "resume" in msg_lower
        is_health = "report" in msg_lower or "health" in msg_lower

        # -------------------------
        # PROMPTS
        # -------------------------
        if not context.strip():
            system_prompt = """
You are a helpful AI assistant.
Answer normally without document context.
"""
        elif is_resume:
            system_prompt = f"""
You are a resume expert.

Extract:
- Skills
- Experience
- Projects
- Summary

Context:
{context}
"""
        elif is_health:
            system_prompt = f"""
You are a medical report assistant.

Analyze:
- Key findings
- Abnormal values
- Explanation in simple language

Context:
{context}
"""
        elif is_summary:
            system_prompt = f"""
You are a document summarizer.

Summarize in bullet points.

Context:
{context}
"""
        else:
            system_prompt = f"""
You are a helpful AI assistant.

Use context if relevant, otherwise answer normally.

Context:
{context}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = ask_llm(messages)

        ChatMessage.objects.create(role="assistant", message=response)

        return JsonResponse({"response": response})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# =========================
# PDF UPLOAD API (FIXED)
# =========================
@csrf_exempt
def upload_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    file = request.FILES.get("file")

    if not file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    doc = Document.objects.create(
        title=file.name,
        file=file
    )

    # -------------------------
    # TEXT EXTRACTION
    # -------------------------
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        return JsonResponse({
            "error": "No text found. PDF may be scanned."
        })

    # -------------------------
    # CHUNKING + EMBEDDINGS
    # -------------------------
    chunks = split_text(text)

    for chunk in chunks:
        embedding = get_embedding(chunk)

        DocumentChunk.objects.create(
            document=doc,
            text=chunk,
            embedding=embedding
        )

    return JsonResponse({
        "message": "PDF uploaded successfully",
        "doc_id": doc.id,
        "text_length": len(text),
        "chunks": len(chunks)
    })


# =========================
# COSINE SIMILARITY
# =========================
import numpy as np

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))