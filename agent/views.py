import json
import uuid
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
        session_id = data.get("session_id")

        # Create new session if missing
        if not session_id:
            session_id = str(uuid.uuid4())

        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        msg_lower = user_message.lower()

        # -------------------------
        # INTENT DETECTION
        # -------------------------
        is_summary = (
            "summarize" in msg_lower or
            "summary" in msg_lower
        )

        is_resume = "resume" in msg_lower

        is_health = (
            "report" in msg_lower or
            "health" in msg_lower
        )

        is_personal = (
            "name" in msg_lower or
            "age" in msg_lower or
            "who am i" in msg_lower
        )

        is_general_chat = (
            "what is" in msg_lower or
            "who is" in msg_lower or
            "define" in msg_lower
        )

        # Save user message
        ChatMessage.objects.create(
            session_id=session_id,
            role="user",
            message=user_message
        )

        # -------------------------
        # RAG RETRIEVAL
        # -------------------------
        context = ""

        skip_rag = is_personal or is_general_chat

        if not skip_rag:
            document = None

            if doc_id:
                document = Document.objects.filter(id=doc_id).first()
            else:
                document = Document.objects.order_by("-id").first()

            if document:
                chunks_qs = DocumentChunk.objects.filter(document=document)

                question_embedding = get_embedding(user_message)

                scored = []

                for chunk in chunks_qs:
                    if not chunk.embedding:
                        continue

                    score = cosine_similarity(
                        question_embedding,
                        chunk.embedding
                    )

                    scored.append((score, chunk.text))

                scored.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                top_chunks = scored[:4]

                context = "\n\n".join(
                    [chunk[1] for chunk in top_chunks]
                )

        # -------------------------
        # SYSTEM PROMPT
        # -------------------------
        if is_personal:
            system_prompt = """
You are a helpful AI assistant.

IMPORTANT:
1. Use conversation history for personal questions.
2. Latest user message overrides old information.
3. Ignore document context.
"""

        elif not context.strip():
            system_prompt = """
You are a helpful AI assistant.
Use conversation history for follow-up questions.
Answer normally.
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

Priority:
1. Use conversation history first
2. Use document context only if relevant

Document Context:
{context}
"""

        # -------------------------
        # CHAT MEMORY
        # -------------------------
        chat_history = ChatMessage.objects.filter(
            session_id=session_id
        ).order_by("-id")[:10]

        chat_history = reversed(chat_history)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        for msg in chat_history:
            messages.append({
                "role": msg.role,
                "content": msg.message
            })

        # Debug
        print(messages)

        # -------------------------
        # LLM CALL
        # -------------------------
        response = ask_llm(messages)

        # Save assistant response
        ChatMessage.objects.create(
            session_id=session_id,
            role="assistant",
            message=response
        )

        return JsonResponse({
            "response": response,
            "session_id": session_id
        })

    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)

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