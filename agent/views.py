import json
import uuid
import numpy as np

from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect

from pypdf import PdfReader

from .models import ChatMessage, Document, DocumentChunk, ChatSession
from .tools import ask_llm, get_embedding


# =========================
# AUTH PAGES
# =========================

def signup_page(request):
    if request.user.is_authenticated:
        return redirect("/api/")
    return render(request, "signup.html")


def login_page(request):
    if request.user.is_authenticated:
        return redirect("/api/")
    return render(request, "login.html")


def home(request):
    if request.user.is_authenticated:
        return render(request, "index.html")
    return redirect("/api/login-page/")


# =========================
# SIGNUP API
# =========================

@csrf_exempt
def signup_user(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"})

    data = json.loads(request.body)

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if User.objects.filter(username=username).exists():
        return JsonResponse({"error": "Username exists"})

    User.objects.create_user(
        username=username,
        email=email,
        password=password
    )

    return JsonResponse({
        "message": "Signup successful"
    })


# =========================
# LOGIN API (FIXED)
# =========================

@csrf_exempt
def login_user(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"})

    data = json.loads(request.body)

    username = data.get("username")
    password = data.get("password")

    user = authenticate(username=username, password=password)

    if user is None:
        return JsonResponse({"error": "Invalid credentials"})

    # 🔥 IMPORTANT FIX (THIS WAS MISSING)
    login(request, user)

    return JsonResponse({
        "message": "Login successful"
    })


# =========================
# LOGOUT
# =========================

def logout_user(request):
    logout(request)
    return JsonResponse({
        "message": "Logged out"
    })


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
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    try:
        data = json.loads(request.body)

        user_message = data.get("message", "").strip()
        doc_id = data.get("doc_id")
        session_id = data.get("session_id")

        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        # =========================
        # SESSION HANDLING
        # =========================
        session = None

        if session_id:
            session = ChatSession.objects.filter(
                session_id=session_id,
                user=request.user
            ).first()
        if not session:
            session = ChatSession.objects.create(
                user=request.user,  # 🔥 REQUIRED FIX
                session_id=str(uuid.uuid4()),
                title=user_message[:30]
            )

        # Save user message
        ChatMessage.objects.create(
            session=session,
            role="user",
            message=user_message
        )

        msg_lower = user_message.lower()

        # =========================
        # INTENT DETECTION
        # =========================
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

        general_keywords = [
            "what is",
            "who is",
            "where is",
            "when",
            "pm",
            "president",
            "capital",
            "explain"
        ]

        is_general_chat = any(
            keyword in msg_lower
            for keyword in general_keywords
        )

        # =========================
        # RAG RETRIEVAL
        # =========================
        context = ""
        source_info = None

        is_greeting = msg_lower in [
            "hi", "hii", "hello", "hey",
            "whats going on", "what's going on"
        ]

        # RAG only when explicitly needed
        force_rag = (
                is_summary or
                is_resume or
                is_health or
                (doc_id is not None)
        )
        print("FORCE RAG:", force_rag)

        if force_rag:
            document = None

            if doc_id:
                document = Document.objects.filter(id=doc_id).first()
            else:
                document = Document.objects.order_by("-id").first()

            if document:
                chunks_qs = DocumentChunk.objects.filter(
                    document=document
                )

                question_embedding = get_embedding(user_message)
                scored = []

                for chunk in chunks_qs:
                    if not chunk.embedding:
                        continue

                    score = cosine_similarity(
                        question_embedding,
                        chunk.embedding
                    )

                    scored.append((
                        score,
                        chunk.text,
                        chunk.id,
                        chunk.document.title
                    ))

                scored.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                top_chunks = scored[:4]
                SIMILARITY_THRESHOLD = 0.25

                if not top_chunks or top_chunks[0][0] < SIMILARITY_THRESHOLD:
                    context = ""
                    source_info = None
                else:
                    context = "\n\n".join([chunk[1] for chunk in top_chunks])

                    best_chunk = top_chunks[0]

                    source_info = {
                        "score": round(best_chunk[0], 3),
                        "chunk_id": best_chunk[2],
                        "document": best_chunk[3]
                    }


        # =========================
        # SYSTEM PROMPT
        # =========================
        if is_personal:
            system_prompt = """
You are a helpful AI assistant.

IMPORTANT:
1. Use conversation history for personal questions
2. Latest user message overrides old info
3. Ignore document context
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

        # =========================
        # MEMORY
        # =========================
        chat_history = ChatMessage.objects.filter(
            session=session
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

        # =========================
        # LLM CALL
        # =========================
        response = ask_llm(messages)

        # Save assistant reply
        ChatMessage.objects.create(
            session=session,
            role="assistant",
            message=response
        )
        print("DOC ID RECEIVED:", doc_id)

        return JsonResponse({
            "response": response,
            "session_id": session.session_id,
            "source": source_info
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

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_sessions(request):
    if not request.user.is_authenticated:
        return JsonResponse({"sessions": []})

    sessions = ChatSession.objects.filter(
        user=request.user
    ).order_by("-created_at")

    data = [
        {
            "session_id": s.session_id,
            "title": s.title
        }
        for s in sessions
    ]

    return JsonResponse({"sessions": data})

def get_messages(request, session_id):
    session = ChatSession.objects.filter(
        session_id=session_id,
        user=request.user
    ).first()

    if not session:
        return JsonResponse({"messages": []})

    messages = ChatMessage.objects.filter(session=session)

    data = [
        {
            "role": msg.role,
            "message": msg.message
        }
        for msg in messages
    ]

    return JsonResponse({"messages": data})

@csrf_exempt
def delete_session(request, session_id):
    if request.method != "DELETE":
        return JsonResponse({"error": "Only DELETE allowed"}, status=405)

    session = ChatSession.objects.filter(
        session_id=session_id,
        user=request.user
    ).first()

    if not session:
        return JsonResponse({"error": "Session not found"}, status=404)

    session.delete()

    return JsonResponse({"message": "Session deleted"})
