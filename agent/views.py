import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import ChatMessage, Document
from .tools import ask_llm

from pypdf import PdfReader


# =========================
# UTILITY: TEXT CHUNKING
# =========================
def split_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# =========================
# CHAT API (MEMORY BOT)
# =========================
@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        # 1. Read input
        data = json.loads(request.body)
        user_message = data.get("message", "")

        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        # 2. Save user message
        ChatMessage.objects.create(
            role="user",
            message=user_message
        )

        # 3. Get memory
        history = ChatMessage.objects.all().order_by("-id")[:10]
        history = reversed(history)

        # 4. Build messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Remember context."
            }
        ]

        for msg in history:
            messages.append({
                "role": msg.role,
                "content": msg.message
            })

        # 5. Call LLM
        response = ask_llm(messages)

        # 6. Save assistant response
        ChatMessage.objects.create(
            role="assistant",
            message=response
        )

        return JsonResponse({"response": response})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# =========================
# PDF UPLOAD API
# =========================
@csrf_exempt
def upload_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    file = request.FILES.get("file")

    if not file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    # Save file
    doc = Document.objects.create(
        title=file.name,
        file=file
    )

    # Extract text
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # CHUNKING STEP
    chunks = split_text(text)

    # Save chunks
    doc.content = json.dumps(chunks)
    doc.save()

    return JsonResponse({
        "message": "PDF uploaded successfully",
        "text_length": len(text),
        "chunks": len(chunks)
    })
