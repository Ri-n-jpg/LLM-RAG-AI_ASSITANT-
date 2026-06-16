import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import ChatMessage, Document, DocumentChunk
from .tools import ask_llm,get_embedding,search_chunks

from pypdf import PdfReader


# =========================
# UTILITY: TEXT CHUNKING
# =========================
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


# =========================
# CHAT API (MEMORY BOT)
# =========================
@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST method allowed"},
            status=405
        )

    try:
        # 1. Read input
        data = json.loads(request.body)
        user_message = data.get("message", "")

        if not user_message:
            return JsonResponse(
                {"error": "Empty message"},
                status=400
            )

        # 2. Save user message
        ChatMessage.objects.create(
            role="user",
            message=user_message
        )

        # 3. Retrieve relevant chunks
        top_chunks = search_chunks(user_message)

        # 4. Create context from chunks
        context = ""

        for score, text in top_chunks:
            context += text + "\n\n"

        # 5. Build prompt
        messages = [
            {
                "role": "system",
                "content": f"""
You are a helpful AI assistant.

Answer using the following PDF context:

{context}

If the answer is not present in the context,
say:
'I could not find this information in the uploaded document.'
"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # 6. Call LLM
        response = ask_llm(messages)

        # 7. Save AI response
        ChatMessage.objects.create(
            role="assistant",
            message=response
        )

        return JsonResponse({
            "response": response
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )

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
    for chunk in chunks:
        embedding = get_embedding(chunk)

        DocumentChunk.objects.create(
            document=doc,
            text=chunk,
            embedding=embedding
        )
    return JsonResponse({
        "message": "PDF uploaded successfully",
        "text_length": len(text),
        "chunks": len(chunks)
    })
