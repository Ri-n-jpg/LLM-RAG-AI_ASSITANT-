from groq import Groq
import os

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception("❌ GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)


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