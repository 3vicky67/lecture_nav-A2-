import os
import cohere

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

def ask_cohere(question: str) -> str:
    if not co:
        raise RuntimeError("Cohere client is not configured. Set COHERE_API_KEY in .env.")
    response = co.chat(model="command-xlarge-nightly", message=question)
    return response.text


