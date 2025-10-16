# api/main.py
import os
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Google GenAI SDK
try:
    from google import genai
except Exception as e:
    genai = None

app = FastAPI(title="Reyad-AI-Backend (FastAPI + Google GenAI)")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reyad_ai")

# Pydantic request models
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-2.5-flash"   # يمكنك تغييره حسب حاجتك
    max_output_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "gemini-embedding-001"

# Initialize Google GenAI client (on startup)
GENAI_CLIENT = None

@app.on_event("startup")
def startup_event():
    global GENAI_CLIENT
    if genai is None:
        logger.warning("google-genai package not installed or couldn't be imported.")
        return

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GOOGLE_API_KEY / GEMINI_API_KEY found in environment. The API will fail until set.")
        # still create client without key if using Vertex env variables later
        GENAI_CLIENT = genai.Client()
        return

    # create client for Gemini Developer API (api_key) or Vertex depending on env
    GENAI_CLIENT = genai.Client(api_key=api_key)
    logger.info("Google GenAI client initialized.")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "reyad-ai-backend"}

@app.post("/api/chat")
def chat(req: ChatRequest):
    if GENAI_CLIENT is None:
        raise HTTPException(status_code=500, detail="GenAI client not initialized. Set GOOGLE_API_KEY env var.")

    try:
        # Use the models.generate_content helper (sync example)
        resp = GENAI_CLIENT.models.generate_content(
            model=req.model,
            contents=req.prompt,
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature
        )
        # response.text is a convenience accessor in the SDK
        generated_text = getattr(resp, "text", None) or str(resp)
        return {
            "model": req.model,
            "prompt": req.prompt,
            "response": generated_text,
            "raw": repr(resp)
        }
    except Exception as e:
        logger.exception("Error calling GenAI")
        raise HTTPException(status_code=500, detail=f"GenAI error: {e}")

@app.post("/api/embeddings")
def embeddings(req: EmbeddingRequest):
    if GENAI_CLIENT is None:
        raise HTTPException(status_code=500, detail="GenAI client not initialized. Set GOOGLE_API_KEY env var.")
    try:
        # The SDK has embedding helpers; interfaces vary by SDK version.
        # Using client.models.embed_content(...) or client.embeddings.create(...) depending on SDK.
        # We'll try common name `models.embed_content` first and fallback to `embeddings.create`.
        if hasattr(GENAI_CLIENT.models, "embed_content"):
            resp = GENAI_CLIENT.models.embed_content(model=req.model, contents=req.texts)
            # resp will likely have embeddings in resp.embeddings or similar
            return {"model": req.model, "raw": repr(resp)}
        elif hasattr(GENAI_CLIENT, "embeddings") and hasattr(GENAI_CLIENT.embeddings, "create"):
            resp = GENAI_CLIENT.embeddings.create(model=req.model, input=req.texts)
            return {"model": req.model, "raw": repr(resp)}
        else:
            # Fallback: raise informative error
            raise RuntimeError("Embeddings API not available on this client version. Check google-genai docs.")
    except Exception as e:
        logger.exception("Error creating embeddings")
        raise HTTPException(status_code=500, detail=f"Embeddings error: {e}")
