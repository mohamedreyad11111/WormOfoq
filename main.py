import os
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Reyad-AI-Backend (FastAPI + Google GenAI)")

# إعداد نظام الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reyad_ai")

# إعداد المفتاح
GENAI_API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("❌ Missing environment variable: GENAI_API_KEY or GOOGLE_API_KEY")

genai.configure(api_key=GENAI_API_KEY)

# نماذج البيانات
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-1.5-flash"
    max_output_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "text-embedding-004"

@app.get("/")
def health():
    return {"status": "ok", "service": "Reyad-AI-Backend"}

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        model = genai.GenerativeModel(req.model)
        response = model.generate_content(
            req.prompt,
            generation_config={
                "max_output_tokens": req.max_output_tokens,
                "temperature": req.temperature
            },
        )
        return {"response": response.text}
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings")
def embeddings(req: EmbeddingRequest):
    try:
        model = genai.get_model(req.model)
        resp = model.embed_content(req.texts)
        return {"embeddings": resp}
    except Exception as e:
        logger.exception("Embedding error")
        raise HTTPException(status_code=500, detail=str(e))
