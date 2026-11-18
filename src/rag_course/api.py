"""
api.py — FastAPI server for your portfolio chatbot.

This creates a REST API that your portfolio website can call.
When a recruiter types a question on your site, the frontend sends
it here, and this server runs the RAG pipeline and returns the answer.

Endpoints:
  POST /chat          → Ask a question, get an answer
  GET  /health        → Check if the server is running

How to run:
  cd RAG-COURSE
  .venv\\Scripts\\activate
  $env:GROQ_API_KEY="your-key"
  uvicorn rag_course.api:app --reload

Then test it:
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question": "What are Seif projects?"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import RAGPipeline

# ──────────────────────────────────────────────
# Create the FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title="Seif's Portfolio Chatbot API",
    description="RAG-powered chatbot that answers questions about Seif's portfolio",
    version="1.0.0",
)

# Allow your portfolio website to call this API (CORS)
# In production, replace ["*"] with your actual domain like ["https://seif.dev"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Initialize the RAG pipeline (runs once when the server starts)
# ──────────────────────────────────────────────

rag = None  # Will be set on startup


@app.on_event("startup")
async def startup():
    """Load the RAG pipeline when the server starts."""
    global rag
    rag = RAGPipeline()


# ──────────────────────────────────────────────
# Request/Response models (what the API expects and returns)
# ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """What the frontend sends us."""
    question: str

    # Example for the auto-generated docs
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What are Seif's main projects?"}
            ]
        }
    }


class SourceInfo(BaseModel):
    """Info about where the answer came from."""
    file: str
    section: str
    subsection: str
    relevance_distance: float
    preview: str


class ChatResponse(BaseModel):
    """What we send back to the frontend."""
    answer: str
    sources: list[SourceInfo]


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question about Seif's portfolio.

    The RAG pipeline will:
      1. Search the knowledge base for relevant info
      2. Use Gemini to generate a grounded answer
      3. Return the answer + sources for transparency
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag.ask(request.question)
    return ChatResponse(**result)


@app.get("/health")
async def health():
    """Check if the server is running and the pipeline is loaded."""
    chunk_count = rag.collection_count if rag else 0
    return {
        "status": "healthy",
        "pipeline_loaded": rag is not None,
        "chunks_indexed": rag.vector_store.collection.count() if rag else 0,
    }
