# Seif's Portfolio RAG Chatbot 🤖

A **Retrieval-Augmented Generation (RAG)** chatbot that answers recruiter questions about my portfolio, grounded in real data — no hallucinations.

## What is RAG?

```
Recruiter asks: "What are Seif's main projects?"
         │
         ▼
   ┌─────────────┐
   │  Embedding   │  Convert question to a vector
   │   Model      │  (list of 384 numbers)
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  ChromaDB    │  Search for the most similar
   │  Vector DB   │  chunks from my portfolio docs
   └──────┬──────┘
          │
          ▼  Top 5 relevant chunks
   ┌─────────────┐
   │  Google      │  "Given this context about Seif,
   │  Gemini LLM  │   answer the recruiter's question"
   └──────┬──────┘
          │
          ▼
   Grounded, accurate answer
   with source references
```

## Project Structure

```
RAG-COURSE/
├── Data/                          # Knowledge base (markdown files about me)
│   ├── profile.md                 # Who I am, positioning
│   ├── education.md               # Degrees and certifications
│   ├── experience.md              # Work experience
│   ├── projects.md                # Detailed project descriptions
│   ├── skills.md                  # Technologies and skills
│   ├── links.md                   # GitHub, LinkedIn, etc.
│   ├── certifications.md          # Professional certificates
│   ├── chatbot_instructions.md    # System prompt for the LLM
│   └── structured_profile.json   # Exact facts (dates, counts)
│
├── src/rag_course/                # The RAG pipeline code
│   ├── config.py                  # All settings in one place
│   ├── document_loader.py         # Step 1: Load & chunk markdown files
│   ├── embeddings.py              # Step 2: Text → vectors
│   ├── vector_store.py            # Step 3: Store & search vectors (ChromaDB)
│   ├── llm.py                     # Step 4: Generate answers (Gemini)
│   ├── rag_pipeline.py            # Ties it all together
│   └── api.py                     # FastAPI REST server
│
├── chroma_db/                     # Vector database (auto-created)
└── Notebook/Documents.ipynb       # Exploration notebook
```

## Setup

### 1. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Set your Gemini API key

Get a free key at: https://aistudio.google.com/apikey

```powershell
# PowerShell
$env:GEMINI_API_KEY="your-key-here"
```

```bash
# Linux/Mac
export GEMINI_API_KEY="your-key-here"
```

### 3. Run the API server

```bash
uvicorn rag_course.api:app --reload
```

### 4. Test it

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Seif main projects?"}'
```

Or open http://localhost:8000/docs for the interactive Swagger UI.

## How Each File Works

| File | What it does | Key concept |
|------|-------------|-------------|
| `config.py` | Central settings | No hardcoded values scattered around |
| `document_loader.py` | Reads .md files, splits by headers | Chunks = small, meaningful pieces of text |
| `embeddings.py` | Converts text to 384-dim vectors | Similar texts → similar vectors |
| `vector_store.py` | Stores/searches vectors in ChromaDB | Semantic search by meaning, not keywords |
| `llm.py` | Builds prompt, calls Gemini | RAG = retrieve first, then generate |
| `rag_pipeline.py` | Connects all steps | The "brain" that orchestrates everything |
| `api.py` | REST API for your website | Frontend calls POST /chat |

## Integrating with Your Portfolio

In your portfolio frontend (React, Next.js, etc.), call the API:

```javascript
async function askChatbot(question) {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();
  return data.answer;
}
```

## Updating Your Knowledge Base

1. Edit any `.md` file in `Data/`
2. Restart the server (it auto-rebuilds the index if empty) — or call the rebuild endpoint
3. The chatbot now knows the updated info!
