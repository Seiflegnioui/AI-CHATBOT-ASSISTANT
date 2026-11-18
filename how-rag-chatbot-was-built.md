# How I Built This RAG Chatbot — A Complete Guide 🧠

> **Goal:** You finish reading this and you can build any RAG pipeline from scratch.

---

## Table of Contents

1. [What is RAG and Why Do We Need It?](#1-what-is-rag-and-why-do-we-need-it)
2. [The Big Picture — Architecture](#2-the-big-picture--architecture)
3. [Step 1 — Document Loading & Chunking](#3-step-1--document-loading--chunking)
4. [Step 2 — Embeddings (Text → Vectors)](#4-step-2--embeddings-text--vectors)
5. [Step 3 — Vector Store (ChromaDB)](#5-step-3--vector-store-chromadb)
6. [Step 4 — LLM Generation (Groq)](#6-step-4--llm-generation-groq)
7. [Step 5 — The RAG Pipeline (Tying It All Together)](#7-step-5--the-rag-pipeline-tying-it-all-together)
8. [Step 6 — FastAPI Server](#8-step-6--fastapi-server)
9. [Step 7 — Frontend Chat Widget](#9-step-7--frontend-chat-widget)
10. [Key Concepts Cheat Sheet](#10-key-concepts-cheat-sheet)
11. [What To Build Next](#11-what-to-build-next)

---

## 1. What is RAG and Why Do We Need It?

### The Problem

LLMs (like GPT, Llama, Gemini) are trained on general internet data. They don't know **your** specific information — your projects, your CV, your company's docs. If you ask them:

> "What are Seif's main projects?"

They'll either **hallucinate** (make up fake projects) or say "I don't know."

### The Solution: RAG

**RAG = Retrieval-Augmented Generation**

Instead of relying on the LLM's memory, we:

1. **RETRIEVE** relevant documents from our own knowledge base
2. **AUGMENT** the LLM's prompt with those documents
3. Let the LLM **GENERATE** an answer grounded in real data

```
                    ┌──────────────────────────────────────────┐
                    │           THE RAG PATTERN                │
                    │                                          │
  User Question ──► │  1. RETRIEVE: Search vector DB           │
                    │       ↓ top 5 relevant chunks            │
                    │  2. AUGMENT: Add chunks to prompt         │
                    │       ↓ "Given this context, answer..."  │
                    │  3. GENERATE: LLM writes grounded answer │
                    │                                          │
                    └──────────────────────────────────────────┘
```

### Why Not Just Put Everything in the Prompt?

You might think: "Why not just paste all my documents into the prompt?" Two reasons:

| Approach | Problem |
|----------|---------|
| Paste everything | LLMs have a **context window limit** (e.g. 8K–128K tokens). Your docs might be too big. |
| Paste everything | Even if it fits, the LLM gets **confused** with too much text. Retrieval focuses it on what matters. |
| RAG | Only sends the **relevant** 5 chunks → faster, cheaper, more accurate. |

---

## 2. The Big Picture — Architecture

```
 ┌─────────────────────────────── INDEXING (one-time) ──────────────────────────┐
 │                                                                              │
 │   Data/*.md files                                                            │
 │        │                                                                     │
 │        ▼                                                                     │
 │   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
 │   │ document_loader   │───►│   embeddings     │───►│  vector_store    │       │
 │   │                  │    │                  │    │   (ChromaDB)     │       │
 │   │ Load .md files   │    │ Text → 384-dim   │    │ Store vectors    │       │
 │   │ Split by headers │    │ number arrays    │    │ on disk          │       │
 │   └──────────────────┘    └──────────────────┘    └──────────────────┘       │
 │                                                                              │
 └──────────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────── QUERYING (every question) ────────────────────┐
 │                                                                              │
 │   "What are Seif's projects?"                                               │
 │        │                                                                     │
 │        ▼                                                                     │
 │   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
 │   │   embeddings     │───►│  vector_store     │───►│      llm         │       │
 │   │                  │    │                  │    │                  │       │
 │   │ Question → vec   │    │ Find 5 closest   │    │ Question +       │       │
 │   │                  │    │ chunks           │    │ chunks → Groq   │       │
 │   └──────────────────┘    └──────────────────┘    └──────────────────┘       │
 │                                                            │                 │
 │                                                            ▼                 │
 │                                                   "Seif's main projects     │
 │                                                    are the MLOps churn      │
 │                                                    platform and the         │
 │                                                    fraud detection..."      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/rag_course/
├── config.py            # All settings (paths, model names, etc.)
├── document_loader.py   # Step 1: Load markdown → split into chunks
├── embeddings.py        # Step 2: Text → vectors (numbers)
├── vector_store.py      # Step 3: Store & search vectors in ChromaDB
├── llm.py               # Step 4: Send question + context → Groq LLM
├── rag_pipeline.py      # Step 5: Orchestrates everything
└── api.py               # Step 6: FastAPI REST server
```

---

## 3. Step 1 — Document Loading & Chunking

> **File:** [`document_loader.py`](file:///c:/Users/crime/OneDrive/Desktop/RAG-COURSE/src/rag_course/document_loader.py)

### What This Does

Takes your raw markdown files and breaks them into small, meaningful **chunks**.

### Why Do We Need Chunks?

Imagine your `projects.md` file is 3 pages long. If a recruiter asks "What is the churn project?", we don't want to send ALL 3 pages to the LLM — we just want the section about the churn project. Chunking solves this.

### Why Split by Markdown Headers?

There are many chunking strategies:

| Strategy | How it works | Problem |
|----------|-------------|---------|
| Fixed size (500 chars) | Cut every 500 characters | Cuts in the middle of sentences |
| Sentence splitting | Split by periods | A project description might span many sentences |
| **Markdown headers** ✅ | Split by #, ##, ### | Each chunk = one logical section (a project, a skill category, etc.) |

We chose **markdown headers** because our data is already organized with headers. Each chunk stays self-contained.

### The Code Explained

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
```

**`DirectoryLoader`** — Scans a folder and loads all files matching a pattern.
**`TextLoader`** — Reads each file as plain text (since .md files are just text).
**`MarkdownHeaderTextSplitter`** — The smart splitter that knows about `#`, `##`, `###`.

```python
def load_documents():
    loader = DirectoryLoader(
        str(config.DATA_DIR),         # Path to Data/ folder
        glob="**/*.md",               # Only .md files
        exclude=config.EXCLUDED_FILES, # Skip README, instructions, etc.
        loader_cls=TextLoader,         # Read as plain text
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()   # Returns list of Document objects
    return documents
```

Each `Document` has:
- `page_content` → the full text of the file
- `metadata` → `{"source": "C:/.../projects.md"}`

```python
def split_into_chunks(documents):
    headers_to_split_on = [
        ("#", "h1"),    # e.g., "# Projects"
        ("##", "h2"),   # e.g., "## Churn Detection"
        ("###", "h3"),  # e.g., "### Technologies"
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    chunks = []
    for doc in documents:
        split_pieces = splitter.split_text(doc.page_content)
        for piece in split_pieces:
            piece.metadata["source"] = doc.metadata["source"]  # Keep track of origin
            chunks.append(piece)
    return chunks
```

### What the Output Looks Like

For a file like:
```markdown
# Projects
## Churn Detection
An end-to-end MLOps platform...
### Technologies
Python, Docker, MLflow
```

We get **2 chunks**:

| Chunk | metadata | content |
|-------|----------|---------|
| 1 | `{h1: "Projects", h2: "Churn Detection"}` | "An end-to-end MLOps platform..." |
| 2 | `{h1: "Projects", h2: "Churn Detection", h3: "Technologies"}` | "Python, Docker, MLflow" |

Each chunk **knows where it came from** thanks to metadata. This is crucial for citing sources later.

---

## 4. Step 2 — Embeddings (Text → Vectors)

> **File:** [`embeddings.py`](file:///c:/Users/crime/OneDrive/Desktop/RAG-COURSE/src/rag_course/embeddings.py)

### What is an Embedding?

An **embedding** converts text into a **list of numbers** (a vector) that captures its **meaning**.

```
"Python programming"  →  [0.12, -0.45, 0.78, 0.33, ..., 0.21]   (384 numbers)
"coding in Python"    →  [0.11, -0.44, 0.77, 0.34, ..., 0.20]   (very similar!)
"banana smoothie"     →  [0.95, 0.23, -0.56, 0.12, ..., -0.87]  (very different!)
```

### Why 384 Numbers?

The model `all-MiniLM-L6-v2` outputs 384-dimensional vectors. Think of it as 384 different "aspects" of meaning. More dimensions = more nuance, but also more memory/compute.

### How Does Similarity Work?

We use **cosine similarity** — it measures the angle between two vectors:
- **1.0** = identical meaning
- **0.0** = completely unrelated
- Vectors close in meaning point in the same "direction" in 384D space

```
                    "Python programming"
                   /
                  /  ← small angle = similar
                 /
    Origin ────/──────── "coding in Python"



                    "banana smoothie"
                   ↗
                  /  ← big angle = different
                 /
    Origin ────/──────── "Python programming"
```

### Why `sentence-transformers` and Not OpenAI?

| Option | Pros | Cons |
|--------|------|------|
| OpenAI embeddings | High quality | Costs money, needs internet |
| **sentence-transformers** ✅ | Free, runs locally, no API key | Slightly lower quality (but great for 50 chunks) |

For a portfolio chatbot with ~50 chunks, `all-MiniLM-L6-v2` is **more than enough**.

### The Code Explained

```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # First time: downloads ~80MB model
        # After that: loads from cache instantly
```

The key method:

```python
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,                      # ["chunk 1 text", "chunk 2 text", ...]
            convert_to_numpy=True,      # Return numpy array (not PyTorch tensor)
            show_progress_bar=len(texts) > 10,
        )
        # Returns shape: (num_texts, 384)
        # e.g., 47 chunks → (47, 384) array
```

**`model.encode()`** does all the magic — tokenizes text, runs it through the neural network, and outputs the embedding vector.

---

## 5. Step 3 — Vector Store (ChromaDB)

> **File:** [`vector_store.py`](file:///c:/Users/crime/OneDrive/Desktop/RAG-COURSE/src/rag_course/vector_store.py)

### What is a Vector Store?

A **vector store** is a database optimized for storing and searching vectors. Instead of SQL `WHERE name = 'Seif'`, you query by **semantic similarity**: "find me the 5 chunks most similar to this question."

### Why ChromaDB?

| Vector Store | Complexity | Use Case |
|-------------|-----------|----------|
| **ChromaDB** ✅ | Simple, local, file-based | Small projects, prototypes, portfolios |
| FAISS | Medium, in-memory | Large datasets, need speed |
| Pinecone | Cloud service, managed | Production apps, millions of vectors |
| Weaviate | Full-featured, server | Enterprise search |

ChromaDB is perfect for learning — zero setup, persists to disk.

### The Code Explained

**Creating the store:**

```python
import chromadb

class VectorStore:
    def __init__(self, embedding_model=None):
        self.embedder = embedding_model or EmbeddingModel()

        # PersistentClient = saves to disk (survives restarts)
        # vs. Client() = in-memory only (lost when program ends)
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

        # A "collection" is like a table in SQL
        self.collection = self.client.get_or_create_collection(
            name="seif_portfolio",
        )
```

**Adding chunks (indexing):**

```python
    def add_chunks(self, chunks):
        # 1. Extract the text from each chunk
        texts = [chunk.page_content for chunk in chunks]

        # 2. Generate embeddings for ALL chunks at once
        #    Shape: (47, 384) — 47 chunks, each with 384 numbers
        vectors = self.embedder.embed(texts)

        # 3. Create unique IDs: ["chunk_0", "chunk_1", ..., "chunk_46"]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        # 4. Store everything in ChromaDB
        self.collection.add(
            ids=ids,                        # Unique identifier
            documents=texts,                # The raw text (returned in search)
            embeddings=vectors.tolist(),     # The vectors (used for search)
            metadatas=metadatas,             # Source file, headers, etc.
        )
```

> [!IMPORTANT]
> ChromaDB stores **three things** per chunk: the text, the vector, and the metadata. The vector is what gets compared during search, but the text is what gets returned.

**Searching (retrieval):**

```python
    def search(self, query: str, top_k: int = 5):
        # 1. Convert the question to a vector (same model as indexing!)
        query_vector = self.embedder.embed([query])

        # 2. ChromaDB finds the 5 closest vectors
        results = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=top_k,
        )

        # 3. Returns the matching chunks with their text + metadata
        return results
```

### How the Search Actually Works (Under the Hood)

```
Question: "What technologies does Seif know?"
   │
   ▼ embed()
Question vector: [0.23, -0.11, 0.67, ...]
   │
   ▼ compare with every stored vector
   │
   ├── chunk_0 (certifications.md)  → distance: 1.45  ❌ far
   ├── chunk_7 (skills.md, AI & ML) → distance: 0.31  ✅ close!
   ├── chunk_8 (skills.md, MLOps)   → distance: 0.38  ✅ close!
   ├── chunk_12 (projects.md)       → distance: 0.52  ✅ decent
   ├── chunk_15 (education.md)      → distance: 1.67  ❌ far
   └── ...
   │
   ▼ sort by distance, take top 5
   │
   Return: [chunk_7, chunk_8, chunk_12, chunk_3, chunk_19]
```

**Distance** = how different the vectors are. Lower = more relevant.

---

## 6. Step 4 — LLM Generation (Groq)

> **File:** [`llm.py`](file:///c:/Users/crime/OneDrive/Desktop/RAG-COURSE/src/rag_course/llm.py)

### What This Does

Takes the question + the retrieved chunks and builds a **prompt** for the LLM. The LLM reads the context and generates a grounded answer.

### The Prompt Structure

This is the most important part of RAG — the prompt tells the LLM **what to do** and **what context to use**:

```
┌──────────────────────────────────────────────────────┐
│ SYSTEM PROMPT (from chatbot_instructions.md)         │
│ "You are Seif's portfolio assistant. Be clear,       │
│  professional. Never invent facts..."                │
├──────────────────────────────────────────────────────┤
│ USER PROMPT:                                         │
│                                                      │
│ Here is the relevant context from Seif's portfolio:  │
│                                                      │
│ [Source: skills.md]                                   │
│ Python, NumPy, Pandas, scikit-learn, PyTorch...      │
│ ---                                                  │
│ [Source: projects.md]                                 │
│ End-to-End MLOps Platform — Telecom Churn...         │
│ ---                                                  │
│ [Source: profile.md]                                  │
│ Seif Eddine Legnioui is a Master's student...        │
│                                                      │
│ Based ONLY on the context above, answer:             │
│ Question: What technologies does Seif know?          │
└──────────────────────────────────────────────────────┘
```

### The Code Explained

**Building the prompt:**

```python
def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    # Format each chunk with its source file name
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"].get("source", "unknown")
        source_name = source.split("\\")[-1].split("/")[-1]  # Get just filename
        context_parts.append(f"[Source: {source_name}]\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    # Load exact facts (dates, counts) from JSON
    structured_facts = _load_structured_profile()

    # The final prompt
    prompt = f"""Here is the relevant context retrieved from Seif's portfolio:

{context_block}

---
Here are structured facts for exact dates, counts, and current status:
{structured_facts}
---

Based ONLY on the context and structured facts above, answer the following question.
If the information is not in the context, say so honestly — do not make things up.

Question: {question}"""

    return prompt
```

> [!TIP]
> **"Based ONLY on the context above"** — this instruction is critical! Without it, the LLM might ignore the context and answer from its general knowledge, which could be wrong.

**Calling Groq:**

```python
def ask(question: str, retrieved_chunks: list[dict]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")

    system_prompt = _load_system_prompt()     # How to behave
    user_prompt = build_prompt(question, retrieved_chunks)  # Context + question

    # Build the API request (OpenAI-compatible format)
    request_body = json.dumps({
        "model": "openai/gpt-oss-120b",   # The LLM to use
        "messages": [
            {"role": "system", "content": system_prompt},  # Instructions
            {"role": "user", "content": user_prompt},      # Context + question
        ],
        "temperature": 0.3,  # 0 = deterministic, 1 = creative
        "max_tokens": 1024,  # Maximum response length
    }).encode("utf-8")

    # Send HTTP POST to Groq's API
    req = urllib.request.Request(
        url="https://api.groq.com/openai/v1/chat/completions",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        return result["choices"][0]["message"]["content"]
```

### Why `temperature: 0.3`?

| Temperature | Behavior | Use Case |
|------------|----------|----------|
| 0.0 | Always picks the most probable word | Math, code |
| **0.3** ✅ | Mostly factual, slight variation | Portfolio chatbot (we want accuracy) |
| 0.7 | More creative and varied | Creative writing |
| 1.0 | Very random, surprising | Brainstorming |

For a portfolio chatbot, we want **factual and consistent** answers → low temperature.

### Why We Use `urllib` Instead of a Library

We used Python's built-in `urllib` instead of installing `openai` or `groq` packages. The Groq API is just a **REST API** — it's literally:

1. Send a JSON POST request to a URL
2. Get a JSON response back

No library needed. This teaches you that **LLM APIs are just HTTP calls**.

---

## 7. Step 5 — The RAG Pipeline (Tying It All Together)

> **File:** [`rag_pipeline.py`](file:///c:/Users/crime/OneDrive/Desktop/RAG-COURSE/src/rag_course/rag_pipeline.py)

### What This Does

This is the **orchestrator**. It connects all the pieces into a single class you can use:

```python
rag = RAGPipeline()
result = rag.ask("What are Seif's projects?")
print(result["answer"])
```

### The Code Explained

**Initialization (runs once):**

```python
class RAGPipeline:
    def __init__(self):
        # 1. Load the embedding model (shared for indexing AND querying)
        self.embedder = EmbeddingModel()

        # 2. Connect to ChromaDB
        self.vector_store = VectorStore(embedding_model=self.embedder)

        # 3. If DB is empty → load docs, chunk them, index them
        if self.vector_store.collection.count() == 0:
            self._index_documents()
```

> [!IMPORTANT]
> The same embedding model must be used for **indexing** (storing chunks) and **querying** (searching). If you index with model A and search with model B, the vectors won't be comparable and search will return garbage.

**Answering a question (runs every time):**

```python
    def ask(self, question: str) -> dict:
        # Step 1: RETRIEVE — find relevant chunks
        retrieved_chunks = self.vector_store.search(
            query=question,
            top_k=5,
        )

        # Step 2: GENERATE — send to LLM with context
        answer = llm_ask(question, retrieved_chunks)

        # Step 3: Return answer + sources (transparency!)
        return {
            "answer": answer,
            "sources": [...]  # Which files were used
        }
```

### The Complete Flow for One Question

```
1. User asks: "What is the churn project?"
                    │
2. embed("What is the churn project?")
   → [0.34, -0.12, 0.56, ...]
                    │
3. ChromaDB.query(vector, top_k=5)
   → Finds 5 closest chunks (from projects.md, skills.md, etc.)
                    │
4. build_prompt(question, chunks)
   → "Here is context from Seif's portfolio:
      [projects.md] End-to-end MLOps platform...
      ...
      Question: What is the churn project?"
                    │
5. Groq API generates answer
   → "The churn project is an end-to-end MLOps platform
      for telecom customer churn detection. It uses..."
                    │
6. Return {answer: "...", sources: ["projects.md", "skills.md"]}
```

---

## 8. Step 6 — FastAPI Server

> **File:** [`api.py`](file:///c:/Users/crime/OneDrive/Desktop/RAG-COURSE/src/rag_course/api.py)

### What This Does

Creates a **REST API** that your portfolio website can call. When a recruiter types a question in the chat widget, the browser sends a POST request to this server.

### The Code Explained

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Seif's Portfolio Chatbot API")

# What the frontend sends us
class ChatRequest(BaseModel):
    question: str                    # e.g., "What are Seif's projects?"

# What we send back
class ChatResponse(BaseModel):
    answer: str                      # The LLM's answer
    sources: list[SourceInfo]        # Where the answer came from
```

**`BaseModel`** (from Pydantic) automatically:
- Validates input (rejects if `question` is missing)
- Converts to/from JSON
- Generates API docs

```python
# The main endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = rag.ask(request.question)   # Calls the RAG pipeline
    return ChatResponse(**result)
```

### CORS — Why It Matters

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Which websites can call this API
)
```

Without CORS, your browser will **block** the request from your portfolio (running on `file://` or `localhost:3000`) to your API (on `localhost:8000`). CORS says "it's OK, let them talk."

> [!NOTE]
> In production, replace `["*"]` with your actual domain like `["https://seif-portfolio.com"]` for security.

### The Request/Response Lifecycle

```
Browser (portfolio)                    FastAPI Server
       │                                     │
       │  POST /chat                         │
       │  {"question": "Seif's projects?"}   │
       │ ───────────────────────────────────► │
       │                                     │  → RAG Pipeline
       │                                     │  → ChromaDB search
       │                                     │  → Groq API call
       │  {"answer": "...", "sources": [...]} │
       │ ◄─────────────────────────────────── │
       │                                     │
       ▼                                     │
  Show answer in chat widget                 │
```

---

## 9. Step 7 — Frontend Chat Widget

> **Files:** [`chatbot.js`](file:///C:/Users/crime/OneDrive/Desktop/portfolio/chatbot.js) + [`chatbot.css`](file:///C:/Users/crime/OneDrive/Desktop/portfolio/chatbot.css)

### The JavaScript — Key Parts

**Sending a message to the API:**

```javascript
async function sendMessage(question) {
    // 1. Show the user's message in the chat
    addMessage(question, 'user');

    // 2. Show "typing..." animation
    const typingEl = addTypingIndicator();

    // 3. Call our FastAPI server
    const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),  // Send the question as JSON
    });

    const data = await response.json();

    // 4. Remove typing indicator, show the answer
    typingEl.remove();
    addMessage(data.answer, 'bot');
}
```

That's it! The frontend is just a simple `fetch()` call. All the intelligence is in the backend.

---

## 10. Key Concepts Cheat Sheet

| Concept | What It Means | Analogy |
|---------|--------------|---------|
| **Embedding** | Text → list of numbers | Like GPS coordinates for meaning |
| **Vector** | The list of numbers itself | The coordinates [lat, long] |
| **Vector Store** | Database for vectors | Google Maps — search by location, not name |
| **Chunk** | A small piece of a document | A paragraph in a book |
| **Retrieval** | Finding relevant chunks | Looking up the right book pages |
| **Context Window** | How much text the LLM can read | The LLM's "working memory" |
| **Temperature** | How creative/random the LLM is | 0 = robot, 1 = poet |
| **System Prompt** | Instructions for the LLM | The LLM's job description |
| **RAG** | Retrieve → Augment → Generate | Open book exam vs. closed book |
| **Cosine Similarity** | How similar two vectors are | The angle between two arrows |
| **Top-K** | How many chunks to retrieve | "Give me the 5 best matches" |

---

## 11. What To Build Next

Now that you understand RAG, here are progressively harder projects:

### Level 1 — Easy
- **PDF RAG chatbot**: Replace markdown files with PDF loading (use `PyPDFLoader`)
- **Multi-language**: Make the chatbot answer in French or English based on the question
- **Conversation memory**: Store chat history so the bot remembers previous questions

### Level 2 — Medium
- **Hybrid search**: Combine vector search + keyword search (BM25) for better retrieval
- **Re-ranking**: After retrieving 20 chunks, use a cross-encoder to re-rank the top 5
- **Evaluation**: Build a test set of questions + expected answers, measure accuracy

### Level 3 — Advanced
- **Agentic RAG**: The LLM decides WHEN to search and WHAT to search for
- **Multi-modal RAG**: Add images (project screenshots) to the knowledge base
- **Production deployment**: Deploy on Railway/Render, add rate limiting, caching, auth

### Level 4 — Expert
- **Knowledge graph RAG**: Extract entities/relationships, combine graph + vector search
- **Self-correcting RAG**: The LLM checks if its answer is supported by the context
- **Fine-tuning**: Train the embedding model on your specific domain for better retrieval

---

> [!TIP]
> **The best way to learn**: take this pipeline, change the data (use your own documents instead of portfolio files), and see what breaks. That's where real understanding comes from.

---

*Built by Seif Eddine Legnioui — 2026*
