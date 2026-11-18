"""
llm.py — Step 4: Generate answers using Groq (free, no credit card!).

RAG Pipeline Flow:
  [Question + Retrieved Chunks] → build_prompt() → Groq (Llama 3.3) → [Answer]

What is RAG (Retrieval-Augmented Generation)?
  Instead of asking the LLM to answer from its general knowledge,
  we FIRST retrieve relevant chunks from our vector store, then
  include them in the prompt as context. This way:
    - The LLM only talks about Seif's actual portfolio ✅
    - It doesn't hallucinate fake projects or skills ✅
    - Answers stay grounded in your real documents ✅

Why Groq?
  - 100% free, no credit card needed
  - Extremely fast (runs on custom LPU chips)
  - Hosts Llama 3.3 70B — a very capable open-source model
  - Simple REST API
"""

import json
import os
import urllib.request
import urllib.error

from . import config


def _load_system_prompt() -> str:
    """Load the system prompt from chatbot_instructions.md."""
    return config.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def _load_structured_profile() -> str:
    """Load the structured profile JSON for exact facts."""
    return config.STRUCTURED_PROFILE_FILE.read_text(encoding="utf-8")


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the prompt that we send to the LLM.

    The prompt has 3 parts:
      1. CONTEXT — The retrieved chunks (the "R" in RAG)
      2. STRUCTURED FACTS — The JSON profile for exact dates/counts
      3. QUESTION — What the recruiter asked

    Args:
        question: The recruiter's question
        retrieved_chunks: Results from vector_store.search()

    Returns:
        A formatted prompt string
    """
    # Format the retrieved chunks into a readable block
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        # Get the filename only (not the full path)
        source_name = source.split("\\")[-1].split("/")[-1]
        context_parts.append(
            f"[Source: {source_name}]\n{chunk['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    # Load the structured profile for exact facts
    structured_facts = _load_structured_profile()

    # Build the final prompt
    prompt = f"""Here is the relevant context retrieved from Seif's portfolio knowledge base:

{context_block}

---

Here are structured facts for exact dates, counts, and current status:

{structured_facts}

---

Based ONLY on the context and structured facts above, answer the following question.
If the information is not in the context, say so honestly — do not make things up.

Question: {question}"""

    return prompt


def ask(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Send the question + context to Groq and get an answer.

    We use urllib (built into Python) instead of an external library
    so there's one less dependency to install. It's just a simple
    HTTP POST request with JSON.

    Args:
        question:          What the recruiter asked
        retrieved_chunks:  The relevant chunks from vector search

    Returns:
        The LLM's answer as a string
    """
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "GROQ_API_KEY not set! Please set it:\n"
            '  PowerShell: $env:GROQ_API_KEY="your-key-here"\n'
            '  CMD:        set GROQ_API_KEY=your-key-here\n'
            "  Get a free key at: https://console.groq.com"
        )

    # Load the system prompt (tells the LLM how to behave)
    system_prompt = _load_system_prompt()

    # Build the user prompt with retrieved context
    user_prompt = build_prompt(question, retrieved_chunks)

    # Build the API request body
    # This follows the OpenAI-compatible chat format that Groq uses
    request_body = json.dumps({
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,  # Low = more factual, less creative
        "max_tokens": 1024,
    }).encode("utf-8")

    # Build the HTTP request
    req = urllib.request.Request(
        url="https://api.groq.com/openai/v1/chat/completions",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "seif-portfolio-chatbot/1.0",
        },
        method="POST",
    )

    # Send the request and get the response
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        return f"Groq API error ({e.code}): {error_body}"
    except Exception as e:
        return f"Error calling Groq: {e}"


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Fake chunks to test prompt building (without needing the full pipeline)
    fake_chunks = [
        {
            "text": "Seif has experience with Python, Docker, MLflow, and FastAPI.",
            "metadata": {"source": "skills.md"},
            "distance": 0.3,
        },
    ]

    prompt = build_prompt("What technologies does Seif know?", fake_chunks)
    print("=== BUILT PROMPT ===")
    print(prompt)
    print("\n=== Calling Groq... ===")
    answer = ask("What technologies does Seif know?", fake_chunks)
    print(answer)
