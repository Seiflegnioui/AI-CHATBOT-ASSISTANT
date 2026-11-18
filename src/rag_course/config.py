"""
config.py — All settings in one place.

Why a config file? So you never have hardcoded paths or model names
scattered across your code. Change something here, and it updates everywhere.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

# Root of the project (two levels up from this file: src/rag_course/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Where your markdown knowledge-base files live
DATA_DIR = PROJECT_ROOT / "Data"

# Where ChromaDB will store the vector database on disk
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# The system prompt file that tells the LLM how to behave
SYSTEM_PROMPT_FILE = DATA_DIR / "chatbot_instructions.md"

# The structured profile JSON (used for exact facts like dates/counts)
STRUCTURED_PROFILE_FILE = DATA_DIR / "structured_profile.json"

# ──────────────────────────────────────────────
# DOCUMENT LOADING
# ──────────────────────────────────────────────

# Which markdown files to skip when loading your knowledge base
EXCLUDED_FILES = [
    "README.md",
    "chatbot_instructions.md",
    "sources_and_verification.md",
]

# ──────────────────────────────────────────────
# EMBEDDING MODEL
# ──────────────────────────────────────────────

# This is a small, fast model that runs locally (no API key needed!)
# It converts text into 384-dimensional vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────

# How many chunks to retrieve for each question
TOP_K = 5

# ChromaDB collection name (think of it like a table name)
COLLECTION_NAME = "seif_portfolio"

# ──────────────────────────────────────────────
# LLM (Language Model for generating answers)
# ──────────────────────────────────────────────

# We use Groq — 100% free, no credit card needed!
# Sign up at: https://console.groq.com
# Then set your API key as an environment variable:
#   export GROQ_API_KEY="your-key-here"     (Linux/Mac)
#   set GROQ_API_KEY=your-key-here           (Windows CMD)
#   $env:GROQ_API_KEY="your-key-here"        (PowerShell)
GROQ_MODEL = "openai/gpt-oss-120b"

# ──────────────────────────────────────────────
# API SERVER
# ──────────────────────────────────────────────

API_HOST = "0.0.0.0"
API_PORT = 8000
