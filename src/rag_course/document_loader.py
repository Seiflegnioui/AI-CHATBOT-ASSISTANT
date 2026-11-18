"""
document_loader.py — Step 1: Load and chunk your documents.

RAG Pipeline Flow:
  [Markdown Files] → load_documents() → split_into_chunks() → [Chunks]

What happens here:
  1. We read all .md files from the Data/ folder (except the excluded ones)
  2. We split each document by its markdown headers (#, ##, ###)
     This keeps related info together — e.g., a project's overview stays
     in one chunk, its technologies in another

Why split by headers instead of fixed character count?
  Because markdown headers are natural boundaries. A recruiter asking
  "What is the churn project?" will get the whole project section, not
  a random cut in the middle of a sentence.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from . import config


def load_documents():
    """
    Load all markdown files from the Data/ directory.

    Returns a list of LangChain Document objects, each with:
      - page_content: the text of the file
      - metadata: {"source": "path/to/file.md"}
    """
    loader = DirectoryLoader(
        str(config.DATA_DIR),
        glob="**/*.md",
        exclude=config.EXCLUDED_FILES,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from {config.DATA_DIR}")
    return documents


def split_into_chunks(documents):
    """
    Split documents into smaller chunks based on markdown headers.

    For example, a document like:
        # Projects
        ## Churn Detection
        Overview text here...
        ### Technologies
        Python, Docker, ...

    Becomes separate chunks, each tagged with its header hierarchy:
        chunk.metadata = {"h1": "Projects", "h2": "Churn Detection", ...}

    This way, when we search for "Docker", we find the chunk AND know
    which project/section it belongs to.
    """
    # Define which header levels to split on
    headers_to_split_on = [
        ("#", "h1"),    # Main title
        ("##", "h2"),   # Section
        ("###", "h3"),  # Subsection
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )

    chunks = []
    for doc in documents:
        # Split this document's text by its headers
        split_pieces = splitter.split_text(doc.page_content)

        for piece in split_pieces:
            # Keep track of which file this chunk came from
            piece.metadata["source"] = doc.metadata["source"]
            chunks.append(piece)

    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ──────────────────────────────────────────────
# Quick test: run this file directly to see it work
# ──────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_documents()
    chunks = split_into_chunks(docs)

    # Show the first 3 chunks so you can see the structure
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content:  {chunk.page_content[:200]}...")
