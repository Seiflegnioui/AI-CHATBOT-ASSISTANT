"""
RAG Pipeline for Seif's Portfolio Chatbot.

This package contains a complete Retrieval-Augmented Generation (RAG) pipeline:
1. Load markdown documents about Seif's portfolio
2. Split them into chunks
3. Embed chunks and store them in ChromaDB
4. Retrieve relevant chunks for a user's question
5. Send the question + context to an LLM to generate an answer
"""
