"""
Create embedding models for FAISS vector search.

Default is TF-IDF (works offline, no PyTorch). Set EMBEDDING_MODE in .env
to huggingface or google only if you have enough RAM and API access.
"""

import os

from langchain_core.embeddings import Embeddings

from utils.tfidf_embeddings import get_tfidf_embeddings


def _huggingface_embeddings() -> Embeddings:
    """Local HuggingFace model (needs PyTorch + ~2GB free RAM)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _google_embeddings() -> Embeddings:
    """Gemini embedding API (same key as chat)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing in .env")

    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


def get_embedding_model() -> Embeddings:
    """
    Return embeddings based on EMBEDDING_MODE:
    tfidf (default) | huggingface | google
    """
    mode = os.getenv("EMBEDDING_MODE", "tfidf").lower().strip()

    if mode in ("tfidf", "auto", ""):
        return get_tfidf_embeddings()

    if mode == "google":
        return _google_embeddings()

    if mode == "huggingface":
        return _huggingface_embeddings()

    # Unknown value — fall back safely
    return get_tfidf_embeddings()


def is_tfidf_mode() -> bool:
    """True when using local TF-IDF embeddings."""
    mode = os.getenv("EMBEDDING_MODE", "tfidf").lower().strip()
    return mode in ("tfidf", "auto", "")
