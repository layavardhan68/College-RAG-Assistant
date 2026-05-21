"""
Build, save, load, and query the FAISS vector database.
"""

import shutil

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.embeddings import get_embedding_model, is_tfidf_mode
from utils.paths import FAISS_INDEX_DIR
from utils.tfidf_embeddings import TfidfEmbeddings, get_tfidf_embeddings, reset_tfidf_embeddings

INDEX_DIR = FAISS_INDEX_DIR
VECTORIZER_FILE = INDEX_DIR / "tfidf_vectorizer.pkl"


def _clear_index_files() -> None:
    """Remove a broken or outdated FAISS index."""
    reset_tfidf_embeddings()
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _index_matches_embeddings(vectorstore: FAISS) -> bool:
    """True when query embedding size matches the FAISS index dimension."""
    try:
        probe = vectorstore.embedding_function.embed_query("dimension check")
        return len(probe) == vectorstore.index.d
    except Exception:
        return False


def _get_embeddings_for_index(chunks: list[Document]):
    """Pick embeddings and fit TF-IDF when that backend is active."""
    reset_tfidf_embeddings()
    if is_tfidf_mode() or VECTORIZER_FILE.exists():
        emb = get_tfidf_embeddings(reload=True)
    else:
        emb = get_embedding_model()

    if isinstance(emb, TfidfEmbeddings):
        texts = [c.page_content for c in chunks]
        emb.fit_corpus(texts)
        emb.save()
    return emb


def _all_documents(existing: FAISS | None, new_chunks: list[Document]) -> list[Document]:
    """Combine stored documents with newly uploaded chunks."""
    if existing is None:
        return new_chunks
    stored = list(existing.docstore._dict.values())
    return stored + new_chunks


def create_vectorstore(chunks: list[Document]) -> FAISS:
    """Embed text chunks and store them in a new FAISS index."""
    embeddings = _get_embeddings_for_index(chunks)
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore: FAISS) -> None:
    """Persist the FAISS index locally so it survives app restarts."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))


def load_vectorstore() -> FAISS | None:
    """Load an existing FAISS index from disk, or return None if missing/corrupt."""
    if not INDEX_DIR.exists():
        return None

    if not (INDEX_DIR / "index.faiss").exists():
        return None

    if VECTORIZER_FILE.exists():
        emb = get_tfidf_embeddings(reload=True)
    else:
        emb = get_embedding_model()

    try:
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            emb,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        _clear_index_files()
        return None

    if not _index_matches_embeddings(vectorstore):
        _clear_index_files()
        return None

    return vectorstore


def merge_vectorstores(existing: FAISS | None, new_chunks: list[Document]) -> FAISS:
    """Add new PDF chunks — rebuilds index so TF-IDF stays consistent."""
    all_docs = _all_documents(existing, new_chunks)
    _clear_index_files()
    embeddings = _get_embeddings_for_index(all_docs)
    return FAISS.from_documents(all_docs, embeddings)


def search_similar_chunks(vectorstore: FAISS, question: str, k: int = 4) -> list[Document]:
    """Retrieve the top-k text chunks most relevant to the user's question."""
    if not _index_matches_embeddings(vectorstore):
        raise ValueError(
            "Embedding dimension mismatch (index built with different data). "
            "Click **Reset knowledge base**, re-upload PDFs, and **Process PDFs** again."
        )
    return vectorstore.similarity_search(question, k=k)
