"""
Lightweight local embeddings using TF-IDF (no PyTorch, no internet).

Works on low-RAM PCs and when SSL blocks HuggingFace / Google downloads.
"""

import pickle
from pathlib import Path

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.paths import FAISS_INDEX_DIR

VECTORIZER_PATH = FAISS_INDEX_DIR / "tfidf_vectorizer.pkl"


class TfidfEmbeddings(Embeddings):
    """Turn text into fixed-size vectors using TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        self._is_fitted = False

    def fit_corpus(self, texts: list[str]) -> None:
        """Learn vocabulary from all PDF chunks (call before building FAISS)."""
        self.vectorizer.fit(texts)
        self._is_fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._is_fitted:
            self.fit_corpus(texts)
        matrix = self.vectorizer.transform(texts).toarray()
        return matrix.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.vectorizer.transform([text]).toarray()
        return vector[0].tolist()

    def save(self, path: Path = VECTORIZER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path = VECTORIZER_PATH) -> None:
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._is_fitted = True


_tfidf_instance: TfidfEmbeddings | None = None


def reset_tfidf_embeddings() -> None:
    """Clear cached vectorizer (use after reset or corrupt index)."""
    global _tfidf_instance
    _tfidf_instance = None


def get_tfidf_embeddings(reload: bool = False) -> TfidfEmbeddings:
    """Reuse one fitted vectorizer across the app session."""
    global _tfidf_instance
    if reload or _tfidf_instance is None:
        _tfidf_instance = TfidfEmbeddings()
        if VECTORIZER_PATH.exists():
            _tfidf_instance.load()
    return _tfidf_instance
