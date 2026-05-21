"""
Store heavy data on D: drive to save C: disk space.
"""

import os
from pathlib import Path

# Root folder on D: for caches and indexes (override with DATA_DRIVE in .env)
DATA_ROOT = Path(os.getenv("DATA_DRIVE", "D:/college-rag-data"))

# HuggingFace / PyTorch / pip caches
CACHE_DIRS = {
    "HF_HOME": DATA_ROOT / "huggingface",
    "TRANSFORMERS_CACHE": DATA_ROOT / "transformers",
    "TORCH_HOME": DATA_ROOT / "torch",
    "SENTENCE_TRANSFORMERS_HOME": DATA_ROOT / "sentence_transformers",
    "PIP_CACHE_DIR": DATA_ROOT / "pip-cache",
}

# FAISS index + TF-IDF vectorizer live here
FAISS_INDEX_DIR = DATA_ROOT / "faiss_index"


def setup_data_paths() -> None:
    """Create D: folders and point environment variables at them."""
    for path in CACHE_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for key, path in CACHE_DIRS.items():
        os.environ[key] = str(path)
