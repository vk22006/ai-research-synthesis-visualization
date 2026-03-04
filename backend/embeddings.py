"""
embeddings.py - Generate sentence embeddings and compute pairwise cosine similarity.

Model: all-MiniLM-L6-v2  (fast, effective for semantic similarity tasks)
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model: SentenceTransformer | None = None  # lazily initialized


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Return the embedding vector for a single piece of text."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def compute_similarity_matrix(texts: list[str]) -> np.ndarray:
    """
    Compute an N×N cosine similarity matrix for a list of texts.

    Returns a 2-D NumPy array where entry [i][j] is the cosine similarity
    between texts[i] and texts[j].
    """
    if not texts:
        return np.array([])

    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix