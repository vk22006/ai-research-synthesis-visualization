"""
summarize.py  Fast, reliable summarization for research paper abstracts.

Strategy:
  1. EXTRACTIVE (default, instant) : selects the best sentences using TF-IDF scoring.
     No model loading, no internet required, works immediately.
  2. ABSTRACTIVE (optional) - uses t5-small if it's already loaded into memory.
     Only attempted if the pipeline is already initialized (e.g., warm cache).
     Enable by setting ENABLE_ABSTRACTIVE=True if you want slower but better summaries.

For a hackathon demo, extractive is recommended as it is instant and deterministic.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────────
# Set to True to attempt loading t5-small for abstractive summaries.
# This adds ~30-60s on first load. Extractive fallback always applies.
ENABLE_ABSTRACTIVE = False

_pipeline = None
_pipeline_attempted = False


def _get_pipeline():
    """Lazily load t5-small only if ENABLE_ABSTRACTIVE is True."""
    global _pipeline, _pipeline_attempted
    if not ENABLE_ABSTRACTIVE or _pipeline_attempted:
        return _pipeline
    _pipeline_attempted = True
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "text2text-generation",
            model="t5-small",
            device=-1,
            framework="pt",
        )
        print("[summarize] t5-small loaded successfully.")
    except Exception as exc:
        print(f"[summarize] Could not load t5-small: {exc}. Using extractive only.")
        _pipeline = None
    return _pipeline


# ── Extractive summariser (TF-IDF sentence scoring) ───────────────────────────
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "with", "by", "from", "as", "it", "its",
    "this", "that", "these", "those", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "we", "our", "us", "they", "their", "which", "also", "such",
}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if w.lower() not in _STOP_WORDS]


def _tfidf_score(sentence: str, doc_tokens: list[str], num_sentences: int) -> float:
    """Score a sentence by summing TF-IDF weights of its tokens."""
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 0.0
    tf = Counter(sent_tokens)
    # Simple IDF approximation: log(N / (1 + count_in_doc))
    doc_freq = Counter(doc_tokens)
    score = sum(
        (tf[t] / len(sent_tokens)) * math.log((num_sentences + 1) / (doc_freq.get(t, 0) + 1))
        for t in tf
    )
    return score


def _extractive_summary(text: str, n: int = 3) -> str:
    """Return the n highest-scored sentences from text as the summary."""
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= n:
        return " ".join(sentences)

    doc_tokens = _tokenize(text)
    scored = [
        (s, _tfidf_score(s, doc_tokens, len(sentences)))
        for s in sentences
    ]
    # Pick top-n by score, preserving original order
    top_indices = sorted(
        range(len(scored)), key=lambda i: scored[i][1], reverse=True
    )[:n]
    top_indices.sort()  # restore reading order
    return " ".join(sentences[i] for i in top_indices)


# ── Public API ─────────────────────────────────────────────────────────────────
def summarize_text(text: str, max_new_tokens: int = 80) -> str:
    """
    Return a summary of *text*.

    Uses TF-IDF extractive summarization (instant) by default.
    Falls back gracefully on any error.
    """
    if not text or not text.strip():
        return "No abstract available."

    try:
        # Try abstractive only if enabled and pipeline is available
        pipe = _get_pipeline()
        if pipe is not None:
            prompt = f"summarize: {text[:800]}"
            result = pipe(prompt, max_new_tokens=max_new_tokens,
                          min_new_tokens=15, do_sample=False, truncation=True)
            summary = result[0].get("generated_text", "").strip()
            if summary:
                return summary
    except Exception as exc:
        print(f"[summarize] Abstractive failed: {exc}")

    # Default: fast extractive
    return _extractive_summary(text, n=3)