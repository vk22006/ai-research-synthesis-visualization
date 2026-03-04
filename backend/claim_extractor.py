"""
claim_extractor.py - Extract the main scientific claim or key insight from an abstract.

Strategy:
1. Split abstract into sentences.
2. Score each sentence by the presence of "claim indicator" verbs/phrases.
3. Return the highest-scoring sentence as the key claim.
4. Fall back to the first sentence if no indicators match.
"""
from __future__ import annotations
import re

# Phrases that commonly introduce scientific claims / contributions
_CLAIM_INDICATORS = [
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe introduce\b",
    r"\bwe demonstrate\b",
    r"\bwe show\b",
    r"\bwe achieve\b",
    r"\bwe develop\b",
    r"\bwe prove\b",
    r"\bthis paper proposes\b",
    r"\bthis paper presents\b",
    r"\bthis work proposes\b",
    r"\bthis work presents\b",
    r"\bin this paper\b",
    r"\bin this work\b",
    r"\bour method\b",
    r"\bour approach\b",
    r"\bour model\b",
    r"\bour results show\b",
    r"\bstate.of.the.art\b",
    r"\bsignificantly outperform\b",
    r"\bnovel\b",
    r"\bnew approach\b",
    r"\bfirst\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _CLAIM_INDICATORS]


def _sentence_score(sentence: str) -> int:
    """Return how many claim indicators appear in *sentence*."""
    return sum(1 for pattern in _COMPILED if pattern.search(sentence))


def extract_claim(text: str) -> str:
    """
    Return the sentence from *text* most likely to contain the main scientific claim.
    """
    if not text or not text.strip():
        return "No claim extracted."

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return text[:200].strip()

    # Score each sentence
    scored = [(s, _sentence_score(s)) for s in sentences]
    best_sentence, best_score = max(scored, key=lambda x: x[1])

    if best_score == 0:
        # No indicators found – fall back to first sentence
        return sentences[0]

    return best_sentence