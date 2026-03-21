"""Familiarity scoring: the ART match function.

This IS the match function |I ∩ w_J| / |I| from Adaptive Resonance
Theory, generalized to multi-component scoring. In classical ART, the
match is a single set intersection ratio. Here, it's a weighted sum of
five components that together measure "how well does this signal fit
this category (worker context)?"

The result is compared against the worker's vigilance criterion ρ
(adaptive_threshold) to determine resonance:
  - score ≥ ρ_high → full resonance (rag_indexed)
  - ρ ≤ score < ρ_high → partial resonance (context_summarized)
  - score < ρ → mismatch (signal propagates to next category)

DETERMINISTIC. Same signal + same context state = same score.
No randomness. No LLM calls. Pure math.

Supports two modes:
- Exact: set intersection for keyword overlap (always correct)
- Fast: bloom filter estimation for keyword overlap (O(1), approximate)

Components (the generalized match function):
  - embedding_similarity: cosine(signal.embedding, context.centroid)
  - keyword_overlap: jaccard(signal.terms, context.terms) or bloom estimate
  - source_affinity: learned weight per source-context pair
  - temporal_proximity: recency of related signals in this context
  - author_affinity: how often this author's signals land in this context
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal
from stigmergy.services.vector_store import cosine_similarity
from stigmergy.structures.bloom import CountingBloomFilter


@dataclass
class FamiliarityWeights:
    """Weights for familiarity score components. Must sum to positive value.

    signal_credibility implements Crawford-Sobel costly signaling theory:
    signals that cost the sender something (commits with diffs, merged PRs)
    carry more information than cheap talk (comments, status updates).
    N* = ⌈-1/2 + 1/2√(1 + 2/b)⌉ — at bias b ≥ 1/4, cheap talk conveys
    zero information (babbling equilibrium). We don't need the exact math;
    the insight is: weight signals by their epistemic cost.
    """

    embedding_similarity: float = 0.35
    keyword_overlap: float = 0.2
    source_affinity: float = 0.15
    temporal_proximity: float = 0.1
    author_affinity: float = 0.1
    signal_credibility: float = 0.1
    sequence_similarity: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "embedding_similarity": self.embedding_similarity,
            "keyword_overlap": self.keyword_overlap,
            "source_affinity": self.source_affinity,
            "temporal_proximity": self.temporal_proximity,
            "author_affinity": self.author_affinity,
            "signal_credibility": self.signal_credibility,
            "sequence_similarity": self.sequence_similarity,
        }


def _embedding_similarity(signal: Signal, centroid: list[float] | None, strategy: str = "semantic") -> float:
    """Cosine similarity between signal embedding and context centroid."""
    if centroid is None:
        return 0.0
    signal_vec = signal.embeddings.get(strategy)
    if signal_vec is None:
        return 0.0
    return max(0.0, cosine_similarity(np.array(signal_vec), np.array(centroid)))


def _keyword_overlap(signal_terms: set[str], context_terms: set[str]) -> float:
    """Jaccard similarity between signal terms and context terms."""
    if not signal_terms or not context_terms:
        return 0.0
    intersection = signal_terms & context_terms
    union = signal_terms | context_terms
    return len(intersection) / len(union)


def _keyword_overlap_bloom(signal_bloom: CountingBloomFilter,
                           context_bloom: CountingBloomFilter) -> float:
    """Fast approximate Jaccard via bloom filter. O(size) but vectorized with numpy."""
    return signal_bloom.estimate_jaccard(context_bloom)


def _temporal_proximity(signal_timestamp: datetime, context_last_signal: datetime,
                        half_life_hours: float = 24.0) -> float:
    """Recency score. Exponential decay with configurable half-life."""
    elapsed = abs((signal_timestamp - context_last_signal).total_seconds())
    decay_rate = math.log(2) / (half_life_hours * 3600)
    return math.exp(-decay_rate * elapsed)


# ── Crawford-Sobel credibility ─────────────────────────────────

# Costly signals: the sender paid a real cost to produce them.
# These carry genuine information (Crawford-Sobel: low bias b → high N*).
_COSTLY_EVENT_TYPES = frozenset({
    "pull_request",      # Code was written, reviewed, tested
    "push",              # Commits pushed — actual code changes
    "release",           # Deployment — real operational cost
    "merge",             # PR was approved and merged
    "review",            # Time spent reviewing code
})

# Cheap talk: low-cost signals that may carry bias.
# Not worthless, but should be discounted (high b → babbling equilibrium).
_CHEAP_TALK_EVENT_TYPES = frozenset({
    "comment",           # Easy to produce, may be noise
    "reaction",          # Almost zero cost
    "status_update",     # Declarative, not verifiable
    "label_change",      # Organizational, not substantive
})


def _signal_credibility(signal: Signal) -> float:
    """Crawford-Sobel credibility score based on signal cost.

    Costly signals (code changes with diffs) score high.
    Cheap talk (comments, reactions) scores low.
    Unknown types get a moderate default.

    The score also considers diff size for code signals:
    a 500-line PR carries more information than a 2-line typo fix.
    """
    event_type = signal.metadata.get("event_type", "")
    source = signal.source

    # Base credibility from event type
    if event_type in _COSTLY_EVENT_TYPES:
        base = 0.8
    elif event_type in _CHEAP_TALK_EVENT_TYPES:
        base = 0.2
    elif source == "github":
        base = 0.6  # GitHub signals are generally code-related
    elif source == "linear":
        base = 0.4  # Issue tracker — moderate cost
    else:
        base = 0.5  # Unknown — moderate default

    # Diff size bonus for code signals: larger diffs = more costly to produce.
    # Logarithmic scaling — 10 lines → +0.05, 100 lines → +0.1, 1000 → +0.15
    additions = signal.metadata.get("additions", 0)
    deletions = signal.metadata.get("deletions", 0)
    diff_lines = (additions or 0) + (deletions or 0)
    if diff_lines > 0:
        diff_bonus = min(0.2, 0.05 * math.log10(max(diff_lines, 1)))
        base = min(1.0, base + diff_bonus)

    return base


# ── Sequence similarity ────────────────────────────────────────

def extract_step_sequence(signal: Signal) -> list[str] | None:
    """Defensively extract a step sequence from signal metadata.

    Returns None if steps key is missing, not a list, or contains
    non-string/empty elements. Never raises on malformed metadata.
    """
    metadata = signal.metadata
    if not isinstance(metadata, dict):
        return None
    steps = metadata.get("steps")
    if not isinstance(steps, list):
        return None
    if not steps:
        return None
    for s in steps:
        if not isinstance(s, str) or not s:
            return None
    return steps


def extract_ngrams(steps: list[str], max_n: int = 3) -> Counter:
    """Extract all n-grams of size 1..max_n from a step sequence.

    O(n * max_n) where n = len(steps).
    """
    ngrams: Counter = Counter()
    n = len(steps)
    for size in range(1, min(max_n, n) + 1):
        for i in range(n - size + 1):
            ngrams[tuple(steps[i:i + size])] += 1
    return ngrams


def compute_sequence_similarity(
    signal_ngrams: Counter, context_ngrams: Counter
) -> float:
    """Multiset Jaccard similarity between two n-gram counters.

    sum(min(A[k], B[k])) / sum(max(A[k], B[k])) over all keys.
    Returns 0.0 when both are empty (division-by-zero guard).
    Pure, deterministic, symmetric.
    """
    all_keys = set(signal_ngrams) | set(context_ngrams)
    if not all_keys:
        return 0.0
    min_sum = 0
    max_sum = 0
    for k in all_keys:
        a = signal_ngrams.get(k, 0)
        b = context_ngrams.get(k, 0)
        min_sum += min(a, b)
        max_sum += max(a, b)
    if max_sum == 0:
        return 0.0
    return min_sum / max_sum


def _sequence_similarity(signal: Signal, context: Context, max_n: int = 3) -> float:
    """Score sequence similarity between a signal and a context.

    Returns 0.0 for signals without steps (backward compat).
    Pure function — does not mutate context.
    """
    steps = extract_step_sequence(signal)
    if steps is None:
        return 0.0
    if not context.sequence_ngrams:
        return 0.0
    signal_ngrams = extract_ngrams(steps, max_n)
    return compute_sequence_similarity(signal_ngrams, context.sequence_ngrams)


def familiarity(signal: Signal, context: Context,
                weights: FamiliarityWeights | None = None,
                centroid: list[float] | None = None,
                strategy: str = "semantic",
                signal_bloom: CountingBloomFilter | None = None) -> float:
    """ART match function: |I ∩ w_J| / |I| generalized to multi-component scoring.

    Pure function. Deterministic. No side effects.

    The returned score is compared against the worker's vigilance criterion ρ
    (adaptive_threshold) to determine the resonance state. This is the core
    routing decision in the mesh — ART's match-based learning guarantee
    depends on this function being deterministic.

    Args:
        signal: The incoming signal (ART input pattern I).
        context: The context to score against (ART category weight w_J).
        weights: Component weights. Defaults to FamiliarityWeights().
        centroid: Pre-computed centroid (avoids async call). If None, embedding
                  similarity is 0.
        strategy: Which embedding strategy to use for similarity.
        signal_bloom: Pre-built bloom filter for signal terms. If provided,
                      uses fast bloom estimation for keyword overlap instead
                      of exact set intersection.

    Returns:
        Float in [0, 1]. Higher = stronger match (closer to resonance).
    """
    w = weights or FamiliarityWeights()

    # Choose keyword overlap method: bloom (fast) or exact
    if signal_bloom is not None and context.term_bloom.count > 0:
        kw_overlap = _keyword_overlap_bloom(signal_bloom, context.term_bloom)
    else:
        kw_overlap = _keyword_overlap(signal.terms, context.terms)

    components = {
        "embedding_similarity": _embedding_similarity(signal, centroid, strategy),
        "keyword_overlap": kw_overlap,
        "source_affinity": context.source_affinity(signal.source),
        "temporal_proximity": _temporal_proximity(signal.timestamp, context.last_signal),
        "author_affinity": context.author_affinity(signal.author),
        "signal_credibility": _signal_credibility(signal),
        "sequence_similarity": _sequence_similarity(signal, context),
    }

    w_dict = w.as_dict()
    total_weight = sum(w_dict.values())
    if total_weight == 0:
        return 0.0

    score = sum(w_dict[k] * components[k] for k in components) / total_weight
    return max(0.0, min(1.0, score))
