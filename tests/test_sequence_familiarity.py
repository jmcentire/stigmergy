"""Tests for sequence-aware familiarity scoring.

Validates:
- N-gram extraction from step sequences
- Multiset Jaccard similarity computation
- Context sequence representation updates
- Integration with familiarity scoring pipeline
- Backward compatibility: signals without steps route normally
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone

import pytest

from stigmergy.core.familiarity import (
    FamiliarityWeights,
    compute_sequence_similarity,
    extract_ngrams,
    extract_step_sequence,
    familiarity,
    _sequence_similarity,
)
from stigmergy.mesh.worker import WorkerNode
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource


# ── Helpers ────────────────────────────────────────────────────


def _make_signal(content: str = "test", metadata: dict | None = None, **kwargs) -> Signal:
    defaults = {
        "source": SignalSource.SLACK,
        "channel": "#test",
        "author": "test.user",
        "timestamp": datetime.now(timezone.utc),
        "metadata": metadata or {},
    }
    defaults.update(kwargs)
    return Signal(content=content, **defaults)


def _make_context(**kwargs) -> Context:
    return Context(**kwargs)


# ── extract_step_sequence ──────────────────────────────────────


class TestExtractStepSequence:
    def test_valid_steps(self):
        signal = _make_signal(metadata={"steps": ["login", "browse", "checkout"]})
        assert extract_step_sequence(signal) == ["login", "browse", "checkout"]

    def test_no_steps_key(self):
        signal = _make_signal(metadata={"event_type": "push"})
        assert extract_step_sequence(signal) is None

    def test_empty_metadata(self):
        signal = _make_signal(metadata={})
        assert extract_step_sequence(signal) is None

    def test_steps_not_a_list(self):
        signal = _make_signal(metadata={"steps": "login,browse"})
        assert extract_step_sequence(signal) is None

    def test_steps_empty_list(self):
        signal = _make_signal(metadata={"steps": []})
        assert extract_step_sequence(signal) is None

    def test_steps_with_non_string_element(self):
        signal = _make_signal(metadata={"steps": ["login", 42, "checkout"]})
        assert extract_step_sequence(signal) is None

    def test_steps_with_empty_string_element(self):
        signal = _make_signal(metadata={"steps": ["login", "", "checkout"]})
        assert extract_step_sequence(signal) is None

    def test_single_step(self):
        signal = _make_signal(metadata={"steps": ["login"]})
        assert extract_step_sequence(signal) == ["login"]


# ── extract_ngrams ─────────────────────────────────────────────


class TestExtractNgrams:
    def test_unigrams(self):
        result = extract_ngrams(["a", "b", "c"], max_n=1)
        assert result == Counter({("a",): 1, ("b",): 1, ("c",): 1})

    def test_bigrams(self):
        result = extract_ngrams(["a", "b", "c"], max_n=2)
        assert result[("a",)] == 1
        assert result[("b",)] == 1
        assert result[("c",)] == 1
        assert result[("a", "b")] == 1
        assert result[("b", "c")] == 1
        assert ("a", "c") not in result

    def test_trigrams(self):
        result = extract_ngrams(["a", "b", "c", "d"], max_n=3)
        # Unigrams: a, b, c, d
        assert result[("a",)] == 1
        assert result[("d",)] == 1
        # Bigrams: ab, bc, cd
        assert result[("a", "b")] == 1
        assert result[("c", "d")] == 1
        # Trigrams: abc, bcd
        assert result[("a", "b", "c")] == 1
        assert result[("b", "c", "d")] == 1

    def test_empty_sequence(self):
        result = extract_ngrams([], max_n=3)
        assert len(result) == 0

    def test_single_element(self):
        result = extract_ngrams(["login"], max_n=3)
        assert result == Counter({("login",): 1})

    def test_repeated_steps(self):
        result = extract_ngrams(["a", "b", "a", "b"], max_n=2)
        assert result[("a",)] == 2
        assert result[("b",)] == 2
        assert result[("a", "b")] == 2
        assert result[("b", "a")] == 1

    def test_max_n_larger_than_sequence(self):
        result = extract_ngrams(["a", "b"], max_n=5)
        # Should only produce unigrams and bigrams
        assert result == Counter({("a",): 1, ("b",): 1, ("a", "b"): 1})


# ── compute_sequence_similarity ────────────────────────────────


class TestComputeSequenceSimilarity:
    def test_identical_counters(self):
        c = Counter({("a",): 2, ("b",): 1})
        assert compute_sequence_similarity(c, c) == pytest.approx(1.0)

    def test_both_empty(self):
        assert compute_sequence_similarity(Counter(), Counter()) == pytest.approx(0.0)

    def test_one_empty(self):
        c = Counter({("a",): 1})
        assert compute_sequence_similarity(c, Counter()) == pytest.approx(0.0)
        assert compute_sequence_similarity(Counter(), c) == pytest.approx(0.0)

    def test_disjoint(self):
        a = Counter({("x",): 1, ("y",): 1})
        b = Counter({("p",): 1, ("q",): 1})
        assert compute_sequence_similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = Counter({("a",): 2, ("b",): 1})
        b = Counter({("a",): 1, ("c",): 1})
        # min_sum = min(2,1) + min(1,0) + min(0,1) = 1
        # max_sum = max(2,1) + max(1,0) + max(0,1) = 2 + 1 + 1 = 4
        assert compute_sequence_similarity(a, b) == pytest.approx(1 / 4)

    def test_symmetric(self):
        a = Counter({("a",): 3, ("b",): 1})
        b = Counter({("a",): 1, ("b",): 2, ("c",): 1})
        assert compute_sequence_similarity(a, b) == pytest.approx(
            compute_sequence_similarity(b, a)
        )

    def test_subset(self):
        a = Counter({("a",): 1})
        b = Counter({("a",): 1, ("b",): 1})
        # min_sum = 1 + 0 = 1, max_sum = 1 + 1 = 2
        assert compute_sequence_similarity(a, b) == pytest.approx(0.5)


# ── _sequence_similarity (integration) ─────────────────────────


class TestSequenceSimilarityScoring:
    def test_returns_zero_without_steps(self):
        """Backward compat: signals without steps score 0.0."""
        signal = _make_signal(metadata={"event_type": "push"})
        ctx = _make_context()
        ctx.sequence_ngrams = Counter({("login",): 1})
        assert _sequence_similarity(signal, ctx) == pytest.approx(0.0)

    def test_returns_zero_with_empty_context(self):
        """No accumulated sequences in context -> 0.0."""
        signal = _make_signal(metadata={"steps": ["login", "checkout"]})
        ctx = _make_context()
        assert _sequence_similarity(signal, ctx) == pytest.approx(0.0)

    def test_returns_one_for_identical_sequence(self):
        """Exact match should score 1.0."""
        steps = ["login", "browse", "add_cart", "checkout"]
        signal = _make_signal(metadata={"steps": steps})
        ctx = _make_context()
        ctx.update_sequences(steps)
        assert _sequence_similarity(signal, ctx) == pytest.approx(1.0)

    def test_intermediate_score_for_partial_match(self):
        """Partial overlap should produce intermediate score."""
        ctx = _make_context()
        ctx.update_sequences(["login", "browse", "add_cart", "checkout"])
        # Similar but not identical
        signal = _make_signal(
            metadata={"steps": ["login", "browse", "search", "checkout"]}
        )
        score = _sequence_similarity(signal, ctx)
        assert 0.0 < score < 1.0

    def test_completely_different_sequences(self):
        """No overlap should score close to 0.0."""
        ctx = _make_context()
        ctx.update_sequences(["alpha", "beta", "gamma"])
        signal = _make_signal(metadata={"steps": ["x", "y", "z"]})
        score = _sequence_similarity(signal, ctx)
        assert score == pytest.approx(0.0)


# ── Context.update_sequences ──────────────────────────────────


class TestContextSequenceUpdate:
    def test_appends_to_step_sequences(self):
        ctx = _make_context()
        assert ctx.step_sequences == []
        ctx.update_sequences(["a", "b", "c"])
        assert len(ctx.step_sequences) == 1
        assert ctx.step_sequences[0] == ["a", "b", "c"]

    def test_updates_ngrams_incrementally(self):
        ctx = _make_context()
        ctx.update_sequences(["a", "b"])
        # Unigrams: a, b. Bigram: (a,b)
        assert ctx.sequence_ngrams[("a",)] == 1
        assert ctx.sequence_ngrams[("b",)] == 1
        assert ctx.sequence_ngrams[("a", "b")] == 1

        # Add another sequence with overlap
        ctx.update_sequences(["a", "c"])
        assert ctx.sequence_ngrams[("a",)] == 2  # incremented
        assert ctx.sequence_ngrams[("c",)] == 1
        assert ctx.sequence_ngrams[("a", "c")] == 1

    def test_multiple_sequences_accumulate(self):
        ctx = _make_context()
        for _ in range(3):
            ctx.update_sequences(["login", "checkout"])
        assert len(ctx.step_sequences) == 3
        assert ctx.sequence_ngrams[("login",)] == 3
        assert ctx.sequence_ngrams[("checkout",)] == 3
        assert ctx.sequence_ngrams[("login", "checkout")] == 3

    def test_stores_copy_of_steps(self):
        """Ensure context stores a copy, not a reference."""
        ctx = _make_context()
        steps = ["a", "b"]
        ctx.update_sequences(steps)
        steps.append("c")  # mutate original
        assert ctx.step_sequences[0] == ["a", "b"]  # copy is unchanged

    def test_default_fields_on_fresh_context(self):
        ctx = _make_context()
        assert ctx.step_sequences == []
        assert isinstance(ctx.sequence_ngrams, Counter)
        assert len(ctx.sequence_ngrams) == 0


# ── Familiarity integration ────────────────────────────────────


class TestFamiliarityIntegration:
    def test_default_weight_zero_no_effect(self):
        """Default sequence_similarity=0.0: existing signals route the same."""
        ctx = _make_context()
        ctx.terms = {"booking", "sync"}
        signal = _make_signal("booking sync error")

        weights = FamiliarityWeights()  # default: sequence_similarity=0.0
        score = familiarity(signal, ctx, weights=weights)
        assert 0.0 <= score <= 1.0

    def test_nonzero_weight_boosts_matching_sequence(self):
        """When weight is nonzero and sequences match, score increases."""
        ctx = _make_context()
        ctx.terms = {"test"}
        steps = ["login", "browse", "checkout"]
        ctx.update_sequences(steps)

        signal_with_steps = _make_signal(
            "test",
            metadata={"steps": steps},
        )
        signal_without_steps = _make_signal("test")

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
            sequence_similarity=1.0,
        )

        score_with = familiarity(signal_with_steps, ctx, weights=weights)
        score_without = familiarity(signal_without_steps, ctx, weights=weights)

        assert score_with == pytest.approx(1.0)
        assert score_without == pytest.approx(0.0)

    def test_backward_compat_no_steps_routes_normally(self):
        """Signals without steps still route based on other components."""
        ctx = _make_context()
        ctx.terms = {"deploy", "release", "production"}
        ctx.source_counts = {"github": 50}
        ctx.signal_count = 50

        signal = _make_signal(
            "deploy release production rollback",
            source="github",
        )

        # Only keyword and source weights
        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.5,
            source_affinity=0.5,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
            sequence_similarity=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score > 0.5

    def test_sequence_only_weight_returns_in_range(self):
        """Score always in [0, 1] even with only sequence weight."""
        ctx = _make_context()
        ctx.update_sequences(["a", "b", "c"])
        signal = _make_signal(metadata={"steps": ["x", "y"]})
        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
            sequence_similarity=1.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert 0.0 <= score <= 1.0


# ── Worker learning integration ────────────────────────────────


class TestWorkerSequenceLearning:
    def test_rag_indexed_updates_context_sequences(self):
        """When a signal with steps is accepted at rag_indexed level,
        context.step_sequences should be updated."""
        ctx = _make_context(capacity=200)
        # Give the context some signals so cold-start doesn't interfere
        ctx.signal_count = 50
        ctx.terms = {"checkout", "flow"}
        worker = WorkerNode(
            ctx,
            base_threshold=0.0,
            max_threshold=0.0,
            high_relevance_offset=0.0,
        )
        signal = _make_signal(
            "checkout flow test",
            metadata={"steps": ["login", "browse", "checkout"]},
        )

        # Score high enough for rag_indexed (threshold is 0.0)
        result = asyncio.run(worker.receive(signal, 0.9))
        assert result.learning_mode == "rag_indexed"
        assert len(ctx.step_sequences) == 1
        assert ctx.step_sequences[0] == ["login", "browse", "checkout"]
        assert ctx.sequence_ngrams[("login",)] == 1

    def test_context_summarized_does_not_update_sequences(self):
        """Signals accepted at context_summarized level should NOT
        update sequence representation."""
        ctx = _make_context(capacity=200)
        ctx.signal_count = 50
        worker = WorkerNode(
            ctx,
            base_threshold=0.0,
            max_threshold=0.0,
            high_relevance_offset=0.5,  # high bar for rag_indexed
        )
        signal = _make_signal(
            "test",
            metadata={"steps": ["a", "b", "c"]},
        )

        # Score above threshold but below high_relevance_threshold
        result = asyncio.run(worker.receive(signal, 0.3))
        assert result.learning_mode == "context_summarized"
        assert len(ctx.step_sequences) == 0

    def test_rag_indexed_no_steps_no_update(self):
        """Signals without steps accepted at rag_indexed level should
        not modify sequence tracking."""
        ctx = _make_context(capacity=200)
        ctx.signal_count = 50
        worker = WorkerNode(
            ctx,
            base_threshold=0.0,
            max_threshold=0.0,
            high_relevance_offset=0.0,
        )
        signal = _make_signal("plain signal without steps")

        result = asyncio.run(worker.receive(signal, 0.9))
        assert result.learning_mode == "rag_indexed"
        assert len(ctx.step_sequences) == 0
        assert len(ctx.sequence_ngrams) == 0
