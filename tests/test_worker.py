"""Tests for WorkerNode — adaptive threshold, receive, learning modes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from stigmergy.mesh.worker import ReceiveResult, WorkerNode
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource


def _make_context(*, signal_count: int = 0, capacity: int = 200, **kwargs) -> Context:
    ctx = Context(capacity=capacity, **kwargs)
    ctx.signal_count = signal_count
    return ctx


def _make_signal(content: str = "test signal", **kwargs) -> Signal:
    defaults = dict(
        content=content,
        source=SignalSource.GITHUB,
        channel="test/repo",
        author="test.user",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Signal(**defaults)


# ── Adaptive Threshold ──────────────────────────────────────────


class TestAdaptiveThreshold:
    def test_empty_worker_uses_cold_start_threshold(self):
        worker = WorkerNode(_make_context(signal_count=0), base_threshold=0.15)
        # Cold start: 0 signals → 25% of base threshold (0.15 * 0.25 = 0.0375)
        assert worker.adaptive_threshold == pytest.approx(0.0375)

    def test_full_worker_uses_max_threshold(self):
        worker = WorkerNode(
            _make_context(signal_count=200, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
        )
        assert worker.adaptive_threshold == pytest.approx(0.8)

    def test_half_full_raises_threshold(self):
        worker = WorkerNode(
            _make_context(signal_count=100, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
            threshold_curve=2.0,
        )
        # fullness = 0.5, curve=2 → 0.15 + 0.65 * 0.25 = 0.3125
        assert worker.adaptive_threshold == pytest.approx(0.3125)

    def test_threshold_curve_linear(self):
        worker = WorkerNode(
            _make_context(signal_count=100, capacity=200),
            base_threshold=0.1,
            max_threshold=0.9,
            threshold_curve=1.0,
        )
        # fullness = 0.5, curve=1 → 0.1 + 0.8 * 0.5 = 0.5
        assert worker.adaptive_threshold == pytest.approx(0.5)

    def test_over_capacity_clamps_to_max(self):
        worker = WorkerNode(
            _make_context(signal_count=300, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
        )
        # fullness clamped to 1.0
        assert worker.adaptive_threshold == pytest.approx(0.8)

    def test_custom_capacity(self):
        worker = WorkerNode(
            _make_context(signal_count=50, capacity=100),
            base_threshold=0.1,
            max_threshold=0.9,
            threshold_curve=1.0,
        )
        # fullness = 0.5, curve=1 → 0.1 + 0.8 * 0.5 = 0.5
        assert worker.adaptive_threshold == pytest.approx(0.5)


# ── Fullness ────────────────────────────────────────────────────


class TestFullness:
    def test_empty_worker(self):
        worker = WorkerNode(_make_context(signal_count=0))
        assert worker.fullness == 0.0
        assert not worker.is_full

    def test_full_worker(self):
        worker = WorkerNode(_make_context(signal_count=200, capacity=200))
        assert worker.fullness == 1.0
        assert worker.is_full

    def test_partial_fill(self):
        worker = WorkerNode(_make_context(signal_count=50, capacity=200))
        assert worker.fullness == pytest.approx(0.25)


# ── Receive ─────────────────────────────────────────────────────


class TestReceive:
    def test_accept_above_threshold(self):
        worker = WorkerNode(
            _make_context(signal_count=0),
            base_threshold=0.1,
        )
        signal = _make_signal()
        # Empty worker: threshold=0.1, score=0.5 → accept
        result = asyncio.run(worker.receive(signal, 0.5))
        assert result.accepted
        assert result.score == 0.5

    def test_reject_below_threshold(self):
        worker = WorkerNode(
            _make_context(signal_count=190, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
        )
        signal = _make_signal()
        # Nearly full: threshold ≈ 0.74, score=0.3 → reject
        result = asyncio.run(worker.receive(signal, 0.3))
        assert not result.accepted
        assert result.learning_mode == "weight_shifted"

    def test_rag_indexed_for_high_score(self):
        worker = WorkerNode(
            _make_context(signal_count=0),
            base_threshold=0.1,
            high_relevance_offset=0.2,
        )
        signal = _make_signal()
        # threshold=0.1, high_relevance=0.3, score=0.8 → rag_indexed
        result = asyncio.run(worker.receive(signal, 0.8))
        assert result.accepted
        assert result.learning_mode == "rag_indexed"

    def test_context_summarized_for_medium_score(self):
        worker = WorkerNode(
            _make_context(signal_count=0),
            base_threshold=0.1,
            high_relevance_offset=0.5,
        )
        signal = _make_signal()
        # threshold=0.1, high_relevance=0.6, score=0.3 → context_summarized
        result = asyncio.run(worker.receive(signal, 0.3))
        assert result.accepted
        assert result.learning_mode == "context_summarized"

    def test_accept_increments_signal_count(self):
        ctx = _make_context(signal_count=0)
        worker = WorkerNode(ctx, base_threshold=0.1)
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.5))
        assert ctx.signal_count == 1

    def test_context_summarize_updates_terms(self):
        ctx = _make_context(signal_count=0)
        worker = WorkerNode(ctx, base_threshold=0.1, high_relevance_offset=0.8)
        signal = _make_signal(content="pricing engine cache invalidation")
        asyncio.run(worker.receive(signal, 0.3))
        assert "pricing" in ctx.terms
        assert "engine" in ctx.terms

    def test_weight_shift_does_not_nudge_source(self):
        """Weight-shift (reject) must NOT increment source_counts.

        Fix #10: rejected signals were incorrectly incrementing source_counts,
        biasing source affinity toward sources that produce low-relevance signals.
        """
        ctx = _make_context(signal_count=190, capacity=200)
        worker = WorkerNode(ctx, base_threshold=0.15, max_threshold=0.8)
        signal = _make_signal(source=SignalSource.SLACK)
        asyncio.run(worker.receive(signal, 0.05))
        assert ctx.source_counts.get("slack", 0) == 0

    def test_receive_tracks_stats(self):
        worker = WorkerNode(_make_context(signal_count=0), base_threshold=0.1)
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.5))
        assert worker.signals_received == 1
        assert worker.signals_accepted == 1
        assert worker.signals_forwarded == 0

    def test_reject_tracks_forwarded(self):
        worker = WorkerNode(
            _make_context(signal_count=200, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
        )
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.1))
        assert worker.signals_received == 1
        assert worker.signals_accepted == 0
        assert worker.signals_forwarded == 1


# ── Score Signal ────────────────────────────────────────────────


class TestScoreSignal:
    def test_score_uses_familiarity(self):
        from stigmergy.core.familiarity import FamiliarityWeights

        ctx = _make_context()
        ctx.terms = {"bug", "error", "fix", "deploy"}
        ctx.signal_count = 50
        ctx.source_counts = {"github": 40, "slack": 10}
        ctx.author_counts = {"test.user": 20}
        worker = WorkerNode(ctx)

        signal = _make_signal(content="bug fix deployed to production")
        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.5,
            source_affinity=0.2,
            temporal_proximity=0.2,
            author_affinity=0.1,
        )
        score = worker.score_signal(signal, weights)
        assert 0.0 <= score <= 1.0
        assert score > 0.1  # should have decent keyword overlap


# ── Label ───────────────────────────────────────────────────────


class TestLabel:
    def test_empty_worker_uses_id(self):
        worker = WorkerNode(_make_context())
        assert worker.label.startswith("worker-")

    def test_worker_with_terms(self):
        ctx = _make_context()
        ctx.terms = {"pricing", "engine", "cache", "availability"}
        worker = WorkerNode(ctx)
        label = worker.label
        assert "/" in label
        assert len(label.split("/")) <= 3


# ── Position ───────────────────────────────────────────────────


class TestPosition:
    def test_empty_worker_has_position(self):
        worker = WorkerNode(_make_context())
        pos = worker.position
        assert pos.worker_id == worker.id
        assert pos.signal_count == 0
        assert pos.bloom_digest.shape == (128,)

    def test_position_reflects_terms(self):
        ctx = _make_context()
        ctx.terms = {"pricing", "engine", "cache", "invalidation"}
        ctx.term_bloom.add_many(ctx.terms)
        ctx.signal_count = 10
        worker = WorkerNode(ctx)
        pos = worker.position
        assert pos.signal_count == 10
        assert pos.bloom_digest.sum() > 0
        assert len(pos.top_terms) > 0

    def test_position_cached(self):
        ctx = _make_context()
        ctx.terms = {"pricing", "engine"}
        ctx.term_bloom.add_many(ctx.terms)
        worker = WorkerNode(ctx)
        pos1 = worker.position
        pos2 = worker.position
        assert pos1 is pos2  # same object — cached

    def test_position_invalidated_on_accept(self):
        ctx = _make_context()
        worker = WorkerNode(ctx, base_threshold=0.0)
        pos_before = worker.position
        signal = _make_signal(content="pricing engine cache invalidation strategy")
        asyncio.run(worker.receive(signal, 0.5))
        pos_after = worker.position
        # Position should be recomputed (different object)
        assert pos_before is not pos_after
        # New position should reflect the accepted signal's terms
        assert pos_after.signal_count == 1
        assert pos_after.bloom_digest.sum() > 0

    def test_position_not_invalidated_on_reject(self):
        ctx = _make_context(signal_count=200, capacity=200)
        worker = WorkerNode(ctx, base_threshold=0.15, max_threshold=0.8)
        pos_before = worker.position
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.05))  # rejected
        pos_after = worker.position
        # Should be same cached object since no accept
        assert pos_before is pos_after


# ── Rolling Average Familiarity ────────────────────────────────


class TestRollingAvgFamiliarity:
    def test_initial_value(self):
        worker = WorkerNode(_make_context())
        assert worker.rolling_avg_familiarity == 0.0

    def test_first_accept_sets_avg(self):
        worker = WorkerNode(_make_context(), base_threshold=0.0)
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.6))
        assert worker.rolling_avg_familiarity == pytest.approx(0.6)

    def test_ema_tracks_trend(self):
        worker = WorkerNode(_make_context(), base_threshold=0.0)
        # Accept several signals with increasing scores
        for score in [0.3, 0.5, 0.7, 0.9]:
            signal = _make_signal(content=f"signal score {score}")
            asyncio.run(worker.receive(signal, score))
        # EMA should be between first and last score, trending up
        assert 0.3 < worker.rolling_avg_familiarity < 0.9

    def test_reject_doesnt_update_avg(self):
        worker = WorkerNode(
            _make_context(signal_count=200, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
        )
        initial_avg = worker.rolling_avg_familiarity
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.05))  # rejected
        assert worker.rolling_avg_familiarity == initial_avg


# ── Recenter Counter ───────────────────────────────────────────


class TestRecenterCounter:
    def test_initial_value(self):
        worker = WorkerNode(_make_context())
        assert worker.recenter_counter == 0

    def test_increments_on_accept(self):
        worker = WorkerNode(_make_context(), base_threshold=0.0)
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.5))
        assert worker.recenter_counter == 1

    def test_no_increment_on_reject(self):
        worker = WorkerNode(
            _make_context(signal_count=200, capacity=200),
            base_threshold=0.15,
            max_threshold=0.8,
        )
        signal = _make_signal()
        asyncio.run(worker.receive(signal, 0.05))
        assert worker.recenter_counter == 0
