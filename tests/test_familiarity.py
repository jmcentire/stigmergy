"""Tests for familiarity scoring.

Key property: DETERMINISTIC. Same inputs → same outputs, always.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from stigmergy.core.familiarity import FamiliarityWeights, familiarity
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal, SignalSource


def _make_signal(content: str, embeddings: dict | None = None, **kwargs) -> Signal:
    defaults = {
        "source": SignalSource.SLACK,
        "channel": "#test",
        "author": "test.user",
        "timestamp": datetime.now(timezone.utc),
        "embeddings": embeddings or {},
    }
    defaults.update(kwargs)
    return Signal(content=content, **defaults)


def _make_centroid(dims: int = 384) -> list[float]:
    rng = np.random.RandomState(42)
    vec = rng.randn(dims)
    return (vec / np.linalg.norm(vec)).tolist()


class TestDeterminism:
    """Same inputs must always produce same outputs."""

    def test_same_inputs_same_output(self):
        ctx = Context()
        ctx.terms = {"booking", "sync", "guesty"}
        signal = _make_signal("booking sync guesty error")

        score1 = familiarity(signal, ctx)
        score2 = familiarity(signal, ctx)
        assert score1 == score2

    def test_deterministic_across_calls(self):
        """Run 100 times, all identical."""
        ctx = Context()
        ctx.terms = {"test", "signal"}
        signal = _make_signal("test signal content")
        scores = [familiarity(signal, ctx) for _ in range(100)]
        assert len(set(scores)) == 1


class TestComponents:
    """Test individual familiarity components."""

    def test_keyword_overlap_high(self):
        ctx = Context()
        ctx.terms = {"booking", "sync", "guesty", "error", "webhook"}
        signal = _make_signal("booking sync guesty error on webhook")

        # Weight keyword overlap heavily
        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=1.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score > 0.5  # High overlap

    def test_keyword_overlap_low(self):
        ctx = Context()
        ctx.terms = {"pricing", "revenue", "forecast"}
        signal = _make_signal("booking sync guesty error")

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=1.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score < 0.2  # No overlap

    def test_source_affinity(self):
        ctx = Context()
        # Simulate context that mostly gets Slack signals
        ctx.source_counts = {"slack": 90, "github": 10}
        ctx.signal_count = 100

        signal = _make_signal("test", source=SignalSource.SLACK)

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=1.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score == pytest.approx(0.9, abs=0.01)

    def test_author_affinity(self):
        ctx = Context()
        ctx.author_counts = {"alice.chen": 50, "bob.martinez": 30, "other": 20}
        ctx.signal_count = 100

        signal = _make_signal("test", author="alice.chen")

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=1.0,
            signal_credibility=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_temporal_proximity_recent(self):
        ctx = Context()
        now = datetime.now(timezone.utc)
        ctx.last_signal = now - timedelta(minutes=5)

        signal = _make_signal("test", timestamp=now)

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=1.0,
            author_affinity=0.0,
            signal_credibility=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score > 0.9  # Very recent

    def test_temporal_proximity_old(self):
        ctx = Context()
        now = datetime.now(timezone.utc)
        ctx.last_signal = now - timedelta(days=30)

        signal = _make_signal("test", timestamp=now)

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=1.0,
            author_affinity=0.0,
            signal_credibility=0.0,
        )
        score = familiarity(signal, ctx, weights=weights)
        assert score < 0.5  # Old context

    def test_embedding_similarity_with_centroid(self):
        rng = np.random.RandomState(99)
        vec = rng.randn(384)
        vec = (vec / np.linalg.norm(vec)).tolist()

        signal = _make_signal("test", embeddings={"semantic": vec})
        ctx = Context()

        weights = FamiliarityWeights(
            embedding_similarity=1.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=0.0,
        )
        # Same vector as centroid = perfect similarity
        score = familiarity(signal, ctx, weights=weights, centroid=vec)
        assert score == pytest.approx(1.0, abs=0.01)


class TestCredibility:
    """Crawford-Sobel credibility scoring: costly signals vs cheap talk."""

    def test_pr_more_credible_than_comment(self):
        """PRs with diffs are costly signals; comments are cheap talk."""
        ctx = Context()
        ctx.terms = {"auth", "middleware"}

        pr_signal = _make_signal(
            "PR #42: Fix auth middleware",
            source="github",
            metadata={"event_type": "pull_request", "additions": 150, "deletions": 30},
        )
        comment_signal = _make_signal(
            "looks good to me",
            source="github",
            metadata={"event_type": "comment"},
        )

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=1.0,
        )
        pr_score = familiarity(pr_signal, ctx, weights=weights)
        comment_score = familiarity(comment_signal, ctx, weights=weights)
        assert pr_score > comment_score

    def test_large_diff_more_credible_than_small(self):
        """Larger diffs are costlier to produce — more credible."""
        ctx = Context()

        large_pr = _make_signal(
            "PR: Major refactor",
            source="github",
            metadata={"event_type": "pull_request", "additions": 500, "deletions": 200},
        )
        small_pr = _make_signal(
            "PR: Fix typo",
            source="github",
            metadata={"event_type": "pull_request", "additions": 2, "deletions": 1},
        )

        weights = FamiliarityWeights(
            embedding_similarity=0.0,
            keyword_overlap=0.0,
            source_affinity=0.0,
            temporal_proximity=0.0,
            author_affinity=0.0,
            signal_credibility=1.0,
        )
        large_score = familiarity(large_pr, ctx, weights=weights)
        small_score = familiarity(small_pr, ctx, weights=weights)
        assert large_score > small_score

    def test_credibility_in_range(self):
        """Credibility is always in [0, 1]."""
        ctx = Context()
        for event_type in ["pull_request", "comment", "push", "reaction", "unknown", ""]:
            signal = _make_signal("test", source="github", metadata={"event_type": event_type})
            weights = FamiliarityWeights(
                embedding_similarity=0.0,
                keyword_overlap=0.0,
                source_affinity=0.0,
                temporal_proximity=0.0,
                author_affinity=0.0,
                signal_credibility=1.0,
            )
            score = familiarity(signal, ctx, weights=weights)
            assert 0.0 <= score <= 1.0


class TestBoundary:
    """Test boundary conditions."""

    def test_empty_context(self):
        ctx = Context()
        signal = _make_signal("anything")
        score = familiarity(signal, ctx)
        assert 0.0 <= score <= 1.0

    def test_empty_signal(self):
        ctx = Context()
        ctx.terms = {"something"}
        signal = _make_signal("")
        score = familiarity(signal, ctx)
        assert 0.0 <= score <= 1.0

    def test_score_always_in_range(self):
        ctx = Context()
        ctx.terms = {"a", "b", "c"}
        for i in range(50):
            signal = _make_signal(f"test content variant {i}")
            score = familiarity(signal, ctx)
            assert 0.0 <= score <= 1.0
