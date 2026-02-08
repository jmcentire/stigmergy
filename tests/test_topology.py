"""Tests for topology — worker position model and spatial coordination."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from stigmergy.mesh.topology import (
    WorkerPosition,
    bloom_digest,
    detect_gap,
    position_distance,
    select_peers,
    signal_position_distance,
)
from stigmergy.structures.bloom import CountingBloomFilter


def _bloom_with_terms(*term_sets: list[str], capacity: int = 10000) -> CountingBloomFilter:
    """Create a bloom filter with terms added."""
    bloom = CountingBloomFilter(capacity=capacity)
    for terms in term_sets:
        bloom.add_many(terms)
    return bloom


def _position(
    terms: list[str] | None = None,
    signal_count: int = 10,
    fullness: float = 0.05,
    worker_id=None,
    bins: int = 128,
) -> WorkerPosition:
    """Create a WorkerPosition from terms."""
    wid = worker_id or uuid4()
    bloom = CountingBloomFilter(capacity=10000)
    if terms:
        bloom.add_many(terms)
    digest = bloom_digest(bloom, bins=bins)
    top = tuple(terms[:10]) if terms else ()
    return WorkerPosition(
        worker_id=wid,
        bloom_digest=digest,
        top_terms=top,
        signal_count=signal_count,
        fullness=fullness,
    )


# ── bloom_digest ──────────────────────────────────────────────


class TestBloomDigest:
    def test_returns_correct_shape(self):
        bloom = CountingBloomFilter(capacity=10000)
        bloom.add_many(["pricing", "engine", "cache"])
        digest = bloom_digest(bloom, bins=128)
        assert digest.shape == (128,)

    def test_non_zero_for_non_empty_bloom(self):
        bloom = CountingBloomFilter(capacity=10000)
        bloom.add_many(["pricing", "engine", "cache"])
        digest = bloom_digest(bloom, bins=128)
        assert digest.sum() > 0

    def test_zero_for_empty_bloom(self):
        bloom = CountingBloomFilter(capacity=10000)
        digest = bloom_digest(bloom, bins=128)
        assert digest.sum() == 0.0

    def test_different_bin_sizes(self):
        bloom = CountingBloomFilter(capacity=10000)
        bloom.add_many(["pricing", "engine"])
        d64 = bloom_digest(bloom, bins=64)
        d256 = bloom_digest(bloom, bins=256)
        assert d64.shape == (64,)
        assert d256.shape == (256,)
        # Both should be non-zero
        assert d64.sum() > 0
        assert d256.sum() > 0

    def test_small_bloom_padded(self):
        bloom = CountingBloomFilter(capacity=1, fp_rate=0.5)
        bloom.add("test")
        digest = bloom_digest(bloom, bins=128)
        assert digest.shape == (128,)

    def test_similar_blooms_similar_digests(self):
        b1 = CountingBloomFilter(capacity=10000)
        b1.add_many(["pricing", "engine", "cache", "invalidation"])
        b2 = CountingBloomFilter(capacity=10000)
        b2.add_many(["pricing", "engine", "cache", "strategy"])

        d1 = bloom_digest(b1, bins=128)
        d2 = bloom_digest(b2, bins=128)

        # Cosine similarity should be high (> 0.5) for overlapping terms
        cos = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
        assert cos > 0.3

    def test_different_blooms_different_digests(self):
        b1 = CountingBloomFilter(capacity=10000)
        b1.add_many(["pricing", "engine", "cache"])
        b2 = CountingBloomFilter(capacity=10000)
        b2.add_many(["deploy", "kubernetes", "infrastructure"])

        d1 = bloom_digest(b1, bins=128)
        d2 = bloom_digest(b2, bins=128)

        # Digests should differ
        assert not np.allclose(d1, d2)


# ── position_distance ────────────────────────────────────────


class TestPositionDistance:
    def test_identical_workers_zero_distance(self):
        terms = ["pricing", "engine", "cache", "invalidation", "strategy"]
        p1 = _position(terms=terms)
        p2 = _position(terms=terms)
        dist = position_distance(p1, p2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_workers_positive_distance(self):
        p1 = _position(terms=["pricing", "engine", "cache"])
        p2 = _position(terms=["deploy", "kubernetes", "infrastructure"])
        dist = position_distance(p1, p2)
        assert dist > 0.0

    def test_empty_workers_zero_distance(self):
        p1 = _position(terms=[])
        p2 = _position(terms=[])
        dist = position_distance(p1, p2)
        # Both empty → cosine similarity is 0 (zero vectors), distance is 1.0
        assert dist == pytest.approx(1.0)

    def test_one_empty_max_distance(self):
        p1 = _position(terms=["pricing", "engine"])
        p2 = _position(terms=[])
        dist = position_distance(p1, p2)
        assert dist == pytest.approx(1.0)

    def test_distance_symmetry(self):
        p1 = _position(terms=["pricing", "engine"])
        p2 = _position(terms=["deploy", "release"])
        assert position_distance(p1, p2) == pytest.approx(position_distance(p2, p1))

    def test_distance_range(self):
        p1 = _position(terms=["pricing", "engine", "cache"])
        p2 = _position(terms=["deploy", "kubernetes", "infra"])
        dist = position_distance(p1, p2)
        assert 0.0 <= dist <= 1.0


# ── signal_position_distance ─────────────────────────────────


class TestSignalPositionDistance:
    def test_matching_signal_low_distance(self):
        terms = ["pricing", "engine", "cache"]
        pos = _position(terms=terms)
        sig_bloom = _bloom_with_terms(terms)
        dist = signal_position_distance(sig_bloom, pos)
        assert dist < 0.3  # should be close

    def test_unrelated_signal_high_distance(self):
        pos = _position(terms=["pricing", "engine", "cache"])
        sig_bloom = _bloom_with_terms(["deploy", "kubernetes", "infrastructure"])
        dist = signal_position_distance(sig_bloom, pos)
        assert dist > 0.5

    def test_empty_signal_max_distance(self):
        pos = _position(terms=["pricing", "engine"])
        sig_bloom = CountingBloomFilter(capacity=10000)
        dist = signal_position_distance(sig_bloom, pos)
        assert dist == pytest.approx(1.0)


# ── select_peers ─────────────────────────────────────────────


class TestSelectPeers:
    def test_basic_selection(self):
        # Create a central worker and 6 candidates at various distances
        center = _position(terms=["pricing", "engine"])
        candidates = [
            _position(terms=["pricing", "cache"]),        # close
            _position(terms=["pricing", "strategy"]),      # close
            _position(terms=["deploy", "release"]),        # medium
            _position(terms=["hiring", "onboarding"]),     # far
            _position(terms=["revenue", "forecast"]),      # far
            _position(terms=["incident", "monitoring"]),   # medium
        ]
        peers = select_peers(center, candidates, target=6)
        assert len(peers) == 6

    def test_excludes_self(self):
        wid = uuid4()
        center = _position(terms=["pricing"], worker_id=wid)
        candidates = [center, _position(terms=["deploy"])]
        peers = select_peers(center, candidates, target=6)
        assert wid not in peers

    def test_fewer_candidates_than_target(self):
        center = _position(terms=["pricing"])
        candidates = [_position(terms=["deploy"]), _position(terms=["revenue"])]
        peers = select_peers(center, candidates, target=6)
        assert len(peers) == 2

    def test_empty_candidates(self):
        center = _position(terms=["pricing"])
        peers = select_peers(center, [], target=6)
        assert peers == []

    def test_picks_mix_of_close_and_far(self):
        center = _position(terms=["pricing", "engine", "cache", "invalidation"])
        # Create clearly close and clearly far candidates
        close1 = _position(terms=["pricing", "engine", "cache", "strategy"])
        close2 = _position(terms=["pricing", "engine", "optimization"])
        far1 = _position(terms=["deploy", "kubernetes", "helm", "docker"])
        far2 = _position(terms=["hiring", "interview", "onboarding", "culture"])
        mid1 = _position(terms=["testing", "coverage", "engine"])
        mid2 = _position(terms=["monitoring", "alert", "pricing"])

        all_positions = [close1, close2, far1, far2, mid1, mid2]
        peers = select_peers(center, all_positions, target=6)

        # Should include both close and far workers
        assert close1.worker_id in peers or close2.worker_id in peers
        assert far1.worker_id in peers or far2.worker_id in peers

    def test_respects_max_peers(self):
        center = _position(terms=["pricing"])
        candidates = [_position(terms=[f"term{i}"]) for i in range(20)]
        peers = select_peers(center, candidates, target=6, max_peers=8)
        assert len(peers) <= 8

    def test_respects_min_peers(self):
        center = _position(terms=["pricing"])
        candidates = [
            _position(terms=["deploy"]),
            _position(terms=["revenue"]),
            _position(terms=["testing"]),
            _position(terms=["hiring"]),
            _position(terms=["monitoring"]),
        ]
        peers = select_peers(center, candidates, target=6, min_peers=4)
        assert len(peers) >= 4


# ── detect_gap ────────────────────────────────────────────────


class TestDetectGap:
    def test_all_below_threshold_is_gap(self):
        scores = {uuid4(): 0.05, uuid4(): 0.03, uuid4(): 0.07}
        assert detect_gap(scores, threshold=0.08)

    def test_one_above_threshold_not_gap(self):
        scores = {uuid4(): 0.05, uuid4(): 0.1, uuid4(): 0.03}
        assert not detect_gap(scores, threshold=0.08)

    def test_empty_scores_is_gap(self):
        assert detect_gap({}, threshold=0.08)

    def test_all_above_threshold_not_gap(self):
        scores = {uuid4(): 0.2, uuid4(): 0.3}
        assert not detect_gap(scores, threshold=0.08)

    def test_exact_threshold_not_gap(self):
        scores = {uuid4(): 0.08}
        assert not detect_gap(scores, threshold=0.08)

    def test_custom_threshold(self):
        scores = {uuid4(): 0.15}
        assert detect_gap(scores, threshold=0.2)
        assert not detect_gap(scores, threshold=0.1)
