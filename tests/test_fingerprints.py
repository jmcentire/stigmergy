"""Tests for spectral fingerprints and structural coupling detection.

Regression tests are annotated with the bug they prevent. Each test
documents WHY it exists so future us understands the failure mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stigmergy.mesh.fingerprints import (
    CouplingDetection,
    FingerprintStore,
    SpectralFingerprint,
    _channel_related,
)


def _make_fp(
    source: str = "github",
    channel: str = "acme-org/backend",
    author: str = "alice",
    terms: frozenset | None = None,
    activation: dict | None = None,
    accepted: frozenset | None = None,
) -> SpectralFingerprint:
    return SpectralFingerprint(
        signal_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        source=source,
        channel=channel,
        author=author,
        terms=terms or frozenset(),
        activation=activation or {},
        accepted_workers=accepted or frozenset(),
    )


class TestFingerprintStore:
    def test_record_and_count(self):
        store = FingerprintStore()
        store.record(_make_fp(activation={"w1": 0.5, "w2": 0.3}))
        store.record(_make_fp(activation={"w1": 0.4, "w2": 0.6}))
        assert store.count == 2

    def test_eviction(self):
        store = FingerprintStore(max_fingerprints=10)
        for _ in range(15):
            store.record(_make_fp(activation={"w1": 0.5}))
        assert store.count <= 10

    def test_worker_universe_grows(self):
        store = FingerprintStore()
        store.record(_make_fp(activation={"w1": 0.5}))
        assert len(store._worker_universe) == 1
        store.record(_make_fp(activation={"w1": 0.3, "w2": 0.7}))
        assert len(store._worker_universe) == 2

    def test_summary(self):
        store = FingerprintStore()
        store.record(_make_fp(source="github", activation={"w1": 0.5}))
        store.record(_make_fp(source="linear", activation={"w1": 0.3}))
        s = store.summary()
        assert s["fingerprints"] == 2
        assert "github" in s["sources"]
        assert "linear" in s["sources"]


class TestCouplingDetection:
    """Coupling detection tests.

    Genuine hidden coupling requires:
    - 5+ workers in the universe (dimensionality threshold)
    - 3+ non-zero dimensions per vector (sparse vector guard)
    - 2+ shared accepted workers (routing coincidence guard)
    - Different channels (within-channel coupling is expected)
    - High activation similarity (cosine > 0.5)
    - Low term overlap (Jaccard < 0.3)
    """

    def test_genuine_hidden_coupling(self):
        """Two signals from different channels with rich activation vectors,
        multiple shared accepted workers, and zero vocabulary overlap.

        This is the paper's core claim: structurally related signals that
        share no vocabulary are detected via their activation patterns.
        """
        store = FingerprintStore()

        # Both signals activate 5 workers with similar patterns,
        # and are ACCEPTED by 2 workers (w1 and w2).
        fp_a = _make_fp(
            channel="acme-org/backend",
            terms=frozenset({"auth", "middleware", "jwt", "token"}),
            activation={"w1": 0.8, "w2": 0.7, "w3": 0.3, "w4": 0.2, "w5": 0.15},
            accepted=frozenset({"w1", "w2"}),
        )
        fp_b = _make_fp(
            channel="acme-org/booking",
            terms=frozenset({"booking", "reservation", "calendar"}),
            activation={"w1": 0.75, "w2": 0.65, "w3": 0.25, "w4": 0.18, "w5": 0.12},
            accepted=frozenset({"w1", "w2"}),
        )
        store.record(fp_a)
        store.record(fp_b)

        couplings = store.detect_couplings(
            min_activation_similarity=0.5,
            max_term_overlap=0.3,
        )
        assert len(couplings) >= 1
        c = couplings[0]
        assert c.activation_similarity > 0.9
        assert c.term_overlap == 0.0
        assert c.is_hidden_coupling
        assert len(c.shared_workers) >= 2

    def test_no_coupling_single_shared_worker(self):
        """BUG REGRESSION: activation=1.00 with shared_workers=1.

        Before this fix, two signals that both routed to the same single
        worker produced cosine similarity ~1.0 with only 1 shared worker.
        This is a routing coincidence, not structural coupling. 41,670
        false positives were generated in a 30-day live run.

        Fix: require 2+ shared accepted workers.
        """
        store = FingerprintStore()

        # Similar activation, but only 1 shared accepted worker
        fp_a = _make_fp(
            channel="acme-org/backend",
            terms=frozenset({"auth", "jwt"}),
            activation={"w1": 0.8, "w2": 0.7, "w3": 0.1, "w4": 0.2, "w5": 0.05},
            accepted=frozenset({"w1"}),  # Only w1 accepted
        )
        fp_b = _make_fp(
            channel="acme-org/booking",
            terms=frozenset({"booking", "calendar"}),
            activation={"w1": 0.75, "w2": 0.65, "w3": 0.15, "w4": 0.18, "w5": 0.08},
            accepted=frozenset({"w1"}),  # Same single worker
        )
        store.record(fp_a)
        store.record(fp_b)

        couplings = store.detect_couplings(min_activation_similarity=0.5)
        assert len(couplings) == 0, (
            "Single shared worker should not produce a coupling detection. "
            "This was the root cause of 41,670 false positives."
        )

    def test_no_coupling_sparse_vectors(self):
        """BUG REGRESSION: sparse vector false positives.

        With 17 workers but signals only visiting 1-2 during BFS routing,
        activation vectors are mostly zeros. Cosine similarity between two
        sparse vectors with overlap in 1-2 positions is trivially high.

        Fix: require 3+ non-zero dimensions per vector.
        """
        store = FingerprintStore()

        # Sparse vectors: only 2 non-zero entries each
        fp_a = _make_fp(
            channel="acme-org/backend",
            terms=frozenset({"deploy"}),
            activation={"w1": 0.9, "w2": 0.1, "w3": 0.0, "w4": 0.0, "w5": 0.0},
            accepted=frozenset({"w1", "w2"}),
        )
        fp_b = _make_fp(
            channel="acme-org/pricing",
            terms=frozenset({"pricing"}),
            activation={"w1": 0.85, "w2": 0.15, "w3": 0.0, "w4": 0.0, "w5": 0.0},
            accepted=frozenset({"w1", "w2"}),
        )
        store.record(fp_a)
        store.record(fp_b)

        couplings = store.detect_couplings(min_activation_similarity=0.5)
        assert len(couplings) == 0, (
            "Vectors with <3 non-zero dimensions produce meaningless cosine "
            "similarity. This should be filtered out."
        )

    def test_no_coupling_when_terms_overlap(self):
        """Similar activation with high term overlap is NOT a hidden coupling.
        If signals share vocabulary, their coupling is expected and findable
        by keyword search — not the hidden structure the paper targets."""
        store = FingerprintStore()

        fp_a = _make_fp(
            channel="acme-org/backend",
            terms=frozenset({"auth", "middleware", "jwt"}),
            activation={"w1": 0.8, "w2": 0.7, "w3": 0.3, "w4": 0.2, "w5": 0.1},
            accepted=frozenset({"w1", "w2"}),
        )
        fp_b = _make_fp(
            channel="acme-org/booking",
            terms=frozenset({"auth", "middleware", "security"}),
            activation={"w1": 0.75, "w2": 0.65, "w3": 0.25, "w4": 0.18, "w5": 0.08},
            accepted=frozenset({"w1", "w2"}),
        )
        store.record(fp_a)
        store.record(fp_b)

        couplings = store.detect_couplings(
            min_activation_similarity=0.5,
            max_term_overlap=0.3,
        )
        assert all(not c.is_hidden_coupling for c in couplings) or len(couplings) == 0

    def test_no_coupling_different_activations(self):
        """Different activation patterns = no structural coupling."""
        store = FingerprintStore()

        fp_a = _make_fp(
            channel="acme-org/backend",
            terms=frozenset({"auth"}),
            activation={"w1": 0.9, "w2": 0.1, "w3": 0.0, "w4": 0.0, "w5": 0.0},
        )
        fp_b = _make_fp(
            channel="acme-org/pricing",
            terms=frozenset({"pricing"}),
            activation={"w1": 0.0, "w2": 0.1, "w3": 0.9, "w4": 0.0, "w5": 0.0},
        )
        store.record(fp_a)
        store.record(fp_b)

        couplings = store.detect_couplings(min_activation_similarity=0.5)
        assert len(couplings) == 0

    def test_no_coupling_same_channel(self):
        """BUG REGRESSION: within-channel coupling is expected, not hidden.

        Signals from the same channel/repo will naturally activate the same
        workers. The paper's insight is about coupling ACROSS channels.
        """
        store = FingerprintStore()

        fp_a = _make_fp(
            channel="acme-org/backend",  # Same channel
            terms=frozenset({"auth"}),
            activation={"w1": 0.8, "w2": 0.7, "w3": 0.3, "w4": 0.2, "w5": 0.1},
            accepted=frozenset({"w1", "w2"}),
        )
        fp_b = _make_fp(
            channel="acme-org/backend",  # Same channel
            terms=frozenset({"deploy"}),
            activation={"w1": 0.75, "w2": 0.65, "w3": 0.25, "w4": 0.18, "w5": 0.08},
            accepted=frozenset({"w1", "w2"}),
        )
        store.record(fp_a)
        store.record(fp_b)

        couplings = store.detect_couplings(min_activation_similarity=0.5)
        assert len(couplings) == 0, "Same-channel pairs are expected couplings, not hidden."


class TestCrossSourceConsistency:
    def test_cross_source_by_author(self):
        """Same author in github + linear = cross-source corroboration."""
        store = FingerprintStore()

        fp_github = _make_fp(
            source="github",
            channel="acme-org/backend",
            author="alice",
            activation={"w1": 0.8, "w2": 0.6, "w3": 0.1, "w4": 0.05, "w5": 0.02},
        )
        fp_linear = _make_fp(
            source="linear",
            channel="PLAT-123",
            author="alice",
            activation={"w1": 0.75, "w2": 0.55, "w3": 0.12, "w4": 0.06, "w5": 0.03},
        )
        store.record(fp_github)
        store.record(fp_linear)

        cross = store.detect_cross_source_consistency(min_similarity=0.4)
        assert len(cross) >= 1
        c = cross[0]
        assert c.is_cross_source
        assert c.source_a != c.source_b

    def test_no_cross_source_different_authors(self):
        """Different authors, different channels — no cross-source match."""
        store = FingerprintStore()

        fp_a = _make_fp(source="github", author="alice", channel="backend",
                        activation={"w1": 0.5, "w2": 0.3, "w3": 0.1, "w4": 0.05, "w5": 0.02})
        fp_b = _make_fp(source="linear", author="bob", channel="PLAT-999",
                        activation={"w1": 0.4, "w2": 0.3, "w3": 0.2, "w4": 0.06, "w5": 0.03})
        store.record(fp_a)
        store.record(fp_b)

        cross = store.detect_cross_source_consistency(min_similarity=0.4)
        assert len(cross) == 0


class TestCouplingDataclass:
    def test_is_hidden_coupling(self):
        c = CouplingDetection(
            signal_a_id=uuid4(),
            signal_b_id=uuid4(),
            activation_similarity=0.8,
            term_overlap=0.05,
            coupling_strength=0.75,
            source_a="github",
            source_b="github",
            channel_a="acme-org/backend",
            channel_b="acme-org/pricing",
            shared_workers=["w1", "w2"],
        )
        assert c.is_hidden_coupling
        assert c.severity == "high"

    def test_is_not_hidden_when_vocab_overlaps(self):
        c = CouplingDetection(
            signal_a_id=uuid4(),
            signal_b_id=uuid4(),
            activation_similarity=0.8,
            term_overlap=0.5,
            coupling_strength=0.3,
            source_a="github",
            source_b="github",
            channel_a="a",
            channel_b="b",
            shared_workers=[],
        )
        assert not c.is_hidden_coupling

    def test_cross_source(self):
        c = CouplingDetection(
            signal_a_id=uuid4(),
            signal_b_id=uuid4(),
            activation_similarity=0.7,
            term_overlap=0.1,
            coupling_strength=0.6,
            source_a="github",
            source_b="linear",
            channel_a="a",
            channel_b="b",
            shared_workers=[],
        )
        assert c.is_cross_source


class TestChannelRelated:
    def test_repo_name_match(self):
        assert _channel_related("acme-org/backend", "backend-PLAT-123")

    def test_no_match(self):
        assert not _channel_related("acme-org/pricing", "events-team")

    def test_empty(self):
        assert not _channel_related("", "")
