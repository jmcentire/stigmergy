"""Tests for CadenceMonitor — silence detection and synthetic signal emission."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from stigmergy.mesh.stability import CadenceMonitor
from stigmergy.primitives.signal import SignalSource


class _FakeContext:
    """Minimal context stub for testing."""

    def __init__(self, signal_count: int = 0, source_counts: dict | None = None):
        self.id = uuid4()
        self.signal_count = signal_count
        self.source_counts = source_counts or {}
        self.energy = 1.0
        self.last_signal = datetime.now(timezone.utc)
        self.terms = set()


class _FakeWorker:
    """Minimal worker stub for testing."""

    def __init__(self, signal_count: int = 0, source_counts: dict | None = None, label: str = "test-worker"):
        self.context = _FakeContext(signal_count=signal_count, source_counts=source_counts)
        self.id = self.context.id
        self.label = label
        self.signals_accepted = signal_count


class _FakeMesh:
    """Minimal mesh stub for testing."""

    def __init__(self, workers: list[_FakeWorker] | None = None):
        self.workers = workers or []


def _empty_src() -> dict[str, int]:
    return {}


class TestBaselineComputation:
    """Test _compute_baseline from snapshot history."""

    def test_baseline_from_regular_activity(self):
        """Worker with 100 signals over 10 check intervals → baseline ~10/interval."""
        worker = _FakeWorker(signal_count=100)
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh, cooldown_signals=100)

        wid = str(worker.id)[:8]
        # Simulate 10 snapshots with steady growth of 10 signals per interval
        for i in range(10):
            monitor._snapshots.setdefault(wid, [])
            monitor._snapshots[wid].append((i * 50, i * 10, _empty_src()))

        baseline = monitor._compute_baseline(wid)
        assert baseline == pytest.approx(10.0 / 50.0, abs=0.01)

    def test_baseline_insufficient_snapshots(self):
        """Worker with <3 snapshots → baseline 0 (not enough data)."""
        worker = _FakeWorker(signal_count=50)
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        monitor._snapshots[wid] = [(0, 0, _empty_src()), (50, 25, _empty_src())]

        baseline = monitor._compute_baseline(wid)
        assert baseline == 0.0


class TestSilenceDetection:
    """Test _detect_silence comparing actual rate to baseline."""

    def test_silence_detected(self):
        """Worker with baseline 10/interval, 0 signals recently → flagged."""
        worker = _FakeWorker(signal_count=100)
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        # Build steady baseline history
        for i in range(8):
            monitor._snapshots.setdefault(wid, []).append((i * 50, i * 10, _empty_src()))
        # Silence period
        monitor._snapshots[wid].append((400, 70, _empty_src()))
        monitor._snapshots[wid].append((450, 70, _empty_src()))

        baseline = monitor._compute_baseline(wid)
        assert baseline > 0
        assert monitor._detect_silence(wid, baseline) is True

    def test_no_silence_when_active(self):
        """Worker still receiving signals → not flagged."""
        worker = _FakeWorker(signal_count=100)
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        for i in range(10):
            monitor._snapshots.setdefault(wid, []).append((i * 50, i * 10, _empty_src()))

        baseline = monitor._compute_baseline(wid)
        assert monitor._detect_silence(wid, baseline) is False


class TestNewWorkerNotFlagged:
    """New workers with insufficient baseline should not be flagged."""

    def test_worker_with_few_signals_not_flagged(self):
        """Worker with <20 signals → not flagged (insufficient baseline)."""
        worker = _FakeWorker(signal_count=15)
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        results = monitor.check(100)
        assert results == []

    def test_worker_at_threshold_not_flagged(self):
        """Worker with exactly MIN_BASELINE_SIGNALS but no snapshots → not flagged."""
        worker = _FakeWorker(signal_count=20)
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        results = monitor.check(50)
        assert results == []


class TestCooldown:
    """Same worker not re-flagged within cooldown window."""

    def test_cooldown_prevents_reflag(self):
        """Worker flagged once should not be re-flagged within cooldown."""
        worker = _FakeWorker(signal_count=100, source_counts={"github": 80, "linear": 20})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh, cooldown_signals=100)

        wid = str(worker.id)[:8]
        for i in range(8):
            monitor._snapshots.setdefault(wid, []).append((i * 50, i * 10, _empty_src()))
        monitor._snapshots[wid].append((400, 70, _empty_src()))
        monitor._snapshots[wid].append((450, 70, _empty_src()))

        results = monitor.check(450)
        assert len(results) == 1

        monitor._snapshots[wid].append((500, 70, _empty_src()))

        results = monitor.check(500)
        assert len(results) == 0

    def test_cooldown_expires(self):
        """Worker can be re-flagged after cooldown expires."""
        worker = _FakeWorker(signal_count=100, source_counts={"github": 80, "linear": 20})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh, cooldown_signals=50)

        wid = str(worker.id)[:8]
        for i in range(8):
            monitor._snapshots.setdefault(wid, []).append((i * 50, i * 10, _empty_src()))
        monitor._snapshots[wid].append((400, 70, _empty_src()))
        monitor._snapshots[wid].append((450, 70, _empty_src()))

        results = monitor.check(450)
        assert len(results) == 1

        monitor._snapshots[wid].append((500, 70, _empty_src()))
        monitor._snapshots[wid].append((550, 70, _empty_src()))

        results = monitor.check(550)
        assert len(results) == 1


class TestSyntheticSignalFormat:
    """Returned signal has correct source, metadata, and content."""

    def test_signal_source_and_metadata(self):
        """Synthetic signal should have source=mesh:cadence and correct metadata."""
        worker = _FakeWorker(
            signal_count=100,
            source_counts={"github": 60, "linear": 40},
            label="pricing-engine",
        )
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh, cooldown_signals=10)

        wid = str(worker.id)[:8]
        for i in range(8):
            monitor._snapshots.setdefault(wid, []).append((i * 50, i * 10, _empty_src()))
        monitor._snapshots[wid].append((400, 70, _empty_src()))
        monitor._snapshots[wid].append((450, 70, _empty_src()))

        results = monitor.check(450)
        assert len(results) == 1

        sig = results[0]
        assert sig.source == SignalSource.CADENCE
        assert sig.source == "mesh:cadence"
        assert sig.author == "cadence-monitor"
        assert sig.channel == f"worker:{wid}"
        assert "cadence_break" in sig.metadata.get("event_type", "")
        assert sig.metadata["worker_id"] == wid
        assert sig.metadata["coherence"] in ("healthy", "incoherent", "unknown")
        assert "baseline_rate" in sig.metadata
        assert "actual_rate" in sig.metadata
        assert "worker_energy" in sig.metadata
        assert "Cadence break detected" in sig.content


class TestSourceEntropy:
    """Test the static entropy computation."""

    def test_uniform_distribution_max_entropy(self):
        """Three equal sources → maximum entropy for 3 sources."""
        import math
        h = CadenceMonitor._source_entropy({"a": 10, "b": 10, "c": 10})
        assert h == pytest.approx(math.log2(3), abs=0.01)

    def test_single_source_zero_entropy(self):
        """One source → zero entropy."""
        h = CadenceMonitor._source_entropy({"a": 100})
        assert h == 0.0

    def test_dominant_source_low_entropy(self):
        """One source dominates → lower entropy than balanced."""
        h_balanced = CadenceMonitor._source_entropy({"a": 50, "b": 50})
        h_skewed = CadenceMonitor._source_entropy({"a": 90, "b": 10})
        assert h_skewed < h_balanced

    def test_empty_zero_entropy(self):
        """No sources → zero entropy."""
        h = CadenceMonitor._source_entropy({})
        assert h == 0.0


class TestAsymmetryScore:
    """Test the continuous asymmetric compression scorer.

    The score replaces hard thresholds with a smooth gradient. sqrt
    amplifies subtle divergences near zero where linear comparison would
    lose the signal in the noise floor.
    """

    def test_clear_asymmetric(self):
        """Entropy collapsed, volume held → high score."""
        score = CadenceMonitor._asymmetry_score(0.0, 0.5)
        assert score > 0.4
        assert score < 0.8

    def test_symmetric_compression(self):
        """Both collapsed → near-zero score (volume presence vanishes)."""
        score = CadenceMonitor._asymmetry_score(0.0, 0.05)
        assert score < 0.1

    def test_no_compression(self):
        """No change → zero score."""
        score = CadenceMonitor._asymmetry_score(1.0, 1.0)
        assert score == 0.0

    def test_volume_dropped_more_than_entropy(self):
        """Volume dropped more → zero (not asymmetric in the bad direction)."""
        score = CadenceMonitor._asymmetry_score(0.8, 0.3)
        assert score == 0.0

    def test_gradient_not_cliff(self):
        """Scores increase smoothly — no jump at a magic threshold."""
        scores = []
        # Sweep entropy_ratio from 1.0 (no compression) to 0.0 (full)
        # with volume holding at 0.5
        for i in range(11):
            e_ratio = 1.0 - i * 0.1
            s = CadenceMonitor._asymmetry_score(e_ratio, 0.5)
            scores.append(s)
        # Should be monotonically non-decreasing
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 0.001
        # First score (no compression) should be 0, last should be > 0
        assert scores[0] == 0.0
        assert scores[-1] > 0.3

    def test_sqrt_amplifies_subtle_asymmetry(self):
        """Small asymmetry gets amplified by sqrt, not buried in noise.

        With entropy_ratio=0.7 and volume_ratio=0.95, the linear gap
        is only 0.25 (entropy compressed 0.3, volume compressed 0.05).
        The sqrt amplification makes this detectable.
        """
        score = CadenceMonitor._asymmetry_score(0.7, 0.95)
        # Linear gap = 0.3-0.05 = 0.25, presence = 0.95
        # sqrt(0.25) * sqrt(0.95) ≈ 0.5 * 0.97 ≈ 0.49
        assert score > 0.3  # sqrt pushed it above the threshold


class TestSourceCoherence:
    """Cross-source coherence via entropy compression detection.

    The coherence check uses source distribution entropy across snapshot
    windows. Asymmetric compression (entropy drops, volume maintained) =
    incoherent. Symmetric compression (entropy drops, volume drops) = healthy.
    Returns (classification, score) where score is continuous 0.0–1.0.
    """

    def test_incoherent_asymmetric_compression(self):
        """One source goes quiet while others maintain rate → incoherent.

        This is the insidious case: volume barely drops because other sources
        compensate, but the source mix loses a dimension. The entropy
        compression is asymmetric.
        """
        worker = _FakeWorker(signal_count=100, source_counts={"github": 50, "linear": 50})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        # Baseline: both sources active, balanced
        monitor._snapshots[wid] = [
            (0, 0, {"github": 0, "linear": 0}),
            (50, 20, {"github": 10, "linear": 10}),
            (100, 40, {"github": 20, "linear": 20}),
            (150, 60, {"github": 30, "linear": 30}),
            (200, 80, {"github": 40, "linear": 40}),
            # GitHub goes quiet, Linear continues
            (250, 90, {"github": 40, "linear": 50}),
            (300, 100, {"github": 40, "linear": 60}),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        assert coherence == "incoherent"
        assert score > 0.35  # continuous score above threshold

    def test_healthy_symmetric_compression(self):
        """All sources go quiet together → healthy.

        Volume drops AND entropy drops together. This is a weekend, sprint
        boundary, or holiday — everyone stopped.
        """
        worker = _FakeWorker(signal_count=100, source_counts={"github": 50, "linear": 50})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        # Baseline: both sources active, balanced
        monitor._snapshots[wid] = [
            (0, 0, {"github": 0, "linear": 0}),
            (50, 20, {"github": 10, "linear": 10}),
            (100, 40, {"github": 20, "linear": 20}),
            (150, 60, {"github": 30, "linear": 30}),
            # Both sources go quiet — volume drops to near zero
            (200, 62, {"github": 31, "linear": 31}),
            (250, 63, {"github": 31, "linear": 32}),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        assert coherence == "healthy"
        assert score < 0.1  # symmetric compression → near-zero score

    def test_unknown_insufficient_history(self):
        """Too few snapshots → unknown."""
        worker = _FakeWorker(signal_count=100, source_counts={"github": 50, "linear": 50})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        monitor._snapshots[wid] = [
            (0, 0, {"github": 0, "linear": 0}),
            (50, 20, {"github": 10, "linear": 10}),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        assert coherence == "unknown"
        assert score == 0.0

    def test_healthy_stable_entropy(self):
        """All sources maintain their rate → healthy (no compression)."""
        worker = _FakeWorker(signal_count=200, source_counts={"github": 100, "linear": 100})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        # Steady balanced flow throughout
        monitor._snapshots[wid] = [
            (0, 0, {"github": 0, "linear": 0}),
            (50, 20, {"github": 10, "linear": 10}),
            (100, 40, {"github": 20, "linear": 20}),
            (150, 60, {"github": 30, "linear": 30}),
            (200, 80, {"github": 40, "linear": 40}),
            (250, 100, {"github": 50, "linear": 50}),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        assert coherence == "healthy"
        assert score == 0.0  # no compression at all


class TestPeriodicBaseline:
    """Period-aware coherence: the mesh learns its own heartbeat.

    The organization has a rhythm. Weekends look different from weekdays.
    One engineer pushes PRs on Saturday nights while Slack and Linear
    sleep. That's the normal Saturday signature, not an anomaly.
    The detector compares Saturday against previous Saturdays.
    The anomaly is an arrhythmia, not a slow pulse.
    """

    # Reference dates: Jan 2024, Monday the 1st
    _MON1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _TUE1 = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    _WED1 = datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
    _THU1 = datetime(2024, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
    _FRI1 = datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
    _SAT1 = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
    _SUN1 = datetime(2024, 1, 7, 12, 0, 0, tzinfo=timezone.utc)
    _MON2 = datetime(2024, 1, 8, 12, 0, 0, tzinfo=timezone.utc)
    _TUE2 = datetime(2024, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
    _WED2 = datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    _SAT2 = datetime(2024, 1, 13, 12, 0, 0, tzinfo=timezone.utc)
    _SUN2 = datetime(2024, 1, 14, 12, 0, 0, tzinfo=timezone.utc)

    def test_weekend_github_only_learned_healthy(self):
        """Weekend with GitHub-only is healthy if previous weekends were the same.

        One engineer works weekends, cranking out PRs while Slack/Linear
        are quiet. The naive entropy detector would flag this as incoherent
        every Saturday. But with learned periodicity, Saturday compares
        against previous Saturdays — and this IS the normal Saturday pattern.
        """
        worker = _FakeWorker(signal_count=200, source_counts={"github": 100, "linear": 100})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        monitor._snapshots[wid] = [
            # Week 1 weekdays: balanced github + linear
            (0,   0,   {"github": 0,  "linear": 0},  self._MON1),
            (50,  20,  {"github": 10, "linear": 10}, self._TUE1),
            (100, 40,  {"github": 20, "linear": 20}, self._WED1),
            # Week 1 weekend: github only (one engineer)
            (150, 50,  {"github": 30, "linear": 20}, self._SAT1),
            (200, 60,  {"github": 40, "linear": 20}, self._SUN1),
            # Week 2 weekdays: balanced again
            (250, 80,  {"github": 50, "linear": 30}, self._MON2),
            (300, 100, {"github": 60, "linear": 40}, self._TUE2),
            # Week 2 weekend: github only again (same learned pattern)
            (350, 110, {"github": 70, "linear": 40}, self._SAT2),
            (400, 120, {"github": 80, "linear": 40}, self._SUN2),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        assert coherence == "healthy"
        assert score == 0.0  # learned weekend pattern matches

    def test_tuesday_github_only_incoherent(self):
        """Tuesday with GitHub-only is incoherent when Tuesdays are balanced.

        Same asymmetric source pattern as the weekend test, but on a day
        when it doesn't match the learned rhythm. The anomaly is the
        arrhythmia — same pattern, wrong beat.
        """
        worker = _FakeWorker(signal_count=200, source_counts={"github": 100, "linear": 100})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        monitor._snapshots[wid] = [
            # Weekday history: balanced github + linear
            (0,   0,   {"github": 0,  "linear": 0},  self._MON1),
            (50,  20,  {"github": 10, "linear": 10}, self._TUE1),
            (100, 40,  {"github": 20, "linear": 20}, self._WED1),
            (150, 60,  {"github": 30, "linear": 30}, self._THU1),
            (200, 80,  {"github": 40, "linear": 40}, self._FRI1),
            (250, 100, {"github": 50, "linear": 50}, self._MON2),
            # Anomalous: GitHub active, Linear goes quiet on a weekday
            (300, 110, {"github": 60, "linear": 50}, self._TUE2),
            (350, 120, {"github": 70, "linear": 50}, self._WED2),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        assert coherence == "incoherent"
        assert score > 0.35  # continuous score confirms asymmetry

    def test_first_weekend_no_learned_pattern(self):
        """First weekend seen: no learned pattern → falls back to all-time.

        The mesh hasn't observed a weekend before. All-time baseline is
        balanced (from weekdays). Current weekend is GitHub-only. Without
        period-matched history, the all-time comparison fires. After this
        weekend's data enters the snapshots, subsequent weekends match.
        """
        worker = _FakeWorker(signal_count=100, source_counts={"github": 50, "linear": 50})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        monitor._snapshots[wid] = [
            # All weekday history: balanced
            (0,   0,  {"github": 0,  "linear": 0},  self._MON1),
            (50,  20, {"github": 10, "linear": 10}, self._TUE1),
            (100, 40, {"github": 20, "linear": 20}, self._WED1),
            (150, 60, {"github": 30, "linear": 30}, self._THU1),
            # First weekend ever: GitHub only
            (200, 70, {"github": 40, "linear": 30}, self._SAT1),
            (250, 80, {"github": 50, "linear": 30}, self._SUN1),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        # No prior weekend data → falls back to all-time (balanced)
        # Current is single-source → incoherent (first occurrence)
        assert coherence == "incoherent"
        assert score > 0.35

    def test_no_timestamps_falls_back_to_alltime(self):
        """Snapshots without timestamps use all-time baseline (backward compat)."""
        worker = _FakeWorker(signal_count=100, source_counts={"github": 50, "linear": 50})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        # 3-element tuples (no timestamps) — same as pre-periodicity format
        monitor._snapshots[wid] = [
            (0, 0, {"github": 0, "linear": 0}),
            (50, 20, {"github": 10, "linear": 10}),
            (100, 40, {"github": 20, "linear": 20}),
            (150, 60, {"github": 30, "linear": 30}),
            (200, 80, {"github": 40, "linear": 40}),
            # Asymmetric: github goes quiet
            (250, 90, {"github": 40, "linear": 50}),
            (300, 100, {"github": 40, "linear": 60}),
        ]

        coherence, score = monitor._check_source_coherence(wid)
        # No timestamps → all-time baseline → detects asymmetric compression
        assert coherence == "incoherent"
        assert score > 0.35

    def test_period_aware_baseline_rate(self):
        """Baseline rate should use period-matched windows when available.

        If weekends naturally have lower signal rate, the baseline for
        Saturday should reflect Saturday history, not the all-time average
        inflated by weekday activity.
        """
        worker = _FakeWorker(signal_count=200, source_counts={"github": 100, "linear": 100})
        mesh = _FakeMesh(workers=[worker])
        monitor = CadenceMonitor(mesh)

        wid = str(worker.id)[:8]
        monitor._snapshots[wid] = [
            # Weekdays: 10 signals per 50-signal interval (rate=0.2)
            (0,   0,  {"github": 0, "linear": 0}, self._MON1),
            (50,  10, {"github": 5, "linear": 5}, self._TUE1),
            (100, 20, {"github": 10, "linear": 10}, self._WED1),
            # Weekend: 2 signals per interval (rate=0.04)
            (150, 22, {"github": 12, "linear": 10}, self._SAT1),
            (200, 24, {"github": 14, "linear": 10}, self._SUN1),
            # Next week
            (250, 34, {"github": 19, "linear": 15}, self._MON2),
            (300, 44, {"github": 24, "linear": 20}, self._TUE2),
            # Current weekend — should compare against weekend rate
            (350, 46, {"github": 26, "linear": 20}, self._SAT2),
            (400, 48, {"github": 28, "linear": 20}, self._SUN2),
        ]

        baseline = monitor._compute_baseline(wid)
        # Period-matched weekends: rate=2/50=0.04 per window
        # All-time average would be much higher (~0.12)
        # With period matching, weekend baseline should be close to 0.04
        assert baseline < 0.08  # Distinctly lower than all-time
        assert baseline > 0.01  # But not zero
