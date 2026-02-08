"""Tests for left-shift (eigenvalue collapse) detection in spectral analysis."""

import pytest

from stigmergy.policy.spectral import SpectralAnalyzer, SpectralSnapshot
from datetime import datetime, timezone


class TestLeftShift:
    def test_normal_not_shifted(self):
        """Normal energy distribution — no shift detected."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.5,
            high_freq_ratio=0.5,
            spectral_gap=1.5,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        assert not s.is_right_shifted
        assert not s.is_left_shifted
        assert s.severity == "low"

    def test_right_shift_detected(self):
        """High-frequency dominant — Discovery signal."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.25,
            high_freq_ratio=0.75,
            spectral_gap=1.5,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        assert s.is_right_shifted
        assert not s.is_left_shifted

    def test_left_shift_detected(self):
        """Low-frequency dominant with small gap — eigenvalue collapse."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.80,
            high_freq_ratio=0.20,
            spectral_gap=0.3,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        assert not s.is_right_shifted
        assert s.is_left_shifted
        assert s.severity == "medium"

    def test_left_shift_requires_small_gap(self):
        """High low-freq but large gap = well-connected, not collapsed."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.80,
            high_freq_ratio=0.20,
            spectral_gap=1.5,  # Large gap = healthy
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        assert not s.is_left_shifted

    def test_effective_dimensionality_healthy(self):
        """Good gap ratio = high effective dimensionality."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.5,
            high_freq_ratio=0.5,
            spectral_gap=2.0,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        assert s.effective_dimensionality == pytest.approx(1.0)

    def test_effective_dimensionality_collapsed(self):
        """Small gap relative to max = low dimensionality."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.8,
            high_freq_ratio=0.2,
            spectral_gap=0.1,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        assert s.effective_dimensionality < 0.1

    def test_effective_dimensionality_edge_cases(self):
        """Zero energy / zero max eigenvalue."""
        s = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=0.0,
            low_freq_ratio=0.5,
            high_freq_ratio=0.5,
            spectral_gap=0.0,
            max_eigenvalue=0.0,
            node_count=1,
            edge_count=0,
        )
        assert s.effective_dimensionality == 1.0


class TestSpectralAnalyzerLeftShift:
    def test_algedonic_alert_right_shift(self):
        analyzer = SpectralAnalyzer()
        snapshot = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.25,
            high_freq_ratio=0.75,
            spectral_gap=1.5,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        analyzer._history.append(snapshot)
        alert = analyzer.algedonic_alert()
        assert alert is not None
        assert alert["subtype"] == "right_shift"

    def test_algedonic_alert_left_shift(self):
        analyzer = SpectralAnalyzer()
        snapshot = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.80,
            high_freq_ratio=0.20,
            spectral_gap=0.3,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        analyzer._history.append(snapshot)
        alert = analyzer.algedonic_alert()
        assert alert is not None
        assert alert["subtype"] == "left_shift"
        assert "eigenvalue collapse" in alert["message"].lower()

    def test_no_alert_normal(self):
        analyzer = SpectralAnalyzer()
        snapshot = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.5,
            high_freq_ratio=0.5,
            spectral_gap=1.5,
            max_eigenvalue=4.0,
            node_count=10,
            edge_count=15,
        )
        analyzer._history.append(snapshot)
        assert analyzer.algedonic_alert() is None

    def test_eigenvalue_trend_collapsing(self):
        analyzer = SpectralAnalyzer()
        # Declining effective dimensionality over 3 snapshots
        for gap in [2.0, 1.0, 0.2]:
            analyzer._history.append(SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=10.0,
                low_freq_ratio=0.5,
                high_freq_ratio=0.5,
                spectral_gap=gap,
                max_eigenvalue=4.0,
                node_count=10,
                edge_count=15,
            ))
        assert analyzer.eigenvalue_trend == "collapsing"

    def test_eigenvalue_trend_recovering(self):
        analyzer = SpectralAnalyzer()
        # Increasing effective dimensionality
        for gap in [0.2, 1.0, 2.0]:
            analyzer._history.append(SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=10.0,
                low_freq_ratio=0.5,
                high_freq_ratio=0.5,
                spectral_gap=gap,
                max_eigenvalue=4.0,
                node_count=10,
                edge_count=15,
            ))
        assert analyzer.eigenvalue_trend == "recovering"

    def test_eigenvalue_trend_stable(self):
        analyzer = SpectralAnalyzer()
        for _ in range(3):
            analyzer._history.append(SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=10.0,
                low_freq_ratio=0.5,
                high_freq_ratio=0.5,
                spectral_gap=1.5,
                max_eigenvalue=4.0,
                node_count=10,
                edge_count=15,
            ))
        assert analyzer.eigenvalue_trend == "stable"
