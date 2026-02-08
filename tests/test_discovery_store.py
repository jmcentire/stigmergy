"""Tests for discovery store — system self-knowledge persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stigmergy.mesh.discovery_store import Discovery, DiscoveryStore


# ── Helpers ──────────────────────────────────────────────────


def _make_discovery(
    obs_type: str = "mesh_health",
    summary: str = "Mesh is converging well",
    confidence: float = 0.8,
    decay_weight: float = 1.0,
) -> Discovery:
    return Discovery(
        observation_type=obs_type,
        summary=summary,
        confidence=confidence,
        decay_weight=decay_weight,
        source_agent="meta_modeler",
    )


# ── Tests ────────────────────────────────────────────────────


class TestDiscovery:
    def test_defaults(self):
        d = Discovery(observation_type="mesh_health", summary="test")
        assert d.decay_weight == 1.0
        assert d.confidence == 0.5
        assert d.signal_index == 0
        assert d.timestamp  # auto-generated


class TestDiscoveryStore:
    def test_deposit_and_sense(self):
        store = DiscoveryStore(path=Path("/tmp/_test_discovery_store.json"))
        store.clear()
        store.deposit(_make_discovery(summary="Observation 1"))
        store.deposit(_make_discovery(summary="Observation 2"))

        results = store.sense()
        assert len(results) == 2
        # Most recent first
        assert results[0]["summary"] == "Observation 2"
        assert results[1]["summary"] == "Observation 1"

    def test_sense_filter_by_type(self):
        store = DiscoveryStore(path=Path("/tmp/_test_discovery_store_filter.json"))
        store.clear()
        store.deposit(_make_discovery(obs_type="mesh_health", summary="A"))
        store.deposit(_make_discovery(obs_type="happy_place", summary="B"))
        store.deposit(_make_discovery(obs_type="mesh_health", summary="C"))

        health = store.sense(observation_type="mesh_health")
        assert len(health) == 2
        happy = store.sense(observation_type="happy_place")
        assert len(happy) == 1
        assert happy[0]["summary"] == "B"

    def test_sense_limit(self):
        store = DiscoveryStore(path=Path("/tmp/_test_discovery_store_limit.json"))
        store.clear()
        for i in range(20):
            store.deposit(_make_discovery(summary=f"Obs {i}"))

        results = store.sense(limit=5)
        assert len(results) == 5

    def test_happy_places_sorted_by_weighted_score(self):
        store = DiscoveryStore(path=Path("/tmp/_test_discovery_store_happy.json"))
        store.clear()
        store.deposit(_make_discovery(
            obs_type="happy_place", summary="Low confidence",
            confidence=0.3, decay_weight=1.0,
        ))
        store.deposit(_make_discovery(
            obs_type="happy_place", summary="High confidence",
            confidence=0.9, decay_weight=1.0,
        ))
        store.deposit(_make_discovery(
            obs_type="happy_place", summary="Decayed",
            confidence=0.9, decay_weight=0.2,
        ))

        happy = store.happy_places(limit=3)
        assert len(happy) == 3
        # Highest confidence * decay_weight first
        assert happy[0]["summary"] == "High confidence"

    def test_persistence_and_session_decay(self, tmp_path):
        path = tmp_path / "discoveries.json"

        # Session 1: deposit discoveries
        store1 = DiscoveryStore(path=path)
        store1.deposit(_make_discovery(
            obs_type="mesh_health", summary="Normal obs",
            decay_weight=1.0,
        ))
        store1.deposit(_make_discovery(
            obs_type="happy_place", summary="Happy snapshot",
            decay_weight=1.0,
        ))
        store1.save()

        # Session 2: reload — decay should be applied
        store2 = DiscoveryStore(path=path)
        results = store2.sense()
        assert len(results) == 2

        # Normal decay: 1.0 * 0.8 = 0.8
        normal = [d for d in results if d["observation_type"] == "mesh_health"]
        assert len(normal) == 1
        assert normal[0]["decay_weight"] == pytest.approx(0.8)

        # Happy-place decay: 1.0 * 0.95 = 0.95
        happy = [d for d in results if d["observation_type"] == "happy_place"]
        assert len(happy) == 1
        assert happy[0]["decay_weight"] == pytest.approx(0.95)

    def test_pruning_below_threshold(self, tmp_path):
        path = tmp_path / "discoveries.json"

        # Write a discovery with very low decay weight
        data = [
            {
                "observation_type": "mesh_health",
                "summary": "Ancient observation",
                "confidence": 0.5,
                "decay_weight": 0.05,  # below 0.1 threshold
                "source_agent": "test",
                "timestamp": "2024-01-01T00:00:00",
                "signal_index": 0,
                "structured": {},
            },
            {
                "observation_type": "mesh_health",
                "summary": "Recent observation",
                "confidence": 0.5,
                "decay_weight": 0.5,
                "source_agent": "test",
                "timestamp": "2024-01-01T00:00:00",
                "signal_index": 0,
                "structured": {},
            },
        ]
        path.write_text(json.dumps(data))

        store = DiscoveryStore(path=path)
        # After session decay: 0.05*0.8=0.04 (pruned), 0.5*0.8=0.4 (kept)
        assert store.count == 1
        results = store.sense()
        assert results[0]["summary"] == "Recent observation"

    def test_min_decay_filter(self):
        store = DiscoveryStore(path=Path("/tmp/_test_discovery_store_mindecay.json"))
        store.clear()
        store.deposit(_make_discovery(summary="Strong", decay_weight=0.9))
        store.deposit(_make_discovery(summary="Weak", decay_weight=0.2))

        strong = store.sense(min_decay=0.5)
        assert len(strong) == 1
        assert strong[0]["summary"] == "Strong"

    def test_empty_store(self):
        store = DiscoveryStore(path=Path("/tmp/_test_discovery_store_empty.json"))
        store.clear()
        assert store.count == 0
        assert store.sense() == []
        assert store.happy_places() == []
