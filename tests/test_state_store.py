"""Tests for mesh state persistence — warm restart without losing self-organization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from stigmergy.mesh.state_store import (
    load_mesh_state,
    restore_agent_state,
    restore_compression_tracker,
    restore_worker_state,
    save_mesh_state,
)


# ---------------------------------------------------------------------------
# Fakes — minimal duck-typed objects matching what state_store expects
# ---------------------------------------------------------------------------

@dataclass
class _FakeContext:
    id: UUID = field(default_factory=uuid4)


@dataclass
class _FakeWorker:
    context: _FakeContext = field(default_factory=_FakeContext)
    signals_received: int = 0
    signals_accepted: int = 0
    signals_forwarded: int = 0
    rolling_avg_familiarity: float = 0.0
    source_name: str | None = None

    @property
    def id(self) -> UUID:
        return self.context.id


@dataclass
class _FakeCompetencies:
    weights: dict[str, float] = field(default_factory=lambda: {
        "similarity_detection": 0.5,
        "temporal_analysis": 0.5,
    })


@dataclass
class _FakeAgent:
    id: UUID = field(default_factory=uuid4)
    competencies: _FakeCompetencies = field(default_factory=_FakeCompetencies)
    confidence: float = 0.5


@dataclass
class _FakeTracker:
    _ema: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _peaks: dict[str, float] = field(default_factory=dict)
    _prev_ema: dict[str, float] = field(default_factory=dict)
    _initial: dict[str, float] = field(default_factory=dict)
    _ld: dict[str, float] = field(default_factory=dict)
    _se: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_creates_file(self, tmp_path):
        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[], path=p)
        assert p.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "deep" / "state.json"
        save_mesh_state(workers=[], agents=[], path=p)
        assert p.exists()

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_mesh_state(tmp_path / "nope.json") == {}

    def test_load_corrupt_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json!!!")
        assert load_mesh_state(p) == {}

    def test_round_trip_empty(self, tmp_path):
        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[], path=p)
        loaded = load_mesh_state(p)
        assert loaded["workers"] == {}
        assert loaded["agents"] == {}
        assert "compression_tracker" not in loaded

    def test_round_trip_workers(self, tmp_path):
        w = _FakeWorker(
            signals_received=42,
            signals_accepted=30,
            signals_forwarded=5,
            rolling_avg_familiarity=0.7123456789,
            source_name="slack",
        )
        p = tmp_path / "state.json"
        save_mesh_state(workers=[w], agents=[], path=p)
        loaded = load_mesh_state(p)
        ws = loaded["workers"][str(w.id)]
        assert ws["signals_received"] == 42
        assert ws["signals_accepted"] == 30
        assert ws["signals_forwarded"] == 5
        assert ws["rolling_avg_familiarity"] == pytest.approx(0.712346, abs=1e-5)
        assert ws["source_name"] == "slack"

    def test_round_trip_agents(self, tmp_path):
        a = _FakeAgent(confidence=0.8123456789)
        a.competencies.weights = {"risk_detection": 0.9, "temporal_analysis": 0.3}
        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[a], path=p)
        loaded = load_mesh_state(p)
        ag = loaded["agents"][str(a.id)]
        assert ag["competencies"]["risk_detection"] == pytest.approx(0.9, abs=1e-5)
        assert ag["competencies"]["temporal_analysis"] == pytest.approx(0.3, abs=1e-5)
        assert ag["confidence"] == pytest.approx(0.812346, abs=1e-5)

    def test_round_trip_compression_tracker(self, tmp_path):
        tracker = _FakeTracker()
        tracker._ema["general"] = 0.45
        tracker._counts["general"] = 100
        tracker._peaks["general"] = 0.78
        tracker._prev_ema["general"] = 0.42
        tracker._initial["general"] = 0.12
        tracker._ld["general"] = 0.65
        tracker._se["general"] = 3.2
        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[], compression_tracker=tracker, path=p)
        loaded = load_mesh_state(p)
        ct = loaded["compression_tracker"]["general"]
        assert ct["ema"] == pytest.approx(0.45, abs=1e-5)
        assert ct["count"] == 100
        assert ct["peak"] == pytest.approx(0.78, abs=1e-5)
        assert ct["prev_ema"] == pytest.approx(0.42, abs=1e-5)
        assert ct["initial"] == pytest.approx(0.12, abs=1e-5)
        assert ct["ld"] == pytest.approx(0.65, abs=1e-5)
        assert ct["se"] == pytest.approx(3.2, abs=1e-5)

    def test_no_tracker_omits_key(self, tmp_path):
        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[], compression_tracker=None, path=p)
        loaded = load_mesh_state(p)
        assert "compression_tracker" not in loaded

    def test_multiple_workers_and_agents(self, tmp_path):
        workers = [_FakeWorker(signals_received=i) for i in range(5)]
        agents = [_FakeAgent(confidence=0.1 * i) for i in range(3)]
        p = tmp_path / "state.json"
        save_mesh_state(workers=workers, agents=agents, path=p)
        loaded = load_mesh_state(p)
        assert len(loaded["workers"]) == 5
        assert len(loaded["agents"]) == 3


# ---------------------------------------------------------------------------
# Restore workers
# ---------------------------------------------------------------------------

class TestRestoreWorker:

    def test_restore_routing_stats(self):
        wid = uuid4()
        w = _FakeWorker()
        w.context.id = wid
        saved = {
            "workers": {
                str(wid): {
                    "signals_received": 100,
                    "signals_accepted": 80,
                    "signals_forwarded": 10,
                    "rolling_avg_familiarity": 0.85,
                }
            }
        }
        restore_worker_state(w, saved)
        assert w.signals_received == 100
        assert w.signals_accepted == 80
        assert w.signals_forwarded == 10
        assert w.rolling_avg_familiarity == pytest.approx(0.85)

    def test_missing_worker_is_noop(self):
        w = _FakeWorker(signals_received=5)
        restore_worker_state(w, {"workers": {}})
        assert w.signals_received == 5  # unchanged

    def test_empty_saved_is_noop(self):
        w = _FakeWorker(signals_received=5)
        restore_worker_state(w, {})
        assert w.signals_received == 5

    def test_partial_state_uses_defaults(self):
        wid = uuid4()
        w = _FakeWorker()
        w.context.id = wid
        w.signals_received = 99
        saved = {"workers": {str(wid): {"signals_received": 50}}}
        restore_worker_state(w, saved)
        assert w.signals_received == 50
        assert w.signals_accepted == 0  # default
        assert w.signals_forwarded == 0
        assert w.rolling_avg_familiarity == 0.0


# ---------------------------------------------------------------------------
# Restore agents
# ---------------------------------------------------------------------------

class TestRestoreAgent:

    def test_restore_competencies(self):
        aid = uuid4()
        a = _FakeAgent(id=aid)
        saved = {
            "agents": {
                str(aid): {
                    "competencies": {"risk_detection": 0.95, "temporal_analysis": 0.7},
                    "confidence": 0.88,
                }
            }
        }
        restore_agent_state(a, saved)
        assert a.competencies.weights["risk_detection"] == 0.95
        assert a.competencies.weights["temporal_analysis"] == 0.7
        assert a.confidence == 0.88

    def test_merge_preserves_new_competencies(self):
        """Saved weights override seeds but new config keys survive."""
        aid = uuid4()
        a = _FakeAgent(id=aid)
        a.competencies.weights["new_skill"] = 0.3
        saved = {
            "agents": {
                str(aid): {
                    "competencies": {"similarity_detection": 0.9},
                    "confidence": 0.7,
                }
            }
        }
        restore_agent_state(a, saved)
        assert a.competencies.weights["similarity_detection"] == 0.9
        assert a.competencies.weights["new_skill"] == 0.3  # preserved

    def test_missing_agent_is_noop(self):
        a = _FakeAgent(confidence=0.5)
        restore_agent_state(a, {"agents": {}})
        assert a.confidence == 0.5

    def test_no_confidence_preserves_current(self):
        aid = uuid4()
        a = _FakeAgent(id=aid, confidence=0.6)
        saved = {"agents": {str(aid): {"competencies": {"x": 0.1}}}}
        restore_agent_state(a, saved)
        assert a.confidence == 0.6  # not overwritten


# ---------------------------------------------------------------------------
# Restore compression tracker
# ---------------------------------------------------------------------------

class TestRestoreTracker:

    def test_restore_all_fields(self):
        tracker = _FakeTracker()
        saved = {
            "compression_tracker": {
                "general": {
                    "ema": 0.5,
                    "count": 200,
                    "peak": 0.9,
                    "prev_ema": 0.48,
                    "initial": 0.1,
                    "ld": 0.7,
                    "se": 3.5,
                }
            }
        }
        restore_compression_tracker(tracker, saved)
        assert tracker._ema["general"] == 0.5
        assert tracker._counts["general"] == 200
        assert tracker._peaks["general"] == 0.9
        assert tracker._prev_ema["general"] == 0.48
        assert tracker._initial["general"] == 0.1
        assert tracker._ld["general"] == 0.7
        assert tracker._se["general"] == 3.5

    def test_multiple_channels(self):
        tracker = _FakeTracker()
        saved = {
            "compression_tracker": {
                "eng-general": {"ema": 0.3, "count": 50, "peak": 0.6,
                                "prev_ema": 0.28, "initial": 0.1, "ld": 0.8, "se": 4.0},
                "eng-incidents": {"ema": 0.7, "count": 20, "peak": 0.9,
                                  "prev_ema": 0.65, "initial": 0.5, "ld": 0.5, "se": 2.5},
            }
        }
        restore_compression_tracker(tracker, saved)
        assert len(tracker._ema) == 2
        assert tracker._ema["eng-general"] == 0.3
        assert tracker._ema["eng-incidents"] == 0.7

    def test_empty_tracker_state_is_noop(self):
        tracker = _FakeTracker()
        tracker._ema["existing"] = 0.5
        restore_compression_tracker(tracker, {"compression_tracker": {}})
        assert tracker._ema["existing"] == 0.5  # untouched

    def test_missing_key_is_noop(self):
        tracker = _FakeTracker()
        restore_compression_tracker(tracker, {})
        assert len(tracker._ema) == 0

    def test_partial_channel_data_uses_defaults(self):
        tracker = _FakeTracker()
        saved = {"compression_tracker": {"ch": {"ema": 0.4, "count": 10}}}
        restore_compression_tracker(tracker, saved)
        assert tracker._ema["ch"] == 0.4
        assert tracker._counts["ch"] == 10
        assert tracker._peaks["ch"] == 0.0  # default
        assert tracker._ld["ch"] == 0.0


# ---------------------------------------------------------------------------
# Full round-trip: save then restore
# ---------------------------------------------------------------------------

class TestFullRoundTrip:

    def test_worker_survives_round_trip(self, tmp_path):
        w = _FakeWorker(
            signals_received=200,
            signals_accepted=150,
            signals_forwarded=20,
            rolling_avg_familiarity=0.92,
            source_name="github",
        )
        p = tmp_path / "state.json"
        save_mesh_state(workers=[w], agents=[], path=p)

        # New worker with same ID but fresh state
        w2 = _FakeWorker()
        w2.context.id = w.id
        saved = load_mesh_state(p)
        restore_worker_state(w2, saved)

        assert w2.signals_received == 200
        assert w2.signals_accepted == 150
        assert w2.signals_forwarded == 20
        assert w2.rolling_avg_familiarity == pytest.approx(0.92, abs=1e-4)

    def test_agent_survives_round_trip(self, tmp_path):
        a = _FakeAgent(confidence=0.9)
        a.competencies.weights = {"risk": 0.95, "temporal": 0.3}
        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[a], path=p)

        # New agent with same ID but seed weights
        a2 = _FakeAgent(id=a.id)
        saved = load_mesh_state(p)
        restore_agent_state(a2, saved)

        assert a2.competencies.weights["risk"] == pytest.approx(0.95, abs=1e-4)
        assert a2.confidence == pytest.approx(0.9, abs=1e-4)

    def test_tracker_survives_round_trip(self, tmp_path):
        tracker = _FakeTracker()
        tracker._ema["ops"] = 0.62
        tracker._counts["ops"] = 75
        tracker._peaks["ops"] = 0.88
        tracker._prev_ema["ops"] = 0.60
        tracker._initial["ops"] = 0.15
        tracker._ld["ops"] = 0.72
        tracker._se["ops"] = 3.8

        p = tmp_path / "state.json"
        save_mesh_state(workers=[], agents=[], compression_tracker=tracker, path=p)

        tracker2 = _FakeTracker()
        saved = load_mesh_state(p)
        restore_compression_tracker(tracker2, saved)

        assert tracker2._ema["ops"] == pytest.approx(0.62, abs=1e-4)
        assert tracker2._counts["ops"] == 75
        assert tracker2._peaks["ops"] == pytest.approx(0.88, abs=1e-4)
        assert tracker2._initial["ops"] == pytest.approx(0.15, abs=1e-4)
        assert tracker2._ld["ops"] == pytest.approx(0.72, abs=1e-4)
        assert tracker2._se["ops"] == pytest.approx(3.8, abs=1e-4)

    def test_full_mesh_round_trip(self, tmp_path):
        """Complete mesh state: workers + agents + tracker all survive."""
        workers = [
            _FakeWorker(signals_received=10, signals_accepted=8, source_name="slack"),
            _FakeWorker(signals_received=20, signals_accepted=15, source_name="github"),
        ]
        agents = [
            _FakeAgent(confidence=0.85),
            _FakeAgent(confidence=0.72),
        ]
        agents[0].competencies.weights = {"risk": 0.9}
        agents[1].competencies.weights = {"temporal": 0.8}

        tracker = _FakeTracker()
        tracker._ema["general"] = 0.5
        tracker._counts["general"] = 50
        tracker._peaks["general"] = 0.7
        tracker._prev_ema["general"] = 0.48
        tracker._initial["general"] = 0.1
        tracker._ld["general"] = 0.6
        tracker._se["general"] = 3.0

        p = tmp_path / "state.json"
        save_mesh_state(workers=workers, agents=agents,
                        compression_tracker=tracker, path=p)

        # Fresh objects with same IDs
        w1_new = _FakeWorker()
        w1_new.context.id = workers[0].id
        w2_new = _FakeWorker()
        w2_new.context.id = workers[1].id
        a1_new = _FakeAgent(id=agents[0].id)
        a2_new = _FakeAgent(id=agents[1].id)
        tracker_new = _FakeTracker()

        saved = load_mesh_state(p)
        restore_worker_state(w1_new, saved)
        restore_worker_state(w2_new, saved)
        restore_agent_state(a1_new, saved)
        restore_agent_state(a2_new, saved)
        restore_compression_tracker(tracker_new, saved)

        assert w1_new.signals_received == 10
        assert w2_new.signals_received == 20
        assert a1_new.competencies.weights["risk"] == pytest.approx(0.9, abs=1e-4)
        assert a2_new.competencies.weights["temporal"] == pytest.approx(0.8, abs=1e-4)
        assert a1_new.confidence == pytest.approx(0.85, abs=1e-4)
        assert tracker_new._ema["general"] == pytest.approx(0.5, abs=1e-4)
