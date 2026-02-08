"""Tests for insight persistence and finding registry."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from stigmergy.mesh.insight_store import (
    FindingRegistry,
    InsightStore,
    _finding_hash,
)
from stigmergy.mesh.insights import Insight


def _make_insight(
    type_: str = "OVERLAP",
    summary: str = "Alice and Bob both modifying pricing engine",
    confidence: float = 0.75,
    actors: list | None = None,
    severity: str = "medium",
) -> Insight:
    return Insight(
        agent_id=uuid4(),
        type=type_,
        summary=summary,
        confidence=confidence,
        signal_ids=[uuid4(), uuid4()],
        details={
            "actors": actors or ["alice", "bob"],
            "severity": severity,
            "bridge_distances": {"alice↔bob": 3.0},
        },
    )


# ── InsightStore ────────────────────────────────────────────


class TestInsightStore:
    def test_record_and_save(self, tmp_path):
        path = tmp_path / "insights.jsonl"
        store = InsightStore(path=path)

        ins = _make_insight()
        store.record(ins, score=0.42, scoring_components={"d_bridge": 0.5})
        assert store.buffered_count == 1

        count = store.save()
        assert count == 1
        assert store.buffered_count == 0
        assert path.exists()

        # Verify JSONL content
        with open(path) as f:
            line = f.readline()
        data = json.loads(line)
        assert data["type"] == "OVERLAP"
        assert data["confidence"] == 0.75
        assert data["score"] == 0.42
        assert data["run_id"] == store.run_id
        assert "created_at" in data

    def test_append_across_saves(self, tmp_path):
        path = tmp_path / "insights.jsonl"
        store = InsightStore(path=path)

        store.record(_make_insight(summary="first"))
        store.save()
        store.record(_make_insight(summary="second"))
        store.save()

        entries = store.load_all()
        assert len(entries) == 2
        assert entries[0]["summary"] == "first"
        assert entries[1]["summary"] == "second"

    def test_load_all(self, tmp_path):
        path = tmp_path / "insights.jsonl"
        store = InsightStore(path=path)

        for i in range(5):
            store.record(_make_insight(summary=f"insight-{i}"))
        store.save()

        loaded = store.load_all()
        assert len(loaded) == 5

    def test_load_by_run(self, tmp_path):
        path = tmp_path / "insights.jsonl"

        # Run 1
        store1 = InsightStore(path=path)
        store1.record(_make_insight(summary="run1-a"))
        store1.record(_make_insight(summary="run1-b"))
        store1.save()
        run1_id = store1.run_id

        # Run 2
        store2 = InsightStore(path=path)
        store2.record(_make_insight(summary="run2-a"))
        store2.save()

        # Load by run
        reader = InsightStore(path=path)
        run1_insights = reader.load_by_run(run1_id)
        assert len(run1_insights) == 2
        assert all(e["run_id"] == run1_id for e in run1_insights)

    def test_run_ids(self, tmp_path):
        path = tmp_path / "insights.jsonl"

        s1 = InsightStore(path=path)
        s1.record(_make_insight())
        s1.save()

        s2 = InsightStore(path=path)
        s2.record(_make_insight())
        s2.save()

        reader = InsightStore(path=path)
        runs = reader.run_ids()
        assert len(runs) == 2
        assert s1.run_id in runs
        assert s2.run_id in runs

    def test_empty_save(self, tmp_path):
        path = tmp_path / "insights.jsonl"
        store = InsightStore(path=path)
        count = store.save()
        assert count == 0
        assert not path.exists()

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        store = InsightStore(path=path)
        assert store.load_all() == []


# ── Finding hash ────────────────────────────────────────────


class TestFindingHash:
    def test_same_content_same_hash(self):
        h1 = _finding_hash("OVERLAP", "Alice and Bob modifying pricing", ["alice", "bob"])
        h2 = _finding_hash("OVERLAP", "Alice and Bob modifying pricing", ["bob", "alice"])
        assert h1 == h2  # Actors are sorted

    def test_different_type_different_hash(self):
        h1 = _finding_hash("OVERLAP", "Same summary", ["alice"])
        h2 = _finding_hash("RISK", "Same summary", ["alice"])
        assert h1 != h2

    def test_case_insensitive(self):
        h1 = _finding_hash("overlap", "Hello World", ["Alice"])
        h2 = _finding_hash("overlap", "hello world", ["alice"])
        assert h1 == h2


# ── FindingRegistry ─────────────────────────────────────────


class TestFindingRegistry:
    def test_new_finding_is_active(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insights = [{
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
        }]
        changed = registry.ingest_insights(insights, "run-001")
        assert len(changed) == 1
        assert changed[0].decay_stage == "active"
        assert changed[0].times_surfaced == 1

    def test_resurfaced_finding_deferred(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight = {
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
        }

        registry.ingest_insights([insight], "run-001")
        changed = registry.ingest_insights([insight], "run-002")

        assert len(changed) == 1
        assert changed[0].times_surfaced == 2
        assert changed[0].decay_stage == "deferred"

    def test_state_change_acknowledged(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight = {
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
        }

        changed = registry.ingest_insights([insight], "run-001")
        fh = changed[0].finding_hash
        registry.mark_state_change(fh)
        registry.ingest_insights([insight], "run-002")

        finding = registry._findings[fh]
        assert finding.decay_stage == "acknowledged"
        assert finding.has_state_change is True

    def test_persistence(self, tmp_path):
        path = tmp_path / "registry.json"
        registry = FindingRegistry(path=path)

        insight = {
            "type": "RISK",
            "summary": "Payment service at risk",
            "confidence": 0.8,
            "score": 0.55,
            "details": {"actors": ["nick"]},
        }
        registry.ingest_insights([insight], "run-001")
        registry.save()

        # Load fresh
        registry2 = FindingRegistry(path=path)
        assert registry2.total_findings == 1
        assert registry2.active_count == 1

    def test_decaying_findings(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight = {
            "type": "OVERLAP",
            "summary": "Duplicate work",
            "confidence": 0.7,
            "score": 0.4,
            "details": {"actors": ["alice", "bob"]},
        }

        registry.ingest_insights([insight], "run-001")
        registry.ingest_insights([insight], "run-002")

        decaying = registry.decaying_findings()
        assert len(decaying) == 1
        assert decaying[0].is_decaying

    def test_signal_ids_preserved_on_new(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insights = [{
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
            "signal_ids": ["sig-001", "sig-002"],
        }]
        changed = registry.ingest_insights(insights, "run-001")
        assert changed[0].signal_ids == ["sig-001", "sig-002"]

    def test_signal_ids_aggregated_on_resurface(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        insight1 = {
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.75,
            "score": 0.42,
            "details": {"actors": ["alice", "bob"]},
            "signal_ids": ["sig-001", "sig-002"],
        }
        insight2 = {
            "type": "OVERLAP",
            "summary": "Alice and Bob on pricing",
            "confidence": 0.80,
            "score": 0.50,
            "details": {"actors": ["alice", "bob"]},
            "signal_ids": ["sig-002", "sig-003"],  # sig-002 overlaps
        }

        registry.ingest_insights([insight1], "run-001")
        changed = registry.ingest_insights([insight2], "run-002")

        # Should have 3 unique signal_ids
        assert len(changed[0].signal_ids) == 3
        assert set(changed[0].signal_ids) == {"sig-001", "sig-002", "sig-003"}

    def test_signal_ids_persist_across_save(self, tmp_path):
        path = tmp_path / "registry.json"
        registry = FindingRegistry(path=path)

        insights = [{
            "type": "RISK",
            "summary": "Payment service at risk",
            "confidence": 0.8,
            "score": 0.55,
            "details": {"actors": ["nick"]},
            "signal_ids": ["sig-aaa", "sig-bbb"],
        }]
        registry.ingest_insights(insights, "run-001")
        registry.save()

        # Load fresh
        registry2 = FindingRegistry(path=path)
        finding = list(registry2._findings.values())[0]
        assert finding.signal_ids == ["sig-aaa", "sig-bbb"]

    def test_summary(self, tmp_path):
        registry = FindingRegistry(path=tmp_path / "registry.json")

        for i in range(3):
            registry.ingest_insights([{
                "type": f"TYPE-{i}",
                "summary": f"Finding {i}",
                "confidence": 0.5,
                "score": 0.3,
                "details": {"actors": [f"actor-{i}"]},
            }], f"run-{i}")

        stats = registry.summary()
        assert stats["total"] == 3
        assert stats["active"] == 3
