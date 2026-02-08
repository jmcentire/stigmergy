"""Tests for the policy engine: traces, structure, signals, budgets, engine, spectral."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from stigmergy.policy.budgets import ActivityBudget, BudgetConfig
from stigmergy.policy.engine import Intervention, PolicyEngine, PolicyEvaluation
from stigmergy.policy.signals import PolicySignal, detect_signals
from stigmergy.policy.spectral import SpectralAnalyzer, SpectralSnapshot
from stigmergy.policy.structure import StructureGraph
from stigmergy.policy.traces import TraceEvent, TraceStore


# ── TraceEvent ──────────────────────────────────────────────────


class TestTraceEvent:
    def test_from_signal_pr(self):
        """TraceEvent extracts PR metadata from a Signal-like object."""

        class FakeSignal:
            id = uuid4()
            content = "PR #42: Fix auth middleware\nSome body text"
            source = "github"
            channel = "acme-org/backend"
            author = "alice"
            timestamp = datetime(2026, 2, 7, tzinfo=timezone.utc)
            metadata = {
                "event_type": "pull_request",
                "pr_number": 42,
                "additions": 150,
                "deletions": 30,
                "files_changed": ["src/auth/middleware.ts", "src/auth/jwt.ts"],
                "url": "https://github.com/acme-org/backend/pull/42",
                "state": "OPEN",
            }

        event = TraceEvent.from_signal(FakeSignal())
        assert event.event_type == "pull_request"
        assert event.repo == "acme-org/backend"
        assert event.author == "alice"
        assert event.diff_lines == 180
        assert len(event.files_changed) == 2
        assert "src/auth/middleware.ts" in event.files_changed

    def test_content_addressable_id(self):
        """Same PR in same repo always gets the same trace ID."""

        class Sig1:
            id = uuid4()
            content = "PR #42: First version"
            source = "github"
            channel = "acme-org/backend"
            author = "alice"
            timestamp = datetime(2026, 2, 7, tzinfo=timezone.utc)
            metadata = {"event_type": "pull_request", "pr_number": 42}

        class Sig2:
            id = uuid4()  # Different signal ID
            content = "PR #42: Updated version"
            source = "github"
            channel = "acme-org/backend"
            author = "alice"
            timestamp = datetime(2026, 2, 7, 12, tzinfo=timezone.utc)
            metadata = {"event_type": "pull_request", "pr_number": 42}

        e1 = TraceEvent.from_signal(Sig1())
        e2 = TraceEvent.from_signal(Sig2())
        assert e1.id == e2.id  # Same PR = same trace ID


# ── TraceStore ──────────────────────────────────────────────────


class TestTraceStore:
    def test_append_and_query(self, tmp_path):
        store = TraceStore(path=tmp_path / "traces.jsonl")
        event = TraceEvent(
            id="test-001",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            title="Fix auth",
        )
        assert store.append(event) is True
        assert store.append(event) is False  # Duplicate
        assert store.count == 1

        # Query
        assert len(store.by_repo("acme-org/backend")) == 1
        assert len(store.by_author("alice")) == 1
        assert store.get("test-001") is not None

    def test_persistence(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        store1 = TraceStore(path=path)
        store1.append(TraceEvent(
            id="persist-001",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
        ))
        store1.save()

        store2 = TraceStore(path=path)
        assert store2.count == 1
        assert store2.get("persist-001") is not None


# ── StructureGraph ──────────────────────────────────────────────


class TestStructureGraph:
    def test_auth_surface_detection(self):
        assert StructureGraph.is_auth_surface("src/auth/middleware.ts")
        assert StructureGraph.is_auth_surface("services/auth/routes.ts")
        assert StructureGraph.is_auth_surface("src/jwt/verify.ts")
        assert StructureGraph.is_auth_surface("middleware/auth.ts")
        assert not StructureGraph.is_auth_surface("src/pricing/engine.ts")
        assert not StructureGraph.is_auth_surface("src/bookings/actions.ts")

    def test_file_domains(self):
        graph = StructureGraph()
        assert "pricing" in graph.file_domains("src/data/pricing/engine.ts")
        assert "booking" in graph.file_domains("src/pms/bookings/actions.ts")
        assert "availability" in graph.file_domains("src/pms/availability/sync.ts")

    def test_boundary_crossing(self):
        graph = StructureGraph()
        # Files in one domain
        assert not graph.is_boundary_crossing(["src/data/pricing/engine.ts"])
        # Files in two domains
        assert graph.is_boundary_crossing([
            "src/data/pricing/engine.ts",
            "src/pms/availability/sync.ts",
        ])

    def test_crossing_domains(self):
        graph = StructureGraph()
        domains = graph.crossing_domains([
            "src/data/pricing/engine.ts",
            "src/pms/bookings/actions.ts",
            "src/pms/availability/sync.ts",
        ])
        assert "pricing" in domains
        assert "booking" in domains
        assert "availability" in domains

    def test_persistence(self, tmp_path):
        path = tmp_path / "structure.json"
        g1 = StructureGraph()
        from stigmergy.policy.structure import Node, Edge
        g1.add_node(Node(id="acme-org/backend", node_type="repo"))
        g1.add_node(Node(id="package:@acme/cache", node_type="package"))
        g1.add_edge(Edge(source="acme-org/backend", target="package:@acme/cache", edge_type="depends_on"))
        g1.save(path)

        g2 = StructureGraph()
        assert g2.load(path) is True
        assert g2.node_count == 2
        assert g2.edge_count == 1
        assert "package:@acme/cache" in g2.neighbors("acme-org/backend")


# ── Policy Signals ──────────────────────────────────────────────


class TestPolicySignals:
    def test_dependency_crossing_detected(self):
        graph = StructureGraph()
        event = TraceEvent(
            id="test-cross",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            files_changed=[
                "src/data/pricing/engine.ts",
                "src/pms/availability/sync.ts",
            ],
            additions=100,
            deletions=50,
        )
        signals = detect_signals(event, graph)
        types = [s.signal_type for s in signals]
        assert "dependency_crossing" in types

    def test_auth_surface_detected(self):
        graph = StructureGraph()
        event = TraceEvent(
            id="test-auth",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            files_changed=["src/auth/middleware.ts", "src/auth/jwt.ts"],
            additions=50,
            deletions=10,
        )
        signals = detect_signals(event, graph)
        types = [s.signal_type for s in signals]
        assert "auth_surface_touched" in types

    def test_high_churn_detected(self):
        graph = StructureGraph()
        event = TraceEvent(
            id="test-churn",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            additions=400,
            deletions=200,
        )
        signals = detect_signals(event, graph)
        types = [s.signal_type for s in signals]
        assert "high_churn" in types

    def test_no_signals_for_normal_pr(self):
        graph = StructureGraph()
        event = TraceEvent(
            id="test-normal",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            files_changed=["src/data/pricing/engine.ts"],
            additions=30,
            deletions=10,
        )
        signals = detect_signals(event, graph)
        # Single domain, small diff, no auth
        assert len(signals) == 0


# ── Activity Budgets ────────────────────────────────────────────


class TestActivityBudget:
    def test_record_and_query(self, tmp_path):
        budget = ActivityBudget(
            config=BudgetConfig(dependency_crossings_per_week=3),
            path=tmp_path / "budgets.json",
        )
        budget.record("acme-org/backend", "dependency_crossings", 1)
        budget.record("acme-org/backend", "dependency_crossings", 1)
        assert budget.current("acme-org/backend", "dependency_crossings") == 2
        assert not budget.is_exceeded("acme-org/backend", "dependency_crossings")

        budget.record("acme-org/backend", "dependency_crossings", 1)
        budget.record("acme-org/backend", "dependency_crossings", 1)
        assert budget.is_exceeded("acme-org/backend", "dependency_crossings")

    def test_utilization(self, tmp_path):
        budget = ActivityBudget(
            config=BudgetConfig(auth_touches_per_week=4),
            path=tmp_path / "budgets.json",
        )
        budget.record("acme-org/backend", "auth_touches", 2)
        assert budget.utilization("acme-org/backend", "auth_touches") == 0.5

    def test_persistence(self, tmp_path):
        path = tmp_path / "budgets.json"
        b1 = ActivityBudget(config=BudgetConfig(), path=path)
        b1.record("acme-org/backend", "diff_lines", 1000)
        b1.save()

        b2 = ActivityBudget(config=BudgetConfig(), path=path)
        assert b2.current("acme-org/backend", "diff_lines") == 1000


# ── PolicyEngine ────────────────────────────────────────────────


class TestPolicyEngine:
    @pytest.mark.asyncio
    async def test_mechanical_flags_exceeded_auth(self, tmp_path):
        """Mechanical fallback flags auth changes when budget exceeded."""
        graph = StructureGraph()
        budget = ActivityBudget(
            config=BudgetConfig(auth_touches_per_week=2),
            path=tmp_path / "budgets.json",
        )
        # Pre-fill budget to exceed threshold
        budget.record("acme-org/backend", "auth_touches", 3)

        trace_store = TraceStore(path=tmp_path / "traces.jsonl")
        engine = PolicyEngine(graph=graph, budget=budget, trace_store=trace_store)

        event = TraceEvent(
            id="engine-auth-test",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            files_changed=["src/auth/middleware.ts"],
            additions=50,
            deletions=10,
        )

        interventions = await engine.evaluate(event)
        assert len(interventions) >= 1
        assert any(i.policy_name == "auth_gate" for i in interventions)

    @pytest.mark.asyncio
    async def test_no_intervention_for_normal_pr(self, tmp_path):
        """No intervention for a routine PR in mechanical mode."""
        graph = StructureGraph()
        budget = ActivityBudget(config=BudgetConfig(), path=tmp_path / "budgets.json")
        trace_store = TraceStore(path=tmp_path / "traces.jsonl")
        engine = PolicyEngine(graph=graph, budget=budget, trace_store=trace_store)

        event = TraceEvent(
            id="engine-normal",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
            files_changed=["src/data/pricing/engine.ts"],
            additions=30,
            deletions=10,
        )

        interventions = await engine.evaluate(event)
        assert len(interventions) == 0

    @pytest.mark.asyncio
    async def test_trace_recorded_even_without_intervention(self, tmp_path):
        """Every event produces a trace, even if no policy triggers."""
        graph = StructureGraph()
        budget = ActivityBudget(config=BudgetConfig(), path=tmp_path / "budgets.json")
        trace_store = TraceStore(path=tmp_path / "traces.jsonl")
        engine = PolicyEngine(graph=graph, budget=budget, trace_store=trace_store)

        event = TraceEvent(
            id="trace-test",
            timestamp="2026-02-07T12:00:00+00:00",
            event_type="pull_request",
            repo="acme-org/backend",
            author="alice",
        )

        await engine.evaluate(event)
        assert trace_store.count == 1


# ── Schema ──────────────────────────────────────────────────────


class TestPolicySchemas:
    def test_policy_evaluation_empty(self):
        pe = PolicyEvaluation(interventions=[], reasoning="nothing to report")
        assert len(pe.interventions) == 0

    def test_intervention_one_line(self):
        intv = Intervention(
            intervention_type="advisory",
            policy_name="auth_gate",
            signal_type="auth_surface_touched",
            repo="acme-org/backend",
            author="alice",
            message="Auth surface modified",
            severity="high",
        )
        assert "!!" in intv.one_line
        assert "auth_gate" in intv.one_line


# ── Spectral Analysis ──────────────────────────────────────────


class TestSpectralAnalyzer:
    def test_simple_graph_energy(self):
        """A simple connected graph has computable spectral energy."""
        analyzer = SpectralAnalyzer()
        adjacency = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        snapshot = analyzer.analyze(adjacency)
        assert snapshot.node_count == 3
        assert snapshot.total_energy > 0
        assert 0.0 <= snapshot.low_freq_ratio <= 1.0
        assert 0.0 <= snapshot.high_freq_ratio <= 1.0
        assert abs(snapshot.low_freq_ratio + snapshot.high_freq_ratio - 1.0) < 0.01

    def test_disconnected_graph(self):
        """Disconnected components have spectral gap of 0."""
        analyzer = SpectralAnalyzer()
        adjacency = {
            "A": ["B"],
            "B": ["A"],
            "C": ["D"],
            "D": ["C"],
        }
        snapshot = analyzer.analyze(adjacency)
        assert snapshot.node_count == 4
        # Disconnected graph has λ_2 = 0
        assert snapshot.spectral_gap < 0.01

    def test_star_graph_high_frequency(self):
        """Star topology concentrates energy in high frequencies."""
        analyzer = SpectralAnalyzer()
        # Star: center connects to all leaves, leaves don't connect to each other
        adjacency = {
            "center": ["leaf1", "leaf2", "leaf3", "leaf4", "leaf5"],
            "leaf1": ["center"],
            "leaf2": ["center"],
            "leaf3": ["center"],
            "leaf4": ["center"],
            "leaf5": ["center"],
        }
        snapshot = analyzer.analyze(adjacency)
        assert snapshot.node_count == 6
        assert snapshot.total_energy > 0
        # Star graphs tend to have high-frequency concentration
        assert snapshot.high_freq_ratio > 0.4

    def test_single_node(self):
        """Single node graph returns zero energy."""
        analyzer = SpectralAnalyzer()
        snapshot = analyzer.analyze({"A": []})
        assert snapshot.total_energy == 0.0
        assert snapshot.node_count == 1

    def test_right_shift_detection(self):
        """SpectralSnapshot correctly detects right-shift."""
        normal = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.55,
            high_freq_ratio=0.45,
            spectral_gap=1.0,
            max_eigenvalue=5.0,
            node_count=10,
            edge_count=15,
        )
        assert not normal.is_right_shifted
        assert normal.severity == "low"

        shifted = SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0,
            low_freq_ratio=0.25,
            high_freq_ratio=0.75,
            spectral_gap=0.5,
            max_eigenvalue=8.0,
            node_count=10,
            edge_count=15,
        )
        assert shifted.is_right_shifted
        assert shifted.severity == "high"

    def test_trend_detection(self):
        """Analyzer detects rising/falling/stable trends."""
        analyzer = SpectralAnalyzer()
        # Stable
        for _ in range(3):
            analyzer._record(SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=10.0, low_freq_ratio=0.5, high_freq_ratio=0.5,
                spectral_gap=1.0, max_eigenvalue=5.0, node_count=10, edge_count=15,
            ))
        assert analyzer.trend == "stable"

        # Rising
        analyzer._history.clear()
        for hi in [0.5, 0.55, 0.65]:
            analyzer._record(SpectralSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_energy=10.0, low_freq_ratio=1-hi, high_freq_ratio=hi,
                spectral_gap=1.0, max_eigenvalue=5.0, node_count=10, edge_count=15,
            ))
        assert analyzer.trend == "rising"

    def test_algedonic_alert(self):
        """Algedonic alert fires on right-shift."""
        analyzer = SpectralAnalyzer()
        analyzer._record(SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0, low_freq_ratio=0.1, high_freq_ratio=0.9,
            spectral_gap=0.3, max_eigenvalue=8.0, node_count=10, edge_count=15,
        ))
        alert = analyzer.algedonic_alert()
        assert alert is not None
        assert alert["type"] == "algedonic"
        assert alert["severity"] == "critical"
        assert "right-shift" in alert["message"].lower()

    def test_no_alert_when_normal(self):
        """No algedonic alert for normal spectral distribution."""
        analyzer = SpectralAnalyzer()
        analyzer._record(SpectralSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_energy=10.0, low_freq_ratio=0.55, high_freq_ratio=0.45,
            spectral_gap=1.0, max_eigenvalue=5.0, node_count=10, edge_count=15,
        ))
        assert analyzer.algedonic_alert() is None

    def test_flow_weights_amplify(self):
        """Flow weights amplify edges with recent activity."""
        analyzer = SpectralAnalyzer()
        adjacency = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        # Without flow
        snap1 = analyzer.analyze(adjacency)
        # With heavy flow on A-B edge
        flow = {("A", "B"): 100.0}
        snap2 = analyzer.analyze(adjacency, flow_weights=flow)
        # Flow changes the energy distribution
        assert snap1.total_energy != snap2.total_energy
