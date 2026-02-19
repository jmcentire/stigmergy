"""Tests for mock adapter signal shapes and enriched metadata."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from stigmergy.adapters.github import MockGitHubAdapter, _MOCK_EVENTS
from stigmergy.adapters.grafana import (
    MockGrafanaAdapter,
    _MOCK_ALERTS,
    _MOCK_ANNOTATIONS,
    _MOCK_METRIC_ANOMALIES,
    _MOCK_TEMPO_TRACES,
)
from stigmergy.adapters.linear import MockLinearAdapter, _MOCK_TICKETS
from stigmergy.primitives.signal import SignalSource


def _run(coro):
    return asyncio.run(coro)


class TestMockGitHubAdapter:
    def test_emits_all_events(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        assert len(signals) == len(_MOCK_EVENTS)

    def test_all_signals_are_github_source(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        for sig in signals:
            assert sig.source == SignalSource.GITHUB

    def test_pr_comment_signal_shape(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        comments = [s for s in signals if s.metadata.get("event_type") == "pr_comment"]
        assert len(comments) >= 1
        c = comments[0]
        assert "pr_number" in c.metadata
        assert "parent_url" in c.metadata
        assert c.author != ""

    def test_review_signal_shape(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        reviews = [s for s in signals if s.metadata.get("event_type") == "review"]
        assert len(reviews) >= 1
        r = reviews[0]
        assert "review_state" in r.metadata
        assert r.metadata["review_state"] in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED")
        assert "pr_number" in r.metadata

    def test_check_run_signal_shape(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        checks = [s for s in signals if s.metadata.get("event_type") == "check_run"]
        assert len(checks) >= 1
        for c in checks:
            assert c.metadata["conclusion"] in ("failure", "timed_out", "cancelled")
            assert "run_name" in c.metadata
            assert "branch" in c.metadata
            assert c.author == "github-actions"

    def test_pr_enriched_metadata(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        prs = [s for s in signals if s.metadata.get("event_type") == "pull_request"]
        assert len(prs) >= 1
        for pr in prs:
            assert "branch" in pr.metadata
            assert "is_draft" in pr.metadata
            assert "review_decision" in pr.metadata
            assert "state" in pr.metadata

    def test_merged_pr_has_merged_metadata(self):
        adapter = MockGitHubAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        merged = [s for s in signals if s.metadata.get("merged_at")]
        assert len(merged) >= 1
        m = merged[0]
        assert "merged_by" in m.metadata


class TestMockLinearAdapter:
    def test_emits_all_tickets(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        assert len(signals) == len(_MOCK_TICKETS)

    def test_all_signals_are_linear_source(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        for sig in signals:
            assert sig.source == SignalSource.LINEAR

    def test_comment_signal_shape(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        comments = [s for s in signals if s.metadata.get("event_type") == "linear_comment"]
        assert len(comments) >= 1
        for c in comments:
            assert "identifier" in c.metadata
            assert "team" in c.metadata
            assert c.author != ""

    def test_enriched_metadata_estimate(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_estimate = [s for s in signals if "estimate" in s.metadata]
        assert len(with_estimate) >= 1

    def test_enriched_metadata_cycle(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_cycle = [s for s in signals if "cycle" in s.metadata]
        assert len(with_cycle) >= 1
        c = with_cycle[0]
        assert "cycle_start" in c.metadata
        assert "cycle_end" in c.metadata

    def test_enriched_metadata_relations(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_relations = [s for s in signals if "relations" in s.metadata]
        assert len(with_relations) >= 1
        r = with_relations[0]
        assert isinstance(r.metadata["relations"], list)
        assert "type" in r.metadata["relations"][0]
        assert "issue" in r.metadata["relations"][0]

    def test_enriched_metadata_parent(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_parent = [s for s in signals if "parent" in s.metadata]
        assert len(with_parent) >= 1
        p = with_parent[0]
        assert "parent_title" in p.metadata

    def test_enriched_metadata_children(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_children = [s for s in signals if "children" in s.metadata]
        assert len(with_children) >= 1
        c = with_children[0]
        assert isinstance(c.metadata["children"], list)
        assert "id" in c.metadata["children"][0]
        assert "title" in c.metadata["children"][0]
        assert "status" in c.metadata["children"][0]

    def test_enriched_metadata_project(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_project = [s for s in signals if "project" in s.metadata]
        assert len(with_project) >= 1

    def test_enriched_metadata_due_date(self):
        adapter = MockLinearAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        with_due = [s for s in signals if "due_date" in s.metadata]
        assert len(with_due) >= 1


class TestMockGrafanaAdapter:
    def test_emits_all_events(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        expected = len(_MOCK_ALERTS) + len(_MOCK_ANNOTATIONS) + len(_MOCK_TEMPO_TRACES) + len(_MOCK_METRIC_ANOMALIES)
        assert len(signals) == expected

    def test_all_signals_are_grafana_source(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        for sig in signals:
            assert sig.source == SignalSource.GRAFANA

    def test_alert_signal_shape(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        alerts = [s for s in signals if s.metadata.get("event_type") == "alert"]
        assert len(alerts) == len(_MOCK_ALERTS)
        for a in alerts:
            assert "state" in a.metadata
            assert a.metadata["state"] in ("firing", "resolved")
            assert "severity" in a.metadata
            assert "fingerprint" in a.metadata

    def test_resolved_alert_has_ends_at(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        resolved = [s for s in signals if s.metadata.get("state") == "resolved"]
        assert len(resolved) >= 1
        for r in resolved:
            assert "ends_at" in r.metadata

    def test_annotation_signal_shape(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        annotations = [s for s in signals if s.metadata.get("event_type") == "annotation"]
        assert len(annotations) == len(_MOCK_ANNOTATIONS)
        for a in annotations:
            assert "annotation_id" in a.metadata
            assert "tags" in a.metadata
            assert isinstance(a.metadata["tags"], list)

    def test_annotation_region_has_time_end(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        annotations = [s for s in signals if s.metadata.get("event_type") == "annotation"]
        with_end = [a for a in annotations if "time_end" in a.metadata]
        assert len(with_end) >= 1

    def test_tempo_trace_signal_shape(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        traces = [s for s in signals if s.metadata.get("event_type") == "tempo_trace"]
        assert len(traces) == len(_MOCK_TEMPO_TRACES)
        for t in traces:
            assert "trace_id" in t.metadata
            assert "root_service" in t.metadata or "services" in t.metadata
            assert "duration_ms" in t.metadata
            assert "status" in t.metadata
            assert t.metadata["status"] in ("ok", "error")

    def test_tempo_cross_service_spans(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        traces = [s for s in signals if s.metadata.get("event_type") == "tempo_trace"]
        multi_service = [t for t in traces if len(t.metadata.get("services", [])) > 1]
        assert len(multi_service) >= 1
        # Verify spans list is present with multiple services
        for t in multi_service:
            assert "spans" in t.metadata
            span_services = {s["service"] for s in t.metadata["spans"]}
            assert len(span_services) > 1

    def test_metric_anomaly_signal_shape(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        anomalies = [s for s in signals if s.metadata.get("event_type") == "metric_anomaly"]
        assert len(anomalies) == len(_MOCK_METRIC_ANOMALIES)
        for a in anomalies:
            assert "metric_name" in a.metadata
            assert "z_score" in a.metadata
            assert a.metadata["z_score"] > 0
            assert "direction" in a.metadata
            assert a.metadata["direction"] in ("above", "below")

    def test_channels_use_grafana_prefix(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        signals = _run(adapter.emit_all())
        for sig in signals:
            assert sig.channel.startswith("grafana:")

    def test_callback_fires_for_each_signal(self):
        adapter = MockGrafanaAdapter()
        _run(adapter.connect())
        received = []
        _run(adapter.subscribe(lambda s: received.append(s)))
        signals = _run(adapter.emit_all())
        assert len(received) == len(signals)
        assert len(received) > 0
