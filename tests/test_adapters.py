"""Tests for mock adapter signal shapes and enriched metadata."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from stigmergy.adapters.github import MockGitHubAdapter, _MOCK_EVENTS
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
