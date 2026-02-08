"""Tests for InsightDeduplicator and stream-level dedup."""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from stigmergy.cli.output import InsightDeduplicator
from stigmergy.cli.run_cmd import _stream_insight
from stigmergy.mesh.insight_store import _finding_hash


@dataclass
class FakeInsight:
    """Minimal insight for testing dedup logic."""

    type: str = "cross_signal_pattern"
    summary: str = "Guesty sync failures across 3 repos"
    confidence: float = 0.8
    signal_ids: list = field(default_factory=list)
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    details: dict[str, Any] = field(default_factory=dict)


class TestInsightDeduplicator:
    def test_first_insight_displayed(self):
        dedup = InsightDeduplicator()
        ins = FakeInsight()
        assert dedup.should_display(ins) is True

    def test_duplicate_suppressed(self):
        dedup = InsightDeduplicator()
        ins = FakeInsight()
        assert dedup.should_display(ins) is True
        assert dedup.should_display(ins) is False

    def test_different_insights_both_shown(self):
        dedup = InsightDeduplicator()
        ins1 = FakeInsight(summary="Guesty sync failures")
        ins2 = FakeInsight(summary="Payment webhook timeouts")
        assert dedup.should_display(ins1) is True
        assert dedup.should_display(ins2) is True

    def test_same_type_different_summary(self):
        dedup = InsightDeduplicator()
        ins1 = FakeInsight(type="risk", summary="Database migration risk")
        ins2 = FakeInsight(type="risk", summary="API deprecation risk")
        assert dedup.should_display(ins1) is True
        assert dedup.should_display(ins2) is True

    def test_different_type_same_summary(self):
        dedup = InsightDeduplicator()
        ins1 = FakeInsight(type="pattern", summary="Guesty sync failures")
        ins2 = FakeInsight(type="risk", summary="Guesty sync failures")
        assert dedup.should_display(ins1) is True
        assert dedup.should_display(ins2) is True

    def test_flush_reports_count(self, capsys):
        dedup = InsightDeduplicator()
        ins = FakeInsight()
        dedup.should_display(ins)
        # Duplicate 5 times
        for _ in range(5):
            dedup.should_display(ins)
        dedup.flush()
        captured = capsys.readouterr()
        assert "5 duplicate insight(s) suppressed" in captured.out

    def test_flush_no_output_when_no_suppression(self, capsys):
        dedup = InsightDeduplicator()
        ins = FakeInsight()
        dedup.should_display(ins)
        dedup.flush()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_flush_resets_counter(self, capsys):
        dedup = InsightDeduplicator()
        ins = FakeInsight()
        dedup.should_display(ins)
        dedup.should_display(ins)  # 1 suppressed
        dedup.flush()
        capsys.readouterr()  # consume output
        dedup.should_display(ins)  # still suppressed (seen set persists)
        dedup.flush()
        captured = capsys.readouterr()
        assert "1 duplicate insight(s) suppressed" in captured.out

    def test_summary_truncation_at_80(self):
        dedup = InsightDeduplicator()
        long_summary = "A" * 100
        ins1 = FakeInsight(summary=long_summary)
        ins2 = FakeInsight(summary=long_summary[:80] + "DIFFERENT_SUFFIX")
        # Both share the same first 80 chars, so second should be suppressed
        assert dedup.should_display(ins1) is True
        assert dedup.should_display(ins2) is False


class TestStreamDedup:
    """Tests for _stream_insight with seen_hashes dedup."""

    def test_first_write_succeeds(self):
        stream = io.StringIO()
        seen = set()
        ins = FakeInsight(details={"actors": ["@alice"]})
        assert _stream_insight(stream, ins, seen_hashes=seen) is True
        assert len(seen) == 1
        lines = stream.getvalue().strip().split("\n")
        assert len(lines) == 1

    def test_duplicate_blocked(self):
        stream = io.StringIO()
        seen = set()
        ins = FakeInsight(details={"actors": ["@alice"]})
        _stream_insight(stream, ins, seen_hashes=seen)
        # Same content, different object
        ins2 = FakeInsight(details={"actors": ["@alice"]})
        assert _stream_insight(stream, ins2, seen_hashes=seen) is False
        lines = stream.getvalue().strip().split("\n")
        assert len(lines) == 1  # Only one entry written

    def test_different_actors_different_hash(self):
        stream = io.StringIO()
        seen = set()
        ins1 = FakeInsight(details={"actors": ["@alice"]})
        ins2 = FakeInsight(details={"actors": ["@bob"]})
        assert _stream_insight(stream, ins1, seen_hashes=seen) is True
        assert _stream_insight(stream, ins2, seen_hashes=seen) is True
        assert len(seen) == 2

    def test_same_content_11_times_only_one_written(self):
        """Reproduces the 11x repeated db_separation_migration bug."""
        stream = io.StringIO()
        seen = set()
        for _ in range(11):
            ins = FakeInsight(
                type="database_separation_migration",
                summary="@alice is conducting a comprehensive audit across four tracking tickets",
                details={"actors": ["@alice"]},
            )
            _stream_insight(stream, ins, seen_hashes=seen)
        lines = [l for l in stream.getvalue().strip().split("\n") if l]
        assert len(lines) == 1
        assert len(seen) == 1

    def test_pre_populated_hashes_block_writes(self):
        """Cross-run dedup: pre-populate from existing stream."""
        stream = io.StringIO()
        # Pre-populate with a known hash
        fh = _finding_hash("overlap", "@alice and @bob both doing X", ["@alice", "@bob"])
        seen = {fh}
        ins = FakeInsight(
            type="overlap",
            summary="@alice and @bob both doing X",
            details={"actors": ["@alice", "@bob"]},
        )
        assert _stream_insight(stream, ins, seen_hashes=seen) is False
        assert stream.getvalue() == ""  # Nothing written

    def test_no_seen_hashes_always_writes(self):
        """Without dedup set, all writes succeed (backward compat)."""
        stream = io.StringIO()
        ins = FakeInsight(details={"actors": ["@alice"]})
        assert _stream_insight(stream, ins, seen_hashes=None) is True
        assert _stream_insight(stream, ins, seen_hashes=None) is True
        lines = [l for l in stream.getvalue().strip().split("\n") if l]
        assert len(lines) == 2

    def test_finding_hash_included_in_record(self):
        stream = io.StringIO()
        ins = FakeInsight(details={"actors": ["@alice"]})
        _stream_insight(stream, ins, seen_hashes=set())
        record = json.loads(stream.getvalue().strip())
        assert "finding_hash" in record
        assert len(record["finding_hash"]) == 16
