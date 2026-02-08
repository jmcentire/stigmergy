"""Tests for pheromone feedback loop — correlator reads prior annotations."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from stigmergy.mesh.annotations import AnnotationStore, Annotation
from stigmergy.services.agent_prompts import correlator_prompt


class TestCorrelatorPromptPriorFindings:
    """correlator_prompt() accepts and includes prior_findings context."""

    def test_prior_findings_included_in_prompt(self):
        """Prior findings text should appear in the prompt."""
        summaries = ["[github/repo] @alice: Fixed bug in pricing"]
        prior = "- [github/repo] (confidence=0.85): OPER-2010 tax risk detected"

        prompt = correlator_prompt(summaries, prior_findings=prior)

        assert "Prior findings already identified" in prompt
        assert "OPER-2010 tax risk detected" in prompt
        assert "genuinely NEW" in prompt
        assert "do NOT re-emit" in prompt

    def test_no_prior_findings_no_section(self):
        """Empty prior_findings should not add the prior findings section."""
        summaries = ["[github/repo] @alice: Fixed bug in pricing"]

        prompt = correlator_prompt(summaries, prior_findings="")

        assert "Prior findings already identified" not in prompt
        # Original prompt structure preserved
        assert "Review these recent signals" in prompt
        assert "Report only patterns" in prompt

    def test_prior_findings_with_graph_context(self):
        """Both prior_findings and graph_context should coexist."""
        summaries = ["[github/repo] @alice: Fixed bug"]
        graph = "  @alice ↔ @bob: 3 hops"
        prior = "- [github/repo] (confidence=0.90): duplicate work detected"

        prompt = correlator_prompt(summaries, graph_context=graph, prior_findings=prior)

        assert "bridge distance" in prompt
        assert "@alice ↔ @bob" in prompt
        assert "duplicate work detected" in prompt
        assert "Prior findings already identified" in prompt


class TestCorrelatorReadsPriorAnnotations:
    """The correlator gathers prior annotations from the annotation store."""

    def test_prior_correlator_annotations_gathered(self, tmp_path):
        """Annotations deposited by a previous correlator call should be gathered."""
        store = AnnotationStore(path=tmp_path / "annotations.json")

        # Deposit a correlator annotation for a resource
        store.annotate(
            "github", "acme-org/pricing#123",
            Annotation(
                agent_type="correlator",
                summary="OPER-2010: tax calculation risk in pricing module",
                confidence=0.85,
            ),
        )
        # Deposit a surfacer annotation (should be ignored)
        store.annotate(
            "github", "acme-org/pricing#123",
            Annotation(
                agent_type="surfacer",
                summary="Surfaced for review",
                confidence=0.7,
            ),
        )

        # Query back — simulate what agent.py does
        anns = store.get("github", "acme-org/pricing#123")
        correlator_anns = [a for a in anns if a.get("agent_type") == "correlator"]

        assert len(correlator_anns) == 1
        assert "OPER-2010" in correlator_anns[0]["summary"]
        assert correlator_anns[0]["confidence"] == 0.85

    def test_no_annotations_yields_empty(self, tmp_path):
        """No annotations for a resource → empty list."""
        store = AnnotationStore(path=tmp_path / "annotations.json")

        anns = store.get("github", "acme-org/unknown-repo#999")
        assert anns == []


class TestStreamDedupSkipsKnownFinding:
    """_stream_insight skips writing when finding hash is already in seen_hashes."""

    def test_known_finding_skipped(self):
        """Finding hash already in seen_hashes → not written to stream."""
        from stigmergy.cli.run_cmd import _stream_insight
        from stigmergy.mesh.insight_store import _finding_hash

        class _MockInsight:
            type = "risk_convergence"
            summary = "OPER-2010: tax calculation risk in pricing module"
            confidence = 0.85
            signal_ids = [uuid4()]
            details = {"actors": ["alice", "bob"], "severity": "high"}
            agent_id = uuid4()

        ins = _MockInsight()

        # Pre-populate the seen_hashes set
        fh = _finding_hash(ins.type, ins.summary, ins.details["actors"])
        seen = {fh}

        stream = io.StringIO()
        result = _stream_insight(stream, ins, seen_hashes=seen)

        assert result is False
        assert stream.getvalue() == ""  # Nothing written

    def test_unknown_finding_written(self):
        """Finding hash NOT in seen_hashes → written to stream."""
        from stigmergy.cli.run_cmd import _stream_insight

        class _MockInsight:
            type = "risk_convergence"
            summary = "Brand new finding never seen before"
            confidence = 0.9
            signal_ids = [uuid4()]
            details = {"actors": ["charlie"], "severity": "medium"}
            agent_id = uuid4()

        ins = _MockInsight()
        seen: set[str] = set()

        stream = io.StringIO()
        result = _stream_insight(stream, ins, seen_hashes=seen)

        assert result is True
        assert stream.getvalue() != ""
        assert "Brand new finding" in stream.getvalue()
        assert len(seen) == 1  # Hash was registered

    def test_no_seen_hashes_always_writes(self):
        """Without seen_hashes, _stream_insight always writes."""
        from stigmergy.cli.run_cmd import _stream_insight

        class _MockInsight:
            type = "overlap"
            summary = "Test finding"
            confidence = 0.5
            signal_ids = [uuid4()]
            details = {}
            agent_id = uuid4()

        stream = io.StringIO()
        result = _stream_insight(stream, _MockInsight(), seen_hashes=None)

        assert result is True
        assert stream.getvalue() != ""
