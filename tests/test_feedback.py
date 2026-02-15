"""Tests for the feedback recording and storage module."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from stigmergy.primitives.feedback import (
    DuplicateFeedbackError,
    FeedbackError,
    FeedbackNotFoundError,
    FeedbackStore,
    FindingRecord,
    mint_feedback_id,
)


class TestMintFeedbackId:
    def test_format(self):
        fid = mint_feedback_id()
        assert fid.startswith("fb-")
        assert len(fid) >= 7  # "fb-" + at least 4 chars

    def test_uniqueness(self):
        ids = {mint_feedback_id() for _ in range(100)}
        assert len(ids) == 100


class TestFindingRecord:
    def test_frozen(self):
        record = FindingRecord(
            feedback_id="fb-abc123",
            signal_id="sig-1",
            context_id="ctx-1",
            agent_id="agent-1",
            timestamp="2026-01-01T00:00:00Z",
            content_summary="Test finding",
        )
        with pytest.raises(Exception):
            record.signal_id = "other"

    def test_invalid_feedback_id(self):
        with pytest.raises(Exception):
            FindingRecord(
                feedback_id="invalid",
                signal_id="sig-1",
                context_id="ctx-1",
                agent_id="agent-1",
                timestamp="2026-01-01T00:00:00Z",
                content_summary="Test",
            )


class TestFeedbackStore:
    @pytest.fixture
    def store(self, tmp_path):
        return FeedbackStore(data_dir=str(tmp_path / ".stigmergy"))

    async def test_record_and_get_finding(self, store):
        await store.load()
        fid = mint_feedback_id()
        record = await store.record_finding(
            feedback_id=fid,
            signal_id="sig-1",
            context_id="ctx-1",
            agent_id="agent-1",
            timestamp="2026-01-01T00:00:00Z",
            content_summary="Test finding",
        )
        assert record.feedback_id == fid
        got = await store.get_finding(fid)
        assert got.feedback_id == fid

    async def test_duplicate_raises(self, store):
        await store.load()
        fid = mint_feedback_id()
        await store.record_finding(
            feedback_id=fid,
            signal_id="sig-1",
            context_id="ctx-1",
            agent_id="agent-1",
            timestamp="2026-01-01T00:00:00Z",
            content_summary="Test",
        )
        with pytest.raises(DuplicateFeedbackError):
            await store.record_finding(
                feedback_id=fid,
                signal_id="sig-2",
                context_id="ctx-2",
                agent_id="agent-2",
                timestamp="2026-01-01T00:00:00Z",
                content_summary="Test 2",
            )

    async def test_not_found(self, store):
        await store.load()
        with pytest.raises(FeedbackNotFoundError):
            await store.get_finding("fb-nonexistent1")

    async def test_record_and_get_feedback(self, store):
        await store.load()
        fid = mint_feedback_id()
        await store.record_finding(
            feedback_id=fid,
            signal_id="sig-1",
            context_id="ctx-1",
            agent_id="agent-1",
            timestamp="2026-01-01T00:00:00Z",
            content_summary="Test",
        )
        fb = await store.record_feedback(fid, "helpful")
        assert fb.rating == "helpful"
        assert fb.feedback_id == fid

        got = await store.get_feedback(fid)
        assert got.rating == "helpful"

    async def test_re_rating(self, store):
        await store.load()
        fid = mint_feedback_id()
        await store.record_finding(
            feedback_id=fid,
            signal_id="sig-1",
            context_id="ctx-1",
            agent_id="agent-1",
            timestamp="2026-01-01T00:00:00Z",
            content_summary="Test",
        )
        await store.record_feedback(fid, "helpful")
        await store.record_feedback(fid, "wrong")
        got = await store.get_feedback(fid)
        assert got.rating == "wrong"  # Latest wins

    async def test_feedback_without_finding_raises(self, store):
        await store.load()
        with pytest.raises(FeedbackNotFoundError):
            await store.record_feedback("fb-nonexistent1", "helpful")

    async def test_list_findings(self, store):
        await store.load()
        for i in range(5):
            await store.record_finding(
                feedback_id=mint_feedback_id(),
                signal_id=f"sig-{i}",
                context_id="ctx-1",
                agent_id="agent-1",
                timestamp=f"2026-01-0{i+1}T00:00:00Z",
                content_summary=f"Finding {i}",
            )
        findings = await store.list_findings()
        assert len(findings) == 5

    async def test_list_findings_with_limit(self, store):
        await store.load()
        for i in range(5):
            await store.record_finding(
                feedback_id=mint_feedback_id(),
                signal_id=f"sig-{i}",
                context_id="ctx-1",
                agent_id="agent-1",
                timestamp=f"2026-01-0{i+1}T00:00:00Z",
                content_summary=f"Finding {i}",
            )
        findings = await store.list_findings(limit=2)
        assert len(findings) == 2

    async def test_summary(self, store):
        await store.load()
        fids = []
        for i in range(4):
            fid = mint_feedback_id()
            fids.append(fid)
            await store.record_finding(
                feedback_id=fid,
                signal_id=f"sig-{i}",
                context_id="ctx-1",
                agent_id="agent-1",
                timestamp="2026-01-01T00:00:00Z",
                content_summary=f"Finding {i}",
            )
        await store.record_feedback(fids[0], "helpful")
        await store.record_feedback(fids[1], "already_knew")
        await store.record_feedback(fids[2], "wrong")

        summary = await store.get_summary()
        assert summary.total_findings == 4
        assert summary.total_rated == 3
        assert summary.helpful_count == 1
        assert summary.already_knew_count == 1
        assert summary.wrong_count == 1
        assert summary.irrelevant_count == 0

    async def test_persistence(self, tmp_path):
        """Test that data survives store reload."""
        data_dir = str(tmp_path / ".stigmergy")
        fid = mint_feedback_id()

        # Write
        store1 = FeedbackStore(data_dir=data_dir)
        await store1.load()
        await store1.record_finding(
            feedback_id=fid,
            signal_id="sig-1",
            context_id="ctx-1",
            agent_id="agent-1",
            timestamp="2026-01-01T00:00:00Z",
            content_summary="Persistent finding",
        )
        await store1.record_feedback(fid, "helpful")

        # Read in new store
        store2 = FeedbackStore(data_dir=data_dir)
        await store2.load()
        finding = await store2.get_finding(fid)
        assert finding.content_summary == "Persistent finding"
        feedback = await store2.get_feedback(fid)
        assert feedback.rating == "helpful"
