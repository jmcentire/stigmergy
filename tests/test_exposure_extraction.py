"""Tests for exposure extraction from each signal type.

Covers GitHub, Linear, and Slack signal types, including @mentions
and emoji reactions for Slack.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from stigmergy.attention.exposure import ExposureEvent, ExposureExtractor
from stigmergy.primitives.signal import Signal, SignalSource


def _now() -> datetime:
    return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _signal(
    source: str = "github",
    channel: str = "backend",
    author: str = "alice",
    content: str = "Fix pricing cache invalidation bug",
    event_type: str = "",
    **meta_kwargs,
) -> Signal:
    meta = {"event_type": event_type}
    meta.update(meta_kwargs)
    return Signal(
        content=content,
        source=source,
        channel=channel,
        author=author,
        timestamp=_now(),
        metadata=meta,
    )


# ── GitHub extraction ────────────────────────────────────────


class TestGitHubExtraction:
    """GitHub signals → exposure events."""

    def test_pr_opened_author_authored(self):
        """PR opened → author gets 'authored' exposure."""
        ex = ExposureExtractor()
        sig = _signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
        )
        events = ex.extract(sig)
        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"

    def test_pr_with_assignees(self):
        """PR opened with assignees → each assignee gets 'assigned'."""
        ex = ExposureExtractor()
        sig = _signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
            assignees=["bob", "carol"],
        )
        events = ex.extract(sig)
        assigned = [e for e in events if e.exposure_type == "assigned"]
        assert len(assigned) == 2
        assert {e.person_id for e in assigned} == {"bob", "carol"}

    def test_pr_self_assignee_no_duplicate(self):
        """Author assigned to own PR → no duplicate assigned event."""
        ex = ExposureExtractor()
        sig = _signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
            assignees=["alice", "bob"],
        )
        events = ex.extract(sig)
        assigned = [e for e in events if e.exposure_type == "assigned"]
        assert len(assigned) == 1
        assert assigned[0].person_id == "bob"

    def test_review_reviewer_reviewed(self):
        """PR review → reviewer gets 'reviewed' exposure."""
        ex = ExposureExtractor()
        # First, open the PR
        ex.extract(_signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
        ))
        # Then review
        sig = _signal(
            event_type="review",
            pr_number=42,
            author="bob",
        )
        events = ex.extract(sig)
        reviewed = [e for e in events if e.exposure_type == "reviewed"]
        assert len(reviewed) == 1
        assert reviewed[0].person_id == "bob"

    def test_review_pr_author_reexposed(self):
        """PR review → PR author gets re-exposed as 'authored'."""
        ex = ExposureExtractor()
        ex.extract(_signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
        ))
        events = ex.extract(_signal(
            event_type="review",
            pr_number=42,
            author="bob",
        ))
        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"

    def test_pr_comment_commenter_commented(self):
        """PR comment → commenter gets 'commented' exposure."""
        ex = ExposureExtractor()
        ex.extract(_signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
        ))
        events = ex.extract(_signal(
            event_type="pr_comment",
            pr_number=42,
            author="carol",
        ))
        commented = [e for e in events if e.exposure_type == "commented"]
        assert len(commented) == 1
        assert commented[0].person_id == "carol"

    def test_pr_comment_pr_author_reexposed(self):
        """PR comment → PR author gets re-exposed as 'authored'."""
        ex = ExposureExtractor()
        ex.extract(_signal(
            event_type="pull_request",
            pr_number=42,
            action="opened",
            author="alice",
        ))
        events = ex.extract(_signal(
            event_type="pr_comment",
            pr_number=42,
            author="carol",
        ))
        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"

    def test_issue_author_and_assignees(self):
        """GitHub issue → author authored, assignees assigned."""
        ex = ExposureExtractor()
        sig = _signal(
            event_type="issue",
            author="alice",
            assignees=["bob"],
        )
        events = ex.extract(sig)
        assert any(e.exposure_type == "authored" and e.person_id == "alice" for e in events)
        assert any(e.exposure_type == "assigned" and e.person_id == "bob" for e in events)


# ── Linear extraction ────────────────────────────────────────


class TestLinearExtraction:
    """Linear signals → exposure events."""

    def test_issue_created_author_authored(self):
        """Linear issue created → author gets 'authored'."""
        ex = ExposureExtractor()
        sig = _signal(
            source="linear",
            event_type="",
            author="alice",
            identifier="PLAT-123",
            team="Platform",
        )
        events = ex.extract(sig)
        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"

    def test_issue_with_assignee(self):
        """Linear issue with assignee → assignee gets 'assigned'."""
        ex = ExposureExtractor()
        sig = _signal(
            source="linear",
            event_type="",
            author="alice",
            assignee="bob",
            identifier="PLAT-123",
            team="Platform",
        )
        events = ex.extract(sig)
        assigned = [e for e in events if e.exposure_type == "assigned"]
        assert len(assigned) == 1
        assert assigned[0].person_id == "bob"

    def test_issue_self_assign_no_duplicate(self):
        """Author assigned to own issue → no assigned event."""
        ex = ExposureExtractor()
        sig = _signal(
            source="linear",
            event_type="",
            author="alice",
            assignee="alice",
            identifier="PLAT-123",
            team="Platform",
        )
        events = ex.extract(sig)
        assigned = [e for e in events if e.exposure_type == "assigned"]
        assert len(assigned) == 0

    def test_linear_comment_commenter_and_author(self):
        """Linear comment → commenter commented, issue author re-exposed."""
        ex = ExposureExtractor()
        # First, create the issue
        ex.extract(_signal(
            source="linear",
            event_type="",
            author="alice",
            identifier="PLAT-123",
            team="Platform",
        ))
        # Then comment on it
        events = ex.extract(_signal(
            source="linear",
            event_type="linear_comment",
            author="bob",
            identifier="PLAT-123",
            team="Platform",
        ))
        commented = [e for e in events if e.exposure_type == "commented"]
        assert len(commented) == 1
        assert commented[0].person_id == "bob"

        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"


# ── Slack extraction ─────────────────────────────────────────


class TestSlackExtraction:
    """Slack signals → exposure events."""

    def test_message_author_authored(self):
        """Slack message → author gets 'authored' exposure."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="The pricing cache is broken",
        )
        events = ex.extract(sig)
        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"

    def test_at_mention_in_content(self):
        """@devon in message content → devon gets 'mentioned' exposure."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="Hey @devon can you look at the pricing bug? cc @bob",
        )
        events = ex.extract(sig)
        mentioned = [e for e in events if e.exposure_type == "mentioned"]
        assert len(mentioned) == 2
        assert {e.person_id for e in mentioned} == {"devon", "bob"}

    def test_at_mention_no_self_mention(self):
        """Author @mentioning themselves doesn't create duplicate."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="I (@alice) fixed the bug",
        )
        events = ex.extract(sig)
        mentioned = [e for e in events if e.exposure_type == "mentioned"]
        assert len(mentioned) == 0

    def test_metadata_mentions(self):
        """Mentions from metadata field (structured by adapter)."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="Check this out",
            mentions=["bob", "carol"],
        )
        events = ex.extract(sig)
        mentioned = [e for e in events if e.exposure_type == "mentioned"]
        assert len(mentioned) == 2
        assert {e.person_id for e in mentioned} == {"bob", "carol"}

    def test_content_and_metadata_mentions_deduped(self):
        """Same person mentioned in both content and metadata → only one event."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="Hey @bob check this",
            mentions=["bob"],
        )
        events = ex.extract(sig)
        mentioned = [e for e in events if e.exposure_type == "mentioned"]
        assert len(mentioned) == 1
        assert mentioned[0].person_id == "bob"

    def test_emoji_reactions(self):
        """People who reacted with emoji get 'reacted' exposure."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="Deployed pricing v3",
            reactors=["bob", "carol", "devon"],
        )
        events = ex.extract(sig)
        reacted = [e for e in events if e.exposure_type == "reacted"]
        assert len(reacted) == 3
        assert {e.person_id for e in reacted} == {"bob", "carol", "devon"}

    def test_emoji_reaction_no_self_react(self):
        """Author reacting to own message doesn't create exposure."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="Deployed pricing v3",
            reactors=["alice", "bob"],
        )
        events = ex.extract(sig)
        reacted = [e for e in events if e.exposure_type == "reacted"]
        assert len(reacted) == 1
        assert reacted[0].person_id == "bob"

    def test_thread_reply_commented(self):
        """Thread reply → replier gets 'commented', parent gets re-exposed."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="bob",
            content="Good point, let me check",
            is_thread_reply=True,
            thread_author="alice",
        )
        events = ex.extract(sig)
        commented = [e for e in events if e.exposure_type == "commented"]
        assert len(commented) == 1
        assert commented[0].person_id == "bob"

        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 1
        assert authored[0].person_id == "alice"

    def test_thread_reply_self_reply_no_duplicate(self):
        """Author replying to own thread → no duplicate authored event."""
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="Actually, never mind",
            is_thread_reply=True,
            thread_author="alice",
        )
        events = ex.extract(sig)
        # Alice gets commented (for the reply), but no separate authored
        # because thread_author == signal.author
        authored = [e for e in events if e.exposure_type == "authored"]
        assert len(authored) == 0
        commented = [e for e in events if e.exposure_type == "commented"]
        assert len(commented) == 1


# ── Identity resolution ──────────────────────────────────────


class TestIdentityResolution:
    """All identities should resolve through the resolver when provided."""

    def test_resolver_applied_to_all_exposures(self):
        """ExposureExtractor uses resolver for all person IDs."""
        class MockResolver:
            def resolve(self, identifier):
                return identifier.lower().replace("@", "")

        ex = ExposureExtractor(resolver=MockResolver())
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="Alice",
            content="Hey @Bob check this",
            mentions=["Carol"],
        )
        events = ex.extract(sig)
        # All IDs should be lowercased with @ stripped
        for e in events:
            assert e.person_id == e.person_id.lower()
            assert "@" not in e.person_id


# ── Channel author tracking ──────────────────────────────────


class TestChannelAuthorTracking:
    """Channel unique author count used for size discount."""

    def test_exposures_have_correct_channel(self):
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#all-hands",
            author="alice",
            content="Weekly update on pricing",
        )
        events = ex.extract(sig)
        assert all(e.channel == "#all-hands" for e in events)
        assert all(e.source == "slack" for e in events)


# ── Topic terms ──────────────────────────────────────────────


class TestTopicTerms:
    """Topic terms extracted from signal content for overlap matching."""

    def test_terms_extracted_from_content(self):
        ex = ExposureExtractor()
        sig = _signal(
            source="slack",
            channel="#engineering",
            author="alice",
            content="The pricing cache invalidation is causing race conditions",
        )
        events = ex.extract(sig)
        assert len(events) > 0
        terms = events[0].topic_terms
        assert "pricing" in terms
        assert "cache" in terms
        assert "invalidation" in terms


# ── End-to-end scenario ──────────────────────────────────────


class TestEndToEndScenario:
    """Full scenario: Alice authored 4, Bob mentioned in 1 large channel.

    Finding about pricing → P(knows|Alice) >> P(knows|Bob).
    Surface value for Bob > surface value for Alice.
    """

    def test_alice_vs_bob(self):
        from stigmergy.attention.model import AttentionModel
        from stigmergy.attention.surfacing import rank_for_person

        model = AttentionModel()
        ex = ExposureExtractor()
        now = _now()

        # Alice authored 4 pricing signals
        for i in range(4):
            sig = _signal(
                source="github",
                channel="pricing-service",
                author="alice",
                content=f"Fix pricing cache invalidation bug part {i}",
                event_type="pull_request",
                pr_number=100 + i,
                action="opened",
            )
            for event in ex.extract(sig):
                model.record_exposure(event)

        # Bob mentioned in 1 Slack message in 50-person channel
        sig = _signal(
            source="slack",
            channel="#all-engineering",
            author="carol",
            content="Hey @bob FYI pricing cache issue",
            mentions=["bob"],
        )
        for event in ex.extract(sig):
            model.record_exposure(event)

        # Set channel size for Bob's channel
        bob_state = model.get_state("bob")
        if bob_state:
            bob_state.channels_seen["slack:#all-engineering"] = 50

        pricing_terms = frozenset({"pricing", "cache", "invalidation"})
        p_alice = model.p_knows("alice", pricing_terms, now)
        p_bob = model.p_knows("bob", pricing_terms, now)

        assert p_alice > p_bob, f"Alice={p_alice:.3f} should >> Bob={p_bob:.3f}"

        # Ranking: finding should rank higher for Bob
        finding = {
            "finding_hash": "pricing-cache",
            "summary": "Pricing cache invalidation race condition",
            "type": "risk",
            "sf_score": 0.78,
            "terms": pricing_terms,
            "signal_ids": [str(sig.id)],
        }

        alice_decisions = rank_for_person([finding], "alice", model, now)
        bob_decisions = rank_for_person([finding], "bob", model, now)

        assert bob_decisions[0].margin > alice_decisions[0].margin
