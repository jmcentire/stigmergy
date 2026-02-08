"""Tests for edge extraction from signals."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stigmergy.graph.extractor import EdgeExtractor
from stigmergy.graph.graph import CommunicationGraph
from stigmergy.graph.identity import IdentityResolver
from stigmergy.primitives.signal import Signal, SignalSource


@pytest.fixture
def resolver():
    r = IdentityResolver()
    r.register_alias("alice", "alice.chen")
    r.register_alias("bob", "bob.martinez")
    r.register_alias("carol", "carol.park")
    r.register_alias("dave", "dave.kim")
    return r


@pytest.fixture
def graph(resolver):
    return CommunicationGraph(resolver)


@pytest.fixture
def extractor(graph):
    return EdgeExtractor(graph)


def _make_signal(
    content: str,
    source: str = "github",
    channel: str = "acme-org/backend",
    author: str = "alice.chen",
    metadata: dict | None = None,
) -> Signal:
    return Signal(
        content=content,
        source=source,
        channel=channel,
        author=author,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


class TestGitHubExtraction:
    def test_pr_opened_stores_author(self, extractor, graph):
        """PR opened stores author for later review/comment attribution."""
        sig = _make_signal(
            "PR #487: Fix availability cache",
            author="alice.chen",
            metadata={
                "event_type": "pull_request",
                "action": "opened",
                "pr_number": 487,
            },
        )
        edges = extractor.extract(sig)
        # No person-to-person edge from PR open alone
        assert len(edges) == 0
        # But author is cached
        assert extractor._pr_authors["acme-org/backend#487"] == "alice.chen"

    def test_review_creates_edge(self, extractor, graph):
        """PR review creates reviewer → PR author edge."""
        # First, open the PR
        extractor.extract(_make_signal(
            "PR #487: Fix",
            author="alice.chen",
            metadata={"event_type": "pull_request", "action": "opened", "pr_number": 487},
        ))
        # Then review it
        edges = extractor.extract(_make_signal(
            "Review: LGTM",
            author="carol.park",
            metadata={"event_type": "review", "pr_number": 487},
        ))
        assert len(edges) == 1
        assert edges[0].source == "carol"
        assert edges[0].target == "alice"
        assert edges[0].edge_type == "pr_review"

    def test_pr_comment_creates_edge(self, extractor, graph):
        """PR comment creates commenter → PR author edge."""
        # Open PR
        extractor.extract(_make_signal(
            "PR #487: Fix",
            author="alice.chen",
            metadata={"event_type": "pull_request", "action": "opened", "pr_number": 487},
        ))
        # Comment on it
        edges = extractor.extract(_make_signal(
            "Looks like a duplicate",
            author="bob.martinez",
            metadata={"event_type": "pr_comment", "pr_number": 487},
        ))
        assert len(edges) == 1
        assert edges[0].source == "bob"
        assert edges[0].target == "alice"
        assert edges[0].edge_type == "pr_comment"

    def test_pr_merge_creates_edge(self, extractor, graph):
        """Merged PR creates merger → PR author edge."""
        # Open PR
        extractor.extract(_make_signal(
            "PR #485: Pricing v3",
            author="bob.martinez",
            metadata={"event_type": "pull_request", "action": "opened", "pr_number": 485},
        ))
        # Merge it
        edges = extractor.extract(_make_signal(
            "Merged PR #485",
            author="bob.martinez",
            channel="acme-org/pricing",
            metadata={
                "event_type": "pull_request",
                "action": "closed",
                "merged": True,
                "pr_number": 485,
                "merged_by": "carol.park",
            },
        ))
        assert len(edges) == 1
        assert edges[0].source == "carol"
        assert edges[0].target == "bob"
        assert edges[0].edge_type == "pr_merge"

    def test_self_review_no_edge(self, extractor, graph):
        """Self-review doesn't create an edge."""
        extractor.extract(_make_signal(
            "PR #487: Fix",
            author="alice.chen",
            metadata={"event_type": "pull_request", "action": "opened", "pr_number": 487},
        ))
        edges = extractor.extract(_make_signal(
            "Review on own PR",
            author="alice.chen",
            metadata={"event_type": "review", "pr_number": 487},
        ))
        assert len(edges) == 0

    def test_review_without_pr_open_no_edge(self, extractor, graph):
        """Review without prior PR open produces no edge (missing author cache)."""
        edges = extractor.extract(_make_signal(
            "Review: LGTM",
            author="carol.park",
            metadata={"event_type": "review", "pr_number": 999},
        ))
        assert len(edges) == 0

    def test_multiple_reviews_accumulate(self, extractor, graph):
        """Multiple reviews on same PR accumulate edge weight."""
        extractor.extract(_make_signal(
            "PR #487: Fix",
            author="alice.chen",
            metadata={"event_type": "pull_request", "action": "opened", "pr_number": 487},
        ))
        extractor.extract(_make_signal(
            "Review 1",
            author="carol.park",
            metadata={"event_type": "review", "pr_number": 487},
        ))
        extractor.extract(_make_signal(
            "Review 2",
            author="carol.park",
            metadata={"event_type": "review", "pr_number": 487},
        ))
        # Edge should have count=2
        edges = graph.all_edges()
        pr_review_edges = [e for e in edges if e.edge_type == "pr_review"]
        assert len(pr_review_edges) == 1
        assert pr_review_edges[0].count == 2
        assert pr_review_edges[0].weight == 2.0


class TestLinearExtraction:
    def test_issue_assign_creates_edge(self, extractor, graph):
        """Linear issue with assignee creates author → assignee edge."""
        edges = extractor.extract(_make_signal(
            "PMS webhook sync failure",
            source="linear",
            channel="ENG-1234",
            author="bob.martinez",
            metadata={
                "identifier": "ENG-1234",
                "assignee": "alice.chen",
                "team": "Engineering",
            },
        ))
        assert len(edges) == 1
        assert edges[0].source == "bob"
        assert edges[0].target == "alice"
        assert edges[0].edge_type == "linear_assign"

    def test_self_assign_no_edge(self, extractor, graph):
        """Self-assignment doesn't create an edge."""
        edges = extractor.extract(_make_signal(
            "My own task",
            source="linear",
            channel="ENG-1234",
            author="alice.chen",
            metadata={
                "identifier": "ENG-1234",
                "assignee": "alice.chen",
                "team": "Engineering",
            },
        ))
        assert len(edges) == 0

    def test_linear_comment_creates_edge(self, extractor, graph):
        """Linear comment creates commenter → issue author edge."""
        # First, create the issue
        extractor.extract(_make_signal(
            "PMS webhook sync failure",
            source="linear",
            channel="ENG-1234",
            author="bob.martinez",
            metadata={
                "identifier": "ENG-1234",
                "assignee": "alice.chen",
                "team": "Engineering",
            },
        ))
        # Then comment on it
        edges = extractor.extract(_make_signal(
            "Already fixed in PR #480",
            source="linear",
            channel="ENG-1234",
            author="alice.chen",
            metadata={
                "event_type": "linear_comment",
                "identifier": "ENG-1234",
                "team": "Engineering",
            },
        ))
        assert len(edges) == 1
        assert edges[0].source == "alice"
        assert edges[0].target == "bob"
        assert edges[0].edge_type == "linear_comment"

    def test_related_issues_create_edge(self, extractor, graph):
        """Related issues create edges between their authors."""
        # Create issue ENG-1236 first
        extractor.extract(_make_signal(
            "Availability cache race condition",
            source="linear",
            channel="ENG-1236",
            author="alice.chen",
            metadata={
                "identifier": "ENG-1236",
                "team": "Engineering",
            },
        ))
        # Create issue ENG-1234 with relation to ENG-1236
        edges = extractor.extract(_make_signal(
            "PMS webhook sync failure",
            source="linear",
            channel="ENG-1234",
            author="bob.martinez",
            metadata={
                "identifier": "ENG-1234",
                "team": "Engineering",
                "relations": [
                    {"type": "related", "issue": "ENG-1236"},
                ],
            },
        ))
        # Should have relation edge: ENG-1236 author (alice) → ENG-1234 author (bob)
        related_edges = [e for e in edges if e.edge_type == "linear_related"]
        assert len(related_edges) == 1
        assert related_edges[0].source == "alice"
        assert related_edges[0].target == "bob"

    def test_no_assignee_no_assign_edge(self, extractor, graph):
        """Issue without assignee creates no assign edge."""
        edges = extractor.extract(_make_signal(
            "Some task",
            source="linear",
            channel="ENG-1238",
            author="bob.martinez",
            metadata={"identifier": "ENG-1238", "team": "Engineering"},
        ))
        assert len(edges) == 0


class TestNonSignalSources:
    def test_slack_signal_no_edges(self, extractor, graph):
        """Slack signals produce no edges (not yet implemented)."""
        edges = extractor.extract(_make_signal(
            "Hey team, standup in 5",
            source="slack",
            channel="#engineering",
            author="alice.chen",
        ))
        assert len(edges) == 0

    def test_unknown_source_no_edges(self, extractor, graph):
        """Unknown sources produce no edges."""
        edges = extractor.extract(_make_signal(
            "Unknown source",
            source="jira",
            channel="JIRA-123",
            author="alice.chen",
        ))
        assert len(edges) == 0


class TestGraphIntegration:
    def test_full_pr_lifecycle(self, extractor, graph):
        """Full PR lifecycle: open → review → comment → merge creates proper graph."""
        # 1. Open PR
        extractor.extract(_make_signal(
            "PR #487: Fix availability cache",
            author="alice.chen",
            metadata={"event_type": "pull_request", "action": "opened", "pr_number": 487},
        ))
        # 2. Review by bob
        extractor.extract(_make_signal(
            "Review: Looks good",
            author="bob.martinez",
            metadata={"event_type": "review", "pr_number": 487},
        ))
        # 3. Comment by carol
        extractor.extract(_make_signal(
            "One nit",
            author="carol.park",
            metadata={"event_type": "pr_comment", "pr_number": 487},
        ))
        # 4. Merge by carol
        extractor.extract(_make_signal(
            "Merged PR #487",
            author="alice.chen",
            metadata={
                "event_type": "pull_request",
                "action": "closed",
                "merged": True,
                "pr_number": 487,
                "merged_by": "carol.park",
            },
        ))

        # Graph should have: bob→alice (review), carol→alice (comment + merge)
        assert graph.node_count == 3  # alice, bob, carol
        assert graph.edge_count == 3  # pr_review, pr_comment, pr_merge

        # Bob reviewed alice's PR
        assert graph.effective_weight("bob", "alice") > 0
        # Carol commented on and merged alice's PR
        assert graph.effective_weight("carol", "alice") > 0
