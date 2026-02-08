"""Tests for action-correlation ratio (Paper Section 5.2)."""

from stigmergy.attention.action_correlation import (
    ActionCorrelation,
    compute_action_correlations,
    _is_action_signal,
    _is_discussion_signal,
)
from stigmergy.primitives.signal import SignalSource


class TestSignalClassification:
    def test_slack_message_is_discussion(self):
        assert _is_discussion_signal(SignalSource.SLACK, {"event_type": "message"})

    def test_github_push_is_action(self):
        assert _is_action_signal(SignalSource.GITHUB, {"event_type": "push"})

    def test_pr_merged_is_action(self):
        assert _is_action_signal(SignalSource.GITHUB, {"event_type": "pull_request_merged"})

    def test_linear_done_is_action(self):
        assert _is_action_signal(SignalSource.LINEAR, {"state": "Done"})

    def test_linear_comment_is_discussion(self):
        assert _is_discussion_signal(SignalSource.LINEAR, {"event_type": "comment"})

    def test_slack_message_not_action(self):
        assert not _is_action_signal(SignalSource.SLACK, {"event_type": "message"})


class TestActionCorrelation:
    def test_healthy_ratio(self):
        """Topics with matching actions and discussion are healthy."""
        signals = [
            {"content": "pricing service deployment", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "pricing service deployment fix", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "pricing service deployment", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "pricing service deployed successfully", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push"}},
            {"content": "pricing service fix merged", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "pull_request_merged"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=2)
        # Should find at least one topic with actions driving a positive ratio
        assert len(results) > 0
        assert any(r.ratio > 0.0 for r in results)

    def test_decoupling_detection(self):
        """Topics with lots of discussion but no actions are decoupling."""
        signals = [
            {"content": "database migration concerns", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "database migration timeline", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "database migration risks", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "database migration planning", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=3)
        db_topics = [r for r in results if "database" in r.topic or "migration" in r.topic]
        assert len(db_topics) > 0
        assert all(r.trend == "decoupling" for r in db_topics)
        assert all(r.ratio == 0.0 for r in db_topics)

    def test_empty_signals(self):
        results = compute_action_correlations([])
        assert results == []

    def test_below_min_discussion(self):
        """Topics with too few discussions are filtered out."""
        signals = [
            {"content": "obscure topic here", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=3)
        assert results == []

    def test_sorted_by_ratio_ascending(self):
        """Results should be sorted worst decoupling first."""
        signals = []
        # Topic A: lots of discussion, no action
        for _ in range(5):
            signals.append({"content": "alpha topic discussion", "source": SignalSource.SLACK,
                           "metadata": {"event_type": "message"}})
        # Topic B: discussion with matching action
        for _ in range(5):
            signals.append({"content": "beta topic work", "source": SignalSource.SLACK,
                           "metadata": {"event_type": "message"}})
        for _ in range(3):
            signals.append({"content": "beta topic work done", "source": SignalSource.GITHUB,
                           "metadata": {"event_type": "push"}})

        results = compute_action_correlations(signals, min_discussion_count=3)
        if len(results) >= 2:
            assert results[0].ratio <= results[-1].ratio

    def test_result_structure(self):
        signals = [
            {"content": "sync service error handling", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "sync service error handling update", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "sync service error handling fix", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=2)
        for r in results:
            assert isinstance(r, ActionCorrelation)
            assert isinstance(r.topic_terms, frozenset)
            assert r.window_days == 7
            assert r.trend in ("healthy", "stable", "decoupling")


class TestExplorationExploitation:
    """Tests for March 1991 exploration/exploitation ratio."""

    def test_all_same_repo_exploitation(self):
        """Actions all on one repo → exploration_ratio low (exploitation)."""
        signals = [
            {"content": "pricing fix deployed", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "pricing fix deployed again", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "pricing fix deployed", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "pricing fix commit", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push", "repo": "pricing"}},
            {"content": "pricing fix merged", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push", "repo": "pricing"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=2)
        assert len(results) > 0
        # All actions on same repo → any topic with actions should have low exploration
        for t in results:
            if t.action_count > 0:
                assert t.exploration_ratio <= 0.5

    def test_multiple_repos_exploration(self):
        """Actions on different repos → higher exploration_ratio."""
        signals = [
            {"content": "auth service refactor", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "auth service refactor needed", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "auth service refactor discussion", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "auth service refactor commit", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push", "repo": "auth-service"}},
            {"content": "auth service refactor deploy", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push", "repo": "api-gateway"}},
            {"content": "auth service refactor config", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push", "repo": "infra-config"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=2)
        assert len(results) > 0
        # Multiple novel repos → any topic with actions should show exploration
        topics_with_actions = [t for t in results if t.action_count > 0]
        assert len(topics_with_actions) > 0
        assert any(t.exploration_ratio > 0.0 for t in topics_with_actions)

    def test_no_actions_no_exploration(self):
        """Zero action signals → exploration_ratio = 0."""
        signals = [
            {"content": "database migration plan", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "database migration plan review", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "database migration plan risks", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=2)
        for r in results:
            assert r.exploration_ratio == 0.0

    def test_exploration_ratio_bounded(self):
        """exploration_ratio should always be in [0, 1]."""
        signals = []
        for i in range(5):
            signals.append({"content": "payment integration work", "source": SignalSource.SLACK,
                           "metadata": {"event_type": "message"}})
        for repo in ["pay-svc", "pay-api", "pay-ui"]:
            signals.append({"content": "payment integration deployed", "source": SignalSource.GITHUB,
                           "metadata": {"event_type": "push", "repo": repo}})
        results = compute_action_correlations(signals, min_discussion_count=3)
        for r in results:
            assert 0.0 <= r.exploration_ratio <= 1.0

    def test_exploration_ratio_present_on_dataclass(self):
        """ActionCorrelation always has exploration_ratio field."""
        signals = [
            {"content": "sync service work", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "sync service work update", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
            {"content": "sync service work progress", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}},
        ]
        results = compute_action_correlations(signals, min_discussion_count=2)
        for r in results:
            assert hasattr(r, "exploration_ratio")
            assert isinstance(r.exploration_ratio, float)
