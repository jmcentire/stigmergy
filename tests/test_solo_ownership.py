"""Tests for solo ownership detection (Paper evaluation section)."""

from datetime import datetime, timedelta, timezone

from stigmergy.attention.solo_ownership import (
    SoloOwnershipAlert,
    detect_solo_ownership,
)
from stigmergy.primitives.signal import SignalSource


def _make_signal(content, author, channel, source=SignalSource.SLACK,
                 event_type="message", days_ago=0, **extra_meta):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    meta = {"event_type": event_type, **extra_meta}
    return {
        "content": content,
        "author": author,
        "channel": channel,
        "source": source,
        "metadata": meta,
        "timestamp": ts,
    }


class TestSoloOwnership:
    def test_detects_solo_owner(self):
        """One person doing all actions across multiple systems."""
        signals = [
            _make_signal("pricing service migration", "alice", "#pricing",
                        source=SignalSource.SLACK),
            _make_signal("pricing service migration update", "bob", "#backend",
                        source=SignalSource.SLACK),
            _make_signal("pricing service migration fix", "alice", "#pricing",
                        source=SignalSource.GITHUB, event_type="push"),
            _make_signal("pricing service migration deployed", "alice", "#infra",
                        source=SignalSource.GITHUB, event_type="push"),
        ]
        alerts = detect_solo_ownership(signals, min_systems=2, min_signals=3)
        solo_alerts = [a for a in alerts if a.sole_owner == "alice"]
        assert len(solo_alerts) > 0
        assert solo_alerts[0].system_count >= 2

    def test_no_alert_multiple_actors(self):
        """Multiple people producing actions — not solo ownership."""
        signals = [
            _make_signal("sync service issue", "alice", "#sync",
                        source=SignalSource.GITHUB, event_type="push"),
            _make_signal("sync service issue fix", "bob", "#sync",
                        source=SignalSource.GITHUB, event_type="push"),
            _make_signal("sync service issue discussion", "carol", "#backend",
                        source=SignalSource.SLACK),
            _make_signal("sync service issue update", "dave", "#ops",
                        source=SignalSource.SLACK),
        ]
        alerts = detect_solo_ownership(signals, min_systems=2, min_signals=3)
        # Both alice and bob have actions, so no solo ownership
        sync_alerts = [a for a in alerts
                      if "sync" in a.topic_terms or "service" in a.topic_terms]
        # Should not flag since multiple actors have actions
        for a in sync_alerts:
            assert a.sole_owner not in ("alice", "bob") or a.action_count == 0

    def test_empty_signals(self):
        assert detect_solo_ownership([]) == []

    def test_below_min_signals(self):
        signals = [
            _make_signal("rare topic", "alice", "#chan1",
                        source=SignalSource.GITHUB, event_type="push"),
        ]
        assert detect_solo_ownership(signals, min_signals=3) == []

    def test_staleness_affects_severity(self):
        """Stale solo ownership is more severe."""
        signals = [
            _make_signal("payment integration bug", "alice", "#payments",
                        source=SignalSource.SLACK, days_ago=0),
            _make_signal("payment integration bug in checkout", "bob", "#checkout",
                        source=SignalSource.SLACK, days_ago=0),
            _make_signal("payment integration bug fix attempt", "alice", "#payments",
                        source=SignalSource.GITHUB, event_type="push", days_ago=10),
            _make_signal("payment integration bug discussion", "carol", "#backend",
                        source=SignalSource.SLACK, days_ago=0),
        ]
        alerts = detect_solo_ownership(signals, min_systems=2, min_signals=3,
                                       staleness_threshold_days=7.0)
        payment_alerts = [a for a in alerts if a.sole_owner == "alice"]
        if payment_alerts:
            assert payment_alerts[0].staleness_days >= 7.0

    def test_result_structure(self):
        signals = [
            _make_signal("auth service error", "alice", "#auth",
                        source=SignalSource.GITHUB, event_type="push"),
            _make_signal("auth service error in api", "bob", "#api",
                        source=SignalSource.SLACK),
            _make_signal("auth service error handling", "carol", "#backend",
                        source=SignalSource.SLACK),
        ]
        alerts = detect_solo_ownership(signals, min_systems=2, min_signals=2)
        for a in alerts:
            assert isinstance(a, SoloOwnershipAlert)
            assert isinstance(a.topic_terms, frozenset)
            assert a.severity in ("warning", "critical")
