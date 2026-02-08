"""Tests for channel migration detection (Paper Section 5.2)."""

from stigmergy.attention.channel_migration import (
    MigrationAlert,
    detect_channel_migration,
    _is_private_channel,
)
from stigmergy.primitives.signal import SignalSource


class TestChannelVisibility:
    def test_private_from_metadata(self):
        meta = {"secret-ops": {"id": "C123", "is_private": True}}
        assert _is_private_channel("#secret-ops", meta)

    def test_public_from_metadata(self):
        meta = {"general": {"id": "C456", "is_private": False}}
        assert not _is_private_channel("#general", meta)

    def test_private_by_naming_convention(self):
        assert _is_private_channel("priv-team", {})
        assert _is_private_channel("dm-alice", {})
        assert _is_private_channel("private-eng", {})

    def test_public_by_default(self):
        assert not _is_private_channel("#engineering", {})


class TestChannelMigration:
    def test_detects_migration(self):
        """Discussion moving to private while work continues publicly."""
        meta = {"secret-team": {"id": "C1", "is_private": True}}
        signals = [
            {"content": "pricing service refactor", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#secret-team"},
            {"content": "pricing service refactor progress", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#secret-team"},
            {"content": "pricing service refactor update", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#secret-team"},
            {"content": "pricing service refactor commit", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push"}, "channel": "pricing-repo"},
        ]
        alerts = detect_channel_migration(signals, channel_meta=meta,
                                          min_total_signals=3, migration_threshold=0.2)
        assert len(alerts) > 0
        assert alerts[0].private_count > 0
        assert alerts[0].migration_score > 0.0

    def test_no_migration_all_public(self):
        """All discussion in public channels — no migration."""
        signals = [
            {"content": "backend service update", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#engineering"},
            {"content": "backend service update progress", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#engineering"},
            {"content": "backend service update fix", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#engineering"},
            {"content": "backend service push", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push"}, "channel": "backend-repo"},
        ]
        alerts = detect_channel_migration(signals, min_total_signals=3,
                                          migration_threshold=0.3)
        assert len(alerts) == 0

    def test_empty_signals(self):
        assert detect_channel_migration([]) == []

    def test_below_threshold(self):
        signals = [
            {"content": "small topic", "source": SignalSource.SLACK,
             "metadata": {}, "channel": "#general"},
        ]
        assert detect_channel_migration(signals, min_total_signals=5) == []

    def test_result_structure(self):
        meta = {"priv-ops": {"id": "C1", "is_private": True}}
        signals = [
            {"content": "sync service investigation", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#priv-ops"},
            {"content": "sync service investigation update", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#priv-ops"},
            {"content": "sync service investigation results", "source": SignalSource.SLACK,
             "metadata": {"event_type": "message"}, "channel": "#priv-ops"},
            {"content": "sync service commit", "source": SignalSource.GITHUB,
             "metadata": {"event_type": "push"}, "channel": "sync-repo"},
        ]
        alerts = detect_channel_migration(signals, channel_meta=meta,
                                          min_total_signals=3, migration_threshold=0.1)
        for a in alerts:
            assert isinstance(a, MigrationAlert)
            assert isinstance(a.topic_terms, frozenset)
            assert 0.0 <= a.migration_score <= 1.0

    def test_sorted_by_score_descending(self):
        meta = {
            "priv-a": {"id": "C1", "is_private": True},
            "priv-b": {"id": "C2", "is_private": True},
        }
        signals = []
        # Topic A: all private discussion
        for _ in range(4):
            signals.append({"content": "alpha private topic", "source": SignalSource.SLACK,
                           "metadata": {"event_type": "message"}, "channel": "#priv-a"})
        signals.append({"content": "alpha private topic code", "source": SignalSource.GITHUB,
                        "metadata": {"event_type": "push"}, "channel": "alpha-repo"})
        # Topic B: mixed
        for _ in range(2):
            signals.append({"content": "beta mixed topic", "source": SignalSource.SLACK,
                           "metadata": {"event_type": "message"}, "channel": "#priv-b"})
        for _ in range(2):
            signals.append({"content": "beta mixed topic", "source": SignalSource.SLACK,
                           "metadata": {"event_type": "message"}, "channel": "#general"})

        alerts = detect_channel_migration(signals, channel_meta=meta,
                                          min_total_signals=3, migration_threshold=0.1)
        if len(alerts) >= 2:
            assert alerts[0].migration_score >= alerts[1].migration_score
