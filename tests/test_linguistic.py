"""Tests for linguistic compression detection (Paper Section 5.3)."""

from stigmergy.attention.linguistic import (
    CompressionSignals,
    CompressionTracker,
    CompressionTrend,
    analyze_text,
)


class TestHedgeDetection:
    def test_clear_hedging(self):
        text = "It seems the deployment may have issues. This could potentially cause downtime."
        result = analyze_text(text)
        assert result.hedging_density > 0.5

    def test_no_hedging(self):
        text = "The deployment failed at 3pm. We rolled back immediately."
        result = analyze_text(text)
        assert result.hedging_density == 0.0

    def test_mixed_hedging(self):
        text = "The API is down. It might be related to the DNS change. We fixed the config."
        result = analyze_text(text)
        assert 0.0 < result.hedging_density < 1.0


class TestPassiveVoice:
    def test_passive_detected(self):
        text = "The issue was identified by the team. The fix has been deployed."
        result = analyze_text(text)
        assert result.passive_voice_ratio > 0.0

    def test_active_voice(self):
        text = "We identified the issue. The team deployed the fix."
        result = analyze_text(text)
        assert result.passive_voice_ratio == 0.0

    def test_mixed_voice(self):
        text = "We found the bug. The patch was applied to production. Testing confirmed the fix."
        result = analyze_text(text)
        assert 0.0 < result.passive_voice_ratio < 1.0


class TestNominalization:
    def test_heavy_nominalization(self):
        text = "The investigation into the implementation revealed a degradation in performance."
        result = analyze_text(text)
        assert result.nominalization_rate > 0.5

    def test_direct_language(self):
        text = "We checked the code. It runs slowly. We will fix it today."
        result = analyze_text(text)
        assert result.nominalization_rate < 0.5


class TestSpecificity:
    def test_specific_language(self):
        text = "We deployed v2.3.1 on 2024-01-15. Response time dropped 40%. We will ship the fix tomorrow."
        result = analyze_text(text)
        assert result.specificity_score > 0.5

    def test_vague_language(self):
        text = "There are some concerns about the approach. The situation is being monitored."
        result = analyze_text(text)
        assert result.specificity_score < 0.5


class TestCompressionIndex:
    def test_direct_text_low_index(self):
        text = (
            "We deployed the hotfix at 2pm. Error rate dropped from 5% to 0.1%. "
            "The root cause was a null pointer in the auth module. "
            "We will add a regression test by Friday."
        )
        result = analyze_text(text)
        assert result.compression_index < 0.3

    def test_compressed_text_high_index(self):
        text = (
            "It appears there may be some degradation in service quality. "
            "The situation is being monitored and an investigation is underway. "
            "It is expected that improvements could potentially be implemented."
        )
        result = analyze_text(text)
        assert result.compression_index > 0.3

    def test_empty_text(self):
        result = analyze_text("")
        assert result.compression_index == 0.0
        assert result.specificity_score == 1.0

    def test_single_sentence(self):
        result = analyze_text("The server crashed.")
        assert isinstance(result, CompressionSignals)
        assert 0.0 <= result.compression_index <= 1.0

    def test_index_bounded(self):
        # Various inputs should always produce index in [0, 1]
        texts = [
            "yes",
            "maybe possibly perhaps",
            "We shipped it. Done. Fixed. Merged.",
            "It seems that it might possibly be the case that something could be wrong.",
        ]
        for text in texts:
            result = analyze_text(text)
            assert 0.0 <= result.compression_index <= 1.0, f"Out of bounds for: {text}"


class TestCompressionTracker:
    def test_single_update(self):
        tracker = CompressionTracker()
        tracker.update("#engineering", 0.4)
        trend = tracker.get_trend("#engineering")
        assert trend is not None
        assert trend.sample_count == 1
        assert trend.current_index == 0.4

    def test_ema_smoothing(self):
        tracker = CompressionTracker(alpha=0.5)
        tracker.update("pricing", 0.2)
        tracker.update("pricing", 0.6)
        trend = tracker.get_trend("pricing")
        assert trend is not None
        # EMA with alpha=0.5: 0.5*0.6 + 0.5*0.2 = 0.4
        assert abs(trend.rolling_avg - 0.4) < 0.01
        assert trend.sample_count == 2

    def test_increasing_trend(self):
        tracker = CompressionTracker(alpha=0.5)
        tracker.update("topic", 0.1)
        tracker.update("topic", 0.3)
        tracker.update("topic", 0.5)
        trend = tracker.get_trend("topic")
        assert trend is not None
        assert trend.trend == "increasing"

    def test_decreasing_trend(self):
        tracker = CompressionTracker(alpha=0.5)
        tracker.update("topic", 0.6)
        tracker.update("topic", 0.4)
        tracker.update("topic", 0.2)
        trend = tracker.get_trend("topic")
        assert trend is not None
        assert trend.trend == "decreasing"

    def test_stable_trend(self):
        tracker = CompressionTracker(alpha=0.3)
        tracker.update("topic", 0.3)
        tracker.update("topic", 0.3)
        tracker.update("topic", 0.3)
        trend = tracker.get_trend("topic")
        assert trend is not None
        assert trend.trend == "stable"

    def test_peak_tracking(self):
        tracker = CompressionTracker()
        tracker.update("topic", 0.3)
        tracker.update("topic", 0.8)
        tracker.update("topic", 0.2)
        trend = tracker.get_trend("topic")
        assert trend is not None
        assert trend.peak_index == 0.8

    def test_unknown_key_returns_none(self):
        tracker = CompressionTracker()
        assert tracker.get_trend("nonexistent") is None

    def test_all_trends_min_samples(self):
        tracker = CompressionTracker()
        tracker.update("a", 0.5)  # 1 sample — below min
        tracker.update("b", 0.3)
        tracker.update("b", 0.4)  # 2 samples — meets min
        trends = tracker.all_trends(min_samples=2)
        assert len(trends) == 1
        assert trends[0].key == "b"

    def test_increasing_trends_filter(self):
        tracker = CompressionTracker(alpha=0.5)
        tracker.update("good", 0.5)
        tracker.update("good", 0.3)  # decreasing
        tracker.update("bad", 0.2)
        tracker.update("bad", 0.6)  # increasing
        increasing = tracker.increasing_trends(min_samples=2)
        assert all(t.trend == "increasing" for t in increasing)

    def test_multiple_topics_sorted(self):
        tracker = CompressionTracker()
        tracker.update("low", 0.1)
        tracker.update("low", 0.15)
        tracker.update("high", 0.8)
        tracker.update("high", 0.85)
        trends = tracker.all_trends(min_samples=2)
        assert len(trends) == 2
        assert trends[0].current_index > trends[1].current_index


# ── Extended metrics tests (variance compression research) ──────


class TestLexicalDiversity:
    def test_all_unique_tokens(self):
        """All unique words → LD = 1.0."""
        text = "Alpha bravo charlie delta echo foxtrot golf hotel."
        result = analyze_text(text)
        assert result.lexical_diversity == 1.0

    def test_repeated_tokens(self):
        """Repeated words → LD < 1.0."""
        text = "the the the the the the the the."
        result = analyze_text(text)
        assert result.lexical_diversity < 0.5

    def test_mixed_repetition(self):
        """Mix of unique and repeated → intermediate LD."""
        text = "The service is slow. The service crashed again. We restarted the service."
        result = analyze_text(text)
        assert 0.3 < result.lexical_diversity < 0.9

    def test_empty_text_ld(self):
        result = analyze_text("")
        assert result.lexical_diversity == 0.0

    def test_case_insensitive(self):
        """LD should be case-insensitive (tokens lowered)."""
        text = "Hello hello HELLO Hello."
        result = analyze_text(text)
        assert result.lexical_diversity < 0.5

    def test_bounded_zero_one(self):
        texts = [
            "one two three four five six seven eight nine ten.",
            "bug bug bug bug bug bug bug bug.",
            "We deployed the fix. The team fixed the deployment. Deploy again.",
        ]
        for text in texts:
            result = analyze_text(text)
            assert 0.0 <= result.lexical_diversity <= 1.0, f"Out of bounds for: {text}"


class TestShannonEntropy:
    def test_uniform_distribution_high_entropy(self):
        """All unique tokens → maximum entropy."""
        text = "Alpha bravo charlie delta echo foxtrot golf hotel."
        result = analyze_text(text)
        # 8 unique tokens → max entropy = log2(8) = 3.0
        assert result.shannon_entropy >= 2.5

    def test_single_token_zero_entropy(self):
        """All identical tokens → zero entropy."""
        text = "test test test test test test test test."
        result = analyze_text(text)
        assert result.shannon_entropy == 0.0

    def test_empty_text_se(self):
        result = analyze_text("")
        assert result.shannon_entropy == 0.0

    def test_entropy_increases_with_vocabulary(self):
        """More vocabulary → higher entropy."""
        small = "the cat sat."
        large = "the cat sat on a blue mat near a tall tree by the old red barn."
        r_small = analyze_text(small)
        r_large = analyze_text(large)
        assert r_large.shannon_entropy > r_small.shannon_entropy

    def test_entropy_non_negative(self):
        texts = ["yes.", "maybe.", "the quick brown fox jumps over the lazy dog."]
        for text in texts:
            result = analyze_text(text)
            assert result.shannon_entropy >= 0.0


class TestTemporalOrientation:
    def test_forward_looking(self):
        """Text with future-oriented language → positive orientation."""
        text = (
            "We will deploy the fix tomorrow. "
            "We plan to launch next week. "
            "The goal is to ship by Friday."
        )
        result = analyze_text(text)
        assert result.temporal_orientation > 0.0

    def test_backward_looking(self):
        """Text with past-oriented language → negative orientation."""
        text = (
            "We were working on it last week. "
            "Previously the system had been stable. "
            "In the past we used to deploy daily."
        )
        result = analyze_text(text)
        assert result.temporal_orientation < 0.0

    def test_balanced_orientation(self):
        """Mix of forward and backward → near zero."""
        text = (
            "Last week we fixed the auth bug. "
            "We will deploy the payment fix tomorrow."
        )
        result = analyze_text(text)
        assert -0.5 <= result.temporal_orientation <= 0.5

    def test_no_temporal_markers(self):
        """Text without temporal language → zero."""
        text = "The system processes requests. Data flows through the pipeline."
        result = analyze_text(text)
        assert result.temporal_orientation == 0.0

    def test_bounded(self):
        """Temporal orientation should be in [-1, +1]."""
        texts = [
            "will will will will.",
            "was was was was.",
            "Hello world.",
        ]
        for text in texts:
            result = analyze_text(text)
            assert -1.0 <= result.temporal_orientation <= 1.0, f"Out of bounds for: {text}"


class TestBureaucraticDensity:
    def test_bureaucratic_language(self):
        """Text with bureaucratic euphemisms → high density."""
        text = (
            "We need to align on the deliverables going forward. "
            "Let's circle back on the action items. "
            "We should socialize this with stakeholders."
        )
        result = analyze_text(text)
        assert result.bureaucratic_density > 0.5

    def test_direct_language(self):
        """Text without bureaucratic terms → zero density."""
        text = (
            "We fixed the bug. The server crashed at 3pm. "
            "We deployed the hotfix and error rate dropped."
        )
        result = analyze_text(text)
        assert result.bureaucratic_density == 0.0

    def test_nasa_vocabulary(self):
        """NASA-specific euphemisms from the aerospace paper."""
        text = (
            "The anomaly was classified as a mishap. "
            "We need to re-plan the contingency procedures. "
            "All systems are nominal."
        )
        result = analyze_text(text)
        assert result.bureaucratic_density > 0.5

    def test_softening_language(self):
        """Politeness-strategy softening language."""
        text = (
            "It would be helpful if we could review this. "
            "We might want to consider an alternative approach. "
            "It could be beneficial to revisit the timeline."
        )
        result = analyze_text(text)
        assert result.bureaucratic_density > 0.5

    def test_bounded(self):
        texts = [
            "normal text here.",
            "leverage synergize operationalize align on.",
            "The server crashed.",
        ]
        for text in texts:
            result = analyze_text(text)
            assert 0.0 <= result.bureaucratic_density <= 1.0


class TestBornCaged:
    def test_initial_index_recorded(self):
        """First update records the initial compression index."""
        tracker = CompressionTracker()
        tracker.update("channel-a", 0.45)
        trend = tracker.get_trend("channel-a")
        assert trend is not None
        assert trend.initial_index == 0.45

    def test_initial_index_not_overwritten(self):
        """Subsequent updates don't change the initial index."""
        tracker = CompressionTracker()
        tracker.update("channel-a", 0.45)
        tracker.update("channel-a", 0.6)
        tracker.update("channel-a", 0.2)
        trend = tracker.get_trend("channel-a")
        assert trend is not None
        assert trend.initial_index == 0.45

    def test_is_born_caged_high_initial(self):
        """Channel starting at >= 0.3 is born caged."""
        tracker = CompressionTracker()
        tracker.update("formal-channel", 0.35)
        tracker.update("formal-channel", 0.4)
        trend = tracker.get_trend("formal-channel")
        assert trend is not None
        assert trend.is_born_caged is True

    def test_not_born_caged_low_initial(self):
        """Channel starting below 0.3 is not born caged."""
        tracker = CompressionTracker()
        tracker.update("casual-channel", 0.1)
        tracker.update("casual-channel", 0.5)
        trend = tracker.get_trend("casual-channel")
        assert trend is not None
        assert trend.is_born_caged is False

    def test_compression_delta(self):
        """Delta is current - initial."""
        tracker = CompressionTracker(alpha=1.0)  # alpha=1.0 → no smoothing
        tracker.update("ch", 0.2)
        tracker.update("ch", 0.5)
        trend = tracker.get_trend("ch")
        assert trend is not None
        assert abs(trend.compression_delta - 0.3) < 0.01

    def test_born_caged_channels_method(self):
        """born_caged_channels() returns only channels with high initial index."""
        tracker = CompressionTracker()
        tracker.update("casual", 0.1)
        tracker.update("casual", 0.15)
        tracker.update("formal", 0.4)
        tracker.update("formal", 0.45)
        caged = tracker.born_caged_channels(min_samples=2)
        assert len(caged) == 1
        assert caged[0].key == "formal"

    def test_ld_se_tracked(self):
        """Tracker stores LD and SE values."""
        tracker = CompressionTracker()
        tracker.update("ch", 0.3, ld=0.17, se=11.5)
        trend = tracker.get_trend("ch")
        assert trend is not None
        assert trend.current_ld == 0.17
        assert trend.current_se == 11.5

    def test_ld_se_updated(self):
        """LD and SE reflect the most recent values."""
        tracker = CompressionTracker()
        tracker.update("ch", 0.3, ld=0.17, se=11.5)
        tracker.update("ch", 0.4, ld=0.14, se=10.2)
        trend = tracker.get_trend("ch")
        assert trend is not None
        assert trend.current_ld == 0.14
        assert trend.current_se == 10.2


# ── Agent path tests (from_agent factory) ──────────────────


class TestFromAgent:
    """Tests for CompressionSignals.from_agent() — the agent-first path.

    These verify the plumbing: agent values get into the data structures
    correctly, the tracker aggregates them, and the system produces
    coherent output regardless of value source.
    """

    def test_basic_creation(self):
        """from_agent() creates valid CompressionSignals."""
        signals = CompressionSignals.from_agent(
            compression_level=0.6,
            temporal_direction=-0.3,
            bureaucratic_index=0.4,
        )
        assert signals.compression_index == 0.6
        assert signals.temporal_orientation == -0.3
        assert signals.bureaucratic_density == 0.4
        assert signals.source == "agent"

    def test_defaults_to_zero(self):
        """Unspecified agent values default to zero/neutral."""
        signals = CompressionSignals.from_agent()
        assert signals.compression_index == 0.0
        assert signals.temporal_orientation == 0.0
        assert signals.bureaucratic_density == 0.0
        assert signals.source == "agent"

    def test_ld_se_computed_from_text(self):
        """LD and SE are computed mechanically even in agent path."""
        signals = CompressionSignals.from_agent(
            compression_level=0.5,
            text="The quick brown fox jumps over the lazy dog near the old barn.",
        )
        assert signals.lexical_diversity > 0.0
        assert signals.shannon_entropy > 0.0
        assert signals.source == "agent"

    def test_no_text_no_ld_se(self):
        """Without text, LD and SE remain zero."""
        signals = CompressionSignals.from_agent(compression_level=0.5)
        assert signals.lexical_diversity == 0.0
        assert signals.shannon_entropy == 0.0

    def test_specificity_inverse_of_compression(self):
        """Specificity is 1 - compression_level in agent path."""
        signals = CompressionSignals.from_agent(compression_level=0.7)
        assert abs(signals.specificity_score - 0.3) < 0.01

    def test_tracker_accepts_agent_values(self):
        """CompressionTracker works identically with agent-sourced values."""
        tracker = CompressionTracker(alpha=0.5)
        # Simulate agent providing values over time
        tracker.update("channel-a", 0.2, ld=0.18, se=11.0)
        tracker.update("channel-a", 0.4, ld=0.15, se=10.5)
        tracker.update("channel-a", 0.6, ld=0.13, se=9.8)
        trend = tracker.get_trend("channel-a")
        assert trend is not None
        assert trend.trend == "increasing"
        assert trend.sample_count == 3
        assert trend.current_ld == 0.13

    def test_agent_and_regex_paths_share_tracker(self):
        """Tracker doesn't distinguish between agent and regex sources."""
        tracker = CompressionTracker()
        # Mix of agent and regex values
        tracker.update("ch", 0.3, ld=0.17, se=11.0)  # could be either source
        tracker.update("ch", 0.5, ld=0.14, se=10.0)  # could be either source
        trend = tracker.get_trend("ch")
        assert trend is not None
        assert trend.sample_count == 2

    def test_regex_analyze_has_source_field(self):
        """analyze_text() produces signals with source='regex'."""
        result = analyze_text("The server crashed at 3pm.")
        assert result.source == "regex"
