"""Tests for feedback calibration: rating→action mapping and application."""

import pytest

from stigmergy.attention.feedback_calibration import (
    CalibrationAction,
    CalibrationApplicationError,
    CalibrationDimension,
    CalibratorProtocol,
    FeedbackCalibrationConfig,
    FeedbackRating,
    FeedbackResult,
    FeedbackStatus,
    apply_feedback,
    filter_enabled_actions,
    map_rating_to_actions,
    process_feedback_batch,
)


# ── Mock Calibrator ──────────────────────────────────────────────────


class MockCalibrator:
    """Records all calibration calls for assertions."""

    def __init__(self, *, fail_on_agent: str | None = None):
        self.calls: list[tuple[str, str, float]] = []  # (method, agent_id, delta)
        self._fail_on_agent = fail_on_agent

    async def calibration_lr_up(self, agent_id: str, delta: float) -> None:
        if agent_id == self._fail_on_agent:
            raise RuntimeError("Calibrator failed")
        self.calls.append(("up", agent_id, delta))

    async def calibration_lr_down(self, agent_id: str, delta: float) -> None:
        if agent_id == self._fail_on_agent:
            raise RuntimeError("Calibrator failed")
        self.calls.append(("down", agent_id, delta))


# ── map_rating_to_actions ────────────────────────────────────────────


class TestMapRatingToActions:
    def test_helpful(self):
        actions = map_rating_to_actions(FeedbackRating.HELPFUL)
        assert len(actions) == 1
        assert actions[0].dimension == CalibrationDimension.AGENT_CONFIDENCE
        assert actions[0].delta > 0

    def test_wrong(self):
        actions = map_rating_to_actions(FeedbackRating.WRONG)
        assert len(actions) == 1
        assert actions[0].dimension == CalibrationDimension.AGENT_CONFIDENCE
        assert actions[0].delta < 0

    def test_irrelevant(self):
        actions = map_rating_to_actions(FeedbackRating.IRRELEVANT)
        assert len(actions) == 1
        assert actions[0].dimension == CalibrationDimension.AGENT_CONFIDENCE
        assert actions[0].delta < 0

    def test_wrong_penalized_more_than_irrelevant(self):
        wrong_actions = map_rating_to_actions(FeedbackRating.WRONG)
        irrel_actions = map_rating_to_actions(FeedbackRating.IRRELEVANT)
        assert abs(wrong_actions[0].delta) > abs(irrel_actions[0].delta)

    def test_already_knew_two_actions(self):
        actions = map_rating_to_actions(FeedbackRating.ALREADY_KNEW)
        assert len(actions) == 2
        dims = {a.dimension for a in actions}
        assert CalibrationDimension.AGENT_CONFIDENCE in dims
        assert CalibrationDimension.SURFACING_PRIORITY in dims

    def test_already_knew_confidence_positive(self):
        actions = map_rating_to_actions(FeedbackRating.ALREADY_KNEW)
        conf = [a for a in actions if a.dimension == CalibrationDimension.AGENT_CONFIDENCE][0]
        assert conf.delta > 0

    def test_already_knew_surfacing_negative(self):
        actions = map_rating_to_actions(FeedbackRating.ALREADY_KNEW)
        surf = [a for a in actions if a.dimension == CalibrationDimension.SURFACING_PRIORITY][0]
        assert surf.delta < 0

    def test_all_actions_have_descriptions(self):
        for rating in FeedbackRating:
            actions = map_rating_to_actions(rating)
            for action in actions:
                assert len(action.description) > 0

    def test_custom_config(self):
        config = FeedbackCalibrationConfig(helpful_confidence_delta=0.10)
        actions = map_rating_to_actions(FeedbackRating.HELPFUL, config)
        assert actions[0].delta == pytest.approx(0.10)


# ── filter_enabled_actions ───────────────────────────────────────────


class TestFilterEnabledActions:
    def test_default_enables_agent_confidence_only(self):
        actions = map_rating_to_actions(FeedbackRating.ALREADY_KNEW)
        partition = filter_enabled_actions(actions)
        assert len(partition.enabled) == 1
        assert partition.enabled[0].dimension == CalibrationDimension.AGENT_CONFIDENCE
        assert len(partition.skipped) == 1
        assert partition.skipped[0].dimension == CalibrationDimension.SURFACING_PRIORITY

    def test_all_enabled(self):
        config = FeedbackCalibrationConfig(
            enabled_dimensions=frozenset({"agent_confidence", "surfacing_priority"})
        )
        actions = map_rating_to_actions(FeedbackRating.ALREADY_KNEW)
        partition = filter_enabled_actions(actions, config)
        assert len(partition.enabled) == 2
        assert len(partition.skipped) == 0

    def test_none_enabled(self):
        config = FeedbackCalibrationConfig(enabled_dimensions=frozenset())
        actions = map_rating_to_actions(FeedbackRating.HELPFUL)
        partition = filter_enabled_actions(actions, config)
        assert len(partition.enabled) == 0
        assert len(partition.skipped) == 1

    def test_preserves_order(self):
        actions = map_rating_to_actions(FeedbackRating.ALREADY_KNEW)
        partition = filter_enabled_actions(actions)
        assert len(partition.enabled) + len(partition.skipped) == len(actions)

    def test_empty_input(self):
        partition = filter_enabled_actions([])
        assert len(partition.enabled) == 0
        assert len(partition.skipped) == 0


# ── apply_feedback ───────────────────────────────────────────────────


class TestApplyFeedback:
    async def test_helpful_calls_lr_up(self):
        cal = MockCalibrator()
        result = await apply_feedback("fb-1", "agent-1", FeedbackRating.HELPFUL, cal)
        assert result.status == FeedbackStatus.APPLIED
        assert len(cal.calls) == 1
        assert cal.calls[0][0] == "up"
        assert cal.calls[0][1] == "agent-1"

    async def test_wrong_calls_lr_down(self):
        cal = MockCalibrator()
        result = await apply_feedback("fb-2", "agent-1", FeedbackRating.WRONG, cal)
        assert result.status == FeedbackStatus.APPLIED
        assert len(cal.calls) == 1
        assert cal.calls[0][0] == "down"

    async def test_already_knew_partial(self):
        cal = MockCalibrator()
        result = await apply_feedback(
            "fb-3", "agent-1", FeedbackRating.ALREADY_KNEW, cal
        )
        # Default config: only agent_confidence enabled
        assert result.status == FeedbackStatus.PARTIAL
        assert len(result.applied_actions) == 1
        assert len(result.skipped_actions) == 1

    async def test_already_knew_all_enabled(self):
        cal = MockCalibrator()
        config = FeedbackCalibrationConfig(
            enabled_dimensions=frozenset({"agent_confidence", "surfacing_priority"})
        )
        result = await apply_feedback(
            "fb-4", "agent-1", FeedbackRating.ALREADY_KNEW, cal, config
        )
        assert result.status == FeedbackStatus.APPLIED
        assert len(result.applied_actions) == 2
        assert len(cal.calls) == 2

    async def test_no_enabled_dimensions(self):
        cal = MockCalibrator()
        config = FeedbackCalibrationConfig(enabled_dimensions=frozenset())
        result = await apply_feedback(
            "fb-5", "agent-1", FeedbackRating.HELPFUL, cal, config
        )
        assert result.status == FeedbackStatus.SKIPPED
        assert len(cal.calls) == 0

    async def test_calibrator_error_raises(self):
        cal = MockCalibrator(fail_on_agent="bad-agent")
        with pytest.raises(CalibrationApplicationError):
            await apply_feedback(
                "fb-6", "bad-agent", FeedbackRating.HELPFUL, cal
            )

    async def test_result_feedback_id_matches(self):
        cal = MockCalibrator()
        result = await apply_feedback("fb-7", "agent-1", FeedbackRating.HELPFUL, cal)
        assert result.feedback_id == "fb-7"


# ── process_feedback_batch ───────────────────────────────────────────


class TestProcessFeedbackBatch:
    async def test_empty_batch(self):
        cal = MockCalibrator()
        results = await process_feedback_batch([], cal)
        assert results == []

    async def test_sequential_processing(self):
        cal = MockCalibrator()
        entries = [
            ("fb-1", "agent-1", FeedbackRating.HELPFUL),
            ("fb-2", "agent-1", FeedbackRating.WRONG),
            ("fb-3", "agent-2", FeedbackRating.IRRELEVANT),
        ]
        results = await process_feedback_batch(entries, cal)
        assert len(results) == 3
        assert results[0].feedback_id == "fb-1"
        assert results[1].feedback_id == "fb-2"
        assert results[2].feedback_id == "fb-3"

    async def test_failure_does_not_halt_batch(self):
        cal = MockCalibrator(fail_on_agent="bad-agent")
        entries = [
            ("fb-1", "agent-1", FeedbackRating.HELPFUL),
            ("fb-2", "bad-agent", FeedbackRating.WRONG),  # Will fail
            ("fb-3", "agent-2", FeedbackRating.HELPFUL),  # Should still run
        ]
        results = await process_feedback_batch(entries, cal)
        assert len(results) == 3
        assert results[0].status == FeedbackStatus.APPLIED
        assert results[1].status == FeedbackStatus.SKIPPED
        assert results[2].status == FeedbackStatus.APPLIED

    async def test_one_result_per_entry(self):
        cal = MockCalibrator()
        entries = [("fb-1", "a", FeedbackRating.HELPFUL)] * 5
        results = await process_feedback_batch(entries, cal)
        assert len(results) == 5
