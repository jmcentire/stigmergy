"""Feedback calibration: map user ratings to agent confidence adjustments.

Maps the four feedback ratings (helpful, wrong, irrelevant, already_knew) to
CalibrationActions that adjust agent confidence scores via a CalibratorProtocol.

Architecture:
  - Pure types: FeedbackRating, CalibrationDimension, CalibrationAction, etc.
  - Pure functions: map_rating_to_actions, filter_enabled_actions
  - Async I/O: apply_feedback, process_feedback_batch

Initially only 'agent_confidence' is enabled. Deferred dimensions
(surfacing_priority, business_weight, familiarity_weight) are computed
but skipped until Features 3 and 4 land.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────


class FeedbackRating(StrEnum):
    """The four possible user feedback ratings for a finding."""

    HELPFUL = "helpful"
    WRONG = "wrong"
    IRRELEVANT = "irrelevant"
    ALREADY_KNEW = "already_knew"


class CalibrationDimension(StrEnum):
    """Target dimension for a calibration action."""

    AGENT_CONFIDENCE = "agent_confidence"
    SURFACING_PRIORITY = "surfacing_priority"
    BUSINESS_WEIGHT = "business_weight"
    FAMILIARITY_WEIGHT = "familiarity_weight"


class FeedbackStatus(StrEnum):
    """Outcome of applying feedback."""

    APPLIED = "applied"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class CalibrationAction(BaseModel):
    """A single calibration adjustment to apply or skip."""

    model_config = ConfigDict(frozen=True)

    dimension: CalibrationDimension
    delta: float = Field(ge=-1.0, le=1.0)
    description: str = Field(min_length=1)


class FeedbackResult(BaseModel):
    """Outcome of applying a single feedback entry."""

    model_config = ConfigDict(frozen=True)

    feedback_id: str
    applied_actions: list[CalibrationAction] = Field(default_factory=list)
    skipped_actions: list[CalibrationAction] = Field(default_factory=list)
    status: FeedbackStatus
    reason: str | None = None


class FeedbackCalibrationConfig(BaseModel):
    """Configuration for rating-to-action mapping and dimension gating."""

    model_config = ConfigDict(frozen=True)

    helpful_confidence_delta: float = Field(default=0.05, gt=0.0, le=1.0)
    wrong_confidence_delta: float = Field(default=0.08, gt=0.0, le=1.0)
    irrelevant_confidence_delta: float = Field(default=0.03, gt=0.0, le=1.0)
    already_knew_confidence_delta: float = Field(default=0.02, gt=0.0, le=1.0)
    already_knew_surfacing_priority_delta: float = Field(
        default=0.04, gt=0.0, le=1.0
    )
    enabled_dimensions: frozenset[str] = frozenset({"agent_confidence"})


class EnabledSkippedPartition(BaseModel):
    """Partition of actions into enabled and skipped."""

    model_config = ConfigDict(frozen=True)

    enabled: list[CalibrationAction] = Field(default_factory=list)
    skipped: list[CalibrationAction] = Field(default_factory=list)


# ── Protocols ────────────────────────────────────────────────────────


@runtime_checkable
class CalibratorProtocol(Protocol):
    """Interface to the concrete calibration system."""

    async def calibration_lr_up(self, agent_id: str, delta: float) -> None:
        ...

    async def calibration_lr_down(self, agent_id: str, delta: float) -> None:
        ...


# ── Errors ───────────────────────────────────────────────────────────


class CalibrationApplicationError(Exception):
    """Raised when the calibrator fails to apply an action."""

    def __init__(
        self, feedback_id: str, agent_id: str, original_error: str
    ) -> None:
        self.feedback_id = feedback_id
        self.agent_id = agent_id
        self.original_error = original_error
        super().__init__(
            f"Calibration failed for feedback={feedback_id}, "
            f"agent={agent_id}: {original_error}"
        )


# ── Pure Functions ───────────────────────────────────────────────────


def map_rating_to_actions(
    rating: FeedbackRating,
    config: FeedbackCalibrationConfig | None = None,
) -> list[CalibrationAction]:
    """Map a feedback rating to calibration actions across all dimensions.

    Returns the full list of intended actions; filtering by enabled
    dimensions happens separately via filter_enabled_actions.
    """
    if config is None:
        config = FeedbackCalibrationConfig()

    match rating:
        case FeedbackRating.HELPFUL:
            return [
                CalibrationAction(
                    dimension=CalibrationDimension.AGENT_CONFIDENCE,
                    delta=config.helpful_confidence_delta,
                    description="Increase agent confidence: rating=helpful",
                ),
            ]
        case FeedbackRating.WRONG:
            return [
                CalibrationAction(
                    dimension=CalibrationDimension.AGENT_CONFIDENCE,
                    delta=-config.wrong_confidence_delta,
                    description="Decrease agent confidence: rating=wrong",
                ),
            ]
        case FeedbackRating.IRRELEVANT:
            return [
                CalibrationAction(
                    dimension=CalibrationDimension.AGENT_CONFIDENCE,
                    delta=-config.irrelevant_confidence_delta,
                    description=(
                        "Decrease agent confidence: rating=irrelevant"
                    ),
                ),
            ]
        case FeedbackRating.ALREADY_KNEW:
            return [
                CalibrationAction(
                    dimension=CalibrationDimension.AGENT_CONFIDENCE,
                    delta=config.already_knew_confidence_delta,
                    description=(
                        "Mild increase agent confidence: "
                        "rating=already_knew (correct but low novelty)"
                    ),
                ),
                CalibrationAction(
                    dimension=CalibrationDimension.SURFACING_PRIORITY,
                    delta=-config.already_knew_surfacing_priority_delta,
                    description=(
                        "Decrease surfacing priority: "
                        "rating=already_knew (reduce low-novelty findings)"
                    ),
                ),
            ]


def filter_enabled_actions(
    actions: list[CalibrationAction],
    config: FeedbackCalibrationConfig | None = None,
) -> EnabledSkippedPartition:
    """Partition actions into enabled and skipped based on dimension gating."""
    if config is None:
        config = FeedbackCalibrationConfig()

    enabled = []
    skipped = []
    for action in actions:
        if action.dimension.value in config.enabled_dimensions:
            enabled.append(action)
        else:
            skipped.append(action)

    return EnabledSkippedPartition(enabled=enabled, skipped=skipped)


# ── Async I/O ────────────────────────────────────────────────────────


async def apply_feedback(
    feedback_id: str,
    agent_id: str,
    rating: FeedbackRating,
    calibrator: CalibratorProtocol,
    config: FeedbackCalibrationConfig | None = None,
) -> FeedbackResult:
    """Process a single feedback event: map, filter, apply.

    For positive deltas calls calibrator.calibration_lr_up;
    for negative deltas calls calibrator.calibration_lr_down with abs(delta).
    """
    if config is None:
        config = FeedbackCalibrationConfig()

    actions = map_rating_to_actions(rating, config)
    partition = filter_enabled_actions(actions, config)

    for action in partition.skipped:
        logger.debug(
            "Skipped deferred calibration: %s (dimension %s not enabled)",
            action.description,
            action.dimension,
        )

    applied = []
    for action in partition.enabled:
        try:
            if action.delta > 0:
                await calibrator.calibration_lr_up(agent_id, action.delta)
            else:
                await calibrator.calibration_lr_down(
                    agent_id, abs(action.delta)
                )
            applied.append(action)
            logger.info(
                "Applied calibration: %s for agent=%s",
                action.description,
                agent_id,
            )
        except Exception as exc:
            raise CalibrationApplicationError(
                feedback_id=feedback_id,
                agent_id=agent_id,
                original_error=str(exc),
            ) from exc

    if not applied and partition.skipped:
        status = FeedbackStatus.SKIPPED
        reason = "All actions targeted disabled dimensions"
    elif partition.skipped:
        status = FeedbackStatus.PARTIAL
        reason = (
            f"{len(partition.skipped)} action(s) skipped "
            f"(disabled dimensions)"
        )
    else:
        status = FeedbackStatus.APPLIED
        reason = None

    return FeedbackResult(
        feedback_id=feedback_id,
        applied_actions=applied,
        skipped_actions=partition.skipped,
        status=status,
        reason=reason,
    )


async def process_feedback_batch(
    entries: list[tuple[str, str, FeedbackRating]],
    calibrator: CalibratorProtocol,
    config: FeedbackCalibrationConfig | None = None,
) -> list[FeedbackResult]:
    """Process a batch of feedback entries sequentially.

    Each entry is (feedback_id, agent_id, rating).
    Failures are captured in results, not raised.
    """
    if config is None:
        config = FeedbackCalibrationConfig()

    results = []
    for feedback_id, agent_id, rating in entries:
        try:
            result = await apply_feedback(
                feedback_id, agent_id, rating, calibrator, config
            )
            results.append(result)
        except CalibrationApplicationError as exc:
            logger.warning(
                "Feedback processing failed for %s: %s",
                feedback_id,
                exc,
            )
            results.append(
                FeedbackResult(
                    feedback_id=feedback_id,
                    applied_actions=[],
                    skipped_actions=[],
                    status=FeedbackStatus.SKIPPED,
                    reason=str(exc),
                )
            )
        except Exception as exc:
            logger.warning(
                "Unexpected error processing feedback %s: %s",
                feedback_id,
                exc,
            )
            results.append(
                FeedbackResult(
                    feedback_id=feedback_id,
                    applied_actions=[],
                    skipped_actions=[],
                    status=FeedbackStatus.SKIPPED,
                    reason=f"Unexpected error: {exc}",
                )
            )

    return results
