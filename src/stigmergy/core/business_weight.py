"""Business weight computation: how important is a context organizationally?

Business weight is a composite score from four metrics:
  - signal_frequency: how often the context generates actionable signals
  - author_seniority: authority level of primary identities
  - channel_gravity: learned importance of the channel/medium
  - engagement_velocity: rate of change of engagement (urgency/momentum)

Used for token budget allocation and routing priority. A context with
higher business weight gets more tokens and is checked earlier in routing.

All compute functions are pure — no side effects, no mutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


class MetricName(StrEnum):
    """The four component metric names."""

    SIGNAL_FREQUENCY = "signal_frequency"
    AUTHOR_SENIORITY = "author_seniority"
    CHANNEL_GRAVITY = "channel_gravity"
    ENGAGEMENT_VELOCITY = "engagement_velocity"


class BusinessWeightMetrics(BaseModel):
    """Mutable tracking state for the four business weight components.

    Each metric defaults to 0.5 (neutral/no-information sentinel).
    Not frozen — metrics update over time as signals arrive.
    """

    signal_frequency: float = Field(default=0.5, ge=0.0, le=1.0)
    author_seniority: float = Field(default=0.5, ge=0.0, le=1.0)
    channel_gravity: float = Field(default=0.5, ge=0.0, le=1.0)
    engagement_velocity: float = Field(default=0.5, ge=0.0, le=1.0)


class BusinessWeightConfig(BaseModel):
    """Configuration for business weight computation."""

    weight_signal_frequency: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_author_seniority: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_channel_gravity: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_engagement_velocity: float = Field(default=0.25, ge=0.0, le=1.0)
    policy_floor: float = Field(default=0.1, ge=0.0, le=1.0)


@dataclass(frozen=True)
class BusinessWeightResult:
    """Result of a business weight computation."""

    raw_weight: float
    effective_weight: float


class MetricUpdateRequest(BaseModel):
    """Partial update to business weight metrics. None = leave unchanged."""

    signal_frequency: float | None = None
    author_seniority: float | None = None
    channel_gravity: float | None = None
    engagement_velocity: float | None = None


def _validate_float_range(value: float, field: str) -> None:
    """Validate a float is in [0.0, 1.0] and not NaN."""
    if math.isnan(value):
        raise ValueError(f"{field} must not be NaN")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{field} must be between 0.0 and 1.0, got {value}")


def validate_metrics(metrics: BusinessWeightMetrics) -> bool:
    """Validate all metrics are in [0.0, 1.0] and not NaN."""
    for name in MetricName:
        value = getattr(metrics, name.value)
        _validate_float_range(value, name.value)
    return True


def compute_business_weight(
    metrics: BusinessWeightMetrics,
    config: BusinessWeightConfig | None = None,
) -> BusinessWeightResult:
    """Pure function: compute composite business weight from metrics.

    Formula: raw = clamp(weighted_avg(metrics, config_weights), 0.0, 1.0)
    effective = max(raw, policy_floor)
    """
    if config is None:
        config = BusinessWeightConfig()

    # Validate metrics
    for name in MetricName:
        value = getattr(metrics, name.value)
        if math.isnan(value) or not (0.0 <= value <= 1.0):
            raise ValueError(
                f"Invalid metric value for {name.value}: {value}"
            )

    # Validate config weights
    for field_name in [
        "weight_signal_frequency",
        "weight_author_seniority",
        "weight_channel_gravity",
        "weight_engagement_velocity",
    ]:
        value = getattr(config, field_name)
        if math.isnan(value) or not (0.0 <= value <= 1.0):
            raise ValueError(
                f"Invalid config weight for {field_name}: {value}"
            )

    if math.isnan(config.policy_floor) or not (0.0 <= config.policy_floor <= 1.0):
        raise ValueError(
            f"Invalid policy_floor: {config.policy_floor}"
        )

    total_weight = (
        config.weight_signal_frequency
        + config.weight_author_seniority
        + config.weight_channel_gravity
        + config.weight_engagement_velocity
    )

    if total_weight == 0.0:
        raw = 0.5  # Neutral default when all weights are zero
    else:
        weighted_sum = (
            config.weight_signal_frequency * metrics.signal_frequency
            + config.weight_author_seniority * metrics.author_seniority
            + config.weight_channel_gravity * metrics.channel_gravity
            + config.weight_engagement_velocity * metrics.engagement_velocity
        )
        raw = max(0.0, min(1.0, weighted_sum / total_weight))

    effective = max(raw, config.policy_floor)

    return BusinessWeightResult(raw_weight=raw, effective_weight=effective)


def apply_metric_update(
    current_metrics: BusinessWeightMetrics,
    update: MetricUpdateRequest,
) -> BusinessWeightMetrics:
    """Apply partial metric update, returning a new copy. Pure function."""
    updates = {}
    for field_name in ["signal_frequency", "author_seniority",
                       "channel_gravity", "engagement_velocity"]:
        new_val = getattr(update, field_name)
        if new_val is not None:
            if math.isnan(new_val) or not (0.0 <= new_val <= 1.0):
                raise ValueError(
                    f"Metric value out of range for {field_name}: {new_val}"
                )
            updates[field_name] = new_val

    return current_metrics.model_copy(update=updates)


def get_effective_business_weight(
    business_weight: float,
    policy_floor: float = 0.1,
) -> float:
    """Convenience: max(business_weight, policy_floor)."""
    if math.isnan(business_weight) or not (0.0 <= business_weight <= 1.0):
        raise ValueError(
            f"business_weight must be between 0.0 and 1.0, got {business_weight}"
        )
    if math.isnan(policy_floor) or not (0.0 <= policy_floor <= 1.0):
        raise ValueError(
            f"policy_floor must be between 0.0 and 1.0, got {policy_floor}"
        )
    return max(business_weight, policy_floor)


def create_default_metrics() -> BusinessWeightMetrics:
    """Factory: all metrics at neutral default (0.5)."""
    return BusinessWeightMetrics()


# ── Routing & Budget Allocation ───────────────────────────────────


def validate_business_weight(raw_weight: float) -> float:
    """Validate and normalize a raw business weight. Floor at 0.1."""
    if math.isnan(raw_weight) or math.isinf(raw_weight):
        raise ValueError(
            f"business_weight must be a finite number, got {raw_weight}"
        )
    if raw_weight < 0.0:
        raise ValueError(
            f"business_weight must be >= 0.0, got {raw_weight}"
        )
    if raw_weight > 1.0:
        raise ValueError(
            f"business_weight must be <= 1.0, got {raw_weight}"
        )
    return max(0.1, raw_weight)


@dataclass(frozen=True)
class TokenBudgetAllocation:
    """Allocation of token budget to a context."""

    context_id: str
    allocated_tokens: int
    business_weight: float
    proportion: float


def weighted_routing_order(
    context_weights: list[tuple[str, float]],
) -> list[str]:
    """Sort context IDs by descending business_weight, stable tiebreak.

    Returns list of context_ids in priority order.
    Pure function — no side effects.
    """
    # Stable sort: enumerate for tiebreak
    indexed = list(enumerate(context_weights))
    indexed.sort(key=lambda x: (-x[1][1], x[0]))
    return [cw[0] for _, cw in indexed]


def allocate_token_budget(
    total_budget: int,
    context_weights: list[tuple[str, float]],
) -> list[TokenBudgetAllocation]:
    """Distribute token budget proportionally by business_weight.

    Uses largest-remainder (Hare quota) rounding so the sum of
    allocated_tokens equals total_budget exactly.
    """
    if total_budget < 0:
        raise ValueError(
            f"total_budget must be non-negative, got {total_budget}"
        )
    if not context_weights:
        raise ValueError(
            "contexts must be non-empty; cannot allocate budget to zero contexts."
        )

    # Check for duplicate IDs
    ids = [cw[0] for cw in context_weights]
    seen = set()
    for cid in ids:
        if cid in seen:
            raise ValueError(f"Duplicate context_id found: {cid}")
        seen.add(cid)

    # Validate and normalize weights
    validated = []
    for cid, w in context_weights:
        vw = validate_business_weight(w)
        validated.append((cid, vw))

    total_weight = sum(w for _, w in validated)
    if total_weight == 0.0:
        raise ValueError(
            "Total business_weight sum is 0.0; cannot compute proportional allocation."
        )

    if total_budget == 0:
        return [
            TokenBudgetAllocation(
                context_id=cid,
                allocated_tokens=0,
                business_weight=w,
                proportion=0.0,
            )
            for cid, w in validated
        ]

    # Compute proportions and floor allocations
    proportions = [(cid, w, w / total_weight) for cid, w in validated]
    ideal = [(cid, w, prop, prop * total_budget) for cid, w, prop in proportions]

    # Largest-remainder method
    floors = [(cid, w, prop, int(tokens), tokens - int(tokens))
              for cid, w, prop, tokens in ideal]
    allocated_sum = sum(f[3] for f in floors)
    remainder = total_budget - allocated_sum

    # Sort by remainder descending for distributing extra tokens
    by_remainder = sorted(enumerate(floors), key=lambda x: -x[1][4])
    result_tokens = [f[3] for f in floors]
    for i in range(remainder):
        idx = by_remainder[i][0]
        result_tokens[idx] += 1

    return [
        TokenBudgetAllocation(
            context_id=floors[i][0],
            allocated_tokens=result_tokens[i],
            business_weight=floors[i][1],
            proportion=floors[i][2],
        )
        for i in range(len(floors))
    ]
