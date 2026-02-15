"""InquiryResult: the output of the active inquiry step.

When a worker accepts a signal (scores above threshold), it decomposes
the signal into dimensions (people, projects, tickets, keywords, sentiment)
and queries its context knowledge base for relevant connections.

The InquiryResult is a frozen (immutable) Pydantic model following the same
pattern as Assessment. It feeds into the downstream assessment/consensus phase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class InquiryDimension(StrEnum):
    """The five inquiry dimensions used to decompose a signal."""

    PEOPLE = "people"
    PROJECTS = "projects"
    TICKETS = "tickets"
    KEYWORDS = "keywords"
    SENTIMENT = "sentiment"


class InquiryResult(BaseModel):
    """Frozen Pydantic model: output of the active inquiry step.

    Produced by the single accepting worker after signal acceptance and
    before consensus. Immutable at the shallow level — nested mutable
    containers (list values in dimensions dict) are not deep-frozen.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    signal_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    dimensions: dict[str, list[str]] = Field(default_factory=dict)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("confidence_scores")
    @classmethod
    def _validate_confidence_range(cls, v: dict[str, float]) -> dict[str, float]:
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(
                    f"All confidence scores must be in the range [0.0, 1.0], "
                    f"got {score} for dimension '{key}'"
                )
        return v

    @field_validator("dimensions")
    @classmethod
    def _validate_dimension_keys(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        valid_keys = {d.value for d in InquiryDimension}
        for key in v:
            if key not in valid_keys:
                raise ValueError(
                    f"All keys must be valid InquiryDimension variants: "
                    f"{', '.join(sorted(valid_keys))}, got '{key}'"
                )
        return v

    @model_validator(mode="after")
    def _validate_key_alignment(self) -> InquiryResult:
        if set(self.confidence_scores.keys()) != set(self.dimensions.keys()):
            raise ValueError(
                "confidence_scores keys must exactly equal dimensions keys"
            )
        return self


def create_inquiry_result(
    signal_id: str,
    context_id: str,
    dimensions: dict[str, list[str]],
    confidence_scores: dict[str, float],
    *,
    id: str | None = None,
    timestamp: str | None = None,
) -> InquiryResult:
    """Construct and validate a new InquiryResult.

    Primary factory entrypoint — callers should prefer this over
    direct constructor access to ensure all cross-field invariants
    are checked.
    """
    kwargs: dict[str, Any] = {
        "signal_id": signal_id,
        "context_id": context_id,
        "dimensions": dimensions,
        "confidence_scores": confidence_scores,
    }
    if id is not None:
        kwargs["id"] = id
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return InquiryResult(**kwargs)


def get_dimension_connections(
    inquiry_result: InquiryResult,
    dimension: InquiryDimension,
) -> list[str]:
    """Retrieve connections for a specific dimension, or empty list if absent."""
    return list(inquiry_result.dimensions.get(dimension.value, []))


def get_dimension_confidence(
    inquiry_result: InquiryResult,
    dimension: InquiryDimension,
) -> float:
    """Retrieve confidence for a dimension, or 0.0 if absent."""
    return inquiry_result.confidence_scores.get(dimension.value, 0.0)


def get_active_dimensions(
    inquiry_result: InquiryResult,
) -> list[InquiryDimension]:
    """Return dimensions that have at least one connection."""
    return [
        InquiryDimension(k)
        for k, v in inquiry_result.dimensions.items()
        if len(v) >= 1
    ]


def get_overall_confidence(
    inquiry_result: InquiryResult,
) -> float:
    """Mean confidence across all present dimensions. 0.0 if empty."""
    scores = inquiry_result.confidence_scores
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def serialize_inquiry_result(inquiry_result: InquiryResult) -> dict:
    """Serialize to a JSON-compatible dict."""
    return inquiry_result.model_dump(mode="json")


def deserialize_inquiry_result(data: dict) -> InquiryResult:
    """Deserialize from a JSON-compatible dict."""
    return InquiryResult.model_validate(data)
