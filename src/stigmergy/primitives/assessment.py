"""Assessment: an agent's evaluation of a signal within a context."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class AssessmentAction:
    """Well-known assessment actions.

    These are starting points, not constraints. Agents may use any
    self-describing string as an action — the system handles unknown
    actions gracefully. Every action string should be readable by a
    fresh pair of eyes without shared context.
    """

    STORE = "store"
    SURFACE = "surface"
    REPLICATE = "replicate"
    BLOCK = "block"
    IGNORE = "ignore"


class Assessment(BaseModel):
    """Produced by an agent after evaluating a signal against a context."""

    agent_id: UUID
    signal_id: UUID
    context_id: UUID
    action: str
    target_user: str | None = None
    target_context: UUID | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    domain: str
    reasoning: str
