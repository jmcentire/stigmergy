"""Append-only trace logging.

Every action the system takes is traceable back through its full decision chain.
Every trace is immutable and stored in an append-only log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from stigmergy.constraints.filter import ConstraintResult
from stigmergy.core.consensus import ConsensusResult
from stigmergy.primitives.assessment import Assessment


@dataclass
class Trace:
    """Full decision chain for a single signal."""

    id: UUID = field(default_factory=uuid4)
    signal_id: UUID | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contexts: set[UUID] = field(default_factory=set)
    familiarity_scores: dict[UUID, float] = field(default_factory=dict)
    assessments: list[Assessment] = field(default_factory=list)
    consensus: ConsensusResult | None = None
    constraint_eval: ConstraintResult | None = None
    output_delivered: bool = False
    output_content: str | None = None


class TraceLog:
    """Append-only trace log. Never delete from it."""

    def __init__(self) -> None:
        self._traces: list[Trace] = []

    def append(self, trace: Trace) -> None:
        self._traces.append(trace)

    def get(self, trace_id: UUID) -> Trace | None:
        for t in self._traces:
            if t.id == trace_id:
                return t
        return None

    def for_signal(self, signal_id: UUID) -> list[Trace]:
        return [t for t in self._traces if t.signal_id == signal_id]

    def recent(self, n: int = 10) -> list[Trace]:
        return list(reversed(self._traces[-n:]))

    def count(self) -> int:
        return len(self._traces)
