"""TemporalScanner: walks backward in time, feeding signals to the mesh.

The scanner starts at NOW and steps backward with widening windows.
Fresh data gets fine-grained attention. Historical data gets coarser
exploration as the mesh fills up. Natural stop conditions:

1. Max depth reached
2. All workers full (the elegant constraint)
3. Budget exhausted
4. No more signals to fetch

The adaptive step width creates exponential time-value decay:
  step *= growth_factor * (1 + avg_fullness)

When the mesh is empty, steps grow slowly (thorough exploration).
When the mesh is full, steps widen rapidly (only significant data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from stigmergy.cli.budget import DollarBudgetTracker
from stigmergy.mesh.mesh import Mesh, MeshTrace
from stigmergy.primitives.signal import Signal

logger = logging.getLogger(__name__)

# A function that fetches signals for a time range: (start, end) -> list[Signal]
SignalFetcher = Callable[[datetime, datetime], Awaitable[list[Signal]]]


@dataclass
class ScanWindow:
    """Record of a single scan window."""

    start: datetime
    end: datetime
    signals_fetched: int
    signals_accepted: int
    step_seconds: float
    mesh_fullness: float


@dataclass
class ScanResult:
    """Summary of a temporal scan."""

    total_signals: int = 0
    total_accepted: int = 0
    total_duplicates: int = 0
    windows: list[ScanWindow] = field(default_factory=list)
    traces: list[MeshTrace] = field(default_factory=list)
    deepest_timestamp: datetime | None = None
    stopped_reason: str = ""  # "max_depth" | "all_full" | "budget" | "no_signals"

    @property
    def windows_scanned(self) -> int:
        return len(self.windows)

    @property
    def depth_seconds(self) -> float:
        if not self.windows:
            return 0.0
        return (self.windows[0].end - self.windows[-1].start).total_seconds()


class TemporalScanner:
    """Walks backward in time, feeding signals to the mesh.

    Adaptive step width widens as:
    1. The scanner goes deeper into history (growth_factor)
    2. The mesh fills up (fullness multiplier)

    This creates natural time-value decay without explicit configuration.
    """

    def __init__(
        self,
        mesh: Mesh,
        *,
        min_step_seconds: float = 60.0,
        max_step_seconds: float = 86400.0,
        step_growth_factor: float = 1.5,
        max_depth_seconds: float = 604800.0,  # 1 week
        fresh_window_seconds: float = 300.0,  # 5 minutes
        budget: DollarBudgetTracker | None = None,
    ) -> None:
        self.mesh = mesh
        self.min_step = min_step_seconds
        self.max_step = max_step_seconds
        self.growth_factor = step_growth_factor
        self.max_depth = max_depth_seconds
        self.fresh_window = fresh_window_seconds
        self.budget = budget

    async def scan(
        self,
        fetcher: SignalFetcher,
        *,
        now: datetime | None = None,
    ) -> ScanResult:
        """Walk backward in time, feeding signals to the mesh.

        Args:
            fetcher: Async function (start, end) -> list[Signal]
            now: Starting timestamp (default: utcnow)

        Returns:
            ScanResult with full scan summary.
        """
        now = now or datetime.now(timezone.utc)
        result = ScanResult()

        cursor = now
        step = self.min_step
        origin = now  # track how far back we've gone

        while True:
            # Compute window
            window_end = cursor
            window_start = cursor - timedelta(seconds=step)
            depth = (origin - window_start).total_seconds()

            # Stop condition: max depth
            if depth > self.max_depth:
                result.stopped_reason = "max_depth"
                break

            # Stop condition: budget exhausted
            if self.budget and not self.budget.can_spend():
                result.stopped_reason = "budget"
                break

            # Stop condition: all workers full
            if self.mesh.all_full:
                result.stopped_reason = "all_full"
                break

            # Fetch signals in this window
            signals = await fetcher(window_start, window_end)

            # Stop condition: no signals and we're past fresh window
            if not signals and depth > self.fresh_window:
                result.stopped_reason = "no_signals"
                break

            # Feed signals to mesh
            window_accepted = 0
            for signal in signals:
                trace = await self.mesh.ingest(signal)
                result.traces.append(trace)
                result.total_signals += 1

                if trace.duplicate:
                    result.total_duplicates += 1
                elif trace.accepted_workers:
                    result.total_accepted += 1
                    window_accepted += 1

            # Record window
            result.windows.append(ScanWindow(
                start=window_start,
                end=window_end,
                signals_fetched=len(signals),
                signals_accepted=window_accepted,
                step_seconds=step,
                mesh_fullness=self.mesh.avg_fullness,
            ))
            result.deepest_timestamp = window_start

            # Advance cursor backward
            cursor = window_start

            # Adaptive step: widens with depth AND mesh fullness
            fullness_multiplier = 1.0 + self.mesh.avg_fullness
            step = min(
                self.max_step,
                step * self.growth_factor * fullness_multiplier,
            )

        return result
