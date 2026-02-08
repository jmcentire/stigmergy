"""Token budget management.

Allocates daily token budget proportionally to context business weight.
Agents request tokens for expensive operations; denied agents fall back
to cheaper heuristic evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class TokenBudget:
    daily_cap: int
    reserve_pool: int
    reserve_threshold: float = 0.8  # signal importance must exceed this to use reserve
    _context_budgets: dict[UUID, int] = field(default_factory=dict)
    _context_spend: dict[UUID, int] = field(default_factory=dict)
    _total_spend: int = 0

    def allocate(self, context_weights: dict[UUID, float]) -> None:
        """Reallocate budgets proportionally to business weights."""
        total_weight = sum(context_weights.values())
        if total_weight == 0:
            return
        allocatable = self.daily_cap - self.reserve_pool
        for ctx_id, weight in context_weights.items():
            self._context_budgets[ctx_id] = int(allocatable * (weight / total_weight))
            self._context_spend.setdefault(ctx_id, 0)

    def request(self, context_id: UUID, estimated_cost: int, signal_importance: float = 0.0) -> bool:
        """Request tokens. Returns True if granted, False if denied."""
        remaining = self._context_budgets.get(context_id, 0) - self._context_spend.get(context_id, 0)
        if remaining >= estimated_cost:
            self._context_spend[context_id] = self._context_spend.get(context_id, 0) + estimated_cost
            self._total_spend += estimated_cost
            return True
        shortfall = estimated_cost - max(remaining, 0)
        if signal_importance > self.reserve_threshold and self.reserve_pool >= shortfall:
            # Context budget covers what it can, reserve covers the rest
            from_context = min(remaining, estimated_cost) if remaining > 0 else 0
            self._context_spend[context_id] = self._context_spend.get(context_id, 0) + from_context
            self.reserve_pool -= shortfall
            self._total_spend += estimated_cost
            return True
        return False

    def remaining(self, context_id: UUID) -> int:
        return self._context_budgets.get(context_id, 0) - self._context_spend.get(context_id, 0)

    def reset(self) -> None:
        """Reset daily spend. Call at start of each budget period."""
        self._context_spend.clear()
        self._total_spend = 0
