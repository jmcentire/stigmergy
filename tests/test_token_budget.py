"""Tests for token budget management.

Allocation proportional to business weight.
Denial triggers fallback behavior.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from stigmergy.services.token_budget import TokenBudget


class TestAllocation:
    def test_proportional_allocation(self):
        budget = TokenBudget(daily_cap=10000, reserve_pool=1000)
        ctx_a = uuid4()
        ctx_b = uuid4()
        budget.allocate({ctx_a: 3.0, ctx_b: 1.0})

        # 9000 allocatable (10000 - 1000 reserve)
        # ctx_a gets 75% = 6750, ctx_b gets 25% = 2250
        assert budget.remaining(ctx_a) == 6750
        assert budget.remaining(ctx_b) == 2250

    def test_equal_weights(self):
        budget = TokenBudget(daily_cap=10000, reserve_pool=0)
        ctx_a = uuid4()
        ctx_b = uuid4()
        budget.allocate({ctx_a: 1.0, ctx_b: 1.0})
        assert budget.remaining(ctx_a) == 5000
        assert budget.remaining(ctx_b) == 5000


class TestRequestGrant:
    def test_grant_within_budget(self):
        budget = TokenBudget(daily_cap=10000, reserve_pool=0)
        ctx = uuid4()
        budget.allocate({ctx: 1.0})
        assert budget.request(ctx, 5000) is True
        assert budget.remaining(ctx) == 5000

    def test_deny_over_budget(self):
        budget = TokenBudget(daily_cap=1000, reserve_pool=0)
        ctx = uuid4()
        budget.allocate({ctx: 1.0})
        assert budget.request(ctx, 1001) is False
        assert budget.remaining(ctx) == 1000  # unchanged

    def test_grant_from_reserve(self):
        budget = TokenBudget(daily_cap=2000, reserve_pool=500, reserve_threshold=0.8)
        ctx = uuid4()
        budget.allocate({ctx: 1.0})
        # Budget is 1500, request 1600 — over budget but high importance
        assert budget.request(ctx, 1600, signal_importance=0.9) is True

    def test_deny_from_reserve_low_importance(self):
        budget = TokenBudget(daily_cap=2000, reserve_pool=500, reserve_threshold=0.8)
        ctx = uuid4()
        budget.allocate({ctx: 1.0})
        # Over budget, but importance too low for reserve
        assert budget.request(ctx, 1600, signal_importance=0.5) is False

    def test_cumulative_spend(self):
        budget = TokenBudget(daily_cap=10000, reserve_pool=0)
        ctx = uuid4()
        budget.allocate({ctx: 1.0})
        budget.request(ctx, 3000)
        budget.request(ctx, 3000)
        assert budget.remaining(ctx) == 4000
        budget.request(ctx, 3000)
        assert budget.remaining(ctx) == 1000
        # This should fail — only 1000 remaining
        assert budget.request(ctx, 2000) is False


class TestReset:
    def test_reset_clears_spend(self):
        budget = TokenBudget(daily_cap=10000, reserve_pool=0)
        ctx = uuid4()
        budget.allocate({ctx: 1.0})
        budget.request(ctx, 8000)
        assert budget.remaining(ctx) == 2000
        budget.reset()
        # After reset, spend should be 0 but budget needs reallocation
        # The _context_budgets persist, only _context_spend resets
        assert budget.remaining(ctx) == 10000
