"""Dollar-based budget tracker wrapping TokenBudget.

Three layers of cost control:
1. DollarBudgetTracker — converts tokens <-> dollars, tracks hourly/daily spend
2. Per-signal cap — max_tokens passed to LLM complete() calls
3. Circuit breaker — when budget exhausted, pipeline runs heuristic-only
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from stigmergy.services.token_budget import TokenBudget

logger = logging.getLogger(__name__)

# Model pricing table — (input_cost_per_million, output_cost_per_million)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
}


def pricing_for_model(model: str) -> tuple[float, float]:
    """Look up pricing for a model. Falls back to Sonnet if unknown."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Partial match — model IDs sometimes have date suffixes
    for key, pricing in MODEL_PRICING.items():
        if key.startswith(model) or model.startswith(key.rsplit("-", 1)[0]):
            return pricing
    logger.warning("Unknown model %r for pricing — defaulting to Sonnet rates", model)
    return MODEL_PRICING["claude-sonnet-4-5-20250929"]


@dataclass
class DollarBudgetTracker:
    """Wraps TokenBudget with dollar-denominated caps and hourly tracking.

    Tracks both estimated costs (from word count heuristic) and actual
    costs (from LLM API response token counts). Uses actual when available.
    """

    daily_cap_usd: float
    hourly_cap_usd: float
    input_cost_per_million: float = 0.0  # 0 = auto-detect from model
    output_cost_per_million: float = 0.0  # 0 = auto-detect from model
    warn_at_percent: int = 80

    # Internal state
    _daily_spend_usd: float = 0.0
    _hourly_spend_usd: float = 0.0
    _hour_start: float = field(default_factory=time.monotonic)
    _day_start: float = field(default_factory=time.monotonic)
    _warnings_emitted: set[str] = field(default_factory=set)
    _token_budget: TokenBudget | None = None
    _llm_ref: object | None = None
    _last_input_tokens: int = 0
    _last_output_tokens: int = 0
    # Multi-client tracking: each entry is a dict with keys
    # {llm, in_rate, out_rate, last_in, last_out}. Lets a tiered setup
    # (cheap per-signal model + quality correlator model) be priced correctly.
    _priced_llms: list = field(default_factory=list)

    def set_llm(self, llm) -> None:
        """Attach LLM service for actual token tracking (priced at the
        tracker's configured rates). Back-compat single-client entry point."""
        self._llm_ref = llm
        self._last_input_tokens = getattr(llm, "total_input_tokens", 0)
        self._last_output_tokens = getattr(llm, "total_output_tokens", 0)

    def add_priced_llm(self, llm, model: str) -> None:
        """Register an LLM client tracked at its OWN model's pricing.

        Use for tiered setups so each client's tokens are costed correctly
        (e.g. Haiku per-signal calls vs Sonnet correlator). Idempotent per
        client identity. Supersedes set_llm when present.
        """
        if llm is None:
            return
        for entry in self._priced_llms:
            if entry["llm"] is llm:
                return
        inp, out = pricing_for_model(model)
        self._priced_llms.append({
            "llm": llm,
            "in_rate": inp,
            "out_rate": out,
            "last_in": getattr(llm, "total_input_tokens", 0),
            "last_out": getattr(llm, "total_output_tokens", 0),
        })

    def set_model_pricing(self, model: str) -> None:
        """Auto-detect pricing from model name if not explicitly configured."""
        if self.input_cost_per_million > 0 and self.output_cost_per_million > 0:
            return  # User explicitly set pricing
        inp, out = pricing_for_model(model)
        self.input_cost_per_million = inp
        self.output_cost_per_million = out
        logger.info("Budget pricing: $%.2f/$%.2f per M tokens (model: %s)", inp, out, model)

    def tokens_to_dollars(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_million / 1_000_000
            + output_tokens * self.output_cost_per_million / 1_000_000
        )

    def dollars_to_tokens(self, dollars: float) -> int:
        """Approximate token count for a dollar amount (using input pricing)."""
        if self.input_cost_per_million <= 0:
            return 1_000_000  # Fallback
        return int(dollars / self.input_cost_per_million * 1_000_000)

    def build_token_budget(self, context_weights: dict[UUID, float]) -> TokenBudget:
        """Create and allocate a TokenBudget from dollar caps."""
        daily_tokens = self.dollars_to_tokens(self.daily_cap_usd)
        reserve = int(daily_tokens * 0.1)
        budget = TokenBudget(daily_cap=daily_tokens, reserve_pool=reserve)
        budget.allocate(context_weights)
        self._token_budget = budget
        return budget

    def _maybe_reset_hour(self) -> None:
        now = time.monotonic()
        if now - self._hour_start >= 3600:
            self._hourly_spend_usd = 0.0
            self._hour_start = now
            self._warnings_emitted.discard("hourly_warn")

    def _maybe_reset_day(self) -> None:
        now = time.monotonic()
        if now - self._day_start >= 86400:
            self._daily_spend_usd = 0.0
            self._day_start = now
            self._warnings_emitted.discard("daily_warn")
            if self._token_budget:
                self._token_budget.reset()

    def sync_from_llm(self) -> None:
        """Pull actual token counts from the LLM service and record the delta.

        Call this after each signal to track real API spend instead of estimates.
        """
        # Tiered path: price each registered client at its own model's rates.
        if self._priced_llms:
            self._maybe_reset_hour()
            self._maybe_reset_day()
            for entry in self._priced_llms:
                llm = entry["llm"]
                current_in = getattr(llm, "total_input_tokens", 0)
                current_out = getattr(llm, "total_output_tokens", 0)
                delta_in = current_in - entry["last_in"]
                delta_out = current_out - entry["last_out"]
                if delta_in > 0 or delta_out > 0:
                    cost = (
                        delta_in * entry["in_rate"] / 1_000_000
                        + delta_out * entry["out_rate"] / 1_000_000
                    )
                    self._daily_spend_usd += cost
                    self._hourly_spend_usd += cost
                entry["last_in"] = current_in
                entry["last_out"] = current_out
            return

        # Single-client path (back-compat): price at the tracker's rates.
        llm = self._llm_ref
        if llm is None:
            return
        current_in = getattr(llm, "total_input_tokens", 0)
        current_out = getattr(llm, "total_output_tokens", 0)
        delta_in = current_in - self._last_input_tokens
        delta_out = current_out - self._last_output_tokens
        if delta_in > 0 or delta_out > 0:
            self.record_spend(delta_in, delta_out)
        self._last_input_tokens = current_in
        self._last_output_tokens = current_out

    def record_spend(self, input_tokens: int, output_tokens: int) -> None:
        """Record a spend event."""
        self._maybe_reset_hour()
        self._maybe_reset_day()
        cost = self.tokens_to_dollars(input_tokens, output_tokens)
        self._daily_spend_usd += cost
        self._hourly_spend_usd += cost

    def can_spend(self) -> bool:
        """Check if we're within both hourly and daily caps."""
        self._maybe_reset_hour()
        self._maybe_reset_day()
        return (
            self._daily_spend_usd < self.daily_cap_usd
            and self._hourly_spend_usd < self.hourly_cap_usd
        )

    def check_warnings(self) -> list[str]:
        """Return any new warning messages."""
        warnings: list[str] = []
        self._maybe_reset_hour()
        self._maybe_reset_day()

        daily_pct = (self._daily_spend_usd / self.daily_cap_usd * 100) if self.daily_cap_usd > 0 else 0
        hourly_pct = (self._hourly_spend_usd / self.hourly_cap_usd * 100) if self.hourly_cap_usd > 0 else 0

        if daily_pct >= self.warn_at_percent and "daily_warn" not in self._warnings_emitted:
            warnings.append(f"Daily budget at {daily_pct:.0f}% (${self._daily_spend_usd:.4f}/${self.daily_cap_usd:.2f})")
            self._warnings_emitted.add("daily_warn")

        if hourly_pct >= self.warn_at_percent and "hourly_warn" not in self._warnings_emitted:
            warnings.append(f"Hourly budget at {hourly_pct:.0f}% (${self._hourly_spend_usd:.4f}/${self.hourly_cap_usd:.2f})")
            self._warnings_emitted.add("hourly_warn")

        if self._daily_spend_usd >= self.daily_cap_usd and "daily_exhausted" not in self._warnings_emitted:
            warnings.append("Daily budget EXHAUSTED — switching to heuristic-only mode")
            self._warnings_emitted.add("daily_exhausted")

        if self._hourly_spend_usd >= self.hourly_cap_usd and "hourly_exhausted" not in self._warnings_emitted:
            warnings.append("Hourly budget EXHAUSTED — switching to heuristic-only mode")
            self._warnings_emitted.add("hourly_exhausted")

        return warnings

    @property
    def daily_spend(self) -> float:
        return self._daily_spend_usd

    @property
    def hourly_spend(self) -> float:
        return self._hourly_spend_usd

    @property
    def total_spend(self) -> float:
        return self._daily_spend_usd
