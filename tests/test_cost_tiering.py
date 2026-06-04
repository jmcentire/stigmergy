"""Cost-reduction regression tests: noise pre-filter, tiered-model budget
pricing, and fast-LLM resolution.

Context: an all-Sonnet run cost ~$0.023/signal and an all-Haiku run cost MORE
(weak correlator destabilized the mesh). The fix routes high-volume per-signal
calls to a cheap tier and keeps the batched correlator on a quality tier, and
drops obvious Slack chaff before any LLM call.
"""

from dataclasses import dataclass

import pytest

from stigmergy.cli.budget import DollarBudgetTracker
from stigmergy.cli.run_cmd import _is_noise_signal, _resolve_fast_llm


@dataclass
class _Sig:
    content: str
    metadata: dict = None


# ── noise pre-filter ──────────────────────────────────────────

@pytest.mark.parametrize("content", ["", "   ", "\n", "👍", ":+1:", "🎉🎉", "..."])
def test_drops_empty_and_emoji(content):
    assert _is_noise_signal(_Sig(content)) is True


@pytest.mark.parametrize("content", ["ok", "thanks", "LGTM", "+1", "done!", "ty", "sounds good"])
def test_drops_short_acks(content):
    assert _is_noise_signal(_Sig(content)) is True


def test_drops_slack_system_subtype():
    assert _is_noise_signal(_Sig("x joined", {"subtype": "channel_join"})) is True


@pytest.mark.parametrize("content", [
    "the booking sync to Guesty is failing for Four Daughters",
    "can someone review PR #1861, it touches the pricing path",
    "lgtm but check the tax edge case in checkout",  # >16 chars: real content
    "ok so the issue is the magic-link redirect handler",  # starts with ok but substantive
])
def test_keeps_substantive_messages(content):
    assert _is_noise_signal(_Sig(content)) is False


# ── tiered-model budget pricing ───────────────────────────────

class _FakeLLM:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0


def test_multi_client_pricing_uses_each_models_rate():
    b = DollarBudgetTracker(daily_cap_usd=100.0, hourly_cap_usd=100.0)
    fast, quality = _FakeLLM(), _FakeLLM()
    b.add_priced_llm(quality, "claude-sonnet-4-5-20250929")  # $3/$15 per M
    b.add_priced_llm(fast, "claude-haiku-4-5-20251001")      # $0.80/$4 per M

    # 1M input tokens on each tier
    fast.total_input_tokens = 1_000_000
    quality.total_input_tokens = 1_000_000
    b.sync_from_llm()

    # Haiku input $0.80 + Sonnet input $3.00 = $3.80
    assert b.daily_spend == pytest.approx(3.80, abs=1e-6)


def test_add_priced_llm_is_idempotent_per_client():
    b = DollarBudgetTracker(daily_cap_usd=100.0, hourly_cap_usd=100.0)
    llm = _FakeLLM()
    b.add_priced_llm(llm, "claude-haiku-4-5-20251001")
    b.add_priced_llm(llm, "claude-haiku-4-5-20251001")  # same client → no double count
    llm.total_output_tokens = 1_000_000
    b.sync_from_llm()
    assert b.daily_spend == pytest.approx(4.00, abs=1e-6)  # Haiku output $4/M, counted once


# ── fast-LLM resolution ───────────────────────────────────────

class _Cfg:
    class llm:
        provider = "anthropic"
        model = "claude-sonnet-4-5-20250929"
        fast_model = ""


def test_resolve_fast_llm_returns_quality_when_no_fast_model():
    sentinel = object()
    assert _resolve_fast_llm(_Cfg(), sentinel) is sentinel


def test_resolve_fast_llm_returns_quality_when_equal():
    cfg = _Cfg()
    cfg.llm.fast_model = cfg.llm.model
    sentinel = object()
    assert _resolve_fast_llm(cfg, sentinel) is sentinel


def test_resolve_fast_llm_none_quality_stays_none():
    cfg = _Cfg()
    cfg.llm.fast_model = "claude-haiku-4-5-20251001"
    # quality_llm None (stub mode) → no fast client built
    assert _resolve_fast_llm(cfg, None) is None
