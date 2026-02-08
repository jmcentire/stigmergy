"""LLM service: interface + stub.

Every LLM reasoning call goes through this interface.
The `extract` method uses structured outputs (tool use) to enforce
schema contracts at generation time — the schema shapes the output,
not validates it after the fact.

Stub returns deterministic mock responses for testing.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMService(Protocol):
    async def complete(self, prompt: str, context: str, max_tokens: int) -> str: ...

    async def classify(self, content: str, categories: list[str]) -> dict[str, float]: ...

    async def extract(
        self,
        schema: type[T],
        prompt: str,
        system: str,
        max_tokens: int = 1024,
    ) -> T:
        """Extract structured data shaped by a Pydantic schema.

        Uses tool use to enforce the schema at generation time.
        The schema IS the guardrail — it constrains what the model
        can output, not what we check afterward.
        """
        ...


class StubLLMService:
    """Returns deterministic stub responses. No real LLM calls."""

    async def complete(self, prompt: str, context: str, max_tokens: int) -> str:
        return f"[stub response | prompt_len={len(prompt)} context_len={len(context)}]"

    async def classify(self, content: str, categories: list[str]) -> dict[str, float]:
        n = len(categories)
        if n == 0:
            return {}
        # Slight bias toward first category for determinism
        weights = {}
        for i, cat in enumerate(categories):
            weights[cat] = (n - i) / sum(range(1, n + 1))
        return weights

    async def extract(
        self,
        schema: type[T],
        prompt: str,
        system: str,
        max_tokens: int = 1024,
    ) -> T:
        """Return a stub instance with default values."""
        # Build minimal valid instance from schema defaults
        return _stub_instance(schema)


def _stub_instance(schema: type[T]) -> T:
    """Construct a minimal valid instance of a Pydantic model for stub mode."""
    from stigmergy.primitives.schemas import (
        CorrelationBatch,
        CorrelationInsight,
        SignalFacts,
        SurfaceDecision,
        WorkerProfile,
    )
    from stigmergy.policy.engine import PolicyEvaluation

    if schema is SignalFacts:
        return schema(  # type: ignore[return-value]
            concepts=["stub-concept"],
            domain="engineering",
            summary="[stub] signal processed in heuristic mode",
        )
    if schema is SurfaceDecision:
        return schema(  # type: ignore[return-value]
            action="store",
            confidence=0.5,
            reasoning="[stub] heuristic fallback",
        )
    if schema is CorrelationInsight:
        return schema(  # type: ignore[return-value]
            pattern_type="trend",
            summary="[stub] no LLM available",
            severity="low",
        )
    if schema is CorrelationBatch:
        return schema(insights=[])  # type: ignore[return-value]
    if schema is WorkerProfile:
        return schema(  # type: ignore[return-value]
            specialization="general",
            domains=["engineering"],
            strength="[stub] no LLM available",
        )
    if schema is PolicyEvaluation:
        return schema(interventions=[], reasoning="[stub] no LLM available")  # type: ignore[return-value]
    # Generic fallback — try to construct with no args
    return schema()  # type: ignore[return-value]
