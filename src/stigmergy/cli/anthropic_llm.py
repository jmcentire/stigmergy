"""Real LLM service via the anthropic SDK.

Implements the LLMService protocol (complete, classify, extract).
Uses structured outputs (tool use) to enforce schema contracts
at generation time. The schema shapes the model's output — it's
a guardrail, not a validator.

Falls back gracefully if the anthropic package is not installed.
Uses AsyncAnthropic to avoid blocking the event loop.
"""

from __future__ import annotations

import json
import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AnthropicLLMService:
    """LLM service using the Anthropic API (async)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic LLM provider. "
                "Install with: pip install -e '.[cli]'"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it with: export ANTHROPIC_API_KEY=your-key"
            )

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            max_retries=5,
            timeout=120.0,
        )
        self._model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def complete(self, prompt: str, context: str, max_tokens: int) -> str:
        """Send a completion request to the Anthropic API."""
        system_msg = context if context else "You are a signal analysis agent."
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        self._total_input_tokens += message.usage.input_tokens
        self._total_output_tokens += message.usage.output_tokens
        return message.content[0].text

    async def classify(self, content: str, categories: list[str]) -> dict[str, float]:
        """Classify content into categories using the LLM."""
        if not categories:
            return {}

        cat_list = ", ".join(categories)
        prompt = (
            f"Classify the following content into these categories: {cat_list}\n\n"
            f"Content: {content}\n\n"
            f"Respond with a JSON object mapping each category to a confidence score "
            f"between 0.0 and 1.0. Only output the JSON, nothing else."
        )

        response = await self.complete(prompt, "", max_tokens=200)

        try:
            scores = json.loads(response)
            result = {}
            for cat in categories:
                result[cat] = float(scores.get(cat, 0.0))
            return result
        except (json.JSONDecodeError, ValueError):
            n = len(categories)
            return {cat: 1.0 / n for cat in categories}

    async def extract(
        self,
        schema: type[T],
        prompt: str,
        system: str,
        max_tokens: int = 1024,
    ) -> T:
        """Extract structured data using tool use to enforce the schema.

        The Pydantic model's JSON schema becomes a tool definition.
        tool_choice forces the model to call that tool, which means
        the output MUST conform to the schema. The schema is the
        guardrail — it shapes generation, not validates after.
        """
        tool_name = schema.__name__
        tool_schema = schema.model_json_schema()

        # Remove title (Pydantic adds it, Anthropic doesn't need it)
        # Keep $defs — needed for nested schemas (e.g. CorrelationBatch)
        tool_schema.pop("title", None)

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "name": tool_name,
                "description": schema.__doc__ or f"Extract {tool_name}",
                "input_schema": tool_schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
        )

        self._total_input_tokens += message.usage.input_tokens
        self._total_output_tokens += message.usage.output_tokens

        # The model is forced to produce a tool call matching our schema
        for block in message.content:
            if block.type == "tool_use" and block.name == tool_name:
                return schema.model_validate(block.input)

        # Should never happen with tool_choice forcing the call,
        # but fall back to stub if it does
        from stigmergy.services.llm import _stub_instance
        return _stub_instance(schema)

    async def probe(self) -> bool:
        """Verify LLM is reachable with a minimal request.

        Returns True if the API responds, raises on failure.
        Used as a preflight check before committing to a full run.
        """
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        self._total_input_tokens += message.usage.input_tokens
        self._total_output_tokens += message.usage.output_tokens
        return True

    async def close(self) -> None:
        """Close the underlying HTTP client so asyncio.run() can exit cleanly."""
        await self._client.close()

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens
