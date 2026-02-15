"""Identity unknown collection and matching.

After each run, collects unknowns from the identity resolver, finds
fuzzy match candidates via a MatchSuggester, and provides an API for
confirming matches (writes to AliasStore).

Mode-awareness (interactive/non-interactive) is decoupled into
ResolutionHandler implementations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ── Value Types (frozen) ──────────────────────────────────────────


class UnknownIdentifier(BaseModel):
    """An identifier that couldn't be resolved."""

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(min_length=1)
    source: str = Field(min_length=1)
    discovered_at: datetime


class MatchCandidate(BaseModel):
    """A candidate match for an unknown identifier."""

    model_config = ConfigDict(frozen=True)

    canonical_id: str = Field(min_length=1)
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_method: str = Field(min_length=1)


class MatchResult(BaseModel):
    """An unknown paired with its match candidates (possibly empty)."""

    model_config = ConfigDict(frozen=True)

    unknown: UnknownIdentifier
    candidates: tuple[MatchCandidate, ...] = ()


class ConfirmationResult(BaseModel):
    """A confirmed mapping from unknown to canonical identity."""

    model_config = ConfigDict(frozen=True)

    unknown_id: str = Field(min_length=1)
    confirmed_id: str = Field(min_length=1)
    confirmed_at: datetime
    source: str = ""


# ── Errors ────────────────────────────────────────────────────────


class IdentityResolutionError(Exception):
    """Base error for identity resolution."""

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.context = context


class StoreWriteError(IdentityResolutionError):
    """Failed to write to alias store."""

    def __init__(self, message: str, unknown_id: str, confirmed_id: str, **context):
        super().__init__(message, **context)
        self.unknown_id = unknown_id
        self.confirmed_id = confirmed_id


class MatchSuggestionError(IdentityResolutionError):
    """Failed to suggest matches for an identifier."""

    def __init__(self, message: str, identifier: str, **context):
        super().__init__(message, **context)
        self.identifier = identifier


# ── Protocols ─────────────────────────────────────────────────────


@runtime_checkable
class MatchSuggester(Protocol):
    async def suggest_match(
        self, identifier: str, limit: int = 5
    ) -> tuple[MatchCandidate, ...]: ...


@runtime_checkable
class ResolutionHandler(Protocol):
    async def handle_results(
        self, results: tuple[MatchResult, ...]
    ) -> tuple[ConfirmationResult, ...]: ...


@runtime_checkable
class AliasStoreProtocol(Protocol):
    async def has_alias(self, identifier: str) -> bool: ...
    async def write_alias(self, unknown_id: str, confirmed_id: str) -> bool: ...


# ── Collector ─────────────────────────────────────────────────────


class UnknownCollector:
    """Collects unknown identifiers and finds match candidates.

    Uses composition — all dependencies injected via constructor.
    """

    def __init__(
        self,
        match_suggester: MatchSuggester,
        alias_store: AliasStoreProtocol,
        resolution_handler: ResolutionHandler | None = None,
    ) -> None:
        self._suggester = match_suggester
        self._store = alias_store
        self._handler = resolution_handler

    async def collect_and_match(
        self,
        unknowns: frozenset[str],
        source: str,
    ) -> tuple[MatchResult, ...]:
        """Collect unknowns, pre-filter known aliases, find match candidates.

        Runs suggest_match() concurrently for all unknowns.
        """
        if not unknowns:
            return ()

        # Pre-filter: skip identifiers already confirmed in the store
        to_check: list[str] = []
        for identifier in sorted(unknowns):
            try:
                if not await self._store.has_alias(identifier):
                    to_check.append(identifier)
            except Exception as e:
                logger.warning("AliasStore.has_alias failed for '%s': %s", identifier, e)
                to_check.append(identifier)  # Include on store failure

        if not to_check:
            return ()

        now = datetime.now(timezone.utc)

        # Run suggest_match concurrently
        async def _suggest(ident: str) -> tuple[str, tuple[MatchCandidate, ...] | Exception]:
            try:
                candidates = await self._suggester.suggest_match(ident)
                # Filter self-matches
                filtered = tuple(
                    c for c in candidates if c.canonical_id != ident
                )
                # Sort by confidence descending
                filtered = tuple(
                    sorted(filtered, key=lambda c: c.confidence, reverse=True)
                )
                return ident, filtered
            except Exception as e:
                return ident, e

        results = await asyncio.gather(
            *[_suggest(ident) for ident in to_check]
        )

        match_results: list[MatchResult] = []
        errors: list[MatchSuggestionError] = []

        for ident, result in results:
            unknown = UnknownIdentifier(
                identifier=ident,
                source=source,
                discovered_at=now,
            )
            if isinstance(result, Exception):
                errors.append(MatchSuggestionError(
                    f"suggest_match failed for '{ident}': {result}",
                    identifier=ident,
                ))
                # Still include with empty candidates
                match_results.append(MatchResult(unknown=unknown, candidates=()))
            else:
                match_results.append(MatchResult(unknown=unknown, candidates=result))

        if errors and len(errors) == len(to_check):
            raise IdentityResolutionError(
                "All suggestion calls failed",
                failed_identifiers=[e.identifier for e in errors],
                errors=[str(e) for e in errors],
            )

        return tuple(match_results)

    async def confirm_match(
        self,
        unknown_id: str,
        confirmed_id: str,
        source: str = "",
    ) -> ConfirmationResult:
        """Confirm a single match by writing to the alias store."""
        if unknown_id == confirmed_id:
            raise IdentityResolutionError(
                "Cannot confirm a self-match: unknown_id equals confirmed_id",
                unknown_id=unknown_id,
                confirmed_id=confirmed_id,
            )

        try:
            await self._store.write_alias(unknown_id, confirmed_id)
        except Exception as e:
            raise StoreWriteError(
                f"Failed to write alias: {e}",
                unknown_id=unknown_id,
                confirmed_id=confirmed_id,
            ) from e

        return ConfirmationResult(
            unknown_id=unknown_id,
            confirmed_id=confirmed_id,
            confirmed_at=datetime.now(timezone.utc),
            source=source,
        )
