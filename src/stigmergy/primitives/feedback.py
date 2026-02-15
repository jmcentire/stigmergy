"""Feedback primitives: types, errors, and utilities for the feedback loop.

Findings are recorded at surfacing time with a unique feedback_id (fb-* prefix).
Users later rate findings (helpful, already_knew, wrong, irrelevant).
Records are immutable frozen Pydantic models persisted to JSONL files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

Rating = Literal["helpful", "already_knew", "wrong", "irrelevant"]
VALID_RATINGS = {"helpful", "already_knew", "wrong", "irrelevant"}

_FEEDBACK_ID_PATTERN = re.compile(r"^fb-[a-zA-Z0-9_-]{4,}$")


def mint_feedback_id() -> str:
    """Generate a unique feedback_id with 'fb-' prefix."""
    return f"fb-{uuid.uuid4().hex[:12]}"


# ── Errors ────────────────────────────────────────────────────────


class FeedbackError(Exception):
    """Base error for the feedback subsystem."""

    def __init__(self, message: str, feedback_id: str | None = None):
        super().__init__(message)
        self.feedback_id = feedback_id


class FeedbackNotFoundError(FeedbackError):
    """Raised when a feedback_id is not found in the store."""

    def __init__(self, feedback_id: str):
        super().__init__(
            f"No record found for feedback_id '{feedback_id}'",
            feedback_id=feedback_id,
        )


class DuplicateFeedbackError(FeedbackError):
    """Raised when recording a finding with a duplicate feedback_id."""

    def __init__(self, feedback_id: str):
        super().__init__(
            f"Finding with feedback_id '{feedback_id}' already exists",
            feedback_id=feedback_id,
        )


# ── Records ───────────────────────────────────────────────────────


class FindingRecord(BaseModel):
    """Immutable record of a surfaced finding."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    feedback_id: str
    signal_id: str
    context_id: str
    agent_id: str
    timestamp: str
    content_summary: str = Field(min_length=1, max_length=500)
    schema_version: int = 1

    @field_validator("feedback_id")
    @classmethod
    def _validate_feedback_id(cls, v: str) -> str:
        if not _FEEDBACK_ID_PATTERN.match(v):
            raise ValueError(f"Invalid feedback_id format: {v}")
        return v


class FeedbackRecord(BaseModel):
    """Immutable record of user feedback on a finding. Denormalized."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    feedback_id: str
    signal_id: str
    context_id: str
    agent_id: str
    timestamp: str
    content_summary: str = Field(min_length=1, max_length=500)
    schema_version: int = 1
    rating: Rating
    feedback_timestamp: str


class FeedbackSummary(BaseModel):
    """Aggregated feedback counts."""

    model_config = ConfigDict(frozen=True)

    total_findings: int = 0
    total_rated: int = 0
    helpful_count: int = 0
    already_knew_count: int = 0
    wrong_count: int = 0
    irrelevant_count: int = 0


# ── Store ─────────────────────────────────────────────────────────


class FeedbackStore:
    """Persistent feedback store backed by JSONL files.

    In-memory dict indexes for O(1) lookups. asyncio.Lock for write
    serialization. Defensive JSONL parsing (skip blank/malformed lines).
    """

    def __init__(
        self,
        data_dir: str = ".stigmergy",
        findings_filename: str = "findings.jsonl",
        feedback_filename: str = "feedback.jsonl",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._findings_path = self._data_dir / findings_filename
        self._feedback_path = self._data_dir / feedback_filename
        self._findings: dict[str, FindingRecord] = {}
        self._feedback: dict[str, FeedbackRecord] = {}
        self._rated_set: set[str] = set()
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load existing records from JSONL files into in-memory indexes."""
        if self._findings_path.exists():
            await self._load_file(
                self._findings_path, FindingRecord, self._findings
            )
        if self._feedback_path.exists():
            await self._load_file(
                self._feedback_path, FeedbackRecord, self._feedback
            )
            self._rated_set = set(self._feedback.keys())

    async def _load_file(
        self,
        path: Path,
        model_type: type,
        index: dict,
    ) -> None:
        """Load a JSONL file into an index dict. Skips malformed lines."""

        def _read():
            lines = []
            with open(path, "r") as f:
                for line in f:
                    lines.append(line)
            return lines

        lines = await asyncio.to_thread(_read)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                record = model_type.model_validate(data)
                index[record.feedback_id] = record
            except Exception as e:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", i + 1, path, e
                )

    async def record_finding(
        self,
        feedback_id: str,
        signal_id: str,
        context_id: str,
        agent_id: str,
        timestamp: str,
        content_summary: str,
    ) -> FindingRecord:
        """Record a new finding. Raises on duplicate or invalid ID."""
        if not _FEEDBACK_ID_PATTERN.match(feedback_id):
            raise FeedbackError("Invalid feedback_id format", feedback_id)
        if len(content_summary) > 500:
            raise FeedbackError(
                "content_summary exceeds maximum length of 500 characters",
                feedback_id,
            )
        if feedback_id in self._findings:
            raise DuplicateFeedbackError(feedback_id)

        record = FindingRecord(
            feedback_id=feedback_id,
            signal_id=signal_id,
            context_id=context_id,
            agent_id=agent_id,
            timestamp=timestamp,
            content_summary=content_summary,
        )

        async with self._lock:
            self._findings[feedback_id] = record
            await self._append_jsonl(self._findings_path, record)

        return record

    async def record_feedback(
        self,
        feedback_id: str,
        rating: Rating,
    ) -> FeedbackRecord:
        """Record user feedback (rating) for an existing finding."""
        if rating not in VALID_RATINGS:
            raise FeedbackError(
                "Invalid rating value. Must be one of: "
                "helpful, already_knew, wrong, irrelevant"
            )
        if feedback_id not in self._findings:
            raise FeedbackNotFoundError(feedback_id)

        finding = self._findings[feedback_id]
        now = datetime.now(timezone.utc).isoformat()

        record = FeedbackRecord(
            feedback_id=feedback_id,
            signal_id=finding.signal_id,
            context_id=finding.context_id,
            agent_id=finding.agent_id,
            timestamp=finding.timestamp,
            content_summary=finding.content_summary,
            rating=rating,
            feedback_timestamp=now,
        )

        async with self._lock:
            self._feedback[feedback_id] = record
            self._rated_set.add(feedback_id)
            await self._append_jsonl(self._feedback_path, record)

        return record

    async def get_finding(self, feedback_id: str) -> FindingRecord:
        """Retrieve a finding by feedback_id. O(1)."""
        if feedback_id not in self._findings:
            raise FeedbackNotFoundError(feedback_id)
        return self._findings[feedback_id]

    async def get_feedback(self, feedback_id: str) -> FeedbackRecord:
        """Retrieve the latest feedback for a feedback_id. O(1)."""
        if feedback_id not in self._feedback:
            raise FeedbackNotFoundError(feedback_id)
        return self._feedback[feedback_id]

    async def list_findings(
        self,
        agent_id: str | None = None,
        rated_only: bool = False,
        unrated_only: bool = False,
        limit: int = 100,
    ) -> list[FindingRecord]:
        """List findings with optional filtering."""
        if rated_only and unrated_only:
            raise FeedbackError("rated_only and unrated_only are mutually exclusive")

        results = list(self._findings.values())

        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]
        if rated_only:
            results = [r for r in results if r.feedback_id in self._rated_set]
        if unrated_only:
            results = [r for r in results if r.feedback_id not in self._rated_set]

        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp, reverse=True)

        if limit > 0:
            results = results[:limit]

        return results

    async def list_feedback(
        self,
        agent_id: str | None = None,
        rating: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackRecord]:
        """List feedback records with optional filtering."""
        results = list(self._feedback.values())

        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]
        if rating:
            results = [r for r in results if r.rating == rating]
        if since:
            results = [r for r in results if r.feedback_timestamp >= since]

        results.sort(key=lambda r: r.feedback_timestamp, reverse=True)

        if limit > 0:
            results = results[:limit]

        return results

    async def get_summary(self, agent_id: str | None = None) -> FeedbackSummary:
        """Compute aggregated feedback summary."""
        findings = list(self._findings.values())
        if agent_id:
            findings = [f for f in findings if f.agent_id == agent_id]

        feedback = list(self._feedback.values())
        if agent_id:
            feedback = [f for f in feedback if f.agent_id == agent_id]

        counts = {"helpful": 0, "already_knew": 0, "wrong": 0, "irrelevant": 0}
        for fb in feedback:
            if fb.rating in counts:
                counts[fb.rating] += 1

        return FeedbackSummary(
            total_findings=len(findings),
            total_rated=len(feedback),
            helpful_count=counts["helpful"],
            already_knew_count=counts["already_knew"],
            wrong_count=counts["wrong"],
            irrelevant_count=counts["irrelevant"],
        )

    async def _append_jsonl(self, path: Path, record: BaseModel) -> None:
        """Append a record to a JSONL file. Auto-creates directory."""

        def _write():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(record.model_dump_json() + "\n")

        await asyncio.to_thread(_write)
