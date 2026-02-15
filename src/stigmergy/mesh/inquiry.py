"""Inquiry engine: decomposes accepted signals into dimensions.

When a worker accepts a signal, the inquiry engine decomposes it into
five dimensions (people, projects, tickets, keywords, sentiment) and
queries the worker's context knowledge base for relevant connections.

Two extraction paths:
  - LLM: uses structured extraction to pull entities from signal content
  - Heuristic: uses keyword overlap and embedding similarity as fallback

Never raises — all errors are captured in the result.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

from stigmergy.primitives.inquiry import (
    InquiryDimension,
    InquiryResult,
    create_inquiry_result,
)

if TYPE_CHECKING:
    from stigmergy.primitives.context import Context
    from stigmergy.primitives.signal import Signal
    from stigmergy.services.llm import LLMService

logger = logging.getLogger(__name__)


class ExtractionMethod(StrEnum):
    LLM = "llm"
    HEURISTIC = "heuristic"


class MatchMethod(StrEnum):
    LLM_EXTRACTION = "llm_extraction"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    KEYWORD_OVERLAP = "keyword_overlap"
    COMBINED = "combined"
    NONE = "none"


@dataclass(frozen=True)
class DimensionMatch:
    """Matches found for a single inquiry dimension."""

    matched_terms: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()
    method: MatchMethod = MatchMethod.NONE


@dataclass(frozen=True)
class SentimentResult:
    """Sentiment analysis of signal content."""

    polarity: float = 0.0  # [-1.0, 1.0]
    magnitude: float = 0.0  # [0.0, 1.0]
    label: str = "neutral"


@dataclass
class EngineInquiryResult:
    """Full inquiry result from the engine layer."""

    signal_id: str
    worker_id: str
    people: DimensionMatch = field(default_factory=DimensionMatch)
    projects: DimensionMatch = field(default_factory=DimensionMatch)
    tickets: DimensionMatch = field(default_factory=DimensionMatch)
    keywords: DimensionMatch = field(default_factory=DimensionMatch)
    sentiment: SentimentResult = field(default_factory=SentimentResult)
    extraction_method: ExtractionMethod = ExtractionMethod.HEURISTIC
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z"
    )
    errors: tuple[str, ...] = ()


# ── Heuristic patterns ────────────────────────────────────────────

# People patterns: @mentions, "Name Name" capitalized sequences
_PEOPLE_PATTERN = re.compile(
    r"@[\w.-]+|(?<!\w)[A-Z][a-z]+ [A-Z][a-z]+(?!\w)"
)

# Project/repo patterns: org/repo, project-name references
_PROJECT_PATTERN = re.compile(
    r"[\w-]+/[\w.-]+|(?:project|repo|service|module)[\s:]+[\w.-]+"
    , re.IGNORECASE
)

# Ticket patterns: PROJ-123, #123, GH-123
_TICKET_PATTERN = re.compile(
    r"[A-Z]{2,}-\d+|#\d{2,}|GH-\d+"
)


def _extract_dimension_heuristic(
    content: str,
    dimension: InquiryDimension,
    context_terms: set[str],
) -> DimensionMatch:
    """Extract dimension matches using heuristic patterns + context overlap."""
    if not content.strip():
        return DimensionMatch(method=MatchMethod.KEYWORD_OVERLAP)

    # Pattern-based extraction
    patterns = {
        InquiryDimension.PEOPLE: _PEOPLE_PATTERN,
        InquiryDimension.PROJECTS: _PROJECT_PATTERN,
        InquiryDimension.TICKETS: _TICKET_PATTERN,
    }

    extracted: set[str] = set()

    if dimension in patterns:
        matches = patterns[dimension].findall(content)
        extracted.update(m.strip() for m in matches if m.strip())

    if dimension == InquiryDimension.KEYWORDS:
        # Extract significant words (>3 chars, lowercase, not stopwords)
        words = set(content.lower().split())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "have",
            "been", "will", "would", "could", "should", "about", "into",
            "when", "what", "which", "where", "there", "their", "they",
            "more", "some", "also", "just", "only", "very", "much",
        }
        extracted = {w for w in words if len(w) > 3 and w not in stopwords}

    # Score against context terms
    matched_terms = []
    scores = []
    for term in sorted(extracted):
        # Check overlap with context knowledge base
        term_lower = term.lower()
        if term_lower in context_terms or term in context_terms:
            matched_terms.append(term)
            scores.append(0.8)  # High score for context match
        elif any(term_lower in ct.lower() or ct.lower() in term_lower
                 for ct in context_terms if len(ct) > 3):
            matched_terms.append(term)
            scores.append(0.5)  # Partial match

    return DimensionMatch(
        matched_terms=tuple(matched_terms),
        scores=tuple(scores),
        method=MatchMethod.KEYWORD_OVERLAP,
    )


def _analyze_sentiment_heuristic(content: str) -> SentimentResult:
    """Simple keyword-based sentiment analysis."""
    if not content.strip():
        return SentimentResult()

    content_lower = content.lower()

    positive = {
        "good", "great", "excellent", "nice", "fixed", "resolved",
        "improved", "success", "working", "done", "merged", "approved",
        "completed", "shipped", "deployed", "ready", "happy", "love",
    }
    negative = {
        "bad", "error", "fail", "bug", "broken", "crash", "issue",
        "problem", "wrong", "blocked", "stuck", "slow", "missing",
        "critical", "urgent", "regression", "revert", "hack",
    }

    words = set(content_lower.split())
    pos_count = len(words & positive)
    neg_count = len(words & negative)
    total = pos_count + neg_count

    if total == 0:
        return SentimentResult(polarity=0.0, magnitude=0.0, label="neutral")

    polarity = (pos_count - neg_count) / total
    magnitude = min(1.0, total / 5.0)  # Scale by word count

    if polarity > 0.2:
        label = "positive"
    elif polarity < -0.2:
        label = "negative"
    else:
        label = "mixed" if magnitude > 0.3 else "neutral"

    return SentimentResult(
        polarity=max(-1.0, min(1.0, polarity)),
        magnitude=max(0.0, min(1.0, magnitude)),
        label=label,
    )


async def inquire(
    signal: Signal,
    context: Context,
    llm: LLMService | None = None,
) -> EngineInquiryResult:
    """Decompose signal into dimensions and query context for connections.

    Never raises — all errors captured in result.errors.
    """
    errors: list[str] = []
    extraction_method = ExtractionMethod.HEURISTIC
    worker_id = str(context.id)
    signal_id = str(signal.id)

    # Handle empty content
    if not signal.content or not signal.content.strip():
        return EngineInquiryResult(
            signal_id=signal_id,
            worker_id=worker_id,
            extraction_method=ExtractionMethod.HEURISTIC,
        )

    context_terms = context.terms if context.terms else set()

    # Try LLM extraction first
    llm_dimensions: dict[str, list[str]] | None = None
    llm_sentiment: SentimentResult | None = None

    if llm is not None:
        try:
            from pydantic import BaseModel, Field as PydField

            class DimensionExtraction(BaseModel):
                people: list[str] = PydField(default_factory=list)
                projects: list[str] = PydField(default_factory=list)
                tickets: list[str] = PydField(default_factory=list)
                keywords: list[str] = PydField(default_factory=list)
                sentiment_polarity: float = 0.0
                sentiment_magnitude: float = 0.0
                sentiment_label: str = "neutral"

            result = await llm.extract(
                DimensionExtraction,
                f"Extract structured dimensions from this signal:\n\n{signal.content}",
                "You are an entity extraction system. Extract people, projects, "
                "tickets, and keywords from the signal content. Also analyze sentiment.",
                max_tokens=500,
            )
            llm_dimensions = {
                "people": result.people,
                "projects": result.projects,
                "tickets": result.tickets,
                "keywords": result.keywords,
            }
            llm_sentiment = SentimentResult(
                polarity=max(-1.0, min(1.0, result.sentiment_polarity)),
                magnitude=max(0.0, min(1.0, result.sentiment_magnitude)),
                label=result.sentiment_label or "neutral",
            )
            extraction_method = ExtractionMethod.LLM
        except Exception as e:
            errors.append(f"LLM extraction failed: {e}")
            logger.debug("LLM extraction failed, falling back to heuristic", exc_info=True)

    # Heuristic extraction for each dimension
    dimension_results: dict[str, DimensionMatch] = {}
    dimensions_to_check = [
        InquiryDimension.PEOPLE,
        InquiryDimension.PROJECTS,
        InquiryDimension.TICKETS,
        InquiryDimension.KEYWORDS,
    ]

    for dim in dimensions_to_check:
        try:
            if llm_dimensions and dim.value in llm_dimensions:
                # Score LLM-extracted terms against context
                llm_terms = llm_dimensions[dim.value]
                matched = []
                match_scores = []
                for term in llm_terms:
                    term_lower = term.lower()
                    if term_lower in context_terms or term in context_terms:
                        matched.append(term)
                        match_scores.append(0.9)
                    elif any(
                        term_lower in ct.lower() or ct.lower() in term_lower
                        for ct in context_terms if len(ct) > 3
                    ):
                        matched.append(term)
                        match_scores.append(0.6)
                    else:
                        # LLM found it but not in context — still valid
                        matched.append(term)
                        match_scores.append(0.4)

                dimension_results[dim.value] = DimensionMatch(
                    matched_terms=tuple(matched),
                    scores=tuple(match_scores),
                    method=MatchMethod.LLM_EXTRACTION,
                )
            else:
                dimension_results[dim.value] = _extract_dimension_heuristic(
                    signal.content, dim, context_terms
                )
        except Exception as e:
            errors.append(f"Dimension query failed for {dim.value}: {e}")
            dimension_results[dim.value] = DimensionMatch()

    # Sentiment
    try:
        sentiment = llm_sentiment or _analyze_sentiment_heuristic(signal.content)
    except Exception as e:
        errors.append(f"Sentiment analysis failed: {e}")
        sentiment = SentimentResult()

    return EngineInquiryResult(
        signal_id=signal_id,
        worker_id=worker_id,
        people=dimension_results.get("people", DimensionMatch()),
        projects=dimension_results.get("projects", DimensionMatch()),
        tickets=dimension_results.get("tickets", DimensionMatch()),
        keywords=dimension_results.get("keywords", DimensionMatch()),
        sentiment=sentiment,
        extraction_method=extraction_method,
        errors=tuple(errors),
    )


def engine_result_to_primitive(
    engine_result: EngineInquiryResult,
    context_id: str,
) -> InquiryResult:
    """Convert engine result to the primitive InquiryResult for the pipeline."""
    dimensions: dict[str, list[str]] = {}
    confidence_scores: dict[str, float] = {}

    for dim_name, dim_match in [
        ("people", engine_result.people),
        ("projects", engine_result.projects),
        ("tickets", engine_result.tickets),
        ("keywords", engine_result.keywords),
    ]:
        if dim_match.matched_terms:
            dimensions[dim_name] = list(dim_match.matched_terms)
            avg_score = (
                sum(dim_match.scores) / len(dim_match.scores)
                if dim_match.scores
                else 0.0
            )
            confidence_scores[dim_name] = avg_score

    # Add sentiment as a dimension if it's non-neutral
    if engine_result.sentiment.magnitude > 0.1:
        dimensions["sentiment"] = [engine_result.sentiment.label]
        confidence_scores["sentiment"] = engine_result.sentiment.magnitude

    return create_inquiry_result(
        signal_id=engine_result.signal_id,
        context_id=context_id,
        dimensions=dimensions,
        confidence_scores=confidence_scores,
    )
