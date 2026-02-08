"""Signal: the atomic unit of information entering the system.

Signals are immutable. Once ingested, a signal is never modified.
Derived state lives in contexts.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# ── Term extraction ───────────────────────────────────────────
#
# Terms drive bloom filter overlap, familiarity scoring, worker
# labels, and domain inference. Quality matters: junk terms
# (HTML, URLs, stopwords) poison bloom filters and prevent
# worker differentiation.

# Characters to strip from token edges. Covers punctuation,
# markdown, HTML angle brackets, backticks, and path separators.
_STRIP_CHARS = ".,!?;:()[]{}\"'*`<>#~^|\\=_"

# Common English stopwords that add no domain signal.
# Kept small — only the most frequent. 3-letter domain keywords
# like "bug", "fix", "api" are intentionally NOT included.
_STOPWORDS = frozenset({
    # 3-letter
    "are", "was", "has", "had", "the", "and", "for", "but", "not",
    "you", "all", "can", "her", "his", "how", "its", "may", "who",
    "did", "got", "him", "she", "our", "out", "own", "too", "yet",
    "any", "few", "now", "say", "way", "via", "per", "use", "see",
    "new", "set", "get", "add", "let", "run", "try",
    # 4-letter
    "also", "been", "does", "each", "even", "from", "have", "here",
    "into", "just", "like", "made", "make", "many", "more", "most",
    "much", "must", "need", "next", "only", "over", "some", "such",
    "take", "than", "that", "them", "then", "they", "this", "very",
    "want", "well", "were", "what", "when", "will", "with", "your",
    "form", "work", "used", "type", "same", "both", "done", "able",
    "keep", "sure", "seem", "show", "look", "give", "good", "part",
    "last", "long", "back", "turn", "help", "left", "case", "call",
    # 5+ letter
    "about", "after", "being", "below", "could", "doing", "every",
    "first", "going", "might", "never", "other", "shall", "since",
    "still", "their", "there", "these", "thing", "those", "under",
    "until", "using", "where", "which", "while", "would", "should",
    "through", "between", "during", "before",
    # Generic words that appear everywhere but carry no domain signal
    "number", "based", "added", "updated", "change", "changed",
    "right", "given", "found", "think", "because", "however",
    "already", "another", "without", "within", "around", "along",
    "either", "rather", "though", "actually", "currently",
    "instead", "whether", "against", "toward", "toward",
    "possible", "potential", "suggested", "related", "expected",
    "required", "included", "following", "according", "different",
    "available", "specific", "several", "example", "general",
    "likely", "please", "ensure", "allow", "allows", "supported",
    "option", "options", "appropriate", "consider", "certain",
    "provide", "provides", "seems", "appears", "become",
})

# Matches version strings (1.2.3), pure integers, and decimal numbers.
_NUMERIC_RE = re.compile(r"^\d[\d.]*$")

# HTML tags: <tag>, </tag>, <tag attr="val">, <!-- comment -->
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Markdown horizontal rules and heading markers (standalone)
_MD_RULE_RE = re.compile(r"^[-#]{2,}$")


def extract_terms(text: str) -> set[str]:
    """Extract domain-relevant terms from raw text.

    Filters out:
    - HTML tags (stripped before tokenization)
    - Tokens shorter than 3 characters
    - Common English stopwords
    - URLs (containing ://)
    - Version numbers and pure numerics (5.102.1, 372)
    - Markdown rules (---, ###)
    - Tokens with no alphanumeric characters after stripping
    """
    # Strip HTML tags before tokenization so <summary>word</summary>
    # becomes " word " rather than "summary>word</summary".
    text = _HTML_TAG_RE.sub(" ", text)

    result: set[str] = set()
    for raw in text.split():
        t = raw.lower().strip(_STRIP_CHARS)
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        if "://" in t:
            continue
        if _NUMERIC_RE.match(t):
            continue
        if _MD_RULE_RE.match(t):
            continue
        if not any(c.isalnum() for c in t):
            continue
        result.add(t)
    return result


class SignalSource:
    """Well-known signal sources.

    These are starting points, not constraints. New adapters may use any
    self-describing string as a source — "jira", "figma", "datadog",
    "pagerduty", whatever. The system handles unknown sources naturally.
    """

    SLACK = "slack"
    LINEAR = "linear"
    GITHUB = "github"
    DOCS = "docs"
    EMAIL = "email"
    DEPLOY = "deploy"
    CADENCE = "mesh:cadence"


class Signal(BaseModel):
    """Immutable signal from any connected source."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    content: str
    source: str
    channel: str
    author: str
    timestamp: datetime
    embeddings: dict[str, list[float]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def terms(self) -> set[str]:
        """Extract domain-relevant terms from content."""
        return extract_terms(self.content)
