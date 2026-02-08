"""Constraint evaluation: kill / redact / pass.

Constraint agents are configured, not coded. Adding a new kill pattern
is a config change, not a code change.

The constraint evaluation is the simplest, most boring code in the system.
No cleverness. No reasoning. Regex, keyword lists, simple classifiers.

Optimized with Aho-Corasick pre-filter: extracts literal keywords from
regex patterns and builds an automaton. If no keywords match the content,
skip all regex evaluation entirely (fast PASS path). Only fall back to
regex for content that hits a keyword.

The principle: the system can know. The system cannot say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from stigmergy.structures.trie import AhoCorasick


class ConstraintAction:
    """Well-known constraint actions.

    The constraint layer is deliberately simple: kill, redact, or pass.
    Future constraint agents may propose additional actions like
    "quarantine", "encrypt", or "escalate".
    """

    KILL = "kill"
    REDACT = "redact"
    PASS = "pass"


@dataclass
class ConstraintResult:
    action: str
    category: str | None = None
    content: str | None = None  # redacted content, or None if killed


@dataclass
class _KillPattern:
    pattern: re.Pattern[str]
    category: str
    keywords: list[str] = field(default_factory=list)  # extracted literal keywords


@dataclass
class _RedactPattern:
    pattern: re.Pattern[str]
    replacement: str
    category: str
    keywords: list[str] = field(default_factory=list)


def _extract_keywords(regex_str: str, min_length: int = 2) -> list[str]:
    """Extract literal keyword substrings from a regex pattern.

    Strips regex syntax (character classes, escape sequences, quantifiers)
    then finds runs of 2+ alphabetic characters from the remaining literals.

    These serve as cheap pre-filter candidates for the Aho-Corasick automaton.
    Patterns with no extractable keywords fall into the "always check" bucket.
    """
    # Strip character classes [...]
    cleaned = re.sub(r'\[.*?\]', '', regex_str)
    # Strip escape sequences \X
    cleaned = re.sub(r'\\[a-zA-Z]', '', cleaned)
    # Strip quantifiers {n,m}
    cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
    # Extract alpha runs from remaining literal content
    pattern = r'[a-zA-Z]{' + str(min_length) + r',}'
    return [m.lower() for m in re.findall(pattern, cleaned)]


class ConstraintEvaluator:
    """Config-driven output constraint evaluation.

    Uses Aho-Corasick as a pre-filter:
    1. Run automaton over content (O(n) single pass)
    2. If no keywords from kill/redact patterns match → PASS immediately
    3. If kill keywords match → run only the kill regexes whose keywords hit
    4. If redact keywords match → run only matching redact regexes

    For clean content (most messages), this is a single-pass O(n) scan
    instead of iterating all regex patterns.
    """

    def __init__(self, kill_patterns: list[_KillPattern] | None = None,
                 redact_patterns: list[_RedactPattern] | None = None) -> None:
        self._kill_patterns = kill_patterns or []
        self._redact_patterns = redact_patterns or []

        # Separate patterns into those with keywords (can be pre-filtered)
        # and those without (must always be checked via regex)
        self._kill_always: list[_KillPattern] = []     # no keywords — always run regex
        self._kill_indexed: list[_KillPattern] = []     # has keywords — only run if keyword hits
        self._redact_always: list[_RedactPattern] = []
        self._redact_indexed: list[_RedactPattern] = []

        # Build Aho-Corasick pre-filter
        self._automaton = AhoCorasick()
        self._kill_by_keyword: dict[str, list[_KillPattern]] = {}
        self._redact_by_keyword: dict[str, list[_RedactPattern]] = {}

        for kp in self._kill_patterns:
            if kp.keywords:
                self._kill_indexed.append(kp)
                for kw in kp.keywords:
                    self._automaton.add_pattern(kw, f"kill:{kp.category}")
                    self._kill_by_keyword.setdefault(kw, []).append(kp)
            else:
                self._kill_always.append(kp)

        for rp in self._redact_patterns:
            if rp.keywords:
                self._redact_indexed.append(rp)
                for kw in rp.keywords:
                    self._automaton.add_pattern(kw, f"redact:{rp.category}")
                    self._redact_by_keyword.setdefault(kw, []).append(rp)
            else:
                self._redact_always.append(rp)

        if self._automaton.pattern_count > 0:
            self._automaton.build()
            self._has_prefilter = True
        else:
            self._has_prefilter = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> ConstraintEvaluator:
        """Load constraint patterns from YAML config."""
        path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        kill_patterns = []
        for entry in config.get("kill_patterns", []):
            pattern_str = entry["pattern"]
            kill_patterns.append(_KillPattern(
                pattern=re.compile(pattern_str, re.IGNORECASE),
                category=entry["category"],
                keywords=_extract_keywords(pattern_str),
            ))

        redact_patterns = []
        for entry in config.get("redact_patterns", []):
            pattern_str = entry["pattern"]
            redact_patterns.append(_RedactPattern(
                pattern=re.compile(pattern_str, re.IGNORECASE),
                replacement=entry.get("replacement", "[REDACTED]"),
                category=entry["category"],
                keywords=_extract_keywords(pattern_str),
            ))

        return cls(kill_patterns=kill_patterns, redact_patterns=redact_patterns)

    def evaluate(self, content: str) -> ConstraintResult:
        """Evaluate content against constraint patterns.

        Two-tier approach:
        1. ALWAYS check patterns with no extractable keywords (numeric regexes etc.)
        2. Use Aho-Corasick to pre-filter keyword-bearing patterns

        Kill is checked first across both tiers. Then redaction.
        """
        # Tier 1: Always-check kill patterns (no keywords, can't pre-filter)
        for kp in self._kill_always:
            if kp.pattern.search(content):
                return ConstraintResult(action=ConstraintAction.KILL, category=kp.category)

        # Tier 2: Pre-filtered patterns (only check if keywords match)
        matched_keywords: set[str] = set()
        if self._has_prefilter:
            matches = self._automaton.search(content)
            matched_keywords = {pattern for _, pattern, _ in matches}

            kill_candidates: set[int] = set()
            for kw in matched_keywords:
                for kp in self._kill_by_keyword.get(kw, []):
                    kill_candidates.add(id(kp))

            for kp in self._kill_indexed:
                if id(kp) in kill_candidates:
                    if kp.pattern.search(content):
                        return ConstraintResult(action=ConstraintAction.KILL, category=kp.category)

        # Redaction: always-check first, then pre-filtered
        redacted = content
        any_redacted = False
        categories: list[str] = []

        for rp in self._redact_always:
            new_content = rp.pattern.sub(rp.replacement, redacted)
            if new_content != redacted:
                any_redacted = True
                categories.append(rp.category)
                redacted = new_content

        if self._has_prefilter and matched_keywords:
            redact_candidates: set[int] = set()
            for kw in matched_keywords:
                for rp in self._redact_by_keyword.get(kw, []):
                    redact_candidates.add(id(rp))

            for rp in self._redact_indexed:
                if id(rp) in redact_candidates:
                    new_content = rp.pattern.sub(rp.replacement, redacted)
                    if new_content != redacted:
                        any_redacted = True
                        categories.append(rp.category)
                        redacted = new_content

        if any_redacted:
            return ConstraintResult(
                action=ConstraintAction.REDACT,
                category=", ".join(categories),
                content=redacted,
            )

        return ConstraintResult(action=ConstraintAction.PASS, content=content)
