"""Trie and Aho-Corasick automaton.

Trie: fast prefix matching for channel/source routing to candidate contexts.
Aho-Corasick: multi-pattern string matching in O(n + m) for constraint
evaluation pre-filtering.

Together, these replace linear scans in the two hottest paths:
signal routing and output constraint checking.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class _TrieNode:
    __slots__ = ("children", "values", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.values: list[Any] = []
        self.is_terminal: bool = False


class Trie:
    """Prefix trie for fast string lookups.

    Supports exact match, prefix search, and multi-value storage per key.
    Used for channel -> context routing (e.g., "#engineering" -> [ctx_1, ctx_5]).

    Performance:
    - insert: O(|key|)
    - search: O(|key|)
    - starts_with: O(|prefix| + |results|)
    """

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    def insert(self, key: str, value: Any = None) -> None:
        """Insert a key with optional associated value."""
        node = self._root
        for ch in key:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        if not node.is_terminal:
            self._size += 1
        node.is_terminal = True
        if value is not None:
            node.values.append(value)

    def search(self, key: str) -> list[Any] | None:
        """Exact match lookup. Returns values or None."""
        node = self._find_node(key)
        if node is None or not node.is_terminal:
            return None
        return node.values

    def __contains__(self, key: str) -> bool:
        node = self._find_node(key)
        return node is not None and node.is_terminal

    def starts_with(self, prefix: str) -> list[tuple[str, list[Any]]]:
        """Find all keys with given prefix. Returns list of (key, values)."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results: list[tuple[str, list[Any]]] = []
        self._collect(node, prefix, results)
        return results

    def longest_prefix(self, text: str) -> tuple[str, list[Any]] | None:
        """Find the longest key that is a prefix of text."""
        node = self._root
        last_match: tuple[str, list[Any]] | None = None
        prefix = []
        for ch in text:
            if ch not in node.children:
                break
            node = node.children[ch]
            prefix.append(ch)
            if node.is_terminal:
                last_match = ("".join(prefix), node.values)
        return last_match

    def _find_node(self, key: str) -> _TrieNode | None:
        node = self._root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node: _TrieNode, prefix: str,
                 results: list[tuple[str, list[Any]]]) -> None:
        if node.is_terminal:
            results.append((prefix, node.values))
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, results)

    def __len__(self) -> int:
        return self._size


# --- Aho-Corasick automaton ---

class _ACNode:
    __slots__ = ("children", "fail", "output", "depth")

    def __init__(self) -> None:
        self.children: dict[str, _ACNode] = {}
        self.fail: _ACNode | None = None
        self.output: list[tuple[str, str]] = []  # (pattern, category)
        self.depth: int = 0


class AhoCorasick:
    """Aho-Corasick automaton for multi-pattern string matching.

    Searches for all occurrences of all patterns in a text in O(n + m)
    where n = text length and m = total match count.

    Used as a pre-filter for constraint evaluation: extract literal
    keywords from regex patterns, build automaton, and skip regex
    evaluation entirely when no keywords match.

    Performance:
    - build: O(sum of pattern lengths)
    - search: O(text length + number of matches)
    """

    __slots__ = ("_root", "_built", "_pattern_count")

    def __init__(self) -> None:
        self._root = _ACNode()
        self._built = False
        self._pattern_count = 0

    def add_pattern(self, pattern: str, category: str = "") -> None:
        """Add a pattern to match against. Must call build() after adding all patterns."""
        self._built = False
        node = self._root
        for ch in pattern.lower():
            if ch not in node.children:
                child = _ACNode()
                child.depth = node.depth + 1
                node.children[ch] = child
            node = node.children[ch]
        node.output.append((pattern, category))
        self._pattern_count += 1

    def build(self) -> None:
        """Build failure links (BFS). Must be called after all patterns are added."""
        queue: deque[_ACNode] = deque()
        # Initialize depth-1 nodes
        for ch, child in self._root.children.items():
            child.fail = self._root
            queue.append(child)

        # BFS to build failure links
        while queue:
            node = queue.popleft()
            for ch, child in node.children.items():
                queue.append(child)
                # Follow failure links to find longest proper suffix
                fail = node.fail
                while fail is not None and ch not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[ch] if fail and ch in fail.children else self._root
                if child.fail is child:
                    child.fail = self._root
                # Merge output from failure chain
                child.output = child.output + child.fail.output

        self._built = True

    def search(self, text: str) -> list[tuple[int, str, str]]:
        """Search text for all pattern matches.

        Returns list of (position, matched_pattern, category).
        Position is the index in text where the match ends.
        """
        if not self._built:
            self.build()

        results: list[tuple[int, str, str]] = []
        node = self._root
        text_lower = text.lower()

        for i, ch in enumerate(text_lower):
            while node is not self._root and ch not in node.children:
                node = node.fail or self._root
            if ch in node.children:
                node = node.children[ch]
            for pattern, category in node.output:
                results.append((i - len(pattern) + 1, pattern, category))

        return results

    def has_any_match(self, text: str) -> bool:
        """Fast check: does text contain ANY pattern? Stops at first match."""
        if not self._built:
            self.build()

        node = self._root
        text_lower = text.lower()

        for ch in text_lower:
            while node is not self._root and ch not in node.children:
                node = node.fail or self._root
            if ch in node.children:
                node = node.children[ch]
            if node.output:
                return True
        return False

    def matched_categories(self, text: str) -> set[str]:
        """Return set of all categories that matched in text."""
        return {cat for _, _, cat in self.search(text)}

    @property
    def pattern_count(self) -> int:
        return self._pattern_count
