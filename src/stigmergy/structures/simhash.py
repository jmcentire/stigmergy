"""SimHash for near-duplicate signal detection.

SimHash generates a fingerprint where similar documents have similar
fingerprints (small Hamming distance). Combined with the counting bloom
filter, this gives us fast dedup in the ingestion pipeline.

Performance:
- fingerprint: O(|terms|)
- Hamming distance: O(1) — single XOR + popcount
- Index query: O(num_blocks * avg_bucket_size) — band-based lookup
"""

from __future__ import annotations

from stigmergy.structures.bloom import _fnv1a_64


class SimHash:
    """64-bit SimHash fingerprint generator."""

    __slots__ = ()

    @staticmethod
    def fingerprint(terms: list[str], weights: list[float] | None = None) -> int:
        """Compute 64-bit SimHash from weighted terms.

        Args:
            terms: List of text tokens.
            weights: Optional weight per term. Defaults to 1.0 each.

        Returns:
            64-bit integer fingerprint.
        """
        if not terms:
            return 0

        v = [0.0] * 64
        w = weights or [1.0] * len(terms)

        for term, weight in zip(terms, w):
            h = _fnv1a_64(term.encode("utf-8"))
            for i in range(64):
                if h & (1 << i):
                    v[i] += weight
                else:
                    v[i] -= weight

        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def hamming_distance(a: int, b: int) -> int:
        """Hamming distance between two fingerprints. O(1)."""
        x = a ^ b
        # popcount
        count = 0
        while x:
            count += 1
            x &= x - 1
        return count

    @staticmethod
    def similarity(a: int, b: int) -> float:
        """Cosine-like similarity from Hamming distance. [0, 1]."""
        d = SimHash.hamming_distance(a, b)
        return 1.0 - (d / 64.0)


class SimHashIndex:
    """Index for finding near-duplicate fingerprints.

    Uses band-based lookup: splits 64-bit fingerprint into blocks,
    indexes each block separately. Two fingerprints match if they
    share at least one identical block.

    Parameters:
    - num_blocks: more blocks = higher recall, more memory
    - distance_threshold: max Hamming distance to consider a match

    Performance:
    - add: O(num_blocks)
    - query: O(num_blocks * avg_bucket_size)
    """

    __slots__ = ("_num_blocks", "_block_bits", "_distance_threshold", "_tables", "_fingerprints")

    def __init__(self, num_blocks: int = 4, distance_threshold: int = 3):
        self._num_blocks = num_blocks
        self._block_bits = 64 // num_blocks
        self._distance_threshold = distance_threshold
        # block_index -> block_value -> set of IDs
        self._tables: list[dict[int, set[str]]] = [{} for _ in range(num_blocks)]
        self._fingerprints: dict[str, int] = {}

    def _get_block(self, fingerprint: int, block_idx: int) -> int:
        """Extract a block of bits from the fingerprint."""
        shift = block_idx * self._block_bits
        mask = (1 << self._block_bits) - 1
        return (fingerprint >> shift) & mask

    def add(self, id: str, fingerprint: int) -> None:
        """Add a fingerprint to the index."""
        self._fingerprints[id] = fingerprint
        for b in range(self._num_blocks):
            block = self._get_block(fingerprint, b)
            self._tables[b].setdefault(block, set()).add(id)

    def remove(self, id: str) -> None:
        """Remove a fingerprint from the index."""
        fp = self._fingerprints.pop(id, None)
        if fp is None:
            return
        for b in range(self._num_blocks):
            block = self._get_block(fp, b)
            bucket = self._tables[b].get(block)
            if bucket:
                bucket.discard(id)

    def query(self, fingerprint: int, exclude_id: str | None = None) -> list[tuple[str, int]]:
        """Find near-duplicates of a fingerprint.

        Returns list of (id, hamming_distance) within distance_threshold,
        sorted by distance ascending.
        """
        candidates: set[str] = set()
        for b in range(self._num_blocks):
            block = self._get_block(fingerprint, b)
            bucket = self._tables[b].get(block, set())
            candidates.update(bucket)

        if exclude_id:
            candidates.discard(exclude_id)

        results: list[tuple[str, int]] = []
        for cid in candidates:
            cfp = self._fingerprints.get(cid)
            if cfp is not None:
                dist = SimHash.hamming_distance(fingerprint, cfp)
                if dist <= self._distance_threshold:
                    results.append((cid, dist))

        results.sort(key=lambda x: x[1])
        return results

    def has_near_duplicate(self, fingerprint: int, exclude_id: str | None = None) -> bool:
        """Fast check: is there any near-duplicate? Stops at first match."""
        for b in range(self._num_blocks):
            block = self._get_block(fingerprint, b)
            for cid in self._tables[b].get(block, set()):
                if cid == exclude_id:
                    continue
                cfp = self._fingerprints.get(cid)
                if cfp is not None and SimHash.hamming_distance(fingerprint, cfp) <= self._distance_threshold:
                    return True
        return False

    def __len__(self) -> int:
        return len(self._fingerprints)

    def save(self, path) -> None:
        """Persist fingerprints to a JSON file for cross-run dedup."""
        import json
        from pathlib import Path
        data = {
            "num_blocks": self._num_blocks,
            "distance_threshold": self._distance_threshold,
            "fingerprints": self._fingerprints,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path) -> bool:
        """Restore fingerprints from a JSON file. Returns True if loaded."""
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return False
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        if data.get("num_blocks") != self._num_blocks:
            return False
        if data.get("distance_threshold") != self._distance_threshold:
            return False
        for id_str, fp in data.get("fingerprints", {}).items():
            self.add(id_str, fp)
        return True
