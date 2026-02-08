"""Mock Redis connector.

Implements the subset of Redis operations the system uses:
- Key-value (get/set/delete)
- Hash maps (hget/hset/hgetall)
- Sorted sets (zadd/zrange/zrangebyscore)
- Expiry (expire/ttl)
- Pub/sub (publish/subscribe)

Same interface a real Redis client would expose. When the team swaps in
redis-py or aioredis, nothing upstream changes.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable, Protocol


class RedisClient(Protocol):
    """Protocol for Redis-like clients. Mock and real implement this."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> None: ...
    async def delete(self, key: str) -> int: ...
    async def exists(self, key: str) -> bool: ...
    async def incr(self, key: str) -> int: ...
    async def expire(self, key: str, seconds: int) -> bool: ...
    async def ttl(self, key: str) -> int: ...
    async def hget(self, key: str, field: str) -> str | None: ...
    async def hset(self, key: str, field: str, value: str) -> None: ...
    async def hgetall(self, key: str) -> dict[str, str]: ...
    async def zadd(self, key: str, mapping: dict[str, float]) -> int: ...
    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list: ...
    async def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list[str]: ...
    async def publish(self, channel: str, message: str) -> int: ...
    async def subscribe(self, channel: str, callback: Callable[[str, str], None]) -> None: ...


class MockRedis:
    """In-memory Redis mock. Thread-safe for single-event-loop async.

    Tracks all operations in an audit log for test assertions.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}  # key -> expiry timestamp
        self._hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self._sorted_sets: dict[str, dict[str, float]] = defaultdict(dict)
        self._subscribers: dict[str, list[Callable[[str, str], None]]] = defaultdict(list)
        self._audit_log: list[dict[str, Any]] = []

    def _log(self, op: str, **kwargs: Any) -> None:
        self._audit_log.append({"op": op, "time": time.monotonic(), **kwargs})

    def _is_expired(self, key: str) -> bool:
        if key in self._expiry:
            if time.monotonic() > self._expiry[key]:
                self._data.pop(key, None)
                self._hashes.pop(key, None)
                self._sorted_sets.pop(key, None)
                del self._expiry[key]
                return True
        return False

    # -- Key-value --

    async def get(self, key: str) -> str | None:
        self._log("GET", key=key)
        if self._is_expired(key):
            return None
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._log("SET", key=key, ex=ex)
        self._data[key] = value
        if ex is not None:
            self._expiry[key] = time.monotonic() + ex

    async def delete(self, key: str) -> int:
        self._log("DEL", key=key)
        existed = key in self._data or key in self._hashes or key in self._sorted_sets
        self._data.pop(key, None)
        self._hashes.pop(key, None)
        self._sorted_sets.pop(key, None)
        self._expiry.pop(key, None)
        return 1 if existed else 0

    async def exists(self, key: str) -> bool:
        self._log("EXISTS", key=key)
        if self._is_expired(key):
            return False
        return key in self._data or key in self._hashes or key in self._sorted_sets

    async def incr(self, key: str) -> int:
        self._log("INCR", key=key)
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int) -> bool:
        self._log("EXPIRE", key=key, seconds=seconds)
        if key in self._data or key in self._hashes or key in self._sorted_sets:
            self._expiry[key] = time.monotonic() + seconds
            return True
        return False

    async def ttl(self, key: str) -> int:
        self._log("TTL", key=key)
        if key not in self._expiry:
            return -1
        remaining = self._expiry[key] - time.monotonic()
        return max(0, int(remaining))

    # -- Hashes --

    async def hget(self, key: str, field: str) -> str | None:
        self._log("HGET", key=key, field=field)
        if self._is_expired(key):
            return None
        return self._hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str) -> None:
        self._log("HSET", key=key, field=field)
        self._hashes[key][field] = value

    async def hgetall(self, key: str) -> dict[str, str]:
        self._log("HGETALL", key=key)
        if self._is_expired(key):
            return {}
        return dict(self._hashes.get(key, {}))

    # -- Sorted sets --

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self._log("ZADD", key=key, count=len(mapping))
        added = 0
        for member, score in mapping.items():
            if member not in self._sorted_sets[key]:
                added += 1
            self._sorted_sets[key][member] = score
        return added

    async def zrange(self, key: str, start: int, stop: int,
                     withscores: bool = False) -> list:
        self._log("ZRANGE", key=key, start=start, stop=stop)
        if self._is_expired(key):
            return []
        items = sorted(self._sorted_sets.get(key, {}).items(), key=lambda x: x[1])
        # Redis ZRANGE is inclusive on both ends, -1 means last
        if stop == -1:
            stop = len(items)
        else:
            stop += 1
        sliced = items[start:stop]
        if withscores:
            return sliced
        return [member for member, _ in sliced]

    async def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list[str]:
        self._log("ZRANGEBYSCORE", key=key, min=min_score, max=max_score)
        if self._is_expired(key):
            return []
        items = sorted(self._sorted_sets.get(key, {}).items(), key=lambda x: x[1])
        return [m for m, s in items if min_score <= s <= max_score]

    # -- Pub/sub --

    async def publish(self, channel: str, message: str) -> int:
        self._log("PUBLISH", channel=channel)
        subs = self._subscribers.get(channel, [])
        for callback in subs:
            callback(channel, message)
        return len(subs)

    async def subscribe(self, channel: str, callback: Callable[[str, str], None]) -> None:
        self._log("SUBSCRIBE", channel=channel)
        self._subscribers[channel].append(callback)

    # -- Test helpers --

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        self._audit_log.clear()

    def reset(self) -> None:
        """Clear all data. For test teardown."""
        self._data.clear()
        self._expiry.clear()
        self._hashes.clear()
        self._sorted_sets.clear()
        self._subscribers.clear()
        self._audit_log.clear()
