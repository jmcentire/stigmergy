"""Live Slack adapter: reads messages from Slack channels via Web API.

Read-only. This adapter fetches messages from configured channels and
converts them to Signal objects. It never posts, reacts, or modifies
anything in Slack — output goes to the CLI or a configured delivery
target (DM to operator), never to Slack channels.

Authentication: requires a Slack Bot Token (xoxb-...) with scopes:
  - channels:history  (read public channel messages)
  - channels:read     (list channels, resolve names to IDs)
  - groups:history    (read private channel messages, if invited)
  - users:read        (resolve user IDs to display names)

Set SLACK_BOT_TOKEN env var or configure in .stigmergy/config.yaml.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from http.client import HTTPResponse
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pathlib import Path

from stigmergy.primitives.signal import Signal, SignalSource

logger = logging.getLogger(__name__)

SLACK_USER_CACHE_PATH = Path(".stigmergy") / "slack_user_cache.json"


class LiveSlackAdapter:
    """Fetches messages from Slack channels. Read-only."""

    def __init__(
        self,
        channels: list[str] | None = None,
        bot_token: str | None = None,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self._channels = channels or []
        self._token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False
        self._progress_callback = progress_callback
        self._error_callback = error_callback
        # Caches
        self._channel_ids: dict[str, str] = {}  # name -> id
        self._user_cache: dict[str, str] = {}  # user_id -> display_name
        self._channel_meta: dict[str, dict] = {}  # name -> {id, is_private, num_members, topic}

    def load_user_cache(self, path: Path = SLACK_USER_CACHE_PATH) -> int:
        """Load user cache from disk. Returns number of entries loaded."""
        if not path.exists():
            return 0
        try:
            with open(path) as f:
                data = json.load(f)
            self._user_cache.update(data)
            return len(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load Slack user cache: %s", exc)
            return 0

    def save_user_cache(self, path: Path = SLACK_USER_CACHE_PATH) -> int:
        """Save user cache to disk. Returns number of entries saved."""
        if not self._user_cache:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._user_cache, f, indent=2)
        return len(self._user_cache)

    async def connect(self) -> None:
        """Verify bot token works and resolve channel names to IDs."""
        if not self._token:
            raise ConnectionError(
                "SLACK_BOT_TOKEN not set. Set it in your environment or "
                ".stigmergy/config.yaml to use live Slack mode."
            )

        # Verify auth
        result = await self._api("auth.test")
        if result is None or not result.get("ok"):
            error = result.get("error", "unknown") if result else "no response"
            raise ConnectionError(f"Slack auth failed: {error}")

        # Resolve channel names to IDs using bot's own membership list
        await self._resolve_channels()
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        """Fetch messages from all configured channels since the given time.

        Fetches both top-level messages AND thread replies. Slack's
        conversations.history only returns top-level messages — thread
        replies require a separate conversations.replies call per thread.
        """
        oldest = str(since.timestamp())
        total = len(self._channel_ids)
        self._stats = {"api_returned": 0, "filtered": 0, "thread_replies": 0}

        for i, (name, channel_id) in enumerate(sorted(self._channel_ids.items()), 1):
            channel_count = 0
            threaded_msgs: list[dict] = []
            cursor = None

            # Phase 1: Fetch top-level messages
            while True:
                params: dict[str, Any] = {
                    "channel": channel_id,
                    "oldest": oldest,
                    "limit": 200,
                    "inclusive": "true",
                }
                if cursor:
                    params["cursor"] = cursor

                result = await self._api("conversations.history", params)
                if result is None or not result.get("ok"):
                    error = result.get("error", "unknown") if result else "no response"
                    if self._error_callback:
                        self._error_callback(name, error)
                    break

                messages = result.get("messages", [])
                self._stats["api_returned"] += len(messages)
                for msg in messages:
                    # Collect threaded messages for phase 2
                    if msg.get("reply_count", 0) > 0 and msg.get("thread_ts"):
                        threaded_msgs.append(msg)

                    signal = await self._message_to_signal(msg, name)
                    if signal is not None:
                        channel_count += 1
                        yield signal
                    else:
                        self._stats["filtered"] += 1

                # Pagination
                metadata = result.get("response_metadata", {})
                next_cursor = metadata.get("next_cursor", "")
                if next_cursor:
                    cursor = next_cursor
                else:
                    break

            # Phase 2: Fetch thread replies for threaded messages
            for msg in threaded_msgs:
                thread_ts = msg["thread_ts"]
                async for signal in self._fetch_thread_replies(
                    channel_id, name, thread_ts, oldest
                ):
                    channel_count += 1
                    self._stats["thread_replies"] += 1
                    yield signal

            if self._progress_callback:
                self._progress_callback(name, i, total, channel_count)

    async def _fetch_thread_replies(
        self, channel_id: str, channel_name: str, thread_ts: str, oldest: str,
    ) -> AsyncIterator[Signal]:
        """Fetch replies in a thread, yielding only those after `oldest`.

        The parent message is returned as the first reply by the API;
        we skip it (already yielded from conversations.history) but
        capture its author for thread_author attribution on replies.
        """
        cursor = None
        thread_author: str | None = None
        while True:
            params: dict[str, Any] = {
                "channel": channel_id,
                "ts": thread_ts,
                "oldest": oldest,
                "inclusive": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            result = await self._api("conversations.replies", params)
            if result is None or not result.get("ok"):
                break

            for msg in result.get("messages", []):
                # The parent message (first in replies) — capture its author
                # but don't re-yield (already yielded from history)
                if msg.get("ts") == thread_ts:
                    parent_user = msg.get("user", "")
                    if parent_user:
                        thread_author = await self._resolve_user(parent_user)
                    continue
                signal = await self._message_to_signal(
                    msg, channel_name, thread_author=thread_author,
                )
                if signal is not None:
                    yield signal

            metadata = result.get("response_metadata", {})
            next_cursor = metadata.get("next_cursor", "")
            if next_cursor:
                cursor = next_cursor
            else:
                break

    async def _message_to_signal(
        self, msg: dict, channel_name: str, thread_author: str | None = None,
    ) -> Signal | None:
        """Convert a Slack message to a Signal. Skips bot messages and joins."""
        subtype = msg.get("subtype", "")
        # Skip system messages (joins, leaves, topic changes, etc.)
        if subtype in ("channel_join", "channel_leave", "channel_topic",
                       "channel_purpose", "channel_name", "bot_add",
                       "bot_remove", "pinned_item", "unpinned_item"):
            return None

        text = msg.get("text", "").strip()
        if not text:
            return None

        # Skip bot messages unless they have meaningful content
        if msg.get("bot_id") and len(text) < 20:
            return None

        # Collect all user IDs upfront for batch resolution
        all_user_ids: list[str] = []
        user_id = msg.get("user", "")
        if user_id:
            all_user_ids.append(user_id)

        reactions = msg.get("reactions", [])
        reactor_uids: list[str] = []
        for reaction in reactions:
            for reactor_uid in reaction.get("users", []):
                if reactor_uid not in reactor_uids:
                    reactor_uids.append(reactor_uid)
                    if reactor_uid not in all_user_ids:
                        all_user_ids.append(reactor_uid)

        mention_uids: list[str] = []
        for match in re.finditer(r"<@(U[A-Z0-9]+)(?:\|[^>]*)?>", text):
            mentioned_uid = match.group(1)
            if mentioned_uid not in mention_uids:
                mention_uids.append(mentioned_uid)
                if mentioned_uid not in all_user_ids:
                    all_user_ids.append(mentioned_uid)

        # Batch-resolve all user IDs concurrently
        resolved_users = await self._batch_resolve_users(all_user_ids)

        author = resolved_users.get(user_id, "unknown") if user_id else "unknown"

        # Parse timestamp
        ts = msg.get("ts", "")
        try:
            timestamp = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            timestamp = datetime.now(timezone.utc)

        # Thread info
        thread_ts = msg.get("thread_ts")
        reply_count = msg.get("reply_count", 0)
        is_thread_reply = thread_ts is not None and thread_ts != ts

        # Reactions — build resolved reactor names from batch results
        reaction_count = sum(r.get("count", 0) for r in reactions)
        reactors: list[str] = []
        for uid in reactor_uids:
            resolved = resolved_users.get(uid, uid)
            if resolved not in reactors:
                reactors.append(resolved)

        # @mentions — build resolved mention names from batch results
        mentions: list[str] = []
        for uid in mention_uids:
            resolved = resolved_users.get(uid, uid)
            if resolved not in mentions:
                mentions.append(resolved)

        # Files/attachments
        files = msg.get("files", [])
        attachments = msg.get("attachments", [])

        meta: dict[str, Any] = {
            "event_type": "message",
            "subtype": subtype or "message",
            "thread_ts": thread_ts,
            "reply_count": reply_count,
            "reaction_count": reaction_count,
            "has_files": len(files) > 0,
            "has_attachments": len(attachments) > 0,
            "is_thread_reply": is_thread_reply,
        }
        if reactors:
            meta["reactors"] = reactors
        if mentions:
            meta["mentions"] = mentions
        if is_thread_reply and thread_author:
            meta["thread_author"] = thread_author

        return Signal(
            content=text,
            source=SignalSource.SLACK,
            channel=f"#{channel_name}",
            author=author,
            timestamp=timestamp,
            metadata=meta,
        )

    async def _resolve_channels(self) -> None:
        """Resolve channel names to IDs using bot's membership list.

        Uses users.conversations (channels the bot is IN) instead of
        conversations.list (ALL workspace channels). This is:
        - Fast: bot is in ~20 channels vs 1600+ in workspace
        - Correct: bot can only read history from channels it's a member of
        - Rate-limit friendly: 1-2 API calls vs 50+

        Stores full channel metadata (is_private, num_members, topic) for
        channel migration detection.
        """
        cursor = None
        bot_channels: dict[str, dict] = {}  # name -> {id, is_private, num_members, topic}

        while True:
            params: dict[str, Any] = {
                "types": "public_channel,private_channel",
                "limit": 200,
                "exclude_archived": "true",
            }
            if cursor:
                params["cursor"] = cursor

            result = await self._api("users.conversations", params)
            if result is None or not result.get("ok"):
                error = result.get("error", "unknown") if result else "no response"
                logger.warning("Failed to list bot channels: %s", error)
                break

            for ch in result.get("channels", []):
                bot_channels[ch["name"]] = {
                    "id": ch["id"],
                    "is_private": ch.get("is_private", False),
                    "num_members": ch.get("num_members", 0),
                    "topic": ch.get("topic", {}).get("value", ""),
                }

            metadata = result.get("response_metadata", {})
            next_cursor = metadata.get("next_cursor", "")
            if next_cursor:
                cursor = next_cursor
            else:
                break

        logger.info("Bot is a member of %d channels", len(bot_channels))

        # Match configured channel names (with or without #)
        not_found: list[str] = []
        for name in self._channels:
            clean = name.lstrip("#")
            if clean in bot_channels:
                meta = bot_channels[clean]
                self._channel_ids[clean] = meta["id"]
                self._channel_meta[clean] = meta
            else:
                not_found.append(clean)
                logger.warning("Slack channel not accessible: %s (bot not a member or doesn't exist)", name)

        if not_found:
            if self._error_callback:
                self._error_callback(
                    "resolve",
                    f"{len(not_found)} channels not accessible: {', '.join(not_found[:5])}"
                    + (f" (+{len(not_found) - 5} more)" if len(not_found) > 5 else "")
                    + " — invite bot with /invite @Stigmergy"
                )

        # If no channels configured, use all bot channels
        if not self._channels:
            self._channel_ids = {name: meta["id"] for name, meta in bot_channels.items()}
            self._channel_meta = bot_channels

    async def _resolve_user(self, user_id: str) -> str:
        """Resolve a Slack user ID to a display name. Cached."""
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        result = await self._api("users.info", {"user": user_id})
        if result and result.get("ok"):
            user = result.get("user", {})
            profile = user.get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or user.get("name")
                or user_id
            )
            self._user_cache[user_id] = name
            return name

        self._user_cache[user_id] = user_id
        return user_id

    async def _batch_resolve_users(self, user_ids: list[str]) -> dict[str, str]:
        """Resolve multiple user IDs concurrently. Returns {user_id: display_name}.

        Fires concurrent _resolve_user() calls via asyncio.gather() for
        uncached IDs. Cached IDs are returned immediately.
        """
        result: dict[str, str] = {}
        uncached: list[str] = []
        for uid in user_ids:
            if uid in self._user_cache:
                result[uid] = self._user_cache[uid]
            elif uid not in uncached:
                uncached.append(uid)

        if uncached:
            resolved = await asyncio.gather(
                *(self._resolve_user(uid) for uid in uncached)
            )
            for uid, name in zip(uncached, resolved):
                result[uid] = name

        return result

    async def _api(self, method: str, params: dict[str, Any] | None = None) -> dict | None:
        """Call a Slack Web API method. Handles rate limiting (429)."""
        url = f"https://slack.com/api/{method}"
        if params:
            parts = []
            for k, v in params.items():
                parts.append(f"{k}={quote(str(v), safe=',')}")
            url = f"{url}?{'&'.join(parts)}"

        req = Request(
            url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        for attempt in range(3):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: urlopen(req, timeout=30))
                return json.loads(response.read().decode())
            except HTTPError as e:
                if e.code == 429:
                    retry_after = int(e.headers.get("Retry-After", 5))
                    logger.info("Slack rate limited (%s), waiting %ds", method, retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                logger.debug("Slack HTTP error (%s): %s %s", method, e.code, e.reason)
                return None
            except (URLError, json.JSONDecodeError, OSError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.debug("Slack API error (%s): %s", method, e)
                return None
        return None
