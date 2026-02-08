"""Slack workspace users provider.

Fetches all users from Slack via users.list API.
Uses the same urllib pattern as cli/slack_live.py.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from stigmergy.identity.provider import PersonRecord

logger = logging.getLogger(__name__)


class SlackUsersProvider:
    """Fetch all workspace users from Slack users.list API.

    Requires SLACK_BOT_TOKEN environment variable with users:read scope.
    """

    def __init__(self, bot_token: str | None = None) -> None:
        self._token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")

    def source_name(self) -> str:
        return "slack-users"

    def available(self) -> bool:
        return bool(self._token)

    def fetch(self) -> list[PersonRecord]:
        if not self._token:
            return []

        records: list[PersonRecord] = []
        cursor: str | None = None

        while True:
            url = "https://slack.com/api/users.list?limit=200"
            if cursor:
                url += f"&cursor={cursor}"

            req = Request(
                url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

            try:
                response = urlopen(req, timeout=15)
                data = json.loads(response.read().decode())
            except (URLError, json.JSONDecodeError, OSError) as exc:
                logger.warning("Slack users.list failed: %s", exc)
                break

            if not data.get("ok"):
                logger.warning("Slack users.list error: %s", data.get("error", "unknown"))
                break

            members = data.get("members", [])
            for member in members:
                # Skip bots, deleted users, and Slackbot
                if member.get("is_bot") or member.get("deleted") or member.get("id") == "USLACKBOT":
                    continue

                profile = member.get("profile", {})
                real_name = profile.get("real_name", "") or member.get("real_name", "")
                display_name = profile.get("display_name", "")
                email = profile.get("email", "")
                user_id = member.get("id", "")

                records.append(
                    PersonRecord(
                        name=real_name,
                        email=email,
                        slack_handle=display_name or real_name,
                        slack_user_id=user_id,
                        source="slack-users",
                    )
                )

            # Pagination
            metadata = data.get("response_metadata", {})
            next_cursor = metadata.get("next_cursor", "")
            if next_cursor:
                cursor = next_cursor
            else:
                break

        logger.info("Fetched %d users from Slack", len(records))
        return records
