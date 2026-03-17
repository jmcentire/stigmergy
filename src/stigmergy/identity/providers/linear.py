"""Linear workspace users provider.

Fetches all users from the Linear workspace via GraphQL API.
Uses the same urllib pattern as cli/linear_live.py.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from stigmergy.identity.provider import PersonRecord

logger = logging.getLogger(__name__)

_GRAPHQL_URL = "https://api.linear.app/graphql"

_USERS_QUERY = """
query($cursor: String) {
  users(first: 100, after: $cursor, includeDisabled: false) {
    nodes {
      id
      name
      displayName
      email
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


class LinearUsersProvider:
    """Fetch all workspace users from Linear GraphQL API.

    Requires LINEAR_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("LINEAR_API_KEY", "")

    def source_name(self) -> str:
        return "linear-users"

    def available(self) -> bool:
        return bool(self._api_key)

    def fetch(self) -> list[PersonRecord]:
        if not self._api_key:
            return []

        records: list[PersonRecord] = []
        cursor: str | None = None

        while True:
            variables: dict[str, str | None] = {"cursor": cursor}
            payload = json.dumps(
                {"query": _USERS_QUERY, "variables": variables}
            ).encode()

            req = Request(
                _GRAPHQL_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._api_key,
                },
            )

            try:
                response = urlopen(req, timeout=15)
                data = json.loads(response.read().decode())
            except (URLError, json.JSONDecodeError, OSError) as exc:
                logger.warning("Linear users query failed: %s", exc)
                break

            users_data = data.get("data", {}).get("users", {})
            nodes = users_data.get("nodes", [])

            for node in nodes:
                records.append(
                    PersonRecord(
                        name=node.get("name", ""),
                        email=node.get("email", ""),
                        linear_uuid=node.get("id", ""),
                        linear_display_name=node.get("displayName", ""),
                        source="linear-users",
                    )
                )

            page_info = users_data.get("pageInfo", {})
            if page_info.get("hasNextPage"):
                cursor = page_info.get("endCursor")
            else:
                break

        logger.info("Fetched %d users from Linear", len(records))
        return records
