"""Live Linear adapter using the Linear GraphQL API.

Fetches real issues from configured teams, converts to Signal objects.
Requires LINEAR_API_KEY environment variable.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Callable
from urllib.request import Request, urlopen
from urllib.error import URLError

from stigmergy.primitives.signal import Signal, SignalSource

_GRAPHQL_URL = "https://api.linear.app/graphql"

_ISSUES_QUERY = """
query($teamId: ID!, $after: DateTimeOrDuration!, $first: Int!, $cursor: String) {
  issues(
    filter: {
      team: { id: { eq: $teamId } }
      updatedAt: { gte: $after }
    }
    first: $first
    after: $cursor
    orderBy: updatedAt
  ) {
    nodes {
      id
      identifier
      title
      description
      state { name }
      priority
      assignee { name displayName }
      creator { name displayName }
      labels { nodes { name } }
      createdAt
      updatedAt
      url
      comments(first: 50) { nodes { body user { name displayName } createdAt } }
      relations { nodes { type relatedIssue { identifier title url } } }
      cycle { name startsAt endsAt }
      estimate
      dueDate
      parent { identifier title }
      children { nodes { identifier title state { name } } }
      project { name }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


class LiveLinearAdapter:
    """Real Linear adapter via GraphQL API."""

    def __init__(
        self,
        teams: list[dict[str, str]] | None = None,
        api_key: str | None = None,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self._teams = teams or []
        self._api_key = api_key or os.environ.get("LINEAR_API_KEY", "")
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False
        self._progress_callback = progress_callback
        self._error_callback = error_callback

    async def connect(self) -> None:
        """Verify API key works."""
        if not self._api_key:
            raise ConnectionError(
                "LINEAR_API_KEY not set. Set it in your environment or "
                ".stigmergy/config.yaml to use live Linear mode."
            )
        # Quick validation: fetch viewer
        result = await self._graphql('query { viewer { id name } }')
        if result is None:
            raise ConnectionError("Linear API authentication failed.")
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        """Fetch issues updated since `since` from all configured teams."""
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        total_teams = len(self._teams)

        for i, team in enumerate(self._teams, 1):
            team_name = team.get("name", "unknown")
            team_id = team.get("id", "")
            team_count = 0

            cursor = None
            while True:
                variables = {
                    "teamId": team_id,
                    "after": since_str,
                    "first": 50,
                }
                if cursor:
                    variables["cursor"] = cursor

                result = await self._graphql(_ISSUES_QUERY, variables)
                if result is None:
                    if self._error_callback:
                        self._error_callback(team_name, "GraphQL query failed")
                    break

                issues_data = result.get("issues", {})
                nodes = issues_data.get("nodes", [])

                for issue in nodes:
                    signal = _issue_to_signal(issue, team_name)
                    team_count += 1
                    yield signal

                    # Yield comment signals
                    for comment_signal in _comment_signals(issue, team_name):
                        team_count += 1
                        yield comment_signal

                page_info = issues_data.get("pageInfo", {})
                if page_info.get("hasNextPage") and page_info.get("endCursor"):
                    cursor = page_info["endCursor"]
                else:
                    break

            if self._progress_callback:
                self._progress_callback(team_name, i, total_teams, team_count)

    async def _graphql(self, query: str, variables: dict | None = None, *, max_retries: int = 3) -> dict | None:
        """Execute a GraphQL query against the Linear API with retry + backoff."""
        payload = json.dumps({"query": query, "variables": variables or {}}).encode()

        for attempt in range(max_retries + 1):
            req = Request(
                _GRAPHQL_URL,
                data=payload,
                headers={
                    "Authorization": self._api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: urlopen(req, timeout=15))
                body = json.loads(response.read().decode())
                if "errors" in body:
                    err_msg = str(body["errors"][:1])
                    # Rate limit errors from Linear contain "rateLimit" or status 429
                    is_transient = any(s in err_msg.lower() for s in ("ratelimit", "rate_limit", "too many", "timeout"))
                    if is_transient and attempt < max_retries:
                        delay = 2 ** attempt
                        if self._error_callback:
                            self._error_callback("linear", f"{err_msg[:80]}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    if self._error_callback:
                        self._error_callback("linear", err_msg)
                    return None
                return body.get("data")
            except (URLError, json.JSONDecodeError, OSError) as e:
                err_msg = str(e)[:200]
                is_transient = any(s in err_msg.lower() for s in ("timeout", "temporary", "connection reset", "502", "503"))
                if is_transient and attempt < max_retries:
                    delay = 2 ** attempt
                    if self._error_callback:
                        self._error_callback("linear", f"{err_msg[:80]}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                if self._error_callback:
                    self._error_callback("linear", err_msg)
                return None

        return None


def _issue_to_signal(issue: dict, team_name: str) -> Signal:
    """Convert a Linear issue dict to a Signal."""
    identifier = issue.get("identifier", "")
    title = issue.get("title", "")
    description = (issue.get("description") or "")[:2000]
    state = issue.get("state", {}).get("name", "")
    priority = issue.get("priority", 0)

    assignee_info = issue.get("assignee") or {}
    assignee = assignee_info.get("displayName") or assignee_info.get("name") or ""
    creator_info = issue.get("creator") or {}
    creator = creator_info.get("displayName") or creator_info.get("name") or "unknown"

    labels = [l.get("name", "") for l in (issue.get("labels", {}).get("nodes") or [])]

    content = f"{title}"
    if description:
        content += f"\n{description}"

    created = issue.get("createdAt", "")
    updated = issue.get("updatedAt", "")
    ts = _parse_timestamp(updated or created)

    metadata = {
        "identifier": identifier,
        "title": title,
        "status": state,
        "priority": priority,
        "labels": labels,
        "assignee": assignee,
        "team": team_name,
        "url": issue.get("url", ""),
    }

    # Enriched metadata — only include when present
    estimate = issue.get("estimate")
    if estimate is not None:
        metadata["estimate"] = estimate

    due_date = issue.get("dueDate")
    if due_date:
        metadata["due_date"] = due_date

    cycle = issue.get("cycle")
    if cycle:
        metadata["cycle"] = cycle.get("name", "")
        metadata["cycle_start"] = cycle.get("startsAt", "")
        metadata["cycle_end"] = cycle.get("endsAt", "")

    parent = issue.get("parent")
    if parent:
        metadata["parent"] = parent.get("identifier", "")
        metadata["parent_title"] = parent.get("title", "")

    children_data = issue.get("children", {}).get("nodes") or []
    if children_data:
        metadata["children"] = [
            {"id": c.get("identifier", ""), "title": c.get("title", ""), "status": c.get("state", {}).get("name", "")}
            for c in children_data
        ]

    project = issue.get("project")
    if project:
        metadata["project"] = project.get("name", "")

    relations_data = issue.get("relations", {}).get("nodes") or []
    if relations_data:
        metadata["relations"] = [
            {"type": r.get("type", ""), "issue": r.get("relatedIssue", {}).get("identifier", ""),
             "title": r.get("relatedIssue", {}).get("title", "")}
            for r in relations_data
        ]

    return Signal(
        content=content,
        source=SignalSource.LINEAR,
        channel=identifier,
        author=creator,
        timestamp=ts,
        metadata=metadata,
    )


def _comment_signals(issue: dict, team_name: str) -> list[Signal]:
    """Extract comment signals from a Linear issue."""
    signals: list[Signal] = []
    identifier = issue.get("identifier", "")
    comments_data = issue.get("comments", {}).get("nodes") or []

    for comment in comments_data:
        body = (comment.get("body") or "").strip()
        if not body:
            continue

        user_info = comment.get("user") or {}
        comment_author = user_info.get("displayName") or user_info.get("name") or "unknown"

        comment_ts = _parse_timestamp(comment.get("createdAt", ""))

        signals.append(Signal(
            content=body[:2000],
            source=SignalSource.LINEAR,
            channel=identifier,
            author=comment_author,
            timestamp=comment_ts,
            metadata={
                "event_type": "linear_comment",
                "identifier": identifier,
                "team": team_name,
            },
        ))

    return signals


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp from Linear."""
    if not ts:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
