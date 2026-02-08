"""GitHub organization member provider.

Fetches org members via `gh api` CLI. Uses the same subprocess pattern
as the rest of the codebase (no requests dependency).
"""

from __future__ import annotations

import json
import logging
import subprocess

from stigmergy.identity.provider import PersonRecord

logger = logging.getLogger(__name__)


class GitHubOrgProvider:
    """Fetch organization members from GitHub via `gh api`.

    Requires `gh auth login` to have been run. Uses the `gh api`
    command to avoid adding HTTP dependencies.
    """

    def __init__(self, org: str) -> None:
        self._org = org

    def source_name(self) -> str:
        return "github-org"

    def available(self) -> bool:
        """Check that `gh` CLI is authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def fetch(self) -> list[PersonRecord]:
        """Fetch org members. Returns empty list on failure."""
        try:
            result = subprocess.run(
                [
                    "gh", "api",
                    f"orgs/{self._org}/members",
                    "--paginate",
                    "--jq", ".[].login",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "gh api orgs/%s/members failed: %s",
                    self._org,
                    result.stderr.strip(),
                )
                return []

            logins = [line.strip() for line in result.stdout.splitlines() if line.strip()]

        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("GitHub org provider failed: %s", exc)
            return []

        # Fetch detailed info for each member
        records: list[PersonRecord] = []
        for login in logins:
            record = self._fetch_user(login)
            if record:
                records.append(record)

        logger.info("Fetched %d members from GitHub org %s", len(records), self._org)
        return records

    def _fetch_user(self, login: str) -> PersonRecord | None:
        """Fetch a single user's profile."""
        try:
            result = subprocess.run(
                ["gh", "api", f"users/{login}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return PersonRecord(
                    name=login,
                    github_handle=login,
                    source="github-org",
                )

            data = json.loads(result.stdout)
            return PersonRecord(
                name=data.get("name") or login,
                email=data.get("email") or "",
                github_handle=login,
                source="github-org",
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return PersonRecord(
                name=login,
                github_handle=login,
                source="github-org",
            )
