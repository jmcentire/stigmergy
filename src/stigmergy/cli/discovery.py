"""Auto-discovery of repos, config files, and environment variables."""

from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from stigmergy.cli.config_schema import LinearTeamConfig


def refresh_repos(
    repos_dir: Path,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, bool]:
    """Git-fetch all repos under repos_dir in parallel.

    Returns {repo_name: success_bool}.
    """
    if not repos_dir.is_dir():
        return {}

    git_dirs = [
        entry for entry in sorted(repos_dir.iterdir())
        if entry.is_dir() and (entry / ".git").exists()
    ]
    if not git_dirs:
        return {}

    results: dict[str, bool] = {}
    total = len(git_dirs)

    def _fetch_one(repo_path: Path) -> tuple[str, bool]:
        name = repo_path.name
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "fetch", "--all", "--prune"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return name, result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return name, False

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, d): d for d in git_dirs}
        done_count = 0
        for future in as_completed(futures):
            name, ok = future.result()
            results[name] = ok
            done_count += 1
            if progress_callback:
                progress_callback(name, done_count, total)

    return results


def discover_repos(repos_dir: Path) -> list[str]:
    """Scan a directory for git repos, return org/repo strings from remotes."""
    repos: list[str] = []
    if not repos_dir.is_dir():
        return repos

    for entry in sorted(repos_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(entry), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                org_repo = _parse_remote_url(result.stdout.strip())
                if org_repo:
                    repos.append(org_repo)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return repos


def _parse_remote_url(url: str) -> str | None:
    """Extract org/repo from a git remote URL (SSH or HTTPS, with or without tokens)."""
    # SSH: git@github.com:org/repo.git  (also github.com-alias variants)
    m = re.match(r"git@[\w.-]+:([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # HTTPS with token: https://x-access-token:TOKEN@github.com/org/repo.git
    m = re.match(r"https?://[^@]+@[\w.-]+/([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    # HTTPS plain: https://github.com/org/repo.git
    m = re.match(r"https?://[\w.-]+/([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def parse_linear_reference(
    path: Path,
) -> tuple[list[LinearTeamConfig], dict[str, str]]:
    """Parse linear-reference.md for teams and members.

    Returns (teams, members) where members is {name: uuid}.
    """
    teams: list[LinearTeamConfig] = []
    members: dict[str, str] = {}

    if not path.exists():
        return teams, members

    text = path.read_text()

    section = ""
    for line in text.splitlines():
        if "## Teams" in line and "## Team Members" not in line:
            section = "teams"
            continue
        if "## Team Members" in line:
            section = "members"
            continue
        if line.startswith("##"):
            section = ""
            continue

        if section == "teams":
            # | Team | Key | UUID |
            m = re.match(
                r"\|\s*(\w[\w\s]*?)\s*\|\s*(\w+)\s*\|\s*`([0-9a-f-]+)`\s*\|",
                line,
            )
            if m:
                name, _key, uuid = m.groups()
                teams.append(LinearTeamConfig(name=name.strip(), id=uuid))

        elif section == "members":
            # | Name | Email | UUID |
            m = re.match(
                r"\|\s*(.+?)\s*\|\s*\S+@\S+\s*\|\s*`([0-9a-f-]+)`\s*\|",
                line,
            )
            if m:
                name, uuid = m.groups()
                members[name.strip()] = uuid

    return teams, members


def read_linear_api_key(path: Path) -> str | None:
    """Read Linear API key from a file."""
    if not path.exists():
        return None
    text = path.read_text().strip()
    if text and text.startswith("lin_api_"):
        return text
    return None


def discover_linear_teams_live(
    api_key: str | None = None,
) -> tuple[list[LinearTeamConfig], str]:
    """Query the Linear GraphQL API to discover all teams in the organization.

    Returns (teams, org_id). Falls back gracefully if API is unreachable.
    """
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    key = api_key or os.environ.get("LINEAR_API_KEY", "")
    if not key:
        return [], ""

    url = "https://api.linear.app/graphql"

    # Fetch teams
    teams_query = '{"query": "query { teams { nodes { id name key } } organization { id name } }"}'
    req = Request(
        url,
        data=teams_query.encode(),
        headers={
            "Authorization": key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        response = urlopen(req, timeout=15)
        body = json.loads(response.read().decode())
        if "errors" in body:
            return [], ""
        data = body.get("data", {})
    except (URLError, json.JSONDecodeError, OSError):
        return [], ""

    org_id = data.get("organization", {}).get("id", "")
    team_nodes = data.get("teams", {}).get("nodes", [])

    teams = [
        LinearTeamConfig(name=t["name"], id=t["id"])
        for t in sorted(team_nodes, key=lambda t: t.get("name", ""))
        if t.get("id") and t.get("name")
    ]
    return teams, org_id


def find_workspace_root() -> Path | None:
    """Try to find the workspace root by looking for repos/ and config/ dirs."""
    cwd = Path.cwd()

    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "repos").is_dir() or (candidate / "config").is_dir():
            return candidate

    return None


class DiscoveredConfig:
    """Aggregated discovered configuration."""

    def __init__(self) -> None:
        self.repos: list[str] = []
        self.linear_teams: list[LinearTeamConfig] = []
        self.linear_members: dict[str, str] = {}
        self.linear_api_key: str | None = None
        self.anthropic_api_key: str | None = None
        self.workspace_root: Path | None = None
        self.config_dir: Path | None = None
        self.repos_dir: Path | None = None

    @property
    def has_repos(self) -> bool:
        return len(self.repos) > 0

    @property
    def has_linear(self) -> bool:
        return len(self.linear_teams) > 0

    @property
    def github_org(self) -> str | None:
        """Infer the GitHub org from discovered repos."""
        if not self.repos:
            return None
        orgs: dict[str, int] = {}
        for r in self.repos:
            org = r.split("/")[0]
            orgs[org] = orgs.get(org, 0) + 1
        return max(orgs, key=orgs.get) if orgs else None


def discover(config_dir: Path | None = None) -> DiscoveredConfig:
    """Run full discovery — repos, config files, env vars."""
    result = DiscoveredConfig()

    # Find workspace root
    result.workspace_root = find_workspace_root()

    # Resolve config dir (explicit > workspace/config)
    if config_dir:
        result.config_dir = config_dir.resolve()
    elif result.workspace_root and (result.workspace_root / "config").is_dir():
        result.config_dir = result.workspace_root / "config"

    # Resolve repos dir
    if result.workspace_root and (result.workspace_root / "repos").is_dir():
        result.repos_dir = result.workspace_root / "repos"

    # Discover repos from local checkouts
    if result.repos_dir:
        result.repos = discover_repos(result.repos_dir)

    # Parse config files
    if result.config_dir:
        api_path = result.config_dir / "linear_api"
        result.linear_api_key = read_linear_api_key(api_path)

        # Try live API discovery first, fall back to static reference file
        api_key = result.linear_api_key or os.environ.get("LINEAR_API_KEY", "")
        if api_key:
            live_teams, _org_id = discover_linear_teams_live(api_key)
            if live_teams:
                result.linear_teams = live_teams

        # Fall back to linear-reference.md if API discovery didn't work
        if not result.linear_teams:
            ref_path = result.config_dir / "linear-reference.md"
            result.linear_teams, result.linear_members = parse_linear_reference(ref_path)

    # Check env vars
    result.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    return result
