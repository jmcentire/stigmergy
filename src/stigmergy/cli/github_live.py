"""Live GitHub adapter using `gh` CLI subprocess.

Converts `gh pr list` / `gh issue list` JSON into Signal objects.
Requires `gh` to be installed and authenticated.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Callable

from stigmergy.primitives.signal import Signal, SignalSource


BOT_AUTHORS = frozenset({
    "github-actions",
    "github-actions[bot]",
    "dependabot",
    "dependabot[bot]",
    "app/dependabot",
    "renovate",
    "renovate[bot]",
    "vercel[bot]",
    "codecov[bot]",
    "snyk[bot]",
    "actions-user",
    "cursor",
    "linear",
    "copilot-pull-request-reviewer",
})


def _is_bot(author: str) -> bool:
    """Check if an author is a known bot/automation account."""
    return author.lower() in BOT_AUTHORS or author.endswith("[bot]")


class LiveGitHubAdapter:
    """Real GitHub adapter via `gh` CLI subprocess."""

    def __init__(
        self,
        repos: list[str] | None = None,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
        include_bots: bool = False,
    ) -> None:
        self._repos = repos or []
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False
        self._progress_callback = progress_callback
        self._error_callback = error_callback
        self._include_bots = include_bots
        self._bot_skipped = 0
        # Signal-level dedup: GitHub search only supports date precision
        # (updated:>=YYYY-MM-DD), so every poll on the same day returns
        # the same items. Track yielded signal keys to avoid re-processing.
        self._seen_keys: set[str] = set()

    @property
    def bot_skipped(self) -> int:
        return self._bot_skipped

    def _signal_key(self, repo: str, event_type: str, number: int, sub_id: str = "") -> str:
        """Unique key for signal-level dedup across poll cycles."""
        return f"{repo}:{event_type}:{number}:{sub_id}"

    def _is_new(self, key: str) -> bool:
        """Check if this signal key is new; register it if so."""
        if key in self._seen_keys:
            return False
        self._seen_keys.add(key)
        return True

    async def connect(self) -> None:
        """Verify gh is authenticated."""
        proc = await asyncio.create_subprocess_exec(
            "gh", "auth", "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ConnectionError(
                f"gh auth failed (exit {proc.returncode}). "
                f"Run `gh auth login` first.\n{stderr.decode()}"
            )
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        """Fetch PRs, issues, and check runs updated since `since` from all repos."""
        since_str = since.strftime("%Y-%m-%d")
        total_repos = len(self._repos)
        self._bot_skipped = 0

        for i, repo in enumerate(self._repos, 1):
            repo_count = 0
            # Fetch PRs
            async for signal in self._fetch_prs(repo, since_str):
                repo_count += 1
                yield signal

            # Fetch issues
            async for signal in self._fetch_issues(repo, since_str):
                repo_count += 1
                yield signal

            # Fetch check run failures (skip entirely if bots filtered —
            # all check_run signals are authored by github-actions)
            if self._include_bots:
                async for signal in self._fetch_check_runs(repo):
                    repo_count += 1
                    yield signal

            if self._progress_callback:
                self._progress_callback(repo, i, total_repos, repo_count)

    async def _fetch_prs(self, repo: str, since_str: str) -> AsyncIterator[Signal]:
        cmd = [
            "gh", "pr", "list",
            "--repo", repo,
            "--json", "title,author,number,createdAt,body,url,state,additions,deletions,files,"
                      "comments,reviews,reviewDecision,mergedAt,mergedBy,closedAt,isDraft,headRefName,assignees",
            "--search", f"updated:>={since_str}",
            "--limit", "200",
        ]
        data = await self._run_gh(cmd)
        if data is None:
            return

        for pr in data:
            author = pr.get("author", {}).get("login", "unknown")
            if not self._include_bots and _is_bot(author):
                self._bot_skipped += 1
                continue
            number = pr.get("number", 0)
            title = pr.get("title", "")
            body = (pr.get("body") or "")[:2000]
            state = pr.get("state", "OPEN")
            content = f"PR #{number}: {title}"
            if body:
                content += f"\n{body}"

            created = pr.get("createdAt", "")
            ts = _parse_gh_timestamp(created)

            files = pr.get("files", [])
            file_paths = [f.get("path", "") for f in files] if files else []

            assignees = pr.get("assignees", [])
            assignee_logins = [a.get("login", "") for a in assignees if a] if assignees else []

            metadata = {
                "event_type": "pull_request",
                "pr_number": number,
                "state": state,
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "files_changed": file_paths[:10],
                "url": pr.get("url", ""),
                "review_decision": pr.get("reviewDecision", ""),
                "is_draft": pr.get("isDraft", False),
                "branch": pr.get("headRefName", ""),
            }
            if assignee_logins:
                metadata["assignees"] = assignee_logins
            merged_at = pr.get("mergedAt")
            if merged_at:
                metadata["merged_at"] = merged_at
                merged_by = pr.get("mergedBy", {})
                if merged_by:
                    metadata["merged_by"] = merged_by.get("login", "")

            # Dedup: PR keyed on repo + number + state (state change = new signal)
            pr_key = self._signal_key(repo, "pr", number, state)
            if not self._is_new(pr_key):
                # Still check comments/reviews — PR body unchanged but
                # new comments may have been added since last poll.
                pass
            else:
                yield Signal(
                    content=content,
                    source=SignalSource.GITHUB,
                    channel=repo,
                    author=author,
                    timestamp=ts,
                    metadata=metadata,
                )

            # PR comments as separate signals
            for comment in (pr.get("comments") or []):
                comment_body = (comment.get("body") or "").strip()
                if not comment_body:
                    continue
                comment_author = comment.get("author", {}).get("login", author)
                if not self._include_bots and _is_bot(comment_author):
                    self._bot_skipped += 1
                    continue
                # Dedup: comment keyed on createdAt (immutable once posted)
                comment_created = comment.get("createdAt", "")
                comment_key = self._signal_key(repo, "pr_comment", number, comment_created)
                if not self._is_new(comment_key):
                    continue
                comment_ts = _parse_gh_timestamp(comment_created)
                yield Signal(
                    content=comment_body[:2000],
                    source=SignalSource.GITHUB,
                    channel=repo,
                    author=comment_author,
                    timestamp=comment_ts,
                    metadata={
                        "event_type": "pr_comment",
                        "pr_number": number,
                        "parent_url": pr.get("url", ""),
                    },
                )

            # PR reviews as separate signals
            for review in (pr.get("reviews") or []):
                review_body = (review.get("body") or "").strip()
                review_state = review.get("state", "COMMENTED")
                review_content = f"Review ({review_state})"
                if review_body:
                    review_content += f": {review_body[:2000]}"
                reviewer = review.get("author", {}).get("login", "unknown")
                if not self._include_bots and _is_bot(reviewer):
                    self._bot_skipped += 1
                    continue
                # Dedup: review keyed on submittedAt (immutable once submitted)
                review_created = review.get("submittedAt", review.get("createdAt", ""))
                review_key = self._signal_key(repo, "review", number, review_created)
                if not self._is_new(review_key):
                    continue
                review_ts = _parse_gh_timestamp(review_created)
                yield Signal(
                    content=review_content,
                    source=SignalSource.GITHUB,
                    channel=repo,
                    author=reviewer,
                    timestamp=review_ts,
                    metadata={
                        "event_type": "review",
                        "review_state": review_state,
                        "pr_number": number,
                        "parent_url": pr.get("url", ""),
                    },
                )

    async def _fetch_issues(self, repo: str, since_str: str) -> AsyncIterator[Signal]:
        cmd = [
            "gh", "issue", "list",
            "--repo", repo,
            "--json", "title,author,number,createdAt,body,url,state,labels,comments,assignees,milestone,closedAt",
            "--search", f"updated:>={since_str}",
            "--limit", "200",
        ]
        data = await self._run_gh(cmd)
        if data is None:
            return

        for issue in data:
            author = issue.get("author", {}).get("login", "unknown")
            if not self._include_bots and _is_bot(author):
                self._bot_skipped += 1
                continue
            number = issue.get("number", 0)
            title = issue.get("title", "")
            body = (issue.get("body") or "")[:2000]
            content = f"Issue #{number}: {title}"
            if body:
                content += f"\n{body}"

            created = issue.get("createdAt", "")
            ts = _parse_gh_timestamp(created)

            labels = [l.get("name", "") for l in (issue.get("labels") or [])]
            assignees = issue.get("assignees", [])
            assignee_logins = [a.get("login", "") for a in assignees if a] if assignees else []

            metadata = {
                "event_type": "issue",
                "issue_number": number,
                "state": issue.get("state", "OPEN"),
                "labels": labels,
                "url": issue.get("url", ""),
            }
            if assignee_logins:
                metadata["assignees"] = assignee_logins
            milestone = issue.get("milestone")
            if milestone:
                metadata["milestone"] = milestone.get("title", "")
            closed_at = issue.get("closedAt")
            if closed_at:
                metadata["closed_at"] = closed_at

            # Dedup: issue keyed on repo + number + state
            issue_state = issue.get("state", "OPEN")
            issue_key = self._signal_key(repo, "issue", number, issue_state)
            if not self._is_new(issue_key):
                pass  # Still check comments below
            else:
                yield Signal(
                    content=content,
                    source=SignalSource.GITHUB,
                    channel=repo,
                    author=author,
                    timestamp=ts,
                    metadata=metadata,
                )

            # Issue comments as separate signals
            for comment in (issue.get("comments") or []):
                comment_body = (comment.get("body") or "").strip()
                if not comment_body:
                    continue
                comment_author = comment.get("author", {}).get("login", author)
                if not self._include_bots and _is_bot(comment_author):
                    self._bot_skipped += 1
                    continue
                comment_created = comment.get("createdAt", "")
                comment_key = self._signal_key(repo, "issue_comment", number, comment_created)
                if not self._is_new(comment_key):
                    continue
                comment_ts = _parse_gh_timestamp(comment_created)
                yield Signal(
                    content=comment_body[:2000],
                    source=SignalSource.GITHUB,
                    channel=repo,
                    author=comment_author,
                    timestamp=comment_ts,
                    metadata={
                        "event_type": "issue_comment",
                        "issue_number": number,
                        "parent_url": issue.get("url", ""),
                    },
                )

    async def _fetch_check_runs(self, repo: str) -> AsyncIterator[Signal]:
        """Fetch recent CI failures as signals."""
        cmd = [
            "gh", "run", "list",
            "--repo", repo,
            "--json", "name,status,conclusion,createdAt,headBranch,url",
            "--limit", "20",
        ]
        data = await self._run_gh(cmd)
        if data is None:
            return

        failure_conclusions = {"failure", "timed_out", "cancelled"}
        for run in data:
            conclusion = run.get("conclusion", "")
            if conclusion not in failure_conclusions:
                continue

            name = run.get("name", "unknown")
            branch = run.get("headBranch", "unknown")
            content = f"CI {conclusion}: {name} on {branch}"

            ts = _parse_gh_timestamp(run.get("createdAt", ""))

            # Dedup: check run keyed on created timestamp + name
            run_key = self._signal_key(repo, "check_run", 0, f"{name}:{run.get('createdAt', '')}")
            if not self._is_new(run_key):
                continue

            yield Signal(
                content=content,
                source=SignalSource.GITHUB,
                channel=repo,
                author="github-actions",
                timestamp=ts,
                metadata={
                    "event_type": "check_run",
                    "conclusion": conclusion,
                    "run_name": name,
                    "branch": branch,
                    "url": run.get("url", ""),
                },
            )

    async def _run_gh(self, cmd: list[str], *, max_retries: int = 3) -> list[dict] | None:
        """Run a gh command and parse JSON output with retry + backoff."""
        repo = cmd[cmd.index("--repo") + 1] if "--repo" in cmd else "unknown"

        for attempt in range(max_retries + 1):
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=60,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=60,
                )
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    delay = 2 ** attempt
                    if self._error_callback:
                        self._error_callback(repo, f"timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                if self._error_callback:
                    self._error_callback(repo, "timeout after all retries")
                return None

            if proc.returncode != 0:
                err_msg = stderr.decode().strip()
                # Rate limit or server error — retry with backoff
                is_transient = any(s in err_msg.lower() for s in ("rate limit", "api rate", "502", "503", "timeout"))
                if is_transient and attempt < max_retries:
                    delay = 2 ** attempt
                    if self._error_callback:
                        self._error_callback(repo, f"{err_msg[:80]}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                if self._error_callback:
                    self._error_callback(repo, err_msg)
                return None

            try:
                return json.loads(stdout.decode())
            except json.JSONDecodeError:
                return None

        return None


def _parse_gh_timestamp(ts: str) -> datetime:
    """Parse GitHub ISO timestamp."""
    if not ts:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
