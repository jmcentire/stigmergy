"""`stigmergy check` — quick connectivity tests for each source.

Exercises connect() on each enabled adapter, reports what it finds.
No backfill, no LLM calls, no tokens spent.
"""

from __future__ import annotations

import asyncio
import argparse
import os
from difflib import get_close_matches
from pathlib import Path

import yaml

from stigmergy.cli.config_schema import StigmergyConfig
from stigmergy.cli.output import heading, info, warn, error, success, separator

CONFIG_PATH = Path(".stigmergy") / "config.yaml"


def _load_config() -> StigmergyConfig:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found. Run `stigmergy init` first.")
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return StigmergyConfig(**data)


async def _check_github(config: StigmergyConfig) -> bool:
    """Check GitHub connectivity via gh CLI."""
    if not config.sources.github.enabled:
        info("  github: disabled in config")
        return True

    from stigmergy.cli.github_live import LiveGitHubAdapter
    adapter = LiveGitHubAdapter(repos=config.sources.github.repos)
    try:
        await adapter.connect()
    except ConnectionError as e:
        error(f"  github: {e}")
        return False

    repo_count = len(config.sources.github.repos)
    success(f"  github: authenticated, {repo_count} repos configured")
    return True


async def _check_linear(config: StigmergyConfig) -> bool:
    """Check Linear connectivity via GraphQL API."""
    if not config.sources.linear.enabled:
        info("  linear: disabled in config")
        return True

    from stigmergy.cli.linear_live import LiveLinearAdapter
    teams = [{"name": t.name, "id": t.id} for t in config.sources.linear.teams]
    api_key = config.sources.linear.api_key or os.environ.get("LINEAR_API_KEY", "")
    adapter = LiveLinearAdapter(teams=teams, api_key=api_key)
    try:
        await adapter.connect()
    except ConnectionError as e:
        error(f"  linear: {e}")
        return False

    team_names = ", ".join(t.name for t in config.sources.linear.teams)
    success(f"  linear: authenticated, teams: {team_names}")
    return True


async def _check_slack(config: StigmergyConfig, verbose: bool = False) -> bool:
    """Check Slack connectivity and channel resolution."""
    if not config.sources.slack.enabled:
        if config.sources.slack.channels:
            ch_count = len(config.sources.slack.channels)
            info(f"  slack: disabled in config ({ch_count} channels configured)")
        else:
            info("  slack: disabled in config")
        return True

    from stigmergy.cli.slack_live import LiveSlackAdapter

    token = config.sources.slack.bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        error("  slack: SLACK_BOT_TOKEN not set")
        return False

    adapter = LiveSlackAdapter(
        channels=config.sources.slack.channels,
        bot_token=token,
    )

    # 1. Verify auth
    try:
        result = await adapter._api("auth.test")
        if result is None or not result.get("ok"):
            err = result.get("error", "unknown") if result else "no response"
            error(f"  slack: auth failed — {err}")
            return False
        bot_name = result.get("user", "unknown")
        bot_id = result.get("user_id", "?")
        success(f"  slack: authenticated as @{bot_name} ({bot_id})")
    except Exception as e:
        error(f"  slack: auth error — {e}")
        return False

    # 2. Get channels the bot is a member of (fast — 1-2 API calls)
    cursor = None
    bot_channels: dict[str, str] = {}
    while True:
        params: dict = {
            "types": "public_channel,private_channel",
            "limit": 200,
            "exclude_archived": "true",
        }
        if cursor:
            params["cursor"] = cursor

        result = await adapter._api("users.conversations", params)
        if result is None or not result.get("ok"):
            err = result.get("error", "unknown") if result else "no response"
            warn(f"  slack: users.conversations failed — {err}")
            break

        for ch in result.get("channels", []):
            bot_channels[ch["name"]] = ch["id"]

        metadata = result.get("response_metadata", {})
        next_cursor = metadata.get("next_cursor", "")
        if next_cursor:
            cursor = next_cursor
        else:
            break

    info(f"  slack: bot is a member of {len(bot_channels)} channels")

    # 3. Match configured channels against bot membership
    configured = config.sources.slack.channels or []
    found = []
    missing = []
    for name in configured:
        clean = name.lstrip("#")
        if clean in bot_channels:
            found.append(clean)
        else:
            missing.append(clean)

    if found:
        success(f"  slack: {len(found)}/{len(configured)} configured channels accessible")
        for ch in sorted(found):
            info(f"         + {ch}  ({bot_channels[ch]})")

    if missing:
        warn(f"  slack: {len(missing)} configured channels NOT accessible:")
        # Fuzzy match against bot's channels
        bot_names = list(bot_channels.keys())
        for ch in sorted(missing):
            matches = get_close_matches(ch, bot_names, n=2, cutoff=0.6)
            if matches:
                suggestions = ", ".join(matches)
                warn(f"         - {ch}  (similar bot channel: {suggestions})")
            else:
                warn(f"         - {ch}")
        print()
        warn("  These channels either don't exist or the bot hasn't been invited.")
        warn("  Use /invite @Stigmergy in each Slack channel to add the bot.")

    # 4. Verbose: show all bot channels for reference
    if verbose and bot_channels:
        print()
        info(f"  All {len(bot_channels)} channels bot is a member of:")
        for name in sorted(bot_channels.keys()):
            marker = " <-- configured" if name in [n.lstrip("#") for n in configured] else ""
            info(f"    {name}  ({bot_channels[name]}){marker}")

    # 5. Quick history test on first found channel
    if found:
        test_ch = found[0]
        test_id = bot_channels[test_ch]
        result = await adapter._api("conversations.history", {
            "channel": test_id,
            "limit": 1,
        })
        if result and result.get("ok"):
            success(f"  slack: history access verified (#{test_ch})")
        else:
            err = result.get("error", "unknown") if result else "no response"
            warn(f"  slack: history access failed for #{test_ch} — {err}")

    return len(missing) == 0


async def _check_grafana(config: StigmergyConfig) -> bool:
    """Check Grafana connectivity via API."""
    if not config.sources.grafana.enabled:
        info("  grafana: disabled in config")
        return True

    from stigmergy.cli.grafana_live import LiveGrafanaAdapter
    api_key = config.sources.grafana.api_key or os.environ.get("GRAFANA_API_KEY", "")
    adapter = LiveGrafanaAdapter(
        base_url=config.sources.grafana.base_url,
        api_key=api_key,
        dashboards=config.sources.grafana.dashboards,
        tempo_services=config.sources.grafana.tempo_services,
    )
    try:
        await adapter.connect()
    except ConnectionError as e:
        error(f"  grafana: {e}")
        return False

    parts = []
    if adapter._tempo_uid:
        parts.append(f"Tempo UID: {adapter._tempo_uid}")
    if config.sources.grafana.dashboards:
        parts.append(f"{len(config.sources.grafana.dashboards)} dashboards")
    if config.sources.grafana.tempo_services:
        parts.append(f"{len(config.sources.grafana.tempo_services)} Tempo services")
    detail = ", ".join(parts) if parts else "no dashboards/services configured"
    success(f"  grafana: authenticated, {detail}")
    return True


async def _run_checks(sources: list[str] | None, config: StigmergyConfig, verbose: bool = False) -> int:
    """Run connectivity checks for specified sources (or all)."""
    heading("stigmergy check")
    separator()

    check_all = not sources
    ok = True

    if check_all or "github" in sources:
        if not await _check_github(config):
            ok = False
        print()

    if check_all or "linear" in sources:
        if not await _check_linear(config):
            ok = False
        print()

    if check_all or "slack" in sources:
        if not await _check_slack(config, verbose=verbose):
            ok = False
        print()

    if check_all or "grafana" in sources:
        if not await _check_grafana(config):
            ok = False
        print()

    separator()
    if ok:
        success("All checks passed.")
    else:
        warn("Some checks failed — see above.")
    return 0 if ok else 1


def run_check(args: argparse.Namespace) -> int:
    """Entry point for `stigmergy check`."""
    try:
        config = _load_config()
    except FileNotFoundError as e:
        error(str(e))
        return 1

    sources = []
    if getattr(args, "github", False):
        sources.append("github")
    if getattr(args, "linear", False):
        sources.append("linear")
    if getattr(args, "slack", False):
        sources.append("slack")
    if getattr(args, "grafana", False):
        sources.append("grafana")

    verbose = getattr(args, "verbose", False)
    return asyncio.run(_run_checks(sources or None, config, verbose=verbose))
