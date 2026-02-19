"""`stigmergy init` — smart setup that auto-discovers repos and config files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from stigmergy.cli.config_schema import (
    AgentConfig,
    BudgetConfig,
    ContextConfig,
    FamiliarityWeightsConfig,
    GitHubSourceConfig,
    GrafanaSourceConfig,
    LLMConfig,
    LinearSourceConfig,
    LinearTeamConfig,
    OutputConfig,
    PipelineConfig,
    SlackSourceConfig,
    SourcesConfig,
    StigmergyConfig,
    WorkspaceConfig,
    default_config,
)
from stigmergy.cli.discovery import DiscoveredConfig, discover
from stigmergy.cli.output import heading, info, success, warn


CONFIG_DIR = ".stigmergy"
CONFIG_FILE = "config.yaml"


def _prompt(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {question}{suffix}: ").strip()
    return answer or default


def _prompt_bool(question: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = input(f"  {question} [{hint}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "true", "1")


def _prompt_list(question: str, default: str = "") -> list[str]:
    answer = _prompt(question, default)
    return [s.strip() for s in answer.split(",") if s.strip()]


def _show_discovery(disc: DiscoveredConfig) -> None:
    """Show what was auto-discovered."""
    if disc.workspace_root:
        info(f"Workspace: {disc.workspace_root}")
    if disc.repos_dir:
        info(f"  repos/   {len(disc.repos)} git repos" + (
            f" ({disc.github_org} org)" if disc.github_org else ""
        ))
    if disc.config_dir:
        parts = []
        if disc.has_linear:
            parts.append(f"{len(disc.linear_teams)} Linear teams")
        if disc.linear_api_key:
            parts.append("Linear API key")
        detail = ", ".join(parts) if parts else "(no recognized files)"
        info(f"  config/  {detail}")
    if disc.anthropic_api_key:
        info("  env      ANTHROPIC_API_KEY set")
    else:
        info("  env      ANTHROPIC_API_KEY not set (will use stub LLM)")


def run_init(args: argparse.Namespace) -> int:
    config_path = Path(CONFIG_DIR) / CONFIG_FILE

    if config_path.exists():
        if not _prompt_bool(f"{config_path} already exists. Overwrite?", default=False):
            info("Aborted.")
            return 0

    heading("stigmergy init")
    print()

    # ── Discovery ──────────────────────────────────────────────────
    config_dir = Path(args.config) if getattr(args, "config", None) else None
    repos_dir = Path(args.repos) if getattr(args, "repos", None) else None

    disc = discover(config_dir=config_dir)

    # Override repos dir if explicitly given
    if repos_dir:
        from stigmergy.cli.discovery import discover_repos
        disc.repos_dir = repos_dir.resolve()
        disc.repos = discover_repos(disc.repos_dir)

    _show_discovery(disc)
    print()

    # ── Goal ───────────────────────────────────────────────────────
    goal = _prompt(
        "What should stigmergy monitor?",
        "Monitor engineering signals for the platform team",
    )

    # ── GitHub ─────────────────────────────────────────────────────
    print()
    info("--- Sources ---")
    gh_enabled = _prompt_bool("Enable GitHub?", default=True)
    gh_repos: list[str] = []
    gh_mode = "mock"

    if gh_enabled:
        if disc.has_repos:
            org = disc.github_org or "unknown"
            info(f"  Discovered {len(disc.repos)} repos ({org} org)")
            use_discovered = _prompt_bool("Use discovered repos?", default=True)
            if use_discovered:
                gh_repos = disc.repos
            else:
                gh_repos = _prompt_list(
                    "GitHub repos (comma-separated)",
                    ", ".join(disc.repos[:3]),
                )
        else:
            defaults = default_config().sources.github
            info(f"  No repos/ directory found. Default: {len(defaults.repos)} repos")
            use_defaults = _prompt_bool("Use default repos?", default=True)
            if use_defaults:
                gh_repos = defaults.repos
            else:
                gh_repos = _prompt_list(
                    "GitHub repos (comma-separated)",
                    "acme-org/backend, acme-org/pricing",
                )

        gh_mode = _prompt("GitHub mode (live/mock)", "live")

    # ── Linear ─────────────────────────────────────────────────────
    linear_enabled = _prompt_bool("Enable Linear?", default=True)
    linear_mode = "mock"
    linear_org_id = default_config().sources.linear.organization_id
    linear_teams: list[LinearTeamConfig] = []

    if linear_enabled:
        linear_mode = _prompt("Linear mode (live/mock)", "live")

        # Try live API discovery first — this is the source of truth
        from stigmergy.cli.discovery import discover_linear_teams_live

        api_key = disc.linear_api_key or os.environ.get("LINEAR_API_KEY", "")
        live_teams, live_org_id = [], ""
        if api_key:
            info("  Querying Linear API for teams...")
            live_teams, live_org_id = discover_linear_teams_live(api_key)

        if live_teams:
            if live_org_id:
                linear_org_id = live_org_id
            info(f"  Found {len(live_teams)} teams in Linear organization:")
            for i, t in enumerate(live_teams, 1):
                info(f"    [{i}] {t.name} ({t.id[:8]}...)")

            # Let user deselect teams they don't want
            exclude = _prompt(
                "Teams to EXCLUDE (comma-separated numbers, or blank for all)",
                "",
            )
            if exclude.strip():
                exclude_indices = set()
                for part in exclude.split(","):
                    part = part.strip()
                    if part.isdigit():
                        exclude_indices.add(int(part))
                linear_teams = [
                    t for i, t in enumerate(live_teams, 1)
                    if i not in exclude_indices
                ]
                excluded_names = [
                    t.name for i, t in enumerate(live_teams, 1)
                    if i in exclude_indices
                ]
                if excluded_names:
                    info(f"  Excluded: {', '.join(excluded_names)}")
            else:
                linear_teams = live_teams
            info(f"  Selected {len(linear_teams)} teams")
        elif disc.has_linear:
            # Fall back to config file discovery (linear-reference.md)
            info(f"  Discovered {len(disc.linear_teams)} Linear teams from config:")
            for t in disc.linear_teams:
                info(f"    {t.name} ({t.id[:8]}...)")
            use_discovered = _prompt_bool("Use discovered teams?", default=True)
            if use_discovered:
                linear_teams = disc.linear_teams
            else:
                linear_teams = _prompt_linear_teams_manual(linear_org_id)
        else:
            defaults = default_config().sources.linear
            info(f"  Org: ({defaults.organization_id[:8]}...)")
            for t in defaults.teams:
                info(f"  Team: {t.name} ({t.id[:8]}...)")
            use_defaults = _prompt_bool("Use default org + teams?", default=True)
            if use_defaults:
                linear_teams = defaults.teams
            else:
                linear_teams = _prompt_linear_teams_manual(defaults.organization_id)

        if not api_key:
            warn("  No Linear API key found. Set LINEAR_API_KEY when linear goes live.")

    # ── Grafana ────────────────────────────────────────────────────
    grafana_enabled = _prompt_bool("Enable Grafana?", default=False)
    grafana_mode = "mock"
    grafana_base_url = ""
    grafana_dashboards: list[str] = []
    grafana_tempo_services: list[str] = []

    if grafana_enabled:
        grafana_mode = _prompt("Grafana mode (live/mock)", "live")
        grafana_base_url = _prompt("Grafana base URL", "https://grafana.example.com")
        grafana_dashboards = _prompt_list(
            "Dashboard UIDs for metric queries (comma-separated, or blank)",
            "",
        )
        grafana_tempo_services = _prompt_list(
            "Tempo service names to query (comma-separated, or blank)",
            "",
        )
        grafana_api_key = os.environ.get("GRAFANA_API_KEY", "")
        if not grafana_api_key:
            warn("  No GRAFANA_API_KEY found. Set it when Grafana goes live.")

    # ── LLM ────────────────────────────────────────────────────────
    print()
    info("--- LLM ---")
    if disc.anthropic_api_key:
        default_provider = "anthropic"
        info("  ANTHROPIC_API_KEY detected — defaulting to anthropic provider")
    else:
        default_provider = "stub"

    provider = _prompt(f"LLM provider (stub/anthropic)", default_provider)

    if provider == "anthropic" and not disc.anthropic_api_key:
        warn("ANTHROPIC_API_KEY not set. You'll need it before running.")
        warn("  export ANTHROPIC_API_KEY=your-key")
        warn("Or switch to stub mode later: stigmergy config set llm.provider stub")

    model = _prompt("Model", "claude-haiku-4-5-20251001")

    # ── Budget ─────────────────────────────────────────────────────
    print()
    info("--- Budget ---")
    daily_cap = float(_prompt("Daily cap (USD)", "5.00"))
    hourly_cap = float(_prompt("Hourly cap (USD)", "1.00"))

    # ── Build config ───────────────────────────────────────────────
    workspace = WorkspaceConfig(
        root=str(disc.workspace_root) if disc.workspace_root else "",
        config_dir=str(disc.config_dir) if disc.config_dir else "",
        repos_dir=str(disc.repos_dir) if disc.repos_dir else "",
    )

    config = StigmergyConfig(
        goal=goal,
        workspace=workspace,
        sources=SourcesConfig(
            github=GitHubSourceConfig(
                enabled=gh_enabled,
                mode=gh_mode,
                repos=gh_repos,
                poll_interval=300,
            ),
            linear=LinearSourceConfig(
                enabled=linear_enabled,
                mode=linear_mode,
                organization_id=linear_org_id,
                teams=linear_teams,
                poll_interval=300,
            ),
            slack=SlackSourceConfig(enabled=False),
            grafana=GrafanaSourceConfig(
                enabled=grafana_enabled,
                mode=grafana_mode,
                base_url=grafana_base_url,
                dashboards=grafana_dashboards,
                tempo_services=grafana_tempo_services,
            ),
        ),
        contexts=default_config().contexts,
        agents=default_config().agents,
        llm=LLMConfig(provider=provider, model=model),
        budget=BudgetConfig(
            daily_cap_usd=daily_cap,
            hourly_cap_usd=hourly_cap,
        ),
        pipeline=default_config().pipeline,
        output=default_config().output,
    )

    # ── Write ──────────────────────────────────────────────────────
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = config.model_dump()
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print()
    success(f"Config written to {config_path}")
    info(f"  {len(gh_repos)} GitHub repos, {len(linear_teams)} Linear teams, provider={provider}")
    info("Run `stigmergy run --once` to process signals.")
    return 0


def _prompt_linear_teams_manual(default_org_id: str) -> list[LinearTeamConfig]:
    """Fallback manual entry for Linear teams."""
    teams: list[LinearTeamConfig] = []
    while True:
        name = _prompt("Team name (blank to finish)", "")
        if not name:
            break
        tid = _prompt(f"  Team ID for {name}", "")
        if tid:
            teams.append(LinearTeamConfig(name=name, id=tid))
    return teams
