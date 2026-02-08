"""stigmergy CLI — entry point and command dispatch."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


_EPILOG = """\
quick start:
  stigmergy init                        # interactive setup
  stigmergy run --once                  # process mock signals (no API key needed)
  stigmergy run --once --live           # process real GitHub data (requires: gh auth login)

llm providers:
  stub (default)   No API calls. Deterministic heuristic-only mode. Free.
  anthropic        Real LLM assessments via Haiku. Requires:
                     pip install -e ".[cli]"
                     export ANTHROPIC_API_KEY=your-key
                     stigmergy config set llm.provider anthropic

live sources:
  GitHub           Uses the gh CLI. Run `gh auth login` first, then:
                     stigmergy run --once --live
  Linear           Uses LINEAR_API_KEY env var or config
  Slack            Uses SLACK_BOT_TOKEN env var or config

connectivity check:
  stigmergy check              # verify all sources (no tokens spent)
  stigmergy check --slack      # Slack only (shows channel resolution)

budget:
  Default caps: $5/day, $1/hour. When exhausted, falls back to heuristic-only.
  Adjust with: stigmergy config set budget.daily_cap_usd 10.00
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stigmergy",
        description="Stigmergic signal processing for organizational knowledge systems",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # stigmergy init
    init_parser = sub.add_parser(
        "init",
        help="Interactive setup — creates .stigmergy/config.yaml",
        description="Walk through an interactive setup to configure sources, LLM provider, "
        "contexts, agents, and budget caps. Auto-discovers repos and config files "
        "from the workspace. Writes .stigmergy/config.yaml.",
    )
    init_parser.add_argument(
        "--config",
        metavar="DIR",
        default=None,
        help="Path to config directory containing linear_api, linear-reference.md, etc.",
    )
    init_parser.add_argument(
        "--repos",
        metavar="DIR",
        default=None,
        help="Path to directory containing git repo checkouts (auto-detected if omitted)",
    )

    # stigmergy config
    config_parser = sub.add_parser(
        "config",
        help="View or update configuration",
        description="Read or modify .stigmergy/config.yaml. Use dotted paths to "
        "target nested values (e.g. sources.github.repos, budget.daily_cap_usd).",
    )
    config_sub = config_parser.add_subparsers(dest="config_action")
    show_parser = config_sub.add_parser("show", help="Show config (or a section)")
    show_parser.add_argument("section", nargs="?", default=None, help="Dotted config path (e.g. sources.github)")
    set_parser = config_sub.add_parser("set", help="Update a config value")
    set_parser.add_argument("key", help="Dotted config path (e.g. budget.daily_cap_usd)")
    set_parser.add_argument("value", help="New value")

    # stigmergy run
    run_parser = sub.add_parser(
        "run",
        help="Start processing signals",
        description="Bootstrap the pipeline from config, connect sources, and process signals. "
        "Runs in continuous poll mode by default (Ctrl+C to stop). "
        "Use --once for a single batch. Use --live to fetch real data from GitHub via `gh`.",
        epilog="examples:\n"
        "  stigmergy run --once              # one batch, mock data, exit\n"
        "  stigmergy run --once --live       # one batch, real GitHub data\n"
        "  stigmergy run --live              # continuous polling (Ctrl+C to stop)\n"
        "  stigmergy run                     # continuous with mock data\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument("--once", action="store_true", help="Process one batch then exit")
    run_parser.add_argument("--clean", action="store_true", help="Clear learned state before running (fresh start)")
    run_parser.add_argument(
        "--live",
        action="store_true",
        help="Use live source adapters instead of mocks (requires `gh auth login` for GitHub)",
    )
    run_parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use legacy centralized pipeline instead of mesh routing",
    )
    run_parser.add_argument(
        "--since",
        type=int,
        default=None,
        metavar="DAYS",
        help="Backfill this many days (default: 7, or since last run)",
    )

    # stigmergy check
    check_parser = sub.add_parser(
        "check",
        help="Quick connectivity test for sources (no tokens spent)",
        description="Verify that each enabled source adapter can authenticate and resolve "
        "its targets (repos, teams, channels). No backfill, no LLM calls.",
        epilog="examples:\n"
        "  stigmergy check                 # check all enabled sources\n"
        "  stigmergy check --slack         # check Slack only\n"
        "  stigmergy check --github        # check GitHub only\n"
        "  stigmergy check --linear        # check Linear only\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check_parser.add_argument("--github", action="store_true", help="Check GitHub only")
    check_parser.add_argument("--linear", action="store_true", help="Check Linear only")
    check_parser.add_argument("--slack", action="store_true", help="Check Slack only")
    check_parser.add_argument("--verbose", "-v", action="store_true", help="Show all visible channels (useful for debugging name mismatches)")

    # stigmergy status
    sub.add_parser(
        "status",
        help="Show pipeline state from last run",
        description="Display the summary from the most recent run, including signal counts, "
        "actions taken, and persisted context IDs.",
    )

    # stigmergy feedback
    feedback_parser = sub.add_parser(
        "feedback",
        help="Provide feedback on a surfaced finding (calibrates attention model)",
        description="Record whether you already knew about a surfaced finding. "
        "This calibrates the per-person attention model so future runs better "
        "predict what you already know.",
    )
    feedback_parser.add_argument(
        "--finding", required=True, metavar="HASH",
        help="Finding hash (shown in surfacing output)",
    )
    feedback_parser.add_argument(
        "--response", required=True, choices=["already_knew", "had_no_idea"],
        help="Whether you already knew about this finding",
    )
    feedback_parser.add_argument(
        "--person", default=None, metavar="ID",
        help="Person ID (defaults to first target_person in config)",
    )

    return parser


def _setup_logging() -> None:
    """Configure logging for beta — file at INFO, stderr at WARNING.

    Override with STIGMERGY_LOG_LEVEL or LOGLEVEL env var.
    Log file: .stigmergy/run.log (rotated by overwrite each run).
    """
    env_level = os.environ.get(
        "STIGMERGY_LOG_LEVEL",
        os.environ.get("LOGLEVEL", ""),
    ).upper()

    file_level = getattr(logging, env_level, None) or logging.INFO
    console_level = getattr(logging, env_level, None) or logging.WARNING

    root = logging.getLogger()
    root.setLevel(min(file_level, console_level))

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # File handler — always INFO+ (captures LLM failures, circuit breaker, etc.)
    log_dir = Path(".stigmergy")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log", mode="w")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler — WARNING+ by default (only important stuff)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set up logging before any command runs
    if args.command == "run":
        _setup_logging()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "init":
        from stigmergy.cli.init_cmd import run_init
        return run_init(args)

    if args.command == "config":
        from stigmergy.cli.config_cmd import run_config
        return run_config(args)

    if args.command == "check":
        from stigmergy.cli.check_cmd import run_check
        return run_check(args)

    if args.command == "run":
        from stigmergy.cli.run_cmd import run_pipeline
        return run_pipeline(args)

    if args.command == "status":
        from stigmergy.cli.run_cmd import run_status
        return run_status()

    if args.command == "feedback":
        return _run_feedback(args)

    parser.print_help()
    return 0


def _run_feedback(args) -> int:
    """Record attention model feedback for a finding."""
    from pathlib import Path

    from stigmergy.attention.model import AttentionModel
    from stigmergy.cli.config_cmd import _load_config

    config = _load_config()
    if config is None:
        print("No config found. Run `stigmergy init` first.")
        return 1

    if not config.attention.enabled:
        print("Attention model not enabled. Set attention.enabled: true in config.")
        return 1

    model = AttentionModel()
    attn_path = Path(".stigmergy/attention.json")
    model.load(attn_path)

    person_id = args.person
    if person_id is None:
        if config.attention.target_persons:
            person_id = config.attention.target_persons[0]
        else:
            print("No --person specified and no target_persons in config.")
            return 1

    model.record_feedback(person_id, args.finding, args.response)
    model.save(attn_path)

    state = model.get_state(person_id)
    rate = state.attention_base_rate if state else 0.5
    print(f"Recorded '{args.response}' for {person_id}. Base rate: {rate:.3f}")
    return 0


def cli() -> None:
    """Entry point for the console script."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
