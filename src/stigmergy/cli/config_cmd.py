"""`stigmergy config show` and `stigmergy config set`."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stigmergy.cli.config_schema import StigmergyConfig
from stigmergy.cli.output import error, heading, info


CONFIG_PATH = Path(".stigmergy") / "config.yaml"


def _load_raw() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found. Run `stigmergy init` first.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_config() -> StigmergyConfig:
    return StigmergyConfig(**_load_raw())


def _resolve_path(data: dict, dotted: str) -> object:
    """Walk a dotted path like 'sources.github.repos' into a nested dict."""
    parts = dotted.split(".")
    current: object = data
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key '{part}' not found in config path '{dotted}'")
            current = current[part]
        else:
            raise KeyError(f"Cannot descend into non-dict at '{part}' in '{dotted}'")
    return current


def _set_path(data: dict, dotted: str, value: str) -> None:
    """Set a value at a dotted path, coercing the type to match existing value."""
    parts = dotted.split(".")
    current = data
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Key '{part}' not found in config path '{dotted}'")

    key = parts[-1]
    if key not in current:
        raise KeyError(f"Key '{key}' not found in config path '{dotted}'")

    existing = current[key]
    # Coerce type
    if isinstance(existing, bool):
        current[key] = value.lower() in ("true", "1", "yes")
    elif isinstance(existing, int):
        current[key] = int(value)
    elif isinstance(existing, float):
        current[key] = float(value)
    elif isinstance(existing, list):
        current[key] = [s.strip() for s in value.split(",")]
    else:
        current[key] = value


def run_config(args: argparse.Namespace) -> int:
    if args.config_action is None:
        error("Usage: stigmergy config {show,set}")
        return 1

    try:
        if args.config_action == "show":
            return _show(args.section)
        if args.config_action == "set":
            return _set(args.key, args.value)
    except (FileNotFoundError, KeyError) as exc:
        error(str(exc))
        return 1

    return 0


def _show(section: str | None) -> int:
    data = _load_raw()
    if section:
        data = _resolve_path(data, section)

    heading("stigmergy config")
    print(yaml.dump(data, default_flow_style=False, sort_keys=False).rstrip())
    return 0


def _set(key: str, value: str) -> int:
    data = _load_raw()
    _set_path(data, key, value)

    # Validate through Pydantic
    StigmergyConfig(**data)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    info(f"Set {key} = {value}")
    return 0
