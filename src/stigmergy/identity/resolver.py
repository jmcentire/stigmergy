"""Person identity resolution — canonical IDs from any alias.

Refactored from graph/identity.py. The same person may appear under
multiple identifiers: work email, personal email, firstname.lastname,
@SlackHandle, github-handle. The resolver unifies these to a single
canonical ID (email prefix, lowercase).

Supports multi-source resolution: static CSV/markdown files and live
API providers (GitHub, Linear, Slack). Providers are pluggable via
the IdentityProvider protocol.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from stigmergy.identity.provider import IdentityProvider, PersonRecord

logger = logging.getLogger(__name__)


@dataclass
class PersonProfile:
    """Canonical profile for a team member."""

    canonical_id: str
    name: str
    email: str
    github_handle: str = ""
    slack_handle: str = ""
    team: str = ""
    linear_uuid: str = ""


class IdentityResolver:
    """Canonical person ID from any alias.

    Aliases include: work emails, personal emails, GitHub handles,
    Slack handles, Linear UUIDs, and author strings from signals.
    """

    def __init__(
        self,
        providers: list[IdentityProvider] | None = None,
        store: object | None = None,
    ) -> None:
        self._canonical: dict[str, str] = {}  # alias -> canonical_id
        self._aliases: dict[str, set[str]] = {}  # canonical_id -> {aliases}
        self._profiles: dict[str, PersonProfile] = {}  # canonical_id -> profile
        self._unknown: set[str] = set()  # identifiers that didn't resolve
        self._store = store  # AliasStore for persistence

        # Load from providers if given
        if providers:
            self._load_providers(providers)

        # Load persisted aliases from store
        if store is not None:
            self._load_from_store(store)

    def _load_providers(self, providers: list[IdentityProvider]) -> None:
        """Fetch records from all available providers and merge."""
        for provider in providers:
            if not provider.available():
                logger.info(
                    "Identity provider %s not available, skipping",
                    provider.source_name(),
                )
                continue
            try:
                records = provider.fetch()
                logger.info(
                    "Identity provider %s returned %d records",
                    provider.source_name(),
                    len(records),
                )
                for record in records:
                    self._merge_record(record)
            except Exception:
                logger.warning(
                    "Identity provider %s failed",
                    provider.source_name(),
                    exc_info=True,
                )

    def _merge_record(self, record: PersonRecord) -> None:
        """Merge a PersonRecord into the resolver's state.

        If the record matches an existing profile (by email or name),
        enrich it. Otherwise create a new profile.
        """
        # Try to find existing canonical ID via email
        canonical_id = None
        if record.email:
            canonical_id = self._canonical.get(record.email.lower())

        # Try by name
        if not canonical_id and record.name:
            canonical_id = self._canonical.get(record.name.lower())

        # Try by github handle
        if not canonical_id and record.github_handle:
            canonical_id = self._canonical.get(record.github_handle.lower())

        if canonical_id:
            # Enrich existing profile
            profile = self._profiles.get(canonical_id)
            if profile:
                if record.github_handle and not profile.github_handle:
                    profile.github_handle = record.github_handle.lower()
                if record.slack_handle and not profile.slack_handle:
                    profile.slack_handle = record.slack_handle
                if record.linear_uuid and not profile.linear_uuid:
                    profile.linear_uuid = record.linear_uuid
                if record.team and not profile.team:
                    profile.team = record.team
        elif record.email:
            # Create new profile — canonical ID from email prefix
            canonical_id = record.email.split("@")[0].lower()
            profile = PersonProfile(
                canonical_id=canonical_id,
                name=record.name,
                email=record.email.lower(),
                github_handle=record.github_handle.lower() if record.github_handle else "",
                slack_handle=record.slack_handle,
                team=record.team,
                linear_uuid=record.linear_uuid,
            )
            self._profiles[canonical_id] = profile
            self._aliases[canonical_id] = set()
        else:
            # No email — can't create canonical ID, skip
            return

        # Register all aliases from the record
        self._register(canonical_id, canonical_id)
        if record.email:
            self._register(canonical_id, record.email.lower())
        if record.name:
            self._register(canonical_id, record.name.lower())
            # firstname.lastname pattern
            parts = record.name.lower().split()
            if len(parts) >= 2:
                self._register(canonical_id, f"{parts[0]}.{parts[-1]}")
        if record.github_handle:
            self._register(canonical_id, record.github_handle.lower())
        if record.slack_handle:
            self._register(canonical_id, record.slack_handle.lower())
            self._register(canonical_id, f"@{record.slack_handle.lower()}")
        if record.slack_user_id:
            self._register(canonical_id, record.slack_user_id.lower())
        if record.linear_uuid:
            self._register(canonical_id, record.linear_uuid.lower())
        if record.linear_display_name:
            self._register(canonical_id, record.linear_display_name.lower())
        for alias in record.aliases:
            if alias:
                self._register(canonical_id, alias.lower())

    def _load_from_store(self, store: object) -> None:
        """Load persisted aliases from AliasStore."""
        # Import here to avoid circular dependency
        from stigmergy.identity.store import AliasStore

        if isinstance(store, AliasStore):
            for canonical_id, aliases in store.all_aliases().items():
                for alias in aliases:
                    self._register(canonical_id, alias)

    # ── Legacy loading methods (wrap provider pattern internally) ──

    def load_team_roster(self, csv_path: Path) -> None:
        """Load from team roster CSV (name, email, slack, github).

        CSV format (no header row in the actual file):
          Name,Name <email>,@Slack Handle,github_handle
        """
        if not csv_path.exists():
            logger.warning("Team roster not found: %s", csv_path)
            return

        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue

                name = row[0].strip()
                email_field = row[1].strip()
                slack_raw = row[2].strip()
                github = row[3].strip()

                # Parse email from "Name <email>" format
                email_match = re.search(r"<([^>]+)>", email_field)
                email = email_match.group(1) if email_match else email_field

                # Slack handle: strip leading @
                slack = slack_raw.lstrip("@").strip()

                # Canonical ID: lowercase email prefix (before @)
                canonical_id = email.split("@")[0].lower()

                profile = PersonProfile(
                    canonical_id=canonical_id,
                    name=name,
                    email=email.lower(),
                    github_handle=github.lower(),
                    slack_handle=slack,
                )
                self._profiles[canonical_id] = profile
                self._aliases[canonical_id] = set()

                # Register all known aliases
                self._register(canonical_id, email.lower())
                self._register(canonical_id, name.lower())
                self._register(canonical_id, canonical_id)

                if github:
                    self._register(canonical_id, github.lower())

                if slack:
                    self._register(canonical_id, slack.lower())
                    # Also register without the @ for matching signal authors
                    self._register(canonical_id, f"@{slack.lower()}")

                # Common author patterns from signals: "firstname.lastname"
                parts = name.lower().split()
                if len(parts) >= 2:
                    self._register(canonical_id, f"{parts[0]}.{parts[-1]}")

        logger.info("Loaded %d team members from %s", len(self._profiles), csv_path)

    def load_email_aliases(self, csv_path: Path) -> None:
        """Load from email_aliases.csv (work_email, alias_email).

        Lines starting with # are comments.
        """
        if not csv_path.exists():
            logger.warning("Email aliases not found: %s", csv_path)
            return

        count = 0
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) < 2:
                    continue

                work_email = row[0].strip().lower()
                alias_email = row[1].strip().lower()

                # Find canonical ID for work email
                canonical_id = self._canonical.get(work_email)
                if canonical_id:
                    self._register(canonical_id, alias_email)
                    # Also register the username part of the alias
                    alias_user = alias_email.split("@")[0]
                    if "+" not in alias_user:  # skip noreply-style usernames
                        self._register(canonical_id, alias_user)
                    count += 1

        logger.info("Loaded %d email aliases from %s", count, csv_path)

    def load_linear_reference(self, md_path: Path) -> None:
        """Parse team assignments and Linear UUIDs from linear-reference.md.

        Extracts:
        - Team member UUIDs -> canonical ID mapping
        - Team assignments for profiles
        """
        if not md_path.exists():
            logger.warning("Linear reference not found: %s", md_path)
            return

        text = md_path.read_text()

        # Parse team member table: | Name | Email | UUID |
        member_pattern = re.compile(
            r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|"
        )

        # Track which section we're in
        in_members = False
        count = 0
        for line in text.splitlines():
            if "Team Members" in line:
                in_members = True
                continue
            if line.startswith("###") or (line.startswith("## ") and in_members):
                in_members = False
                continue
            if not in_members:
                continue

            match = member_pattern.match(line)
            if not match:
                continue

            name = match.group(1).strip()
            email = match.group(2).strip().lower()
            linear_uuid = match.group(3).strip()

            # Skip header row
            if name.lower() == "name" or email.lower() == "email":
                continue

            canonical_id = self._canonical.get(email)
            if canonical_id:
                self._register(canonical_id, linear_uuid)
                profile = self._profiles.get(canonical_id)
                if profile:
                    profile.linear_uuid = linear_uuid
                count += 1

        logger.info("Loaded %d Linear UUIDs from %s", count, md_path)

    # ── Public interface ──

    # Identifiers matching signal source names — these are system/bot
    # identities, not people. Detected generically so we don't need
    # to hard-code every integration.
    _SYSTEM_NAMES = frozenset({
        "unknown", "@unknown", "linear", "github-actions", "github-actions[bot]",
        "vercel", "vercel[bot]", "slack", "cursor", "dependabot",
        "dependabot[bot]", "app/dependabot", "renovate", "renovate[bot]",
        "codecov[bot]", "snyk[bot]", "actions-user", "incident",
        "copilot-pull-request-reviewer",
    })

    def resolve(self, identifier: str) -> str:
        """Return canonical ID for any alias.

        Applies progressive normalization:
        1. Exact match
        2. Strip leading @ (signals use @handle, aliases registered without)
        3. Strip trailing underscores (Linear display names like erikalmaraz_)
        4. CamelCase decomposition (CamTosh → cam.tosh, cam)
        5. Username decomposition (jsmith → try first-initial + last-name
           against known profiles)

        Falls through to the identifier itself (lowercased) if unknown.
        System/bot names are returned as-is without tracking as unresolved.
        """
        key = identifier.strip().lower()

        # 1. Exact match
        result = self._canonical.get(key)
        if result is not None:
            return result

        # 2. Strip leading @
        bare = key.lstrip("@")
        if bare != key:
            result = self._canonical.get(bare)
            if result is not None:
                return result

        # 3. Strip trailing underscores/special chars (Linear display name variants)
        stripped = bare.rstrip("_-.")
        if stripped != bare:
            result = self._canonical.get(stripped)
            if result is not None:
                return result

        # 4. CamelCase decomposition: CamTosh → cam, cam.tosh
        camel_parts = re.findall(r"[A-Z][a-z]+|[a-z]+", identifier.strip())
        if len(camel_parts) >= 2:
            lowered = [p.lower() for p in camel_parts]
            # Try firstname.lastname
            dotted = f"{lowered[0]}.{lowered[-1]}"
            result = self._canonical.get(dotted)
            if result is not None:
                return result
            # Try just first name
            result = self._canonical.get(lowered[0])
            if result is not None:
                return result

        # 5. Username decomposition: try first-initial + lastname against
        #    known profiles (e.g., jsmith → j + smith → john.smith)
        if len(bare) > 3 and bare.isalnum():
            result = self._match_by_name_parts(bare)
            if result is not None:
                return result

        # System/bot identifiers — return as-is, don't track as unresolved
        if key in self._SYSTEM_NAMES or key.endswith("[bot]"):
            return key

        self._unknown.add(key)
        return key

    def _match_by_name_parts(self, handle: str) -> str | None:
        """Try to match a handle against known profile names.

        Handles common username patterns: firstlast, first.last,
        firstinitiallast (e.g., jsmith → j + smith → john smith).
        """
        for canonical_id, profile in self._profiles.items():
            if not profile.name:
                continue
            name_parts = profile.name.lower().split()
            if len(name_parts) < 2:
                continue
            first = name_parts[0]
            last = name_parts[-1]

            # firstlast (e.g., johnsmith)
            if handle == f"{first}{last}":
                return canonical_id

            # firstinitiallast (e.g., jsmith)
            if handle == f"{first[0]}{last}":
                return canonical_id

            # lastinitialfirst (e.g., smithj)
            if handle == f"{last}{first[0]}":
                return canonical_id

            # lastfirst (e.g., smithjohn)
            if handle == f"{last}{first}":
                return canonical_id

        return None

    def register_alias(self, canonical_id: str, alias: str) -> None:
        """Runtime learning — new aliases discovered from signals."""
        self._register(canonical_id, alias.lower())
        # Persist to store if available
        if self._store is not None:
            from stigmergy.identity.store import AliasStore

            if isinstance(self._store, AliasStore):
                self._store.add(canonical_id, alias.lower())

    def get_profile(self, canonical_id: str) -> PersonProfile | None:
        """Get the profile for a canonical ID."""
        return self._profiles.get(canonical_id)

    def all_profiles(self) -> list[PersonProfile]:
        """All known person profiles."""
        return list(self._profiles.values())

    @property
    def person_count(self) -> int:
        return len(self._profiles)

    @property
    def alias_count(self) -> int:
        return len(self._canonical)

    @property
    def unknown_identifiers(self) -> frozenset[str]:
        """Identifiers that were resolved but had no mapping."""
        return frozenset(self._unknown)

    def suggest_match(self, identifier: str) -> list[tuple[str, float]]:
        """Fuzzy-match an identifier against known canonical IDs.

        Returns up to 5 candidates with similarity scores (0-1).
        Useful for diagnostics on unresolved identifiers.
        """
        key = identifier.strip().lower()
        candidates: list[tuple[str, float]] = []

        for alias, canonical_id in self._canonical.items():
            ratio = SequenceMatcher(None, key, alias).ratio()
            if ratio >= 0.6:
                candidates.append((canonical_id, ratio))

        # Deduplicate by canonical_id, keeping highest score
        best: dict[str, float] = {}
        for cid, score in candidates:
            if cid not in best or score > best[cid]:
                best[cid] = score

        return sorted(best.items(), key=lambda x: x[1], reverse=True)[:5]

    def _register(self, canonical_id: str, alias: str) -> None:
        """Register an alias -> canonical_id mapping."""
        alias = alias.strip().lower()
        if not alias:
            return
        self._canonical[alias] = canonical_id
        if canonical_id not in self._aliases:
            self._aliases[canonical_id] = set()
        self._aliases[canonical_id].add(alias)
