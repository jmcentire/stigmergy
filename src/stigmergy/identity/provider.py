"""Identity provider protocol — pluggable sources of person data.

Each provider fetches person records from a single source (CSV file,
GitHub org API, Linear workspace, Slack workspace). The resolver
merges records from multiple providers into canonical profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class PersonRecord:
    """A single person's data from one provider.

    Providers populate whichever fields they have. The resolver merges
    records with matching identifiers into a unified PersonProfile.
    """

    name: str
    email: str = ""
    github_handle: str = ""
    slack_handle: str = ""
    slack_user_id: str = ""
    linear_uuid: str = ""
    linear_display_name: str = ""
    team: str = ""
    aliases: list[str] = field(default_factory=list)
    source: str = ""


@runtime_checkable
class IdentityProvider(Protocol):
    """Protocol for identity data sources.

    Providers are synchronous and one-shot: fetch() is called once
    during resolver initialization and returns all known records.
    """

    def source_name(self) -> str:
        """Human-readable name for this provider (e.g. 'github-org')."""
        ...

    def available(self) -> bool:
        """Check prerequisites (API key, CLI auth, file exists).

        Returns False if the provider cannot fetch — enables graceful
        degradation when credentials are missing.
        """
        ...

    def fetch(self) -> list[PersonRecord]:
        """Fetch all person records from this source.

        Returns empty list on failure (never raises).
        """
        ...
