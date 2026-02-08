"""Data classification and redaction for identity data.

Applied at output boundaries, not access control. This ensures PII
doesn't leak into logs, CLI output, or external integrations beyond
the configured sensitivity level.

Classification levels:
- PUBLIC:       canonical_id, github_handle, team
- INTERNAL:     name, slack_handle, linear_display_name
- CONFIDENTIAL: email, phone, slack_user_id
- RESTRICTED:   SSN, compensation (future HR integrations)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stigmergy.identity.resolver import PersonProfile


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# Which profile fields belong to each classification level
_FIELD_CLASSIFICATION: dict[str, DataClassification] = {
    "canonical_id": DataClassification.PUBLIC,
    "github_handle": DataClassification.PUBLIC,
    "team": DataClassification.PUBLIC,
    "name": DataClassification.INTERNAL,
    "slack_handle": DataClassification.INTERNAL,
    "linear_uuid": DataClassification.INTERNAL,
    "email": DataClassification.CONFIDENTIAL,
}

# Ordered from least to most sensitive
_LEVEL_ORDER = [
    DataClassification.PUBLIC,
    DataClassification.INTERNAL,
    DataClassification.CONFIDENTIAL,
    DataClassification.RESTRICTED,
]


def _level_allows(max_level: DataClassification, field_level: DataClassification) -> bool:
    """Check if field_level is at or below max_level."""
    return _LEVEL_ORDER.index(field_level) <= _LEVEL_ORDER.index(max_level)


def redact_profile(
    profile: PersonProfile,
    max_level: DataClassification = DataClassification.INTERNAL,
) -> dict[str, str]:
    """Return profile as dict with fields above max_level redacted.

    Redacted fields are replaced with "[REDACTED]".
    """
    result: dict[str, str] = {}
    for field_name, classification in _FIELD_CLASSIFICATION.items():
        value = getattr(profile, field_name, "")
        if _level_allows(max_level, classification):
            result[field_name] = value
        else:
            result[field_name] = "[REDACTED]" if value else ""
    return result


def redact_identifier(
    identifier: str,
    max_level: DataClassification = DataClassification.INTERNAL,
) -> str:
    """Redact an identifier if it looks like PII above the allowed level.

    Heuristic: email addresses are CONFIDENTIAL, everything else PUBLIC.
    """
    if max_level == DataClassification.RESTRICTED:
        return identifier
    if "@" in identifier and "." in identifier.split("@")[-1]:
        # Looks like an email
        if not _level_allows(max_level, DataClassification.CONFIDENTIAL):
            user = identifier.split("@")[0]
            return f"{user[0]}***@***"
    return identifier


def safe_for_log(
    profile: PersonProfile,
    max_level: DataClassification = DataClassification.PUBLIC,
) -> str:
    """Return a log-safe string representation of a profile.

    At PUBLIC level: "canonical_id (team)"
    At INTERNAL level: "name [canonical_id] (team)"
    At CONFIDENTIAL+: "name [canonical_id] <email> (team)"
    """
    parts = [profile.canonical_id]

    if _level_allows(max_level, DataClassification.INTERNAL):
        parts = [f"{profile.name} [{profile.canonical_id}]"]

    if _level_allows(max_level, DataClassification.CONFIDENTIAL) and profile.email:
        parts.append(f"<{profile.email}>")

    if profile.team:
        parts.append(f"({profile.team})")

    return " ".join(parts)
