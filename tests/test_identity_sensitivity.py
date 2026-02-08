"""Tests for identity sensitivity — data classification and redaction."""

from __future__ import annotations

import pytest

from stigmergy.identity.resolver import PersonProfile
from stigmergy.identity.sensitivity import (
    DataClassification,
    redact_identifier,
    redact_profile,
    safe_for_log,
)


@pytest.fixture
def profile():
    return PersonProfile(
        canonical_id="alice",
        name="Alice Smith",
        email="alice@example.com",
        github_handle="alice-gh",
        slack_handle="Alice",
        team="Platform",
        linear_uuid="11111111-2222-3333-4444-555555555555",
    )


class TestDataClassification:
    def test_enum_values(self):
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.INTERNAL.value == "internal"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.RESTRICTED.value == "restricted"


class TestRedactProfile:
    def test_public_level(self, profile):
        result = redact_profile(profile, DataClassification.PUBLIC)
        assert result["canonical_id"] == "alice"
        assert result["github_handle"] == "alice-gh"
        assert result["team"] == "Platform"
        # Internal and above should be redacted
        assert result["name"] == "[REDACTED]"
        assert result["slack_handle"] == "[REDACTED]"
        assert result["email"] == "[REDACTED]"

    def test_internal_level(self, profile):
        result = redact_profile(profile, DataClassification.INTERNAL)
        assert result["canonical_id"] == "alice"
        assert result["name"] == "Alice Smith"
        assert result["slack_handle"] == "Alice"
        # Confidential should be redacted
        assert result["email"] == "[REDACTED]"

    def test_confidential_level(self, profile):
        result = redact_profile(profile, DataClassification.CONFIDENTIAL)
        assert result["canonical_id"] == "alice"
        assert result["name"] == "Alice Smith"
        assert result["email"] == "alice@example.com"

    def test_restricted_level(self, profile):
        result = redact_profile(profile, DataClassification.RESTRICTED)
        # Everything visible
        assert result["canonical_id"] == "alice"
        assert result["name"] == "Alice Smith"
        assert result["email"] == "alice@example.com"

    def test_empty_fields_not_marked_redacted(self):
        profile = PersonProfile(
            canonical_id="bob",
            name="Bob",
            email="",
        )
        result = redact_profile(profile, DataClassification.PUBLIC)
        # Empty email should stay empty, not "[REDACTED]"
        assert result["email"] == ""


class TestRedactIdentifier:
    def test_email_redacted_at_public(self):
        result = redact_identifier("alice@example.com", DataClassification.PUBLIC)
        assert "@" not in result or "***" in result
        assert result.startswith("a")

    def test_email_redacted_at_internal(self):
        result = redact_identifier("alice@example.com", DataClassification.INTERNAL)
        assert "***" in result

    def test_email_visible_at_confidential(self):
        result = redact_identifier("alice@example.com", DataClassification.CONFIDENTIAL)
        assert result == "alice@example.com"

    def test_non_email_not_redacted(self):
        result = redact_identifier("alice-gh", DataClassification.PUBLIC)
        assert result == "alice-gh"

    def test_restricted_shows_everything(self):
        result = redact_identifier("alice@example.com", DataClassification.RESTRICTED)
        assert result == "alice@example.com"


class TestSafeForLog:
    def test_public_level(self, profile):
        result = safe_for_log(profile, DataClassification.PUBLIC)
        assert "alice" in result
        assert "Platform" in result
        assert "Alice Smith" not in result
        assert "alice@example.com" not in result

    def test_internal_level(self, profile):
        result = safe_for_log(profile, DataClassification.INTERNAL)
        assert "Alice Smith" in result
        assert "alice" in result
        assert "alice@example.com" not in result

    def test_confidential_level(self, profile):
        result = safe_for_log(profile, DataClassification.CONFIDENTIAL)
        assert "Alice Smith" in result
        assert "alice@example.com" in result

    def test_no_team(self):
        profile = PersonProfile(canonical_id="bob", name="Bob", email="bob@example.com")
        result = safe_for_log(profile, DataClassification.PUBLIC)
        assert "bob" in result
        assert "()" not in result
