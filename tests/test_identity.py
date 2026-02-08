"""Tests for person identity resolution."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from stigmergy.graph.identity import IdentityResolver


@pytest.fixture
def team_csv(tmp_path):
    """Synthetic team roster CSV."""
    csv = tmp_path / "team.csv"
    csv.write_text(dedent("""\
        Alice Smith,Alice Smith <alice@example.com>,@Alice Smith,alice-gh
        Bob Jones,Bob Jones <bob@example.com>,@Bob,bobjones
        Carol Davis,Carol Davis <carol@example.com>,@Carol Davis,carol-d
    """))
    return csv


@pytest.fixture
def alias_csv(tmp_path):
    """Synthetic email alias CSV."""
    csv = tmp_path / "aliases.csv"
    csv.write_text(dedent("""\
        # Email aliases
        alice@example.com,alice.personal@gmail.com
        alice@example.com,12345+alice-gh@users.noreply.github.com
        bob@example.com,bob.jones.personal@gmail.com
    """))
    return csv


@pytest.fixture
def linear_md(tmp_path):
    """Synthetic linear reference markdown."""
    md = tmp_path / "linear-reference.md"
    md.write_text(dedent("""\
        # Linear API Reference

        ## Teams

        | Team | Key | UUID |
        |------|-----|------|
        | Platform | PLAT | `aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee` |

        ## Team Members

        | Name | Email | UUID |
        |------|-------|------|
        | Alice Smith | alice@example.com | `11111111-2222-3333-4444-555555555555` |
        | Bob Jones | bob@example.com | `66666666-7777-8888-9999-000000000000` |

        ### Not found in Linear

        | Name | Email | Note |
        |------|-------|------|
        | Carol Davis | carol@example.com | No Linear account |
    """))
    return md


@pytest.fixture
def resolver(team_csv, alias_csv, linear_md):
    """Resolver loaded with synthetic test data."""
    r = IdentityResolver()
    r.load_team_roster(team_csv)
    r.load_email_aliases(alias_csv)
    r.load_linear_reference(linear_md)
    return r


@pytest.fixture
def empty_resolver():
    """Resolver with no data loaded."""
    return IdentityResolver()


class TestIdentityResolver:
    def test_empty_resolver_passthrough(self, empty_resolver):
        """Unknown identifiers pass through as-is (lowercased)."""
        assert empty_resolver.resolve("unknown.person") == "unknown.person"
        assert empty_resolver.resolve("UPPER@EMAIL.COM") == "upper@email.com"

    def test_resolve_by_email(self, resolver):
        """Resolve by work email."""
        assert resolver.resolve("alice@example.com") == "alice"
        assert resolver.resolve("bob@example.com") == "bob"

    def test_resolve_by_github_handle(self, resolver):
        """Resolve by GitHub handle."""
        assert resolver.resolve("alice-gh") == "alice"
        assert resolver.resolve("bobjones") == "bob"

    def test_resolve_by_slack_handle(self, resolver):
        """Resolve by Slack handle."""
        assert resolver.resolve("Alice Smith") == "alice"
        assert resolver.resolve("Bob") == "bob"

    def test_resolve_by_name(self, resolver):
        """Resolve by full name."""
        assert resolver.resolve("Alice Smith") == "alice"
        assert resolver.resolve("Bob Jones") == "bob"

    def test_resolve_by_dot_name(self, resolver):
        """Resolve by firstname.lastname pattern (common in signals)."""
        assert resolver.resolve("alice.smith") == "alice"
        assert resolver.resolve("bob.jones") == "bob"

    def test_resolve_email_aliases(self, resolver):
        """Resolve personal emails to canonical IDs."""
        assert resolver.resolve("alice.personal@gmail.com") == "alice"
        assert resolver.resolve("bob.jones.personal@gmail.com") == "bob"

    def test_resolve_case_insensitive(self, resolver):
        """Resolution is case-insensitive."""
        assert resolver.resolve("ALICE@EXAMPLE.COM") == "alice"
        assert resolver.resolve("Alice-GH") == "alice"

    def test_unknown_falls_through(self, resolver):
        """Unknown identifiers return themselves lowercased."""
        assert resolver.resolve("completely.unknown") == "completely.unknown"

    def test_register_alias_runtime(self, resolver):
        """Runtime alias registration works."""
        resolver.register_alias("alice", "alice_new_alias")
        assert resolver.resolve("alice_new_alias") == "alice"

    def test_profile_access(self, resolver):
        """Can access profiles by canonical ID."""
        profile = resolver.get_profile("alice")
        assert profile is not None
        assert profile.name == "Alice Smith"
        assert profile.email == "alice@example.com"
        assert profile.github_handle == "alice-gh"

    def test_person_count(self, resolver):
        """Loaded all team members from fixture."""
        assert resolver.person_count == 3

    def test_alias_count(self, resolver):
        """Has many aliases registered."""
        # Each person has at least email + name + github + canonical + dot-name = 5+ aliases
        assert resolver.alias_count >= resolver.person_count * 3

    def test_all_profiles(self, resolver):
        """Can list all profiles."""
        profiles = resolver.all_profiles()
        assert len(profiles) == resolver.person_count
        names = {p.name for p in profiles}
        assert "Alice Smith" in names
        assert "Bob Jones" in names
        assert "Carol Davis" in names


class TestLinearUUIDs:
    def test_linear_uuid_resolution(self, resolver):
        """Linear UUIDs resolve to canonical IDs."""
        assert resolver.resolve("11111111-2222-3333-4444-555555555555") == "alice"
        assert resolver.resolve("66666666-7777-8888-9999-000000000000") == "bob"

    def test_profile_has_linear_uuid(self, resolver):
        """Profile includes Linear UUID."""
        profile = resolver.get_profile("alice")
        assert profile is not None
        assert profile.linear_uuid == "11111111-2222-3333-4444-555555555555"

    def test_member_without_linear(self, resolver):
        """Member not in Linear has empty UUID."""
        profile = resolver.get_profile("carol")
        assert profile is not None
        assert profile.linear_uuid == ""


class TestMissingFiles:
    def test_load_missing_roster(self, empty_resolver, tmp_path):
        """Loading from non-existent roster is a no-op."""
        empty_resolver.load_team_roster(tmp_path / "nonexistent.csv")
        assert empty_resolver.person_count == 0

    def test_load_missing_aliases(self, empty_resolver, tmp_path):
        """Loading from non-existent alias file is a no-op."""
        empty_resolver.load_email_aliases(tmp_path / "nonexistent.csv")
        assert empty_resolver.alias_count == 0

    def test_load_missing_linear(self, empty_resolver, tmp_path):
        """Loading from non-existent linear reference is a no-op."""
        empty_resolver.load_linear_reference(tmp_path / "nonexistent.md")
        assert empty_resolver.person_count == 0
