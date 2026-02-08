"""Tests for identity providers — static, mocked GitHub/Linear/Slack."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from stigmergy.identity.provider import IdentityProvider, PersonRecord
from stigmergy.identity.providers.static import StaticCSVProvider, StaticLinearMdProvider
from stigmergy.identity.providers.github import GitHubOrgProvider
from stigmergy.identity.providers.linear import LinearUsersProvider
from stigmergy.identity.providers.slack import SlackUsersProvider
from stigmergy.identity.resolver import IdentityResolver


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def team_csv(tmp_path):
    csv = tmp_path / "team.csv"
    csv.write_text(dedent("""\
        Alice Smith,Alice Smith <alice@example.com>,@Alice Smith,alice-gh
        Bob Jones,Bob Jones <bob@example.com>,@Bob,bobjones
        Carol Davis,Carol Davis <carol@example.com>,@Carol Davis,carol-d
    """))
    return csv


@pytest.fixture
def alias_csv(tmp_path):
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


# ── StaticCSVProvider ────────────────────────────────────────


class TestStaticCSVProvider:
    def test_source_name(self, team_csv):
        p = StaticCSVProvider(team_csv)
        assert p.source_name() == "static-csv"

    def test_available_when_file_exists(self, team_csv):
        p = StaticCSVProvider(team_csv)
        assert p.available() is True

    def test_not_available_when_missing(self, tmp_path):
        p = StaticCSVProvider(tmp_path / "nonexistent.csv")
        assert p.available() is False

    def test_fetch_returns_records(self, team_csv):
        p = StaticCSVProvider(team_csv)
        records = p.fetch()
        assert len(records) == 3
        assert all(isinstance(r, PersonRecord) for r in records)

    def test_record_fields(self, team_csv):
        p = StaticCSVProvider(team_csv)
        records = p.fetch()
        alice = [r for r in records if r.name == "Alice Smith"][0]
        assert alice.email == "alice@example.com"
        assert alice.github_handle == "alice-gh"
        assert alice.slack_handle == "Alice Smith"
        assert alice.source == "static-csv"

    def test_with_aliases(self, team_csv, alias_csv):
        p = StaticCSVProvider(team_csv, alias_csv)
        records = p.fetch()
        alice = [r for r in records if r.name == "Alice Smith"][0]
        assert "alice.personal@gmail.com" in alice.aliases

    def test_fetch_missing_file_returns_empty(self, tmp_path):
        p = StaticCSVProvider(tmp_path / "nonexistent.csv")
        assert p.fetch() == []

    def test_protocol_compliance(self, team_csv):
        p = StaticCSVProvider(team_csv)
        assert isinstance(p, IdentityProvider)


# ── StaticLinearMdProvider ───────────────────────────────────


class TestStaticLinearMdProvider:
    def test_source_name(self, linear_md):
        p = StaticLinearMdProvider(linear_md)
        assert p.source_name() == "static-linear-md"

    def test_available(self, linear_md):
        p = StaticLinearMdProvider(linear_md)
        assert p.available() is True

    def test_fetch_returns_records(self, linear_md):
        p = StaticLinearMdProvider(linear_md)
        records = p.fetch()
        assert len(records) == 2

    def test_record_has_uuid(self, linear_md):
        p = StaticLinearMdProvider(linear_md)
        records = p.fetch()
        alice = [r for r in records if r.name == "Alice Smith"][0]
        assert alice.linear_uuid == "11111111-2222-3333-4444-555555555555"
        assert alice.email == "alice@example.com"

    def test_missing_file(self, tmp_path):
        p = StaticLinearMdProvider(tmp_path / "nonexistent.md")
        assert p.fetch() == []


# ── GitHubOrgProvider (mocked) ───────────────────────────────


class TestGitHubOrgProvider:
    def test_source_name(self):
        p = GitHubOrgProvider("testorg")
        assert p.source_name() == "github-org"

    @patch("subprocess.run")
    def test_available_when_gh_authed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        p = GitHubOrgProvider("testorg")
        assert p.available() is True

    @patch("subprocess.run")
    def test_not_available_when_gh_not_authed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        p = GitHubOrgProvider("testorg")
        assert p.available() is False

    @patch("subprocess.run")
    def test_fetch_members(self, mock_run):
        # First call: list members
        members_result = MagicMock(returncode=0, stdout="alice-gh\nbob-gh\n", stderr="")
        # User detail calls
        alice_detail = MagicMock(
            returncode=0,
            stdout=json.dumps({"login": "alice-gh", "name": "Alice", "email": "alice@example.com"}),
        )
        bob_detail = MagicMock(
            returncode=0,
            stdout=json.dumps({"login": "bob-gh", "name": "Bob", "email": ""}),
        )
        mock_run.side_effect = [members_result, alice_detail, bob_detail]

        p = GitHubOrgProvider("testorg")
        records = p.fetch()
        assert len(records) == 2
        assert records[0].github_handle == "alice-gh"
        assert records[0].email == "alice@example.com"
        assert records[1].github_handle == "bob-gh"
        assert records[1].email == ""

    @patch("subprocess.run")
    def test_fetch_handles_api_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        p = GitHubOrgProvider("testorg")
        records = p.fetch()
        assert records == []


# ── LinearUsersProvider (mocked) ─────────────────────────────


class TestLinearUsersProvider:
    def test_source_name(self):
        p = LinearUsersProvider("test-key")
        assert p.source_name() == "linear-users"

    def test_available_with_key(self):
        p = LinearUsersProvider("test-key")
        assert p.available() is True

    def test_not_available_without_key(self):
        p = LinearUsersProvider("")
        assert p.available() is False

    @patch("stigmergy.identity.providers.linear.urlopen")
    def test_fetch_users(self, mock_urlopen):
        response_data = {
            "data": {
                "users": {
                    "nodes": [
                        {"id": "uuid-1", "name": "Alice Smith", "displayName": "Alice", "email": "alice@example.com"},
                        {"id": "uuid-2", "name": "Bob Jones", "displayName": "Bob", "email": "bob@example.com"},
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_urlopen.return_value = mock_response

        p = LinearUsersProvider("test-key")
        records = p.fetch()
        assert len(records) == 2
        assert records[0].linear_uuid == "uuid-1"
        assert records[0].email == "alice@example.com"
        assert records[0].linear_display_name == "Alice"

    @patch("stigmergy.identity.providers.linear.urlopen")
    def test_fetch_handles_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("connection failed")
        p = LinearUsersProvider("test-key")
        records = p.fetch()
        assert records == []


# ── SlackUsersProvider (mocked) ──────────────────────────────


class TestSlackUsersProvider:
    def test_source_name(self):
        p = SlackUsersProvider("xoxb-test")
        assert p.source_name() == "slack-users"

    def test_available_with_token(self):
        p = SlackUsersProvider("xoxb-test")
        assert p.available() is True

    def test_not_available_without_token(self):
        p = SlackUsersProvider("")
        assert p.available() is False

    @patch("stigmergy.identity.providers.slack.urlopen")
    def test_fetch_users(self, mock_urlopen):
        response_data = {
            "ok": True,
            "members": [
                {
                    "id": "U12345",
                    "real_name": "Alice Smith",
                    "is_bot": False,
                    "deleted": False,
                    "profile": {
                        "real_name": "Alice Smith",
                        "display_name": "alice",
                        "email": "alice@example.com",
                    },
                },
                {
                    "id": "USLACKBOT",
                    "real_name": "Slackbot",
                    "is_bot": False,
                    "deleted": False,
                    "profile": {"real_name": "Slackbot", "display_name": "slackbot"},
                },
                {
                    "id": "U99999",
                    "real_name": "Bot User",
                    "is_bot": True,
                    "deleted": False,
                    "profile": {"real_name": "Bot User", "display_name": "bot"},
                },
            ],
            "response_metadata": {"next_cursor": ""},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_urlopen.return_value = mock_response

        p = SlackUsersProvider("xoxb-test")
        records = p.fetch()
        # Should skip Slackbot and bot
        assert len(records) == 1
        assert records[0].slack_user_id == "U12345"
        assert records[0].slack_handle == "alice"
        assert records[0].email == "alice@example.com"


# ── Multi-provider resolver integration ──────────────────────


class TestMultiProviderResolver:
    def test_static_providers(self, team_csv, alias_csv, linear_md):
        csv_provider = StaticCSVProvider(team_csv, alias_csv)
        linear_provider = StaticLinearMdProvider(linear_md)

        resolver = IdentityResolver(providers=[csv_provider, linear_provider])
        assert resolver.person_count == 3
        assert resolver.resolve("alice-gh") == "alice"
        assert resolver.resolve("alice@example.com") == "alice"
        assert resolver.resolve("alice.personal@gmail.com") == "alice"

        # Linear UUID should resolve (merged from second provider)
        assert resolver.resolve("11111111-2222-3333-4444-555555555555") == "alice"

    def test_resolver_with_no_providers(self):
        resolver = IdentityResolver(providers=[])
        assert resolver.person_count == 0
        assert resolver.resolve("unknown") == "unknown"

    def test_legacy_loading_still_works(self, team_csv, alias_csv, linear_md):
        """Backward compat: legacy load methods still work."""
        resolver = IdentityResolver()
        resolver.load_team_roster(team_csv)
        resolver.load_email_aliases(alias_csv)
        resolver.load_linear_reference(linear_md)
        assert resolver.resolve("alice-gh") == "alice"
        assert resolver.resolve("11111111-2222-3333-4444-555555555555") == "alice"

    def test_unknown_tracking(self, team_csv):
        provider = StaticCSVProvider(team_csv)
        resolver = IdentityResolver(providers=[provider])
        resolver.resolve("known_person_alice-gh")  # won't match (full string)
        resolver.resolve("alice-gh")  # will match
        resolver.resolve("completely_unknown")
        assert "completely_unknown" in resolver.unknown_identifiers
        assert "alice-gh" not in resolver.unknown_identifiers

    def test_suggest_match(self, team_csv):
        provider = StaticCSVProvider(team_csv)
        resolver = IdentityResolver(providers=[provider])
        suggestions = resolver.suggest_match("alice-g")
        assert len(suggestions) > 0
        assert suggestions[0][0] == "alice"  # best match canonical ID
        assert suggestions[0][1] >= 0.6  # similarity score
