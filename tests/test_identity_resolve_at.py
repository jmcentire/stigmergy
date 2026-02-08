"""Tests for identity resolution @ prefix stripping and supplementary aliases."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from stigmergy.identity.providers.static import StaticCSVProvider
from stigmergy.identity.resolver import IdentityResolver


@pytest.fixture
def team_csv(tmp_path):
    csv = tmp_path / "team.csv"
    csv.write_text(dedent("""\
        Alice Wang,Alice Wang <alice@example.com>,@Alice Wang,alice212
        Bob Kim,Bob Kim <bob.kim@example.com>,@Bob,bobkim
        Carlos Santos,Carlos Santos <carlos@example.com>,@Carlos Santos,carlossantos
        Dana Parker,Dana Parker <dana@example.com>,@Dana Parker,Inventor
    """))
    return csv


@pytest.fixture
def local_csv(tmp_path):
    csv = tmp_path / "local.csv"
    csv.write_text(dedent("""\
        Frank Torres (Linear),franktorres_ <frank@example.com>,@franktorres_,franktorres
        Grace Chen,Grace Chen <grace@example.com>,@Grace Chen,gchen
    """))
    return csv


@pytest.fixture
def resolver(team_csv):
    provider = StaticCSVProvider(team_csv)
    return IdentityResolver(providers=[provider])


@pytest.fixture
def resolver_with_local(team_csv, local_csv):
    return IdentityResolver(providers=[
        StaticCSVProvider(team_csv),
        StaticCSVProvider(local_csv),
    ])


class TestAtPrefixStripping:
    def test_resolve_at_alice(self, resolver):
        assert resolver.resolve("@alice212") == "alice"

    def test_resolve_at_bobkim(self, resolver):
        assert resolver.resolve("@bobkim") == "bob.kim"

    def test_resolve_at_carlos(self, resolver):
        assert resolver.resolve("@carlossantos") == "carlos"

    def test_resolve_at_dana(self, resolver):
        assert resolver.resolve("@Inventor") == "dana"

    def test_resolve_without_at_still_works(self, resolver):
        assert resolver.resolve("alice212") == "alice"

    def test_resolve_at_with_explicit_slack_registration(self, resolver):
        # Slack handles are registered both with and without @
        # resolve("@Alice Wang") should work via the explicit @slack registration
        assert resolver.resolve("@Alice Wang") == "alice"

    def test_unknown_at_identifier_tracked(self, resolver):
        result = resolver.resolve("@totally_unknown")
        assert result == "@totally_unknown"
        assert "@totally_unknown" in resolver.unknown_identifiers


class TestSupplementaryAliases:
    def test_franktorres_underscore(self, resolver_with_local):
        assert resolver_with_local.resolve("franktorres_") == "frank"

    def test_gchen_github(self, resolver_with_local):
        assert resolver_with_local.resolve("gchen") == "grace"

    def test_bot_passthrough(self, resolver_with_local):
        result = resolver_with_local.resolve("github-actions")
        assert result == "github-actions"
        # System/bot names are returned as-is but NOT tracked as unresolved
        assert "github-actions" not in resolver_with_local.unknown_identifiers
