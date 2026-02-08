"""Identity providers — pluggable sources of person data."""

from stigmergy.identity.providers.github import GitHubOrgProvider
from stigmergy.identity.providers.linear import LinearUsersProvider
from stigmergy.identity.providers.slack import SlackUsersProvider
from stigmergy.identity.providers.static import StaticCSVProvider, StaticLinearMdProvider

__all__ = [
    "GitHubOrgProvider",
    "LinearUsersProvider",
    "SlackUsersProvider",
    "StaticCSVProvider",
    "StaticLinearMdProvider",
]
