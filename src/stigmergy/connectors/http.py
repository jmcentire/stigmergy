"""Mock HTTP client for external API calls.

Mocks the HTTP interface that real adapters (Slack API, Linear API,
GitHub API, LLM providers) would use. Supports:
- Request/response recording for test assertions
- Configurable response fixtures
- Latency simulation
- Error injection for resilience testing

Same interface as httpx.AsyncClient or aiohttp.ClientSession.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class HTTPResponse:
    status_code: int = 200
    body: dict[str, Any] | str = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def json(self) -> dict[str, Any]:
        if isinstance(self.body, str):
            return json.loads(self.body)
        return self.body

    @property
    def text(self) -> str:
        if isinstance(self.body, dict):
            return json.dumps(self.body)
        return self.body

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass
class HTTPRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | str | None = None
    timestamp: float = field(default_factory=time.monotonic)


class HTTPClient(Protocol):
    """Protocol for HTTP clients. Mock and real implement this."""

    async def get(self, url: str, headers: dict[str, str] | None = None) -> HTTPResponse: ...
    async def post(self, url: str, body: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> HTTPResponse: ...
    async def put(self, url: str, body: dict[str, Any] | None = None,
                  headers: dict[str, str] | None = None) -> HTTPResponse: ...
    async def delete(self, url: str, headers: dict[str, str] | None = None) -> HTTPResponse: ...


class MockHTTPClient:
    """Configurable HTTP mock with request recording and response fixtures.

    Usage:
        client = MockHTTPClient()
        client.add_fixture("GET", "https://api.slack.com/channels", HTTPResponse(
            body={"channels": [{"id": "C1", "name": "engineering"}]}
        ))
        response = await client.get("https://api.slack.com/channels")
        assert response.json["channels"][0]["name"] == "engineering"
    """

    def __init__(self, default_latency: float = 0.0) -> None:
        self._fixtures: dict[tuple[str, str], HTTPResponse] = {}
        self._pattern_fixtures: list[tuple[str, str, HTTPResponse]] = []  # method, url_prefix, response
        self._requests: list[HTTPRequest] = []
        self._default_latency = default_latency
        self._error_injection: dict[str, Exception] = {}  # url -> exception to raise

    def add_fixture(self, method: str, url: str, response: HTTPResponse) -> None:
        """Add an exact-match response fixture."""
        self._fixtures[(method.upper(), url)] = response

    def add_prefix_fixture(self, method: str, url_prefix: str, response: HTTPResponse) -> None:
        """Add a prefix-match response fixture (matches any URL starting with prefix)."""
        self._pattern_fixtures.append((method.upper(), url_prefix, response))

    def inject_error(self, url: str, error: Exception) -> None:
        """Make requests to this URL raise an error (for resilience testing)."""
        self._error_injection[url] = error

    def clear_error_injection(self, url: str | None = None) -> None:
        if url:
            self._error_injection.pop(url, None)
        else:
            self._error_injection.clear()

    async def _request(self, method: str, url: str,
                       body: dict[str, Any] | str | None = None,
                       headers: dict[str, str] | None = None) -> HTTPResponse:
        # Record request
        req = HTTPRequest(method=method, url=url, headers=headers or {}, body=body)
        self._requests.append(req)

        # Simulate latency
        if self._default_latency > 0:
            await asyncio.sleep(self._default_latency)

        # Error injection
        if url in self._error_injection:
            raise self._error_injection[url]

        # Exact match
        fixture = self._fixtures.get((method, url))
        if fixture:
            return fixture

        # Prefix match
        for fix_method, prefix, response in self._pattern_fixtures:
            if method == fix_method and url.startswith(prefix):
                return response

        # Default 404
        return HTTPResponse(status_code=404, body={"error": "no fixture configured"})

    async def get(self, url: str, headers: dict[str, str] | None = None) -> HTTPResponse:
        return await self._request("GET", url, headers=headers)

    async def post(self, url: str, body: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> HTTPResponse:
        return await self._request("POST", url, body=body, headers=headers)

    async def put(self, url: str, body: dict[str, Any] | None = None,
                  headers: dict[str, str] | None = None) -> HTTPResponse:
        return await self._request("PUT", url, body=body, headers=headers)

    async def delete(self, url: str, headers: dict[str, str] | None = None) -> HTTPResponse:
        return await self._request("DELETE", url, headers=headers)

    # -- Test helpers --

    @property
    def requests(self) -> list[HTTPRequest]:
        """All recorded requests."""
        return list(self._requests)

    def requests_to(self, url: str) -> list[HTTPRequest]:
        """Filter requests by URL."""
        return [r for r in self._requests if r.url == url]

    def requests_by_method(self, method: str) -> list[HTTPRequest]:
        """Filter requests by HTTP method."""
        return [r for r in self._requests if r.method == method.upper()]

    @property
    def request_count(self) -> int:
        return len(self._requests)

    def reset(self) -> None:
        """Clear all fixtures, requests, and error injections."""
        self._fixtures.clear()
        self._pattern_fixtures.clear()
        self._requests.clear()
        self._error_injection.clear()
