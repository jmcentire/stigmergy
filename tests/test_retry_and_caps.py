"""Tests for adapter retry logic and Agent.reset_batch_state()."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from stigmergy.primitives.agent import Agent


# ── Agent.reset_batch_state() ────────────────────────────────


class TestAgentResetBatchState:
    def test_reset_clears_batch_ids(self):
        agent = Agent()
        agent._last_batch_ids = frozenset([uuid4(), uuid4()])
        agent._last_batch_insights = [MagicMock(), MagicMock()]
        agent.reset_batch_state()
        assert agent._last_batch_ids == frozenset()
        assert agent._last_batch_insights == []

    def test_reset_preserves_counters(self):
        agent = Agent()
        agent._batch_dedup_count = 5
        agent._llm_reflect_count = 3
        agent.reset_batch_state()
        # Counters are NOT reset — they're diagnostic accumulators
        assert agent._batch_dedup_count == 5
        assert agent._llm_reflect_count == 3

    def test_reset_allows_fresh_evaluation(self):
        """After reset, the same batch should NOT be dedup-skipped."""
        agent = Agent()
        ids = frozenset([uuid4(), uuid4(), uuid4()])
        agent._last_batch_ids = ids
        agent._last_batch_insights = [MagicMock()]
        # Before reset: overlap would be 100%, triggering cache return
        overlap = len(ids & agent._last_batch_ids)
        assert overlap / len(ids) >= 0.8
        # After reset: no overlap possible
        agent.reset_batch_state()
        overlap_after = len(ids & agent._last_batch_ids)
        assert overlap_after == 0

    def test_reset_idempotent(self):
        agent = Agent()
        agent.reset_batch_state()
        agent.reset_batch_state()
        assert agent._last_batch_ids == frozenset()
        assert agent._last_batch_insights == []


# ── GitHub adapter retry ─────────────────────────────────────


class TestGitHubRetry:
    @pytest.fixture
    def errors(self):
        return []

    @pytest.fixture
    def adapter(self, errors):
        from stigmergy.cli.github_live import LiveGitHubAdapter
        return LiveGitHubAdapter(
            repos=["org/repo"],
            error_callback=lambda repo, msg: errors.append((repo, msg)),
        )

    async def test_succeeds_on_first_attempt(self, adapter):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate.return_value = (b'[{"title": "test"}]', b"")
            proc.returncode = 0
            mock_exec.return_value = proc
            result = await adapter._run_gh(["gh", "pr", "list", "--repo", "org/repo"])
            assert result == [{"title": "test"}]

    async def test_retries_on_rate_limit(self, adapter, errors):
        with patch("asyncio.create_subprocess_exec") as mock_exec, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fail_proc = AsyncMock()
            fail_proc.communicate.return_value = (b"", b"API rate limit exceeded")
            fail_proc.returncode = 1

            ok_proc = AsyncMock()
            ok_proc.communicate.return_value = (b'[]', b"")
            ok_proc.returncode = 0

            mock_exec.side_effect = [fail_proc, ok_proc]
            result = await adapter._run_gh(["gh", "pr", "list", "--repo", "org/repo"])
            assert result == []
            mock_sleep.assert_called_once()  # backed off once

    async def test_gives_up_after_max_retries(self, adapter, errors):
        with patch("asyncio.create_subprocess_exec") as mock_exec, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            fail_proc = AsyncMock()
            fail_proc.communicate.return_value = (b"", b"API rate limit exceeded")
            fail_proc.returncode = 1

            mock_exec.return_value = fail_proc
            result = await adapter._run_gh(["gh", "pr", "list", "--repo", "org/repo"], max_retries=2)
            assert result is None
            # 3 attempts total (initial + 2 retries)
            assert mock_exec.call_count == 3

    async def test_no_retry_on_permanent_error(self, adapter, errors):
        with patch("asyncio.create_subprocess_exec") as mock_exec, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            fail_proc = AsyncMock()
            fail_proc.communicate.return_value = (b"", b"Could not resolve repository")
            fail_proc.returncode = 1

            mock_exec.return_value = fail_proc
            result = await adapter._run_gh(["gh", "pr", "list", "--repo", "org/repo"])
            assert result is None
            mock_sleep.assert_not_called()  # no retry for permanent errors


# ── Linear adapter retry ─────────────────────────────────────


class TestLinearRetry:
    @pytest.fixture
    def errors(self):
        return []

    @pytest.fixture
    def adapter(self, errors):
        from stigmergy.cli.linear_live import LiveLinearAdapter
        return LiveLinearAdapter(
            teams=[{"name": "Platform", "id": "test-id"}],
            api_key="test-key",
            error_callback=lambda source, msg: errors.append((source, msg)),
        )

    async def test_succeeds_on_first_attempt(self, adapter):
        response_data = {"data": {"viewer": {"id": "1", "name": "Test"}}}
        with patch("stigmergy.cli.linear_live.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_urlopen.return_value = mock_response
            result = await adapter._graphql("query { viewer { id name } }")
            assert result == {"viewer": {"id": "1", "name": "Test"}}

    async def test_retries_on_timeout(self, adapter, errors):
        from urllib.error import URLError
        response_data = {"data": {"viewer": {"id": "1"}}}
        with patch("stigmergy.cli.linear_live.urlopen") as mock_urlopen, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_urlopen.side_effect = [URLError("timeout"), mock_response]
            result = await adapter._graphql("query { viewer { id } }")
            assert result == {"viewer": {"id": "1"}}
            mock_sleep.assert_called_once()

    async def test_gives_up_after_max_retries(self, adapter, errors):
        from urllib.error import URLError
        with patch("stigmergy.cli.linear_live.urlopen") as mock_urlopen, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            mock_urlopen.side_effect = URLError("timeout")
            result = await adapter._graphql("query { viewer { id } }", max_retries=2)
            assert result is None
            assert mock_urlopen.call_count == 3

    async def test_no_retry_on_permanent_graphql_error(self, adapter, errors):
        response_data = {"errors": [{"message": "Unknown field 'foo'"}]}
        with patch("stigmergy.cli.linear_live.urlopen") as mock_urlopen, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(response_data).encode()
            mock_urlopen.return_value = mock_response
            result = await adapter._graphql("query { foo }")
            assert result is None
            mock_sleep.assert_not_called()
