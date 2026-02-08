"""Tests for mocked connectors: Redis, Stream, HTTP, Metrics."""

from __future__ import annotations

import asyncio

import pytest

from stigmergy.connectors.http import HTTPResponse, MockHTTPClient
from stigmergy.connectors.metrics import MetricsCollector, MetricType
from stigmergy.connectors.redis import MockRedis
from stigmergy.connectors.stream import BackpressureError, MockStream


# --- Mock Redis ---

class TestMockRedis:
    @pytest.mark.asyncio
    async def test_get_set(self):
        r = MockRedis()
        await r.set("key", "value")
        assert await r.get("key") == "value"
        assert await r.get("missing") is None

    @pytest.mark.asyncio
    async def test_delete(self):
        r = MockRedis()
        await r.set("key", "value")
        deleted = await r.delete("key")
        assert deleted == 1
        assert await r.get("key") is None

    @pytest.mark.asyncio
    async def test_exists(self):
        r = MockRedis()
        assert await r.exists("key") is False
        await r.set("key", "val")
        assert await r.exists("key") is True

    @pytest.mark.asyncio
    async def test_incr(self):
        r = MockRedis()
        v1 = await r.incr("counter")
        v2 = await r.incr("counter")
        assert v1 == 1
        assert v2 == 2

    @pytest.mark.asyncio
    async def test_hash_operations(self):
        r = MockRedis()
        await r.hset("ctx:1", "energy", "0.85")
        await r.hset("ctx:1", "signal_count", "42")
        assert await r.hget("ctx:1", "energy") == "0.85"
        all_fields = await r.hgetall("ctx:1")
        assert all_fields == {"energy": "0.85", "signal_count": "42"}

    @pytest.mark.asyncio
    async def test_sorted_set(self):
        r = MockRedis()
        await r.zadd("scores", {"agent_a": 0.9, "agent_b": 0.7, "agent_c": 0.3})
        top = await r.zrange("scores", 0, -1, withscores=True)
        assert top[0][0] == "agent_c"  # lowest score first
        assert top[-1][0] == "agent_a"

    @pytest.mark.asyncio
    async def test_zrangebyscore(self):
        r = MockRedis()
        await r.zadd("scores", {"low": 0.1, "med": 0.5, "high": 0.9})
        result = await r.zrangebyscore("scores", 0.4, 1.0)
        assert "med" in result
        assert "high" in result
        assert "low" not in result

    @pytest.mark.asyncio
    async def test_pubsub(self):
        r = MockRedis()
        received = []
        await r.subscribe("signals", lambda ch, msg: received.append((ch, msg)))
        count = await r.publish("signals", "new_signal")
        assert count == 1
        assert received == [("signals", "new_signal")]

    @pytest.mark.asyncio
    async def test_audit_log(self):
        r = MockRedis()
        await r.set("key", "val")
        await r.get("key")
        assert len(r.audit_log) == 2
        assert r.audit_log[0]["op"] == "SET"
        assert r.audit_log[1]["op"] == "GET"

    @pytest.mark.asyncio
    async def test_reset(self):
        r = MockRedis()
        await r.set("key", "val")
        r.reset()
        assert len(r.audit_log) == 0  # audit log cleared by reset
        assert await r.get("key") is None  # data cleared too (this adds to log)


# --- Mock Stream ---

class TestMockStream:
    @pytest.mark.asyncio
    async def test_send_receive(self):
        s = MockStream()
        msg_id = await s.send("signals", {"content": "test"})
        msg = await s.receive("signals", timeout=1.0)
        assert msg is not None
        assert msg.payload["content"] == "test"
        assert msg.id == msg_id

    @pytest.mark.asyncio
    async def test_ack(self):
        s = MockStream()
        await s.send("signals", {"content": "test"})
        msg = await s.receive("signals", timeout=1.0)
        await s.ack(msg.id)
        stats = s.stats("signals")
        assert stats["acked"] == 1

    @pytest.mark.asyncio
    async def test_nack_retry(self):
        s = MockStream(max_retries=3)
        await s.send("signals", {"content": "fail"})
        msg = await s.receive("signals", timeout=1.0)
        await s.nack(msg.id)
        # Message should be re-enqueued
        retry = await s.receive("signals", timeout=1.0)
        assert retry is not None
        assert retry.attempts == 2

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self):
        s = MockStream(max_retries=1)
        await s.send("signals", {"content": "permanent_fail"})
        msg = await s.receive("signals", timeout=1.0)
        await s.nack(msg.id)  # first attempt = max_retries, goes to DLQ
        assert len(s.dead_letter_queue) == 1

    @pytest.mark.asyncio
    async def test_backpressure(self):
        s = MockStream(max_queue_depth=2)
        await s.send("signals", {"content": "1"})
        await s.send("signals", {"content": "2"})
        with pytest.raises(BackpressureError):
            await s.send("signals", {"content": "3"})

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        s = MockStream()
        msg = await s.receive("empty_topic", timeout=0.01)
        assert msg is None

    @pytest.mark.asyncio
    async def test_stats(self):
        s = MockStream()
        await s.send("t", {"x": 1})
        await s.send("t", {"x": 2})
        stats = s.stats("t")
        assert stats["produced"] == 2
        assert stats["queue_depth"] == 2


# --- Mock HTTP ---

class TestMockHTTP:
    @pytest.mark.asyncio
    async def test_fixture_response(self):
        c = MockHTTPClient()
        c.add_fixture("GET", "https://api.example.com/data", HTTPResponse(
            status_code=200,
            body={"items": [1, 2, 3]}
        ))
        resp = await c.get("https://api.example.com/data")
        assert resp.ok
        assert resp.json["items"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_no_fixture_404(self):
        c = MockHTTPClient()
        resp = await c.get("https://api.example.com/missing")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_prefix_fixture(self):
        c = MockHTTPClient()
        c.add_prefix_fixture("GET", "https://api.slack.com/", HTTPResponse(
            body={"ok": True}
        ))
        resp = await c.get("https://api.slack.com/channels")
        assert resp.json["ok"] is True

    @pytest.mark.asyncio
    async def test_error_injection(self):
        c = MockHTTPClient()
        c.inject_error("https://api.example.com/fail", ConnectionError("simulated"))
        with pytest.raises(ConnectionError):
            await c.get("https://api.example.com/fail")

    @pytest.mark.asyncio
    async def test_request_recording(self):
        c = MockHTTPClient()
        c.add_fixture("POST", "https://api.example.com/send", HTTPResponse())
        await c.post("https://api.example.com/send", body={"message": "hello"})
        assert c.request_count == 1
        assert c.requests[0].method == "POST"
        assert c.requests[0].body["message"] == "hello"


# --- Metrics Collector ---

class TestMetricsCollector:
    def test_counter(self):
        m = MetricsCollector()
        m.count("signals.processed")
        m.count("signals.processed")
        m.count("signals.processed", value=3)
        assert m.counter_value("signals.processed") == 5.0

    def test_counter_with_tags(self):
        m = MetricsCollector()
        m.count("signals.processed", source="slack")
        m.count("signals.processed", source="github")
        assert m.counter_value("signals.processed", source="slack") == 1.0
        assert m.counter_value("signals.processed", source="github") == 1.0

    def test_gauge(self):
        m = MetricsCollector()
        m.gauge("queue.depth", 42)
        assert m.gauge_value("queue.depth") == 42
        m.gauge("queue.depth", 10)
        assert m.gauge_value("queue.depth") == 10  # latest value

    def test_histogram_percentiles(self):
        m = MetricsCollector()
        for i in range(100):
            m.histogram("latency_ms", float(i))
        assert m.percentile("latency_ms", 50) == pytest.approx(50, abs=2)
        assert m.percentile("latency_ms", 95) == pytest.approx(95, abs=2)

    def test_summary(self):
        m = MetricsCollector()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.histogram("scores", v)
        s = m.summary("scores")
        assert s["count"] == 5
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert s["mean"] == 3.0

    def test_timer_context_manager(self):
        m = MetricsCollector()
        with m.start_timer()("test_op"):
            pass  # instant
        assert len(m.events("test_op")) == 1

    def test_event_query(self):
        m = MetricsCollector()
        m.count("a", source="x")
        m.count("a", source="y")
        m.count("b", source="x")
        assert len(m.events("a")) == 2
        assert len(m.events("a", source="x")) == 1

    def test_reset(self):
        m = MetricsCollector()
        m.count("x")
        m.reset()
        assert m.event_count == 0
        assert m.counter_value("x") == 0.0
