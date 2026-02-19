"""Mock Grafana adapter. Returns signals shaped like real Grafana events.

Covers four signal types:
  - Alerts (firing/resolved from Grafana Alertmanager)
  - Annotations (deploy markers, incident tags)
  - Tempo traces (cross-service spans, errors, high latency)
  - Metric anomalies (z-score deviations on dashboard panels)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Callable

from stigmergy.primitives.signal import Signal, SignalSource

# ── Mock data ────────────────────────────────────────────────

_MOCK_ALERTS = [
    {
        "content": "FIRING: High error rate on pricing-service — 5xx rate > 5% for 10 minutes. Labels: severity=critical, service=pricing-service, namespace=production.",
        "channel": "grafana:alerts/platform",
        "author": "grafana-alertmanager",
        "metadata": {
            "event_type": "alert",
            "state": "firing",
            "severity": "critical",
            "fingerprint": "abc123def456",
            "alert_name": "HighErrorRate",
            "labels": {
                "service": "pricing-service",
                "namespace": "production",
                "severity": "critical",
            },
            "starts_at": "2026-02-17T14:30:00Z",
            "generator_url": "https://grafana.example.com/alerting/grafana/abc123def456/view",
        },
    },
    {
        "content": "RESOLVED: High memory usage on sync-service — memory dropped below 85% threshold. Labels: severity=warning, service=sync-service.",
        "channel": "grafana:alerts/platform",
        "author": "grafana-alertmanager",
        "metadata": {
            "event_type": "alert",
            "state": "resolved",
            "severity": "warning",
            "fingerprint": "def789ghi012",
            "alert_name": "HighMemoryUsage",
            "labels": {
                "service": "sync-service",
                "namespace": "production",
                "severity": "warning",
            },
            "starts_at": "2026-02-17T10:00:00Z",
            "ends_at": "2026-02-17T11:45:00Z",
            "generator_url": "https://grafana.example.com/alerting/grafana/def789ghi012/view",
        },
    },
    {
        "content": "FIRING: Database connection pool exhaustion on property-service — active connections 48/50. Labels: severity=critical, service=property-service.",
        "channel": "grafana:alerts/infrastructure",
        "author": "grafana-alertmanager",
        "metadata": {
            "event_type": "alert",
            "state": "firing",
            "severity": "critical",
            "fingerprint": "ghi345jkl678",
            "alert_name": "DBConnectionPoolExhaustion",
            "labels": {
                "service": "property-service",
                "namespace": "production",
                "severity": "critical",
            },
            "starts_at": "2026-02-18T08:15:00Z",
            "generator_url": "https://grafana.example.com/alerting/grafana/ghi345jkl678/view",
        },
    },
]

_MOCK_ANNOTATIONS = [
    {
        "content": "Deploy: pricing-service v2.4.1 rolled out to production. Commit: a1b2c3d. Deployer: devon.stewart.",
        "channel": "grafana:annotations/platform-overview",
        "author": "grafana-deploy",
        "metadata": {
            "event_type": "annotation",
            "annotation_id": 1042,
            "tags": ["deploy", "pricing-service", "production"],
            "dashboard_uid": "platform-overview",
            "time": "2026-02-17T16:00:00Z",
        },
    },
    {
        "content": "Incident: SEV2 — booking failures spike correlated with sync-service deploy. Region annotation from 14:30 to 15:45 UTC.",
        "channel": "grafana:annotations/platform-overview",
        "author": "grafana-incident",
        "metadata": {
            "event_type": "annotation",
            "annotation_id": 1043,
            "tags": ["incident", "sev2", "sync-service", "booking"],
            "dashboard_uid": "platform-overview",
            "time": "2026-02-17T14:30:00Z",
            "time_end": "2026-02-17T15:45:00Z",
        },
    },
]

_MOCK_TEMPO_TRACES = [
    {
        "content": "Trace: booking-create spans pricing-service, property-service, payment-service. Duration 2340ms (p99 threshold 1500ms). 12 spans across 4 services.",
        "channel": "grafana:tempo/booking-service",
        "author": "grafana-tempo",
        "metadata": {
            "event_type": "tempo_trace",
            "trace_id": "abc123def456789012345678abcdef01",
            "root_service": "booking-service",
            "root_operation": "POST /api/v2/bookings",
            "duration_ms": 2340,
            "span_count": 12,
            "status": "ok",
            "services": ["booking-service", "pricing-service", "property-service", "payment-service"],
            "spans": [
                {"service": "booking-service", "operation": "POST /api/v2/bookings", "duration_ms": 2340},
                {"service": "pricing-service", "operation": "calculatePrice", "duration_ms": 890},
                {"service": "property-service", "operation": "getAvailability", "duration_ms": 450},
                {"service": "payment-service", "operation": "authorizePayment", "duration_ms": 670},
            ],
        },
    },
    {
        "content": "Trace: sync-webhook ERROR — sync-service to property-service call failed with 503. Duration 5120ms. 8 spans, 2 errors.",
        "channel": "grafana:tempo/sync-service",
        "author": "grafana-tempo",
        "metadata": {
            "event_type": "tempo_trace",
            "trace_id": "def456789012345678abcdef01234567",
            "root_service": "sync-service",
            "root_operation": "POST /webhooks/pms",
            "duration_ms": 5120,
            "span_count": 8,
            "status": "error",
            "error_message": "upstream connect error: 503 Service Unavailable",
            "services": ["sync-service", "property-service"],
            "spans": [
                {"service": "sync-service", "operation": "POST /webhooks/pms", "duration_ms": 5120},
                {"service": "sync-service", "operation": "processWebhook", "duration_ms": 5100},
                {"service": "property-service", "operation": "updateProperty", "duration_ms": 0, "error": "503 Service Unavailable"},
            ],
        },
    },
    {
        "content": "Trace: availability-check high latency — availability-service query took 3200ms. 6 spans. Redis cache miss forced DB fallback.",
        "channel": "grafana:tempo/availability-service",
        "author": "grafana-tempo",
        "metadata": {
            "event_type": "tempo_trace",
            "trace_id": "789012345678abcdef0123456789abcd",
            "root_service": "availability-service",
            "root_operation": "GET /api/v1/availability",
            "duration_ms": 3200,
            "span_count": 6,
            "status": "ok",
            "services": ["availability-service"],
            "spans": [
                {"service": "availability-service", "operation": "GET /api/v1/availability", "duration_ms": 3200},
                {"service": "availability-service", "operation": "redis.get", "duration_ms": 2, "error": "cache miss"},
                {"service": "availability-service", "operation": "postgres.query", "duration_ms": 3100},
            ],
        },
    },
]

_MOCK_METRIC_ANOMALIES = [
    {
        "content": "Metric anomaly: pricing-service request latency p95 spiked to 1850ms (baseline 450ms, z-score 3.1). Dashboard: platform-overview, panel: Request Latency.",
        "channel": "grafana:metrics/platform-overview/request-latency",
        "author": "grafana-metrics",
        "metadata": {
            "event_type": "metric_anomaly",
            "metric_name": "http_request_duration_seconds_p95",
            "dashboard_uid": "platform-overview",
            "panel_title": "Request Latency",
            "value": 1.85,
            "baseline": 0.45,
            "z_score": 3.1,
            "direction": "above",
            "labels": {"service": "pricing-service"},
        },
    },
    {
        "content": "Metric anomaly: sync-service error rate jumped to 8.2% (baseline 0.3%, z-score 4.7). Dashboard: sync-health, panel: Error Rate.",
        "channel": "grafana:metrics/sync-health/error-rate",
        "author": "grafana-metrics",
        "metadata": {
            "event_type": "metric_anomaly",
            "metric_name": "http_requests_errors_ratio",
            "dashboard_uid": "sync-health",
            "panel_title": "Error Rate",
            "value": 0.082,
            "baseline": 0.003,
            "z_score": 4.7,
            "direction": "above",
            "labels": {"service": "sync-service"},
        },
    },
]


class MockGrafanaAdapter:
    """Generates realistic Grafana-shaped signals for testing.

    Covers alerts (firing/resolved), annotations (deploy/incident),
    Tempo traces (cross-service, errors, high latency), and metric
    anomalies (z-score deviations).
    """

    def __init__(self) -> None:
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        base_time = since
        all_events = (
            _MOCK_ALERTS
            + _MOCK_ANNOTATIONS
            + _MOCK_TEMPO_TRACES
            + _MOCK_METRIC_ANOMALIES
        )
        for i, event in enumerate(all_events):
            yield Signal(
                content=event["content"],
                source=SignalSource.GRAFANA,
                channel=event["channel"],
                author=event["author"],
                timestamp=base_time + timedelta(hours=i * 2),
                metadata=event["metadata"],
            )

    async def emit_all(self) -> list[Signal]:
        signals = []
        async for signal in self.backfill(datetime.now(timezone.utc) - timedelta(days=1)):
            signals.append(signal)
            if self._callback:
                self._callback(signal)
        return signals
