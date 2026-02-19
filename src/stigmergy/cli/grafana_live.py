"""Live Grafana adapter: reads alerts, annotations, Tempo traces, and metric anomalies.

Read-only. This adapter fetches operational signals from Grafana and
converts them to Signal objects. It never modifies anything in Grafana.

Authentication: requires a Grafana service account token (glsa_...).
Set GRAFANA_API_KEY env var or configure in .stigmergy/config.yaml.

Design directive: Avoid heavy queries. Prefer Tempo search index and
targeted dashboard queries over full datasource scans.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from stigmergy.primitives.signal import Signal, SignalSource

logger = logging.getLogger(__name__)


class LiveGrafanaAdapter:
    """Fetches alerts, annotations, traces, and metric anomalies from Grafana."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str | None = None,
        dashboards: list[str] | None = None,
        tempo_services: list[str] | None = None,
        anomaly_stddev_threshold: float = 2.5,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("GRAFANA_API_KEY", "")
        self._dashboards = dashboards or []
        self._tempo_services = tempo_services or []
        self._anomaly_threshold = anomaly_stddev_threshold
        self._callback: Callable[[Signal], None] | None = None
        self._connected = False
        self._progress_callback = progress_callback
        self._error_callback = error_callback
        self._seen_keys: set[str] = set()
        self._tempo_uid: str | None = None

    async def connect(self) -> None:
        """Verify API key and discover Tempo datasource UID."""
        if not self._api_key:
            raise ConnectionError(
                "GRAFANA_API_KEY not set. Set it in your environment or "
                ".stigmergy/config.yaml to use live Grafana mode."
            )

        result = await self._api("GET", "/api/org")
        if result is None:
            raise ConnectionError("Grafana auth failed: no response from /api/org")

        # Auto-discover Tempo datasource UID
        datasources = await self._api("GET", "/api/datasources")
        if datasources and isinstance(datasources, list):
            for ds in datasources:
                if ds.get("type") == "tempo":
                    self._tempo_uid = ds.get("uid")
                    break

        self._connected = True

    async def subscribe(self, callback: Callable[[Signal], None]) -> None:
        self._callback = callback

    async def backfill(self, since: datetime) -> AsyncIterator[Signal]:
        """Fetch all signal types from Grafana since the given time."""
        since_ms = int(since.timestamp() * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_epoch = int(since.timestamp())
        now_epoch = int(datetime.now(timezone.utc).timestamp())

        phases = [
            ("alerts", self._fetch_alerts),
            ("annotations", lambda: self._fetch_annotations(since_ms, now_ms)),
            ("tempo", lambda: self._fetch_tempo_traces(since_epoch, now_epoch)),
            ("metrics", lambda: self._fetch_metric_anomalies(since_epoch, now_epoch)),
        ]
        total_phases = len(phases)

        for i, (name, fetcher) in enumerate(phases, 1):
            count = 0
            try:
                async for signal in fetcher():
                    count += 1
                    yield signal
            except Exception as e:
                logger.warning("Grafana %s fetch failed: %s", name, e)
                if self._error_callback:
                    self._error_callback(name, str(e)[:120])
            if self._progress_callback:
                self._progress_callback(name, i, total_phases, count)

    async def _fetch_alerts(self) -> AsyncIterator[Signal]:
        """Fetch firing and recently resolved alerts."""
        data = await self._api("GET", "/api/alertmanager/grafana/api/v2/alerts")
        if not data or not isinstance(data, list):
            return

        for alert in data:
            fingerprint = alert.get("fingerprint", "")
            key = f"alert:{fingerprint}"
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)

            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})
            state = alert.get("status", {}).get("state", "firing")
            severity = labels.get("severity", "warning")
            alert_name = labels.get("alertname", "Unknown")
            service = labels.get("service", "unknown")

            summary = annotations.get("summary", "")
            description = annotations.get("description", "")
            content_text = summary or description or f"{state.upper()}: {alert_name}"
            content = f"{state.upper()}: {alert_name} on {service} — {content_text}"

            folder = labels.get("grafana_folder", "default")

            meta: dict[str, Any] = {
                "event_type": "alert",
                "state": state,
                "severity": severity,
                "fingerprint": fingerprint,
                "alert_name": alert_name,
                "labels": labels,
                "starts_at": alert.get("startsAt", ""),
                "generator_url": alert.get("generatorURL", ""),
            }
            if alert.get("endsAt") and state == "resolved":
                meta["ends_at"] = alert["endsAt"]

            yield Signal(
                content=content,
                source=SignalSource.GRAFANA,
                channel=f"grafana:alerts/{folder}",
                author="grafana-alertmanager",
                timestamp=_parse_timestamp(alert.get("startsAt", "")),
                metadata=meta,
            )

    async def _fetch_annotations(self, since_ms: int, now_ms: int) -> AsyncIterator[Signal]:
        """Fetch deploy/incident annotations."""
        params = urlencode({"from": since_ms, "to": now_ms})
        data = await self._api("GET", f"/api/annotations?{params}")
        if not data or not isinstance(data, list):
            return

        for ann in data:
            ann_id = ann.get("id", 0)
            key = f"annotation:{ann_id}"
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)

            text = ann.get("text", "")
            tags = ann.get("tags", [])
            dashboard_uid = ann.get("dashboardUID", "unknown")

            meta: dict[str, Any] = {
                "event_type": "annotation",
                "annotation_id": ann_id,
                "tags": tags,
                "dashboard_uid": dashboard_uid,
                "time": _ms_to_iso(ann.get("time", 0)),
            }
            if ann.get("timeEnd") and ann["timeEnd"] != ann.get("time"):
                meta["time_end"] = _ms_to_iso(ann["timeEnd"])

            yield Signal(
                content=text or f"Annotation on {dashboard_uid}: {', '.join(tags)}",
                source=SignalSource.GRAFANA,
                channel=f"grafana:annotations/{dashboard_uid}",
                author="grafana-annotation",
                timestamp=_ms_to_datetime(ann.get("time", 0)),
                metadata=meta,
            )

    async def _fetch_tempo_traces(self, since_epoch: int, now_epoch: int) -> AsyncIterator[Signal]:
        """Fetch traces from Tempo search index. Lightweight — uses index, not trace scanning."""
        if not self._tempo_uid:
            logger.info("No Tempo datasource found, skipping trace fetch")
            return

        # Query for error traces
        async for sig in self._tempo_search(since_epoch, now_epoch, "status=error"):
            yield sig

        # Query for high-duration traces per configured service
        for service in self._tempo_services:
            traceql = f'{{ resource.service.name="{service}" && duration>1s }}'
            async for sig in self._tempo_search(since_epoch, now_epoch, traceql):
                yield sig

    async def _tempo_search(
        self, since_epoch: int, now_epoch: int, traceql: str,
    ) -> AsyncIterator[Signal]:
        """Execute a Tempo search query and yield signals."""
        params = urlencode({
            "start": since_epoch,
            "end": now_epoch,
            "q": traceql,
            "limit": 50,
        })
        path = f"/api/datasources/proxy/uid/{self._tempo_uid}/api/search?{params}"
        data = await self._api("GET", path)
        if not data:
            return

        traces = data.get("traces", [])
        for trace in traces:
            trace_id = trace.get("traceID", "")
            key = f"trace:{trace_id}"
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)

            root_service = trace.get("rootServiceName", "unknown")
            root_operation = trace.get("rootTraceName", "")
            duration_ms = trace.get("durationMs", 0)
            span_count = trace.get("spanCount", 0) if "spanCount" in trace else len(trace.get("spanSets", []))

            # Determine status from span sets
            status = "ok"
            services: list[str] = []
            if root_service and root_service not in services:
                services.append(root_service)

            for span_set in trace.get("spanSets", []):
                for span in span_set.get("spans", []):
                    attrs = {a["key"]: a.get("value", {}).get("stringValue", "")
                             for a in span.get("attributes", [])}
                    svc = attrs.get("service.name", "")
                    if svc and svc not in services:
                        services.append(svc)
                    if attrs.get("status") == "error" or attrs.get("otel.status_code") == "ERROR":
                        status = "error"

            content = (
                f"Trace: {root_operation} — {root_service}. "
                f"Duration {duration_ms}ms, {span_count} spans"
            )
            if status == "error":
                content = f"Trace ERROR: {root_operation} — {root_service}. Duration {duration_ms}ms"
            elif duration_ms > 1500:
                content = f"Trace HIGH LATENCY: {root_operation} — {root_service}. Duration {duration_ms}ms"

            meta: dict[str, Any] = {
                "event_type": "tempo_trace",
                "trace_id": trace_id,
                "root_service": root_service,
                "root_operation": root_operation,
                "duration_ms": duration_ms,
                "span_count": span_count,
                "status": status,
                "services": services,
            }

            yield Signal(
                content=content,
                source=SignalSource.GRAFANA,
                channel=f"grafana:tempo/{root_service}",
                author="grafana-tempo",
                timestamp=_ms_to_datetime(trace.get("startTimeUnixNano", 0) // 1_000_000)
                if trace.get("startTimeUnixNano")
                else datetime.now(timezone.utc),
                metadata=meta,
            )

    async def _fetch_metric_anomalies(
        self, since_epoch: int, now_epoch: int,
    ) -> AsyncIterator[Signal]:
        """For each configured dashboard, fetch panels and detect z-score anomalies."""
        for dash_uid in self._dashboards:
            dash_data = await self._api("GET", f"/api/dashboards/uid/{dash_uid}")
            if not dash_data:
                continue

            dashboard = dash_data.get("dashboard", {})
            panels = dashboard.get("panels", [])

            for panel in panels:
                panel_id = panel.get("id")
                panel_title = panel.get("title", "Unknown")
                if not panel_id:
                    continue

                # Skip non-timeseries panels
                panel_type = panel.get("type", "")
                if panel_type not in ("timeseries", "graph", "stat", "gauge", ""):
                    continue

                # Query panel data
                query_body = {
                    "queries": [{
                        "datasourceId": panel.get("datasource", {}).get("id", 1),
                        "refId": "A",
                        "expr": "",  # uses panel's saved query
                    }],
                    "from": str(since_epoch * 1000),
                    "to": str(now_epoch * 1000),
                }
                result = await self._api("POST", "/api/ds/query", body=query_body)
                if not result:
                    continue

                # Extract time series and compute z-scores
                for frame_key, frame in result.get("results", {}).items():
                    for series in frame.get("frames", []):
                        values = []
                        schema = series.get("schema", {})
                        data = series.get("data", {})
                        if "values" in data and len(data["values"]) >= 2:
                            values = [v for v in data["values"][1] if v is not None]

                        if len(values) < 10:
                            continue

                        # Z-score on last 100 points
                        recent = values[-100:]
                        mean = sum(recent) / len(recent)
                        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
                        std = math.sqrt(variance) if variance > 0 else 0

                        if std == 0:
                            continue

                        last_val = recent[-1]
                        z = (last_val - mean) / std

                        if abs(z) < self._anomaly_threshold:
                            continue

                        # Dedup by dashboard:panel:hour
                        hour_key = now_epoch // 3600
                        metric_name = ""
                        fields = schema.get("fields", [])
                        if len(fields) >= 2:
                            metric_name = fields[1].get("name", "value")
                        dedup_key = f"metric:{dash_uid}:{panel_id}:{metric_name}:{hour_key}"
                        if dedup_key in self._seen_keys:
                            continue
                        self._seen_keys.add(dedup_key)

                        direction = "above" if z > 0 else "below"
                        content = (
                            f"Metric anomaly: {metric_name} on {panel_title} "
                            f"{'spiked' if z > 0 else 'dropped'} to {last_val:.4g} "
                            f"(baseline {mean:.4g}, z-score {abs(z):.1f}). "
                            f"Dashboard: {dash_uid}."
                        )

                        meta: dict[str, Any] = {
                            "event_type": "metric_anomaly",
                            "metric_name": metric_name,
                            "dashboard_uid": dash_uid,
                            "panel_title": panel_title,
                            "value": last_val,
                            "baseline": mean,
                            "z_score": abs(z),
                            "direction": direction,
                            "labels": {},
                        }

                        # Slugify panel title for channel
                        panel_slug = panel_title.lower().replace(" ", "-")

                        yield Signal(
                            content=content,
                            source=SignalSource.GRAFANA,
                            channel=f"grafana:metrics/{dash_uid}/{panel_slug}",
                            author="grafana-metrics",
                            timestamp=datetime.now(timezone.utc),
                            metadata=meta,
                        )

    async def _api(
        self, method: str, path: str, body: dict | None = None,
    ) -> dict | list | None:
        """Call a Grafana API endpoint. Handles rate limiting and retries."""
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "stigmergy/1.0",
        }

        data = json.dumps(body).encode() if body else None
        req = Request(url, data=data, headers=headers, method=method)

        for attempt in range(3):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: urlopen(req, timeout=30)
                )
                return json.loads(response.read().decode())
            except HTTPError as e:
                if e.code in (429, 502, 503):
                    retry_after = int(e.headers.get("Retry-After", 2 ** (attempt + 1)))
                    logger.info(
                        "Grafana %d on %s, waiting %ds (attempt %d)",
                        e.code, path, retry_after, attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                    continue
                logger.debug("Grafana HTTP error (%s): %s %s", path, e.code, e.reason)
                return None
            except (URLError, json.JSONDecodeError, OSError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.debug("Grafana API error (%s): %s", path, e)
                return None
        return None


def _parse_timestamp(iso_str: str) -> datetime:
    """Parse ISO timestamp from Grafana, with fallback to now."""
    if not iso_str:
        return datetime.now(timezone.utc)
    try:
        # Handle both Z and +00:00 suffixes
        clean = iso_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond epoch to datetime."""
    if not ms:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _ms_to_iso(ms: int) -> str:
    """Convert millisecond epoch to ISO string."""
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
