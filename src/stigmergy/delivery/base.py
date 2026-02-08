"""Delivery interface and local implementation.

Two delivery modes:
- LocalDelivery (implemented): write to .stigmergy/ mirror
- ArtifactDelivery (stubbed): create organizational artifacts
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stigmergy.mesh.insights import Insight

logger = logging.getLogger(__name__)


class DeliveryChannel(ABC):
    """Abstract delivery channel for findings."""

    @abstractmethod
    def deliver(self, finding: dict) -> bool:
        """Deliver a finding. Returns True if delivered successfully."""

    @abstractmethod
    def can_observe_state_change(self) -> bool:
        """Whether this channel can detect if the finding was acted on."""


class LocalDelivery(DeliveryChannel):
    """Deliver to the .stigmergy/ mirror.

    This is the current and primary delivery mode. Findings are written
    to local files that leadership consults deliberately. No work
    artifacts are modified.
    """

    def __init__(self, base_path: Path = Path(".stigmergy")) -> None:
        self._base_path = base_path

    def deliver(self, finding: dict) -> bool:
        # LocalDelivery is implicit — insights.jsonl and annotations.json
        # are written by InsightStore and AnnotationStore respectively.
        # This method exists for interface completeness.
        return True

    def can_observe_state_change(self) -> bool:
        # Local delivery cannot observe whether anyone acted on a finding.
        # State change detection requires external observation (future).
        return False


class ArtifactDelivery(DeliveryChannel):
    """Create organizational artifacts from findings. STUBBED.

    Future implementation will create:
    - Linear tickets for findings that need triage
    - Interface contracts (code artifacts) for coupling findings
    - Dependency diagrams for structural findings

    These artifacts close the feedback loop: a ticket that gets closed
    is a state change. A contract that gets imported is a state change.
    The decay tracker observes the artifact's lifecycle to determine
    whether the finding was acted on or normalized.

    NOT the same as annotating existing artifacts (PR comments, ticket
    annotations). This creates NEW artifacts whose existence IS the
    missing organizational knowledge.
    """

    def deliver(self, finding: dict) -> bool:
        # STUB: not implemented yet.
        # When implemented, this will:
        # 1. Determine artifact type from finding type/severity
        # 2. Create the artifact (Linear ticket, code contract, etc.)
        # 3. Record the artifact ID for state change observation
        logger.debug("ArtifactDelivery.deliver() called but not yet implemented")
        return False

    def can_observe_state_change(self) -> bool:
        # When implemented, this returns True — the created artifact's
        # lifecycle (ticket closed, contract imported, diagram referenced)
        # provides the state change signal the decay tracker needs.
        return False  # Will be True when implemented


def create_delivery_channel(config: dict | None = None) -> DeliveryChannel:
    """Factory for delivery channels.

    Currently always returns LocalDelivery. When artifact delivery
    is implemented, config will determine the channel.
    """
    return LocalDelivery()
