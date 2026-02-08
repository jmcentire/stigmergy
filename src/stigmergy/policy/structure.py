"""Structure graph: the organizational dependency map.

Models repos, services, and teams as nodes with dependency edges.
Auto-discovers edges from package.json files and known architectural patterns.
Used by policy signals to detect boundary crossings and auth surface changes.

The graph answers: "does this change cross a service boundary?"
and "does this file belong to a security-sensitive surface?"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """A node in the structure graph: repo, service, or team."""

    id: str  # e.g. "acme-org/backend", "service:pricing", "team:platform"
    node_type: str  # "repo", "service", "team", "package"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed dependency edge."""

    source: str  # node id
    target: str  # node id
    edge_type: str  # "depends_on", "owns", "publishes"
    metadata: dict[str, Any] = field(default_factory=dict)


# File patterns that indicate auth/security surface
_AUTH_PATTERNS = [
    re.compile(r"(^|/)auth[/.]", re.IGNORECASE),
    re.compile(r"(^|/)middleware/auth", re.IGNORECASE),
    re.compile(r"(^|/)security[/.]", re.IGNORECASE),
    re.compile(r"(^|/)session[/.]", re.IGNORECASE),
    re.compile(r"(^|/)jwt[/.]", re.IGNORECASE),
    re.compile(r"(^|/)token[/.]", re.IGNORECASE),
    re.compile(r"(^|/)acl[/.]", re.IGNORECASE),
    re.compile(r"(^|/)permissions?[/.]", re.IGNORECASE),
    re.compile(r"(^|/)rbac[/.]", re.IGNORECASE),
    re.compile(r"(^|/)oauth[/.]", re.IGNORECASE),
    re.compile(r"(^|/)credentials?[/.]", re.IGNORECASE),
    re.compile(r"(^|/)firewall[/.]", re.IGNORECASE),
]

# Service domain patterns: map file paths to service domains
_DOMAIN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"src/pms/bookings/"), "booking"),
    (re.compile(r"src/pms/availability/"), "availability"),
    (re.compile(r"src/data/pricing/"), "pricing"),
    (re.compile(r"src/payment/"), "payment"),
    (re.compile(r"src/pms/pms/"), "pms-integration"),
    (re.compile(r"src/pms/sync/"), "sync"),
    (re.compile(r"src/auth/"), "auth"),
    (re.compile(r"src/search/"), "search"),
    (re.compile(r"src/saas/"), "saas"),
    (re.compile(r"src/growth/"), "growth"),
    (re.compile(r"src/ai/"), "ai"),
    (re.compile(r"services/api/"), "api-server"),
    (re.compile(r"services/auth/"), "auth-server"),
    (re.compile(r"services/scheduler/"), "scheduler"),
    (re.compile(r"packages/libraries/prisma/"), "prisma-schema"),
    (re.compile(r"packages/clients/"), "external-client"),
]


class StructureGraph:
    """Organizational dependency graph with auto-discovery.

    Nodes: repos, services, teams
    Edges: depends_on, owns, publishes
    """

    def __init__(self, org_prefix: str = "org") -> None:
        self._org_prefix = org_prefix
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._adjacency: dict[str, list[str]] = {}  # source -> [targets]
        self._reverse: dict[str, list[str]] = {}  # target -> [sources]

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self._edges.append(edge)
        self._adjacency.setdefault(edge.source, []).append(edge.target)
        self._reverse.setdefault(edge.target, []).append(edge.source)

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Nodes this node depends on."""
        return self._adjacency.get(node_id, [])

    def dependents(self, node_id: str) -> list[str]:
        """Nodes that depend on this node."""
        return self._reverse.get(node_id, [])

    def edges_from(self, source: str) -> list[Edge]:
        return [e for e in self._edges if e.source == source]

    def edges_to(self, target: str) -> list[Edge]:
        return [e for e in self._edges if e.target == target]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ── Auto-discovery ──────────────────────────────────────────

    def discover_from_repos(self, repos_dir: str | Path) -> int:
        """Scan a directory of repo checkouts and build the graph.

        Returns number of edges discovered.
        """
        repos_path = Path(repos_dir)
        if not repos_path.is_dir():
            logger.warning("Repos directory not found: %s", repos_dir)
            return 0

        edges_before = len(self._edges)

        for repo_dir in sorted(repos_path.iterdir()):
            if not repo_dir.is_dir():
                continue
            if repo_dir.name.startswith("."):
                continue

            repo_id = f"{self._org_prefix}/{repo_dir.name}"
            self.add_node(Node(id=repo_id, node_type="repo", metadata={"path": str(repo_dir)}))

            # Check package.json for npm dependencies
            pkg_json = repo_dir / "package.json"
            if pkg_json.exists():
                self._discover_npm_deps(repo_id, pkg_json)

            # Check for workspace packages (monorepo)
            for nested_pkg in repo_dir.glob("packages/*/package.json"):
                self._discover_npm_deps(repo_id, nested_pkg)
            for nested_pkg in repo_dir.glob("src/*/module/package.json"):
                self._discover_npm_deps(repo_id, nested_pkg)
            for nested_pkg in repo_dir.glob("services/*/package.json"):
                self._discover_npm_deps(repo_id, nested_pkg)

        return len(self._edges) - edges_before

    def _discover_npm_deps(self, repo_id: str, pkg_path: Path) -> None:
        """Parse package.json and extract organization-internal dependencies."""
        try:
            data = json.loads(pkg_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        all_deps = {}
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            all_deps.update(data.get(key, {}))

        for dep_name in all_deps:
            # Only track internal (scoped) packages — @org/*
            if dep_name.startswith(f"@{self._org_prefix}/"):
                pkg_node_id = f"package:{dep_name}"
                if pkg_node_id not in self._nodes:
                    self.add_node(Node(id=pkg_node_id, node_type="package"))
                self.add_edge(Edge(
                    source=repo_id,
                    target=pkg_node_id,
                    edge_type="depends_on",
                    metadata={"discovered_from": str(pkg_path)},
                ))

    def discover_from_config(self, repos: list[str]) -> None:
        """Ensure all configured repos exist as nodes."""
        for repo in repos:
            if repo not in self._nodes:
                self.add_node(Node(id=repo, node_type="repo"))

    # ── Policy signal helpers ───────────────────────────────────

    def file_domains(self, file_path: str) -> list[str]:
        """Which service domains does this file path belong to?"""
        domains = []
        for pattern, domain in _DOMAIN_PATTERNS:
            if pattern.search(file_path):
                domains.append(domain)
        return domains

    def crossing_domains(self, files: list[str]) -> set[str]:
        """Which distinct domains are touched by this set of file changes?"""
        all_domains: set[str] = set()
        for f in files:
            all_domains.update(self.file_domains(f))
        return all_domains

    def is_boundary_crossing(self, files: list[str]) -> bool:
        """Do these files span more than one service domain?"""
        return len(self.crossing_domains(files)) > 1

    def shared_packages(self, repo_a: str, repo_b: str) -> list[str]:
        """Packages that both repos depend on (shared dependencies)."""
        deps_a = set(self.neighbors(repo_a))
        deps_b = set(self.neighbors(repo_b))
        return sorted(deps_a & deps_b)

    @staticmethod
    def is_auth_surface(file_path: str) -> bool:
        """Does this file path touch an authentication/authorization surface?"""
        return any(p.search(file_path) for p in _AUTH_PATTERNS)

    @staticmethod
    def auth_files(files: list[str]) -> list[str]:
        """Filter to only auth-surface files."""
        return [f for f in files if StructureGraph.is_auth_surface(f)]

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        target = path or Path(".stigmergy/structure.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: {"node_type": n.node_type, "metadata": n.metadata}
                      for nid, n in self._nodes.items()},
            "edges": [{"source": e.source, "target": e.target,
                        "edge_type": e.edge_type, "metadata": e.metadata}
                       for e in self._edges],
        }
        target.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: Path | None = None) -> bool:
        target = path or Path(".stigmergy/structure.json")
        if not target.exists():
            return False
        try:
            data = json.loads(target.read_text())
            for nid, ndata in data.get("nodes", {}).items():
                self.add_node(Node(id=nid, **ndata))
            for edata in data.get("edges", []):
                self.add_edge(Edge(**edata))
            return True
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.debug("Failed to load structure graph: %s", exc)
            return False

    def summary(self) -> str:
        """Human-readable summary."""
        repos = sum(1 for n in self._nodes.values() if n.node_type == "repo")
        packages = sum(1 for n in self._nodes.values() if n.node_type == "package")
        return f"{repos} repos, {packages} packages, {self.edge_count} edges"
