"""Graph metrics — structural analysis of the communication graph.

These metrics power three downstream detectors:
1. Bridge distance → Discovery detection (2OI = high distance)
2. Burt's constraint → Normalized Deviance detection (clique groupthink)
3. Structural holes → Channel migration detection (lost edges)
"""

from __future__ import annotations

import math
from collections import deque

from stigmergy.graph.graph import CommunicationGraph


def bridge_distance(graph: CommunicationGraph, person_a: str, person_b: str) -> float:
    """Shortest path distance in the communication graph.

    Uses BFS on the undirected neighbor graph (ignoring edge direction,
    since communication is inherently bidirectional for reachability).

    Returns float('inf') if disconnected.
    """
    a = graph.resolver.resolve(person_a)
    b = graph.resolver.resolve(person_b)

    if a == b:
        return 0.0

    if graph.get_node(a) is None or graph.get_node(b) is None:
        return float("inf")

    # BFS
    visited: set[str] = {a}
    queue: deque[tuple[str, int]] = deque([(a, 0)])

    while queue:
        current, dist = queue.popleft()
        neighbors = graph.neighbors(current)
        for neighbor in neighbors:
            if neighbor == b:
                return float(dist + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return float("inf")


def burt_constraint(graph: CommunicationGraph, person: str) -> float:
    """Burt's network constraint — measures structural embedding.

    High constraint = person is embedded in a single dense clique.
    Low constraint = person bridges multiple disconnected groups.

    Formula: C_i = sum_j (p_ij + sum_q p_iq * p_qj)^2
    where p_ij = proportional strength of tie i→j
    """
    pid = graph.resolver.resolve(person)
    neighbors = graph.neighbors(pid)

    if not neighbors:
        return 0.0

    total_weight = sum(neighbors.values())
    if total_weight == 0:
        return 0.0

    # Proportional tie strengths
    p: dict[str, float] = {n: w / total_weight for n, w in neighbors.items()}

    constraint = 0.0
    for j in neighbors:
        # Direct proportion
        direct = p[j]
        # Indirect: sum over mutual contacts
        indirect = 0.0
        j_neighbors = graph.neighbors(j)
        for q in neighbors:
            if q == j:
                continue
            p_iq = p[q]
            # Proportion of q's ties that go to j
            q_total = sum(j_neighbors.values()) if j_neighbors else 0
            p_qj = j_neighbors.get(q, 0.0) / q_total if q_total > 0 else 0.0
            indirect += p_iq * p_qj

        constraint += (direct + indirect) ** 2

    return constraint


def effective_size(graph: CommunicationGraph, person: str) -> float:
    """Burt's effective size — non-redundant contacts.

    Effective size = n - sum_j(sum_q p_iq * m_jq) for j != i
    where n = number of contacts, p_iq = proportional tie strength,
    m_jq = j's tie to q divided by j's strongest tie.

    High effective size = many non-redundant (structurally diverse) contacts.
    """
    pid = graph.resolver.resolve(person)
    neighbors = graph.neighbors(pid)
    n = len(neighbors)

    if n == 0:
        return 0.0

    total_weight = sum(neighbors.values())
    if total_weight == 0:
        return float(n)

    p: dict[str, float] = {nb: w / total_weight for nb, w in neighbors.items()}

    redundancy = 0.0
    for j in neighbors:
        j_neighbors = graph.neighbors(j)
        if not j_neighbors:
            continue
        j_max = max(j_neighbors.values()) if j_neighbors else 1.0

        marginal_redundancy = 0.0
        for q in neighbors:
            if q == j:
                continue
            p_iq = p[q]
            m_jq = j_neighbors.get(q, 0.0) / j_max if j_max > 0 else 0.0
            marginal_redundancy += p_iq * m_jq

        redundancy += marginal_redundancy

    return n - redundancy


def structural_holes(
    graph: CommunicationGraph,
    min_gap_distance: int = 3,
) -> list[tuple[str, str, float]]:
    """Identify pairs of people/clusters with weak or no connections.

    Returns list of (person_a, person_b, distance) sorted by distance descending.
    Only returns pairs where both people have at least one connection
    (isolated nodes are excluded).
    """
    nodes = graph.all_nodes()
    connected_nodes = [n for n in nodes if graph.neighbors(n)]

    holes: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()

    for a in connected_nodes:
        for b in connected_nodes:
            if a >= b:  # skip duplicates and self-pairs
                continue
            pair = (a, b) if a < b else (b, a)
            if pair in seen:
                continue
            seen.add(pair)

            dist = bridge_distance(graph, a, b)
            if dist >= min_gap_distance:
                holes.append((a, b, dist))

    holes.sort(key=lambda x: x[2], reverse=True)
    return holes


def clustering_coefficient(graph: CommunicationGraph, person: str) -> float:
    """Local clustering coefficient — how connected are a person's contacts to each other.

    Returns value in [0, 1]. 1.0 = all contacts are connected to each other (dense clique).
    0.0 = none of the contacts know each other (pure bridge position).
    """
    pid = graph.resolver.resolve(person)
    neighbors_map = graph.neighbors(pid)
    neighbor_ids = list(neighbors_map.keys())
    k = len(neighbor_ids)

    if k < 2:
        return 0.0

    # Count edges between neighbors
    links = 0
    for i, a in enumerate(neighbor_ids):
        a_neighbors = graph.neighbors(a)
        for b in neighbor_ids[i + 1:]:
            if b in a_neighbors:
                links += 1

    # Maximum possible edges between k neighbors
    max_links = k * (k - 1) / 2
    return links / max_links


def connected_components(graph: CommunicationGraph) -> list[set[str]]:
    """Find disconnected subgroups — islands in the org.

    Uses undirected BFS (treats all edges as bidirectional).
    Returns components sorted by size (largest first).
    """
    nodes = set(graph.all_nodes())
    visited: set[str] = set()
    components: list[set[str]] = []

    for node in nodes:
        if node in visited:
            continue

        # BFS from this node
        component: set[str] = set()
        queue: deque[str] = deque([node])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)

            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    components.sort(key=len, reverse=True)
    return components
