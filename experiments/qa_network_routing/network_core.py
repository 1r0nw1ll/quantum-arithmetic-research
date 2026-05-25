"""
QA Network/Routing Benchmark — core.

Network: directed generator graph on (b,e) ∈ {1,...,N}².
  Nodes  = QA state pairs.
  Edges  = legal generator moves: sigma (b,e+1), mu (e,b), lambda2 (2b,2e), nu (b/2,e/2).

Packet: immutable descriptor (src, dst_orbit, workload_type).
  Delivered when a packet reaches any cell with orbit_9 == dst_orbit.

Routing simulation: fresh _PacketState objects per run — Packet descriptors are
never mutated, so the same Packet list can be passed to multiple routers safely.

Congestion: a node occupied by ≥ congestion_threshold packets in the same tick.

Precomputed distance table: dist_table[b-1][e-1][target_orbit] = min hops from (b,e)
to reach any cell at target_orbit. Built via reverse-graph multi-source BFS:
for each of the 9 target orbits, BFS in the reverse generator graph from all
target-orbit cells gives forward distances — O(9 × |V+E|) total.
"""
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass


# ── Generator graph ───────────────────────────────────────────────────────────

def legal_neighbors(b: int, e: int, N: int) -> list[tuple[int, int]]:
    """All (b',e') reachable in one generator move from (b,e) within {1,...,N}."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(nb: tuple[int, int]) -> None:
        if nb not in seen:
            seen.add(nb)
            out.append(nb)

    if e + 1 <= N:
        _add((b, e + 1))                         # sigma
    if b != e:
        _add((e, b))                              # mu
    if 2 * b <= N and 2 * e <= N:
        _add((2 * b, 2 * e))                     # lambda2
    if b % 2 == 0 and e % 2 == 0:
        _add((b // 2, e // 2))                   # nu
    return out


def orbit_9(b: int, e: int) -> int:
    return (b + e) % 9


# ── Precomputed orbit-distance table ─────────────────────────────────────────

def build_reverse_graph(N: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """rev[v] = list of nodes u such that v is a forward neighbor of u."""
    rev: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for b in range(1, N + 1):
        for e in range(1, N + 1):
            for nb in legal_neighbors(b, e, N):
                rev.setdefault(nb, []).append((b, e))
    return rev


def precompute_orbit_distances(N: int) -> list[list[list[int]]]:
    """
    Returns dist_table where dist_table[b-1][e-1][t] = min hops from (b,e)
    to reach any cell at orbit_9 == t. Value = N*N+1 if unreachable.
    """
    INF = N * N + 1
    rev = build_reverse_graph(N)
    dist_table = [[[INF] * 9 for _ in range(N)] for _ in range(N)]

    for t in range(9):
        sources = [(b, e) for b in range(1, N + 1) for e in range(1, N + 1)
                   if (b + e) % 9 == t]
        dist: dict[tuple[int, int], int] = {s: 0 for s in sources}
        queue: deque[tuple[int, int]] = deque(sources)
        while queue:
            node = queue.popleft()
            d = dist[node]
            for pred in rev.get(node, []):
                if pred not in dist:
                    dist[pred] = d + 1
                    queue.append(pred)
        for (b, e), d in dist.items():
            dist_table[b - 1][e - 1][t] = d

    return dist_table


# ── Packet (immutable descriptor) ─────────────────────────────────────────────

@dataclass(frozen=True)
class Packet:
    """Immutable. Multiple router runs can share the same Packet list safely."""
    packet_id: str
    src: tuple[int, int]
    dst_orbit: int
    workload_type: str


# ── Internal per-packet simulation state ─────────────────────────────────────

class _PacketState:
    __slots__ = ("packet", "pos", "steps", "delivered", "delivery_tick",
                 "congestion_events")

    def __init__(self, pkt: Packet) -> None:
        self.packet = pkt
        self.pos: tuple[int, int] = pkt.src
        self.steps: int = 0
        self.congestion_events: int = 0
        at_target = orbit_9(*pkt.src) == pkt.dst_orbit
        self.delivered: bool = at_target
        self.delivery_tick: int = 0 if at_target else -1


# ── Simulation engine ─────────────────────────────────────────────────────────

def run_simulation(
    packets: list[Packet],
    router_fn,          # (pos, dst_orbit, orbit_load, dist_table, N, rng) -> next_pos
    dist_table: list[list[list[int]]],
    N: int,
    seed: int,
    max_ticks: int,
    congestion_threshold: int = 2,
) -> list:
    """Returns list[PacketResult]. Packets are never mutated."""
    from metrics import PacketResult  # local import to avoid circular

    rng = random.Random(seed)
    states = [_PacketState(p) for p in packets]

    peak_node_load: int = 0
    orbit_saturation_samples: list[float] = []

    for tick in range(max_ticks):
        # Count occupancy for active (undelivered) packets
        orbit_load = [0] * 9
        node_load: dict[tuple[int, int], int] = {}
        for s in states:
            if not s.delivered:
                orbit_load[orbit_9(*s.pos)] += 1
                node_load[s.pos] = node_load.get(s.pos, 0) + 1

        n_active = sum(node_load.values())
        if n_active == 0:
            break

        tick_peak = max(node_load.values())
        if tick_peak > peak_node_load:
            peak_node_load = tick_peak

        sat = max(orbit_load) / n_active
        orbit_saturation_samples.append(sat)

        congested = {node for node, cnt in node_load.items()
                     if cnt >= congestion_threshold}
        for s in states:
            if not s.delivered and s.pos in congested:
                s.congestion_events += 1

        # Move all undelivered packets simultaneously
        for s in states:
            if s.delivered:
                continue
            next_pos = router_fn(s.pos, s.packet.dst_orbit, orbit_load,
                                 dist_table, N, rng)
            s.pos = next_pos
            s.steps += 1
            if orbit_9(*next_pos) == s.packet.dst_orbit:
                s.delivered = True
                s.delivery_tick = tick + 1

        if all(s.delivered for s in states):
            break

    mean_sat = (sum(orbit_saturation_samples) / len(orbit_saturation_samples)
                if orbit_saturation_samples else 0.0)
    peak_sat = max(orbit_saturation_samples) if orbit_saturation_samples else 0.0

    return [
        PacketResult(
            packet_id=s.packet.packet_id,
            router="",
            workload_mode="",
            delivered=s.delivered,
            steps=s.steps,
            delivery_tick=s.delivery_tick,
            congestion_events=s.congestion_events,
            peak_node_load=peak_node_load,
            mean_orbit_saturation=mean_sat,
            peak_orbit_saturation=peak_sat,
        )
        for s in states
    ]
