"""
QA-native router.

At each hop, picks the legal neighbor that minimizes (orbit_distance_to_dst, orbit_load).
Orbit_distance is the precomputed table value; orbit_load is the live count of packets
in each orbit_9 class at this tick.

At each hop, picks the legal neighbor minimizing (orbit_distance, orbit_load_at_neighbor).
The orbit_load tiebreaker fires only when two neighbors have equal orbit_distance to dst.
On qa_lawful, optimal paths are almost always unique → QA routes identically to greedy
(H2 confirmed null). On adversarial_congestion, all packets share the same source cell
→ orbit_load is identical for all → same deterministic choice as greedy (H4 confirmed).
"""
from __future__ import annotations
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from network_core import legal_neighbors, orbit_9, run_simulation, precompute_orbit_distances


def _route(
    pos: tuple[int, int],
    dst_orbit: int,
    orbit_load: list[int],
    dist_table: list[list[list[int]]],
    N: int,
    rng: random.Random,
) -> tuple[int, int]:
    nbrs = legal_neighbors(*pos, N)
    if not nbrs:
        return pos
    # Score: (orbit_distance, orbit_load_at_neighbor_orbit)
    return min(nbrs, key=lambda nb: (
        dist_table[nb[0] - 1][nb[1] - 1][dst_orbit],
        orbit_load[orbit_9(*nb)],
    ))


def run(packets: list, N: int = 20, seed: int = 42,
        workload_mode: str = "qa_lawful") -> list:
    dist_table = precompute_orbit_distances(N)
    max_ticks = len(packets) * 30
    results = run_simulation(packets, _route, dist_table, N, seed, max_ticks)
    for r in results:
        r.router = "qa_router"
        r.workload_mode = workload_mode
    return results
