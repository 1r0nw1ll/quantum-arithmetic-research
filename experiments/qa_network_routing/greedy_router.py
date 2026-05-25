"""
Greedy (shortest orbit-distance) router.

At each hop, picks the legal neighbor with minimum precomputed orbit-distance to
dst_orbit. No congestion/load awareness. Ties broken by list order (deterministic).

Establishes the path-length baseline: how well pure orbit-distance minimization
performs without QA load balancing.
"""
from __future__ import annotations
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from network_core import legal_neighbors, run_simulation, precompute_orbit_distances

_INF = 10 ** 9


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
    return min(nbrs, key=lambda nb: dist_table[nb[0] - 1][nb[1] - 1][dst_orbit])


def run(packets: list, N: int = 20, seed: int = 42,
        workload_mode: str = "qa_lawful") -> list:
    dist_table = precompute_orbit_distances(N)
    max_ticks = len(packets) * 30
    results = run_simulation(packets, _route, dist_table, N, seed, max_ticks)
    for r in results:
        r.router = "greedy"
        r.workload_mode = workload_mode
    return results
