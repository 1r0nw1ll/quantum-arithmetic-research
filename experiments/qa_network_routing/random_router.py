"""
Random router.

At each hop, picks a uniformly random legal neighbor. No structural awareness.
Used as the baseline that QA and greedy routers must beat.
"""
from __future__ import annotations
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from network_core import legal_neighbors, run_simulation, precompute_orbit_distances


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
    return rng.choice(nbrs)


def run(packets: list, N: int = 20, seed: int = 42,
        workload_mode: str = "qa_lawful") -> list:
    dist_table = precompute_orbit_distances(N)
    max_ticks = len(packets) * 30
    results = run_simulation(packets, _route, dist_table, N, seed, max_ticks)
    for r in results:
        r.router = "random"
        r.workload_mode = workload_mode
    return results
