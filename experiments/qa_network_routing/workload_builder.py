"""
Workload builder for routing benchmark.

Four modes:
  qa_lawful            — packets have structured source/destination (src_orbit ≠ dst_orbit).
                         QA orbit-distance routing gives direct paths; QA load balancing
                         reduces orbit saturation vs greedy.
  random_opaque        — source and dst_orbit chosen uniformly at random.
                         No structured relationship between src_orbit and dst_orbit.
  adversarial_congestion — all packets start at the same source cell, same dst_orbit.
                         Greedy/QA make identical deterministic choices → packets stay
                         clustered → peak node congestion stays maximal.
                         Random routing disperses packets → lower peak congestion.
  mixed                — 50% qa_lawful + 50% random_opaque.
"""
from __future__ import annotations
import random

from network_core import orbit_9, legal_neighbors

VALID_MODES = {"qa_lawful", "random_opaque", "adversarial_congestion", "mixed"}


def _has_neighbors(b: int, e: int, N: int) -> bool:
    return len(legal_neighbors(b, e, N)) > 0


def build_workload(
    N: int = 20,
    n_packets: int = 200,
    seed: int = 42,
    workload_mode: str = "qa_lawful",
) -> list:
    from network_core import Packet  # avoid circular at module level

    rng = random.Random(seed)

    def _random_cell() -> tuple[int, int]:
        while True:
            b, e = rng.randint(1, N), rng.randint(1, N)
            if _has_neighbors(b, e, N):
                return b, e

    if workload_mode == "qa_lawful":
        packets = []
        for i in range(n_packets):
            src = _random_cell()
            src_orbit = orbit_9(*src)
            # dst_orbit ≠ src_orbit: pick +3 mod 9 (always reachable via sigma chain)
            dst_orbit = (src_orbit + 3) % 9
            packets.append(Packet(f"p{i:04d}", src, dst_orbit, "qa_lawful"))
        return packets

    elif workload_mode == "random_opaque":
        packets = []
        for i in range(n_packets):
            src = _random_cell()
            dst_orbit = rng.randint(0, 8)
            packets.append(Packet(f"p{i:04d}", src, dst_orbit, "random_opaque"))
        return packets

    elif workload_mode == "adversarial_congestion":
        # All packets: same source (1,1), same dst_orbit = 5.
        # Greedy/QA: (1,1) orbit=2 → best move is lambda2→(2,2) orbit=4, dist=1 to orbit=5.
        # All packets move identically → clustered at (2,2) tick 1 → max node congestion.
        # Random: ~50% pick sigma→(1,2), ~50% pick lambda2→(2,2) → dispersed.
        src = (1, 1)
        dst_orbit = 5
        packets = []
        for i in range(n_packets):
            packets.append(Packet(f"p{i:04d}", src, dst_orbit, "adversarial_congestion"))
        return packets

    elif workload_mode == "mixed":
        packets = []
        for i in range(n_packets):
            if i % 2 == 0:
                src = _random_cell()
                src_orbit = orbit_9(*src)
                dst_orbit = (src_orbit + 3) % 9
                wt = "qa_lawful"
            else:
                src = _random_cell()
                dst_orbit = rng.randint(0, 8)
                wt = "random_opaque"
            packets.append(Packet(f"p{i:04d}", src, dst_orbit, wt))
        return packets

    else:
        raise ValueError(f"Unknown workload_mode: {workload_mode!r}")
