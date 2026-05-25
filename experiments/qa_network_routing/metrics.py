"""
Routing benchmark metrics.
"""
from __future__ import annotations
import json
import statistics
from dataclasses import dataclass, asdict


@dataclass
class PacketResult:
    packet_id: str
    router: str
    workload_mode: str
    delivered: bool
    steps: int
    delivery_tick: int
    congestion_events: int
    peak_node_load: int          # max packets at any node, any tick (run-level)
    mean_orbit_saturation: float # avg fraction of active packets in busiest orbit
    peak_orbit_saturation: float # max fraction of active packets in busiest orbit


@dataclass
class RouterSummary:
    router: str
    workload_mode: str
    total_packets: int
    delivered_packets: int
    delivery_rate: float
    mean_steps: float
    mean_congestion_events: float
    peak_node_load: int
    mean_orbit_saturation: float
    peak_orbit_saturation: float


def aggregate(results: list[PacketResult]) -> RouterSummary:
    n = len(results)
    delivered = [r for r in results if r.delivered]
    return RouterSummary(
        router=results[0].router,
        workload_mode=results[0].workload_mode,
        total_packets=n,
        delivered_packets=len(delivered),
        delivery_rate=len(delivered) / n,
        mean_steps=(statistics.mean(r.steps for r in delivered) if delivered else 0.0),
        mean_congestion_events=statistics.mean(r.congestion_events for r in results),
        peak_node_load=max(r.peak_node_load for r in results),
        mean_orbit_saturation=statistics.mean(r.mean_orbit_saturation for r in results),
        peak_orbit_saturation=max(r.peak_orbit_saturation for r in results),
    )


def print_comparison(mode: str, summaries: list[RouterSummary]) -> None:
    print(f"\n{'='*90}")
    print(f"  Mode: {mode}")
    print(f"{'='*90}")
    hdr = (f"  {'router':<22} {'del%':>6} {'steps':>7} "
           f"{'cong/pkt':>9} {'pk_node':>8} {'orb_sat':>8} {'pk_sat':>8}")
    print(hdr)
    print(f"  {'-'*88}")
    for s in summaries:
        print(
            f"  {s.router:<22} {s.delivery_rate:>6.3f} {s.mean_steps:>7.1f} "
            f"{s.mean_congestion_events:>9.2f} {s.peak_node_load:>8} "
            f"{s.mean_orbit_saturation:>8.3f} {s.peak_orbit_saturation:>8.3f}"
        )


def save_results(
    path: str,
    all_results: dict[str, dict[str, list[PacketResult]]],
    all_summaries: dict[str, dict[str, RouterSummary]],
) -> None:
    payload = {
        "summaries": {
            mode: {r: asdict(s) for r, s in routers.items()}
            for mode, routers in all_summaries.items()
        },
        "per_packet": {
            mode: {
                router: [asdict(r) for r in results]
                for router, results in routers.items()
            }
            for mode, routers in all_results.items()
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
