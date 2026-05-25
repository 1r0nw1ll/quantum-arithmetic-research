"""
QA Substrate Ladder Level 4: Network/Routing Benchmark.

Usage:
  python experiments/qa_network_routing/run_routing_benchmark.py
  python experiments/qa_network_routing/run_routing_benchmark.py --N 20 --packets 200 --seed 42

Routers: random, greedy, qa_router
Modes: qa_lawful, random_opaque, adversarial_congestion, mixed

Hypotheses:
  H1: QA/greedy routers deliver packets in fewer mean hops than random router
      (orbit-distance minimization finds direct paths; random walk meanders).
  H2: QA router achieves no lower orbit saturation than greedy on qa_lawful (CONFIRMED
      NULL): orbit_distance ties are rare → orbit load tiebreaker rarely fires → QA
      routes identically to greedy; abs(qa_sat - gr_sat) < 0.01.
  H3: QA orbit load balancing provides no additional orbit-saturation benefit vs
      greedy router on random_opaque workloads (no exploitable orbit structure).
  H4: On adversarial_congestion, QA and greedy route identically (all packets share the
      same source → identical orbit_load → QA load balancing is inert). Random wanders,
      taking more mean steps than greedy/QA.
"""
from __future__ import annotations
import sys
import os
import argparse
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from workload_builder import build_workload, VALID_MODES
from metrics import aggregate, print_comparison, save_results
import random_router
import greedy_router
import qa_router


def main() -> None:
    parser = argparse.ArgumentParser(description="QA Network/Routing Benchmark")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--packets", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    args = parser.parse_args()

    modes = [args.mode] if args.mode else sorted(VALID_MODES)
    routers = {
        "random":    random_router.run,
        "greedy":    greedy_router.run,
        "qa_router": qa_router.run,
    }

    all_results: dict[str, dict[str, list]] = {}
    all_summaries: dict[str, dict[str, object]] = {}

    for mode in modes:
        print(f"\nBuilding workload: mode={mode}, N={args.N}, "
              f"packets={args.packets}, seed={args.seed}")
        pkts = build_workload(N=args.N, n_packets=args.packets,
                              seed=args.seed, workload_mode=mode)
        print(f"  {len(pkts)} packets generated.")

        all_results[mode] = {}
        all_summaries[mode] = {}

        for rname, rfn in routers.items():
            results = rfn(pkts, N=args.N, seed=args.seed, workload_mode=mode)
            summary = aggregate(results)
            all_results[mode][rname] = results
            all_summaries[mode][rname] = summary
            print(f"  {rname}: del={summary.delivery_rate:.3f} "
                  f"steps={summary.mean_steps:.1f} "
                  f"cong={summary.mean_congestion_events:.2f} "
                  f"pk_node={summary.peak_node_load} "
                  f"pk_sat={summary.peak_orbit_saturation:.3f}")

    print("\n" + "=" * 90)
    print("QA NETWORK/ROUTING BENCHMARK RESULTS")
    for mode in modes:
        sums = [all_summaries[mode][r] for r in routers]
        print_comparison(mode, sums)

    _print_hypotheses(all_summaries)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, f"routing_benchmark_{ts}.json")
    save_results(out_path, all_results, all_summaries)
    print(f"\nResults saved → {out_path}")


def _print_hypotheses(all_summaries: dict) -> None:
    print("\n--- Hypothesis Verdicts ---")

    def _get(mode: str, router: str, field: str):
        try:
            return getattr(all_summaries[mode][router], field)
        except (KeyError, AttributeError):
            return None

    # H1: QA/greedy fewer mean hops than random on qa_lawful
    qa_steps  = _get("qa_lawful", "qa_router", "mean_steps")
    gr_steps  = _get("qa_lawful", "greedy",    "mean_steps")
    rnd_steps = _get("qa_lawful", "random",    "mean_steps")
    if qa_steps is not None and rnd_steps is not None:
        h1 = qa_steps < rnd_steps and gr_steps < rnd_steps
        print(f"H1 (greedy/QA fewer hops than random on qa_lawful): "
              f"{'SUPPORTED' if h1 else 'NOT SUPPORTED'}  "
              f"qa={qa_steps:.1f}  greedy={gr_steps:.1f}  random={rnd_steps:.1f}")
    else:
        print("H1: missing data")

    # H2 CONFIRMED NULL: QA orbit tiebreaker is inert on qa_lawful — orbit_distance
    # ties are rare → QA routes identically to greedy → orbit saturation is equal.
    qa_sat  = _get("qa_lawful", "qa_router", "mean_orbit_saturation")
    gr_sat  = _get("qa_lawful", "greedy",    "mean_orbit_saturation")
    rnd_sat = _get("qa_lawful", "random",    "mean_orbit_saturation")
    if qa_sat is not None and gr_sat is not None:
        h2 = abs(qa_sat - gr_sat) < 0.01
        print(f"H2 (QA orbit tiebreaker inert; QA≈greedy on qa_lawful): "
              f"{'CONFIRMED' if h2 else 'NOT CONFIRMED'}  "
              f"qa={qa_sat:.4f}  greedy={gr_sat:.4f}  random={rnd_sat:.4f}")
    else:
        print("H2: missing data")

    # H3: QA load balancing no benefit vs greedy on random_opaque
    qa_sat_op  = _get("random_opaque", "qa_router", "peak_orbit_saturation")
    gr_sat_op  = _get("random_opaque", "greedy",    "peak_orbit_saturation")
    if qa_sat_op is not None and gr_sat_op is not None:
        # H3 confirmed: gap < 5% of max saturation
        h3 = abs(qa_sat_op - gr_sat_op) < 0.05
        print(f"H3 (QA no orbit-sat advantage vs greedy on random_opaque): "
              f"{'CONFIRMED' if h3 else 'NOT CONFIRMED'}  "
              f"qa={qa_sat_op:.3f}  greedy={gr_sat_op:.3f}")
    else:
        print("H3: missing data")

    # H4: QA degenerates to greedy on adversarial_congestion (all packets identical pos →
    #     identical orbit_load → identical choices). Random wanders → more mean steps.
    qa_steps  = _get("adversarial_congestion", "qa_router", "mean_steps")
    gr_steps  = _get("adversarial_congestion", "greedy",    "mean_steps")
    rnd_steps = _get("adversarial_congestion", "random",    "mean_steps")
    if qa_steps is not None and gr_steps is not None:
        h4 = abs(qa_steps - gr_steps) < 0.5 and rnd_steps > gr_steps + 1.0
        print(f"H4 (QA==greedy on adversarial_congestion; random wanders): "
              f"{'CONFIRMED' if h4 else 'NOT CONFIRMED'}  "
              f"qa={qa_steps:.1f}  greedy={gr_steps:.1f}  random={rnd_steps:.1f}")
    else:
        print("H4: missing data")

    print()


if __name__ == "__main__":
    main()
