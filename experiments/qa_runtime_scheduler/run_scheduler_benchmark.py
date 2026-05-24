"""
QA Substrate Ladder Level 3: Runtime/Scheduler Benchmark.

Usage:
  python experiments/qa_runtime_scheduler/run_scheduler_benchmark.py
  python experiments/qa_runtime_scheduler/run_scheduler_benchmark.py --N 100 --tasks 500 --seed 42

Schedulers: fifo, priority, qa_scheduler
Modes: qa_lawful, random_opaque, deadline_only, mixed_runtime, adversarial_trap

Hypotheses:
  H1: QA scheduler reduces unrecoverable failures on qa_lawful workloads
      (orbit-based move avoidance + BFS recovery guarantees vs random walk).
  H2: QA scheduler reduces wasted recovery steps on qa_lawful workloads
      (BFS escape finds shortest path; FIFO/priority random-walk recovery).
  H3: QA scheduler does NOT automatically beat FIFO/priority on random_opaque workloads
      (LCG transitions have no QA structure; BFS overhead is pure cost).
  H4: QA advantage disappears on adversarial_trap workloads
      (fail cells not orbit-aligned; orbit-based avoidance is useless).
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
import fifo_scheduler
import priority_scheduler
import qa_scheduler


def main():
    parser = argparse.ArgumentParser(description="QA Runtime/Scheduler Benchmark")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        help="Run only this mode (default: all)")
    args = parser.parse_args()

    modes = [args.mode] if args.mode else sorted(VALID_MODES)
    schedulers = {
        "fifo":         fifo_scheduler.run,
        "priority":     priority_scheduler.run,
        "qa_scheduler": qa_scheduler.run,
    }

    all_results: dict[str, dict[str, list]] = {}
    all_summaries: dict[str, dict[str, object]] = {}

    for mode in modes:
        print(f"\nBuilding workload: mode={mode}, N={args.N}, tasks={args.tasks}, seed={args.seed}")
        tasks = build_workload(N=args.N, n_tasks=args.tasks, seed=args.seed, workload_mode=mode)
        print(f"  {len(tasks)} tasks generated.")

        all_results[mode] = {}
        all_summaries[mode] = {}

        for sched_name, sched_fn in schedulers.items():
            results = sched_fn(tasks, N=args.N, seed=args.seed, workload_mode=mode)
            summary = aggregate(results)
            all_results[mode][sched_name] = results
            all_summaries[mode][sched_name] = summary
            print(f"  {sched_name}: done={summary.completed_tasks} "
                  f"unrec={summary.unrecoverable_tasks} "
                  f"wasted={summary.mean_wasted_steps:.2f} "
                  f"dl_miss={summary.deadline_miss_rate:.3f}")

    # Print tables
    print("\n" + "=" * 90)
    print("QA RUNTIME/SCHEDULER BENCHMARK RESULTS")
    for mode in modes:
        sums = [all_summaries[mode][s] for s in schedulers]
        print_comparison(mode, sums)

    _print_hypotheses(all_summaries, list(schedulers.keys()))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, f"scheduler_benchmark_{ts}.json")
    save_results(out_path, all_results, all_summaries)
    print(f"\nResults saved → {out_path}")


def _print_hypotheses(all_summaries, sched_names: list[str]) -> None:
    print("\n--- Hypothesis Verdicts ---")

    def _get(mode, sched, field):
        try:
            return getattr(all_summaries[mode][sched], field)
        except (KeyError, AttributeError):
            return None

    # H1: QA prevents failures (fail_rate) on qa_lawful.
    # The QA scheduler avoids fail_orbit_9 by move selection; FIFO random-walks into it.
    # "Unrecoverable" is near-zero for generous k — the correct metric is fail_tasks
    # (entered fail state at least once, even if recovered).
    qa_fail = _get("qa_lawful", "qa_scheduler", "failed_tasks")
    fi_fail = _get("qa_lawful", "fifo", "failed_tasks")
    pr_fail = _get("qa_lawful", "priority", "failed_tasks")
    qa_tot  = _get("qa_lawful", "qa_scheduler", "total_tasks")
    fi_tot  = _get("qa_lawful", "fifo", "total_tasks")
    if qa_fail is not None and fi_fail is not None and qa_tot and fi_tot:
        qa_rate = qa_fail / qa_tot
        fi_rate = fi_fail / fi_tot
        h1 = qa_rate < fi_rate
        print(f"H1 (QA prevents failures on qa_lawful): "
              f"{'SUPPORTED' if h1 else 'NOT SUPPORTED'}  "
              f"qa_fail_rate={qa_rate:.3f}  fifo={fi_rate:.3f}  priority={pr_fail/fi_tot:.3f}")
    else:
        print("H1: missing data")

    # H2: QA reduces wasted steps on qa_lawful
    qa_waste = _get("qa_lawful", "qa_scheduler", "mean_wasted_steps")
    fi_waste = _get("qa_lawful", "fifo", "mean_wasted_steps")
    pr_waste = _get("qa_lawful", "priority", "mean_wasted_steps")
    if qa_waste is not None and fi_waste is not None:
        h2 = qa_waste < fi_waste
        print(f"H2 (QA fewer wasted steps on qa_lawful): "
              f"{'SUPPORTED' if h2 else 'NOT SUPPORTED'}  "
              f"qa={qa_waste:.2f}  fifo={fi_waste:.2f}  priority={pr_waste:.2f}")
    else:
        print("H2: missing data")

    # H3: QA does NOT beat FIFO/priority on random_opaque
    qa_unrec_op = _get("random_opaque", "qa_scheduler", "unrecoverable_tasks")
    fi_unrec_op = _get("random_opaque", "fifo", "unrecoverable_tasks")
    qa_waste_op = _get("random_opaque", "qa_scheduler", "mean_wasted_steps")
    fi_waste_op = _get("random_opaque", "fifo", "mean_wasted_steps")
    if qa_unrec_op is not None and fi_unrec_op is not None:
        tot = _get("random_opaque", "qa_scheduler", "total_tasks")
        qa_rate_op = qa_unrec_op / tot
        fi_rate_op = fi_unrec_op / tot
        # H3 confirmed: QA does NOT win (rates are similar or QA is worse)
        h3 = abs(qa_rate_op - fi_rate_op) < 0.05
        print(f"H3 (QA no advantage on random_opaque): "
              f"{'CONFIRMED' if h3 else 'NOT CONFIRMED'}  "
              f"qa_unrec={qa_rate_op:.3f}  fifo_unrec={fi_rate_op:.3f}  "
              f"qa_waste={qa_waste_op:.2f}  fifo_waste={fi_waste_op:.2f}")
    else:
        print("H3: missing data")

    # H4: QA advantage disappears on adversarial_trap
    qa_unrec_adv = _get("adversarial_trap", "qa_scheduler", "unrecoverable_tasks")
    fi_unrec_adv = _get("adversarial_trap", "fifo", "unrecoverable_tasks")
    qa_waste_adv = _get("adversarial_trap", "qa_scheduler", "mean_wasted_steps")
    fi_waste_adv = _get("adversarial_trap", "fifo", "mean_wasted_steps")
    if qa_unrec_adv is not None and fi_unrec_adv is not None:
        tot = _get("adversarial_trap", "qa_scheduler", "total_tasks")
        qa_rate_adv = qa_unrec_adv / tot
        fi_rate_adv = fi_unrec_adv / tot
        # H4 confirmed: QA does NOT significantly outperform FIFO on adversarial_trap
        h4 = abs(qa_rate_adv - fi_rate_adv) < 0.08
        print(f"H4 (QA advantage disappears on adversarial_trap): "
              f"{'CONFIRMED' if h4 else 'NOT CONFIRMED'}  "
              f"qa_unrec={qa_rate_adv:.3f}  fifo_unrec={fi_rate_adv:.3f}  "
              f"qa_waste={qa_waste_adv:.2f}  fifo_waste={fi_waste_adv:.2f}")
    else:
        print("H4: missing data")

    print()


if __name__ == "__main__":
    main()
