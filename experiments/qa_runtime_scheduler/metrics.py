"""
Scheduler benchmark metrics.
"""
from __future__ import annotations
import json
import statistics
from dataclasses import dataclass, asdict


@dataclass
class TaskResult:
    task_id: str
    scheduler: str
    workload_mode: str
    completed: bool
    failed_at_least_once: bool
    recovered: bool
    unrecoverable: bool
    steps_to_completion: int
    deadline_missed: bool
    wasted_steps: int
    recovery_attempts: int
    scheduler_latency_ns: int
    component_traps: int
    return_in_k_attempts: int
    return_in_k_successes: int


@dataclass
class SchedulerSummary:
    scheduler: str
    workload_mode: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    recovered_tasks: int
    unrecoverable_tasks: int
    mean_steps_to_completion: float
    deadline_miss_rate: float
    mean_wasted_steps: float
    recovery_success_rate: float
    mean_scheduler_latency_ns: float
    total_runtime_steps: int
    component_trap_count: int
    return_in_k_success_rate: float


def aggregate(results: list[TaskResult]) -> SchedulerSummary:
    n = len(results)
    completed = [r for r in results if r.completed]
    with_recovery = [r for r in results if r.recovery_attempts > 0]
    recovered = [r for r in results if r.recovered]
    rik_attempts = sum(r.return_in_k_attempts for r in results)
    rik_successes = sum(r.return_in_k_successes for r in results)
    return SchedulerSummary(
        scheduler=results[0].scheduler,
        workload_mode=results[0].workload_mode,
        total_tasks=n,
        completed_tasks=len(completed),
        failed_tasks=sum(1 for r in results if r.failed_at_least_once),
        recovered_tasks=len(recovered),
        unrecoverable_tasks=sum(1 for r in results if r.unrecoverable),
        mean_steps_to_completion=(
            statistics.mean(r.steps_to_completion for r in completed) if completed else 0.0
        ),
        deadline_miss_rate=sum(1 for r in results if r.deadline_missed) / n,
        mean_wasted_steps=statistics.mean(r.wasted_steps for r in results),
        recovery_success_rate=(
            len(recovered) / len(with_recovery) if with_recovery else 1.0
        ),
        mean_scheduler_latency_ns=statistics.mean(r.scheduler_latency_ns for r in results),
        total_runtime_steps=sum(r.steps_to_completion for r in results),
        component_trap_count=sum(r.component_traps for r in results),
        return_in_k_success_rate=rik_successes / max(1, rik_attempts),
    )


def print_comparison(mode: str, summaries: list[SchedulerSummary]) -> None:
    print(f"\n{'='*90}")
    print(f"  Mode: {mode}")
    print(f"{'='*90}")
    hdr = (f"  {'scheduler':<22} {'done':>5} {'unrec':>6} {'fail':>5} "
           f"{'rec':>5} {'dl_miss':>8} {'wasted':>8} {'rik%':>6} {'steps':>7}")
    print(hdr)
    print(f"  {'-'*88}")
    for s in summaries:
        rik_pct = f"{s.return_in_k_success_rate*100:.0f}%" if s.return_in_k_success_rate >= 0 else "n/a"
        print(
            f"  {s.scheduler:<22} {s.completed_tasks:>5} {s.unrecoverable_tasks:>6} "
            f"{s.failed_tasks:>5} {s.recovered_tasks:>5} "
            f"{s.deadline_miss_rate:>8.3f} {s.mean_wasted_steps:>8.2f} "
            f"{rik_pct:>6} {s.mean_steps_to_completion:>7.1f}"
        )


def save_results(path: str,
                 all_results: dict[str, dict[str, list[TaskResult]]],
                 all_summaries: dict[str, dict[str, SchedulerSummary]]) -> None:
    payload = {
        "summaries": {
            mode: {sched: asdict(s) for sched, s in scheds.items()}
            for mode, scheds in all_summaries.items()
        },
        "per_task": {
            mode: {
                sched: [asdict(r) for r in results]
                for sched, results in scheds.items()
            }
            for mode, scheds in all_results.items()
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
