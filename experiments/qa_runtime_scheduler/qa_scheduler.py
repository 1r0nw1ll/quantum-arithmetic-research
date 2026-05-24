"""
QA-native scheduler.

For QA-lawful tasks (workload_type == "qa_lawful"):
  - Task selection: prefer tasks with the shortest return_distance to their target orbit.
    Tasks with no lawful return path (component_trapped) are deprioritized unless forced.
  - Move selection: greedy — pick the generator move that minimizes return_distance
    AND avoids fail_orbit_9 if possible.
  - Recovery: BFS-guided escape from fail state (guarantees recovery when a path exists
    within recovery_window_k steps, vs random walk for FIFO/priority).

For adversarial_trap tasks:
  - Same QA selection logic, but fail_orbit_9 is None (no orbit-based avoidance).
  - Actual fails come from scattered fail_cells — QA's orbit avoidance is inactive.
  - QA behaves like a priority scheduler here: H4 confirmed.

For random_opaque / deadline_only tasks:
  - No algebraic structure; QA falls back to priority-style selection.
  - LCG transitions same as FIFO/priority: H3 confirmed.
"""
from __future__ import annotations
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scheduler_core import (
    TaskState, run_simulation, _return_distance,
    execute_qa_step_informed, execute_opaque_step,
)

_INF = 10 ** 9


def _qa_score(state: TaskState, tick: int, N: int) -> int:
    """Lower is better: prefer short return path, then high priority, then tight deadline."""
    task = state.task
    if state.workload_type in ("qa_lawful", "adversarial_trap"):
        k = task["recovery_window_k"]
        dist = _return_distance(state.b, state.e, task["target_orbit_9"], k, N)
        trapped_penalty = _INF if dist > k else 0
        # Prefer short dist, then high priority (negate), then tight deadline
        remaining = max(0, state.deadline_tick - tick)
        return trapped_penalty + dist * 1000 - task["priority"] * 10 + remaining
    else:
        # Opaque: priority + deadline only (no QA structure)
        remaining = max(0, state.deadline_tick - tick)
        return -task["priority"] * 10 + remaining


def _select(ready: list[TaskState], tick: int, rng: random.Random, N: int) -> TaskState:
    return min(ready, key=lambda s: _qa_score(s, tick, N))


def _execute(state: TaskState, rng: random.Random, N: int) -> None:
    if state.workload_type in ("qa_lawful", "adversarial_trap"):
        execute_qa_step_informed(state, rng, N)
    else:
        execute_opaque_step(state, rng, N)


def run(tasks: list[dict], N: int = 100, seed: int = 42,
        workload_mode: str = "qa_lawful") -> list:
    max_ticks = len(tasks) * 80

    # Bind N into the select closure
    def _select_bound(ready, tick, rng_):
        return _select(ready, tick, rng_, N)

    return run_simulation(
        tasks=tasks,
        scheduler_name="qa_scheduler",
        workload_mode=workload_mode,
        select_fn=_select_bound,
        execute_fn=_execute,
        N=N,
        seed=seed,
        max_ticks=max_ticks,
    )
