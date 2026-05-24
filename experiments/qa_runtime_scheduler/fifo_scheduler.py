"""
FIFO scheduler.

Tasks are queued in task_id order.  Generator selection is random for QA-lawful
tasks; opaque tasks use the shared LCG transition.  No structural QA awareness.
"""
from __future__ import annotations
import random
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scheduler_core import (
    TaskState, run_simulation,
    execute_qa_step_random, execute_opaque_step,
)


def _select(ready: list[TaskState], tick: int, rng: random.Random) -> TaskState:
    # FIFO: pick the task with the smallest task_id (queue order)
    return min(ready, key=lambda s: s.task_id)


def _execute(state: TaskState, rng: random.Random, N: int) -> None:
    if state.workload_type in ("qa_lawful", "adversarial_trap"):
        execute_qa_step_random(state, rng, N)
    else:
        execute_opaque_step(state, rng, N)


def run(tasks: list[dict], N: int = 100, seed: int = 42,
        workload_mode: str = "qa_lawful") -> list:
    max_ticks = len(tasks) * 80
    return run_simulation(
        tasks=tasks,
        scheduler_name="fifo",
        workload_mode=workload_mode,
        select_fn=_select,
        execute_fn=_execute,
        N=N,
        seed=seed,
        max_ticks=max_ticks,
    )
