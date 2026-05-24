"""
Priority + deadline scheduler.

Selects the READY task with the highest (priority * 10 - deadline_remaining) score.
Tasks with tighter deadlines are preferred; within equal deadlines, higher priority wins.
Generator selection is random for QA-lawful tasks.
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
    execute_qa_step_random, execute_qa_step_informed, execute_opaque_step,
)


def _urgency(state: TaskState, tick: int) -> int:
    # Higher score = more urgent.  deadline_remaining counts down.
    remaining = max(0, state.deadline_tick - tick)
    return state.task["priority"] * 100 - remaining


def _select(ready: list[TaskState], tick: int, rng: random.Random) -> TaskState:
    return max(ready, key=lambda s: _urgency(s, tick))


def _execute(state: TaskState, rng: random.Random, N: int) -> None:
    if state.workload_type == "adversarial_trap":
        # Use greedy execution so adversarial_trap isolates task-selection order,
        # not execution quality. All schedulers use the same moves here.
        execute_qa_step_informed(state, rng, N)
    elif state.workload_type == "qa_lawful":
        execute_qa_step_random(state, rng, N)
    else:
        execute_opaque_step(state, rng, N)


def run(tasks: list[dict], N: int = 100, seed: int = 42,
        workload_mode: str = "qa_lawful") -> list:
    max_ticks = len(tasks) * 80
    return run_simulation(
        tasks=tasks,
        scheduler_name="priority",
        workload_mode=workload_mode,
        select_fn=_select,
        execute_fn=_execute,
        N=N,
        seed=seed,
        max_ticks=max_ticks,
    )
