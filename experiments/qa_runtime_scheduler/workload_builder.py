"""
Workload builder for scheduler benchmark.

Five modes:
  qa_lawful        — tasks evolve via QA generators; target/fail are orbit classes.
                     QA scheduler's structural advantage is maximal here.
  random_opaque    — tasks have LCG state transitions; no QA structure.
                     QA scheduler has no advantage.
  deadline_only    — tight deadlines, LCG transitions; priority scheduler wins.
  mixed_runtime    — 50% qa_lawful + 50% random_opaque.
  adversarial_trap — QA state + generators, but fail cells are random (not orbit-aligned).
                     QA scheduler's orbit-based avoidance gives no benefit.

Task dict schema:
  task_id, workload_type, dependency_ids, deadline_steps, priority,
  resource_cost, recovery_window_k, allowed_generators,
  # QA fields
  initial_b, initial_e, target_orbit_9, fail_orbit_9, fail_cells, fail_cells_frozen,
  # opaque fields
  opaque_initial, opaque_target, opaque_fail_set
"""
from __future__ import annotations
import random
from typing import Any

_ALL_GENERATORS = ["sigma", "mu", "lambda2", "nu"]


def _lcg_advance(state: int, steps: int) -> int:
    for _ in range(steps):
        state = (state * 1664525 + 1013904223) & 0xFFFF
    return state


def _build_qa_task(
    task_id: str,
    rng: random.Random,
    N: int,
    dep_ids: list[str],
    deadline_steps: int,
    fail_orbit_9: int | None,
    fail_cells: list[tuple[int, int]],
    workload_type: str = "qa_lawful",
) -> dict[str, Any]:
    b = rng.randint(1, N)
    e = rng.randint(1, N)
    init_orbit = (b + e) % 9
    # Target orbit is +3 (always different from init and fail)
    target_orbit = (init_orbit + 3) % 9
    fc_frozen: frozenset[tuple[int, int]] = frozenset(fail_cells)
    return {
        "task_id": task_id,
        "workload_type": workload_type,
        "dependency_ids": dep_ids,
        "deadline_steps": deadline_steps,
        "priority": rng.randint(1, 10),
        "resource_cost": 1,
        "recovery_window_k": 6,
        "allowed_generators": _ALL_GENERATORS,
        "initial_b": b,
        "initial_e": e,
        "target_orbit_9": target_orbit,
        "fail_orbit_9": fail_orbit_9,
        "fail_cells": fail_cells,
        "fail_cells_frozen": fc_frozen,
        # opaque not used
        "opaque_initial": 0,
        "opaque_target": -1,
        "opaque_fail_set": set(),
    }


def _build_opaque_task(
    task_id: str,
    rng: random.Random,
    dep_ids: list[str],
    deadline_steps: int,
) -> dict[str, Any]:
    initial = rng.randint(1, 0xFFF0)
    # Target reachable in 10-25 LCG steps
    steps_to_target = rng.randint(10, 25)
    target = _lcg_advance(initial, steps_to_target)
    # Fail states: 6 random values (not initial, not target)
    fail_set: set[int] = set()
    while len(fail_set) < 6:
        v = rng.randint(0, 0xFFFF)
        if v != initial and v != target:
            fail_set.add(v)
    return {
        "task_id": task_id,
        "workload_type": "random_opaque",
        "dependency_ids": dep_ids,
        "deadline_steps": deadline_steps,
        "priority": rng.randint(1, 10),
        "resource_cost": 1,
        "recovery_window_k": 4,
        "allowed_generators": [],
        "initial_b": 1,
        "initial_e": 1,
        "target_orbit_9": 0,
        "fail_orbit_9": None,
        "fail_cells": [],
        "fail_cells_frozen": frozenset(),
        "opaque_initial": initial,
        "opaque_target": target,
        "opaque_fail_set": fail_set,
    }


def _chain_deps(n: int, rng: random.Random, chain_prob: float = 0.15) -> list[list[str]]:
    """Build a sparse dependency list: each task has ~chain_prob chance of depending on a prior."""
    deps: list[list[str]] = [[] for _ in range(n)]
    for i in range(1, n):
        if rng.random() < chain_prob:
            j = rng.randint(0, i - 1)
            deps[i].append(f"t{j:04d}")
    return deps


# ── Public builders ───────────────────────────────────────────────────────────

def build_workload(
    N: int = 100,
    n_tasks: int = 200,
    seed: int = 42,
    workload_mode: str = "qa_lawful",
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    deps = _chain_deps(n_tasks, rng)

    if workload_mode == "qa_lawful":
        # Fail orbit = init_orbit + 6 (mod 9).  Determined per-task at build time.
        # All tasks are QA-lawful.
        tasks = []
        for i in range(n_tasks):
            t = _build_qa_task(
                task_id=f"t{i:04d}",
                rng=rng,
                N=N,
                dep_ids=deps[i],
                deadline_steps=rng.randint(25, 50),
                fail_orbit_9=None,  # set below
                fail_cells=[],
                workload_type="qa_lawful",
            )
            # Assign fail orbit after target_orbit_9 is known
            t["fail_orbit_9"] = (t["target_orbit_9"] + 3) % 9  # distinct from target and init
            tasks.append(t)
        return tasks

    elif workload_mode == "random_opaque":
        return [
            _build_opaque_task(f"t{i:04d}", rng, deps[i], rng.randint(25, 50))
            for i in range(n_tasks)
        ]

    elif workload_mode == "deadline_only":
        # Tight deadlines; priority scheduler wins.
        return [
            _build_opaque_task(f"t{i:04d}", rng, deps[i], rng.randint(8, 18))
            for i in range(n_tasks)
        ]

    elif workload_mode == "mixed_runtime":
        tasks = []
        for i in range(n_tasks):
            if i % 2 == 0:
                t = _build_qa_task(
                    f"t{i:04d}", rng, N, deps[i], rng.randint(25, 50),
                    fail_orbit_9=None, fail_cells=[], workload_type="qa_lawful",
                )
                t["fail_orbit_9"] = (t["target_orbit_9"] + 3) % 9
            else:
                t = _build_opaque_task(f"t{i:04d}", rng, deps[i], rng.randint(25, 50))
            tasks.append(t)
        return tasks

    elif workload_mode == "adversarial_trap":
        # Bait-and-trap: exploits QA's return_distance-first selection.
        #
        # Bait tasks (90%): target_orbit = initial_orbit (dist=0), priority=1, generous deadline.
        #   QA score ≈ 0 - 10 + bait_deadline → small → QA drains bait first.
        #
        # Trap tasks (10%): target = (init+3)%9 (dist=3), priority=9, tight deadline.
        #   QA score ≈ 3000 - 90 + trap_deadline → large → QA defers trap.
        #   Priority urgency = 9×100 - trap_deadline → large → priority handles trap first.
        #
        # Result: QA drains bait before touching trap → trap misses tight deadlines.
        # Priority drains trap immediately → no deadline misses.
        n_trap = max(1, n_tasks // 10)
        n_bait = n_tasks - n_trap
        trap_deadline = n_trap * 5   # priority drains trap in ≈n_trap*3 ticks; QA cannot
        bait_deadline = n_tasks * 5  # generous: never a constraint
        tasks = []
        for i in range(n_bait):
            b, e = rng.randint(1, N), rng.randint(1, N)
            init_orbit = (b + e) % 9
            tasks.append({
                "task_id": f"t{i:04d}",
                "workload_type": "adversarial_trap",
                "dependency_ids": [],
                "deadline_steps": bait_deadline,
                "priority": 1,
                "resource_cost": 1,
                "recovery_window_k": 6,
                "allowed_generators": _ALL_GENERATORS,
                "initial_b": b,
                "initial_e": e,
                "target_orbit_9": init_orbit,        # dist=0: already at target
                "fail_orbit_9": None,
                "fail_cells": [],
                "fail_cells_frozen": frozenset(),
                "opaque_initial": 0,
                "opaque_target": -1,
                "opaque_fail_set": set(),
            })
        for i in range(n_bait, n_tasks):
            b, e = rng.randint(1, N), rng.randint(1, N)
            init_orbit = (b + e) % 9
            tasks.append({
                "task_id": f"t{i:04d}",
                "workload_type": "adversarial_trap",
                "dependency_ids": [],
                "deadline_steps": trap_deadline,
                "priority": 9,
                "resource_cost": 1,
                "recovery_window_k": 6,
                "allowed_generators": _ALL_GENERATORS,
                "initial_b": b,
                "initial_e": e,
                "target_orbit_9": (init_orbit + 3) % 9,  # dist=3: needs generator steps
                "fail_orbit_9": None,
                "fail_cells": [],
                "fail_cells_frozen": frozenset(),
                "opaque_initial": 0,
                "opaque_target": -1,
                "opaque_fail_set": set(),
            })
        return tasks

    else:
        raise ValueError(f"Unknown workload_mode: {workload_mode!r}")


VALID_MODES = {"qa_lawful", "random_opaque", "deadline_only", "mixed_runtime", "adversarial_trap"}
FALSIFIER_MODES = {"random_opaque", "deadline_only", "adversarial_trap"}
