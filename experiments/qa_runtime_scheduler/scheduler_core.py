"""
QA arithmetic, task state machine, and simulation engine shared by all schedulers.

QA canonical fields (integer, no float):
  d = b + e          a = e + d  (= b + 2e)
  C = 2*e*d          F = a*b
  G = d*d + e*e      J = b*d     X = e*d     K = d*a    D = d*d
  I = |C - F|
  orbit_9 = (b+e) % 9    orbit_24 = (b+e) % 24

Generators: sigma (b,e+1), mu (e,b), lambda2 (2b,2e), nu (b/2,e/2 if both even)
"""
from __future__ import annotations
import random
import time
from collections import deque
from dataclasses import dataclass

from metrics import TaskResult


# ── QA packet ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QAPacket:
    b: int
    e: int

    @property
    def d(self) -> int: return self.b + self.e
    @property
    def a(self) -> int: return self.e + self.d
    @property
    def C(self) -> int: return 2 * self.e * self.d
    @property
    def F(self) -> int: return self.a * self.b
    @property
    def G(self) -> int: d = self.d; return d * d + self.e * self.e
    @property
    def J(self) -> int: return self.b * self.d
    @property
    def X(self) -> int: return self.e * self.d
    @property
    def K(self) -> int: return self.d * self.a
    @property
    def D(self) -> int: d = self.d; return d * d
    @property
    def I(self) -> int: return abs(self.C - self.F)
    @property
    def orbit_9(self) -> int: return (self.b + self.e) % 9
    @property
    def orbit_24(self) -> int: return (self.b + self.e) % 24

    def legal_neighbors(self, N: int) -> list["QAPacket"]:
        out: list[QAPacket] = []
        # sigma
        if self.e + 1 <= N:
            out.append(QAPacket(self.b, self.e + 1))
        # mu (swap) — skip if symmetric
        mu = QAPacket(self.e, self.b)
        if mu != self and 1 <= mu.b <= N and 1 <= mu.e <= N:
            out.append(mu)
        # lambda2
        if 2 * self.b <= N and 2 * self.e <= N:
            out.append(QAPacket(2 * self.b, 2 * self.e))
        # nu (halve) — only if both even
        if self.b % 2 == 0 and self.e % 2 == 0:
            out.append(QAPacket(self.b // 2, self.e // 2))
        return out


def _legal_moves(b: int, e: int, allowed: list[str], N: int) -> list[QAPacket]:
    pkt = QAPacket(b, e)
    seen: set[tuple[int, int]] = set()
    out: list[QAPacket] = []
    for g in allowed:
        if g == "sigma" and pkt.e + 1 <= N:
            nb = QAPacket(pkt.b, pkt.e + 1)
        elif g == "mu":
            nb = QAPacket(pkt.e, pkt.b)
            if nb == pkt or not (1 <= nb.b <= N and 1 <= nb.e <= N):
                continue
        elif g == "lambda2":
            if 2 * pkt.b > N or 2 * pkt.e > N:
                continue
            nb = QAPacket(2 * pkt.b, 2 * pkt.e)
        elif g == "nu":
            if pkt.b % 2 != 0 or pkt.e % 2 != 0:
                continue
            nb = QAPacket(pkt.b // 2, pkt.e // 2)
        else:
            continue
        key = (nb.b, nb.e)
        if key not in seen:
            seen.add(key)
            out.append(nb)
    return out


def _return_distance(b: int, e: int, target_orbit_9: int, k: int, N: int) -> int:
    """BFS: min steps to reach orbit_9 == target_orbit_9 within k, else k+1."""
    if (b + e) % 9 == target_orbit_9:
        return 0
    visited: set[tuple[int, int]] = {(b, e)}
    frontier: list[tuple[int, int]] = [(b, e)]
    for dist in range(1, k + 1):
        nf: list[tuple[int, int]] = []
        for (cb, ce) in frontier:
            for nb in QAPacket(cb, ce).legal_neighbors(N):
                key = (nb.b, nb.e)
                if key not in visited:
                    if (nb.b + nb.e) % 9 == target_orbit_9:
                        return dist
                    visited.add(key)
                    nf.append(key)
        frontier = nf
    return k + 1


def _bfs_escape_path(
    b: int, e: int,
    fail_orbit_9: int | None,
    fail_cells: frozenset[tuple[int, int]],
    k: int,
    N: int,
) -> list[tuple[int, int]]:
    """BFS from fail state; return path to first safe (b,e), or [] if none within k."""
    def _fail(pb: int, pe: int) -> bool:
        if fail_orbit_9 is not None and (pb + pe) % 9 == fail_orbit_9:
            return True
        return (pb, pe) in fail_cells

    if not _fail(b, e):
        return []
    queue: deque[tuple[int, int, list[tuple[int, int]]]] = deque([(b, e, [])])
    visited: set[tuple[int, int]] = {(b, e)}
    while queue:
        cb, ce, path = queue.popleft()
        if len(path) >= k:
            continue
        for nb in QAPacket(cb, ce).legal_neighbors(N):
            key = (nb.b, nb.e)
            if key in visited:
                continue
            visited.add(key)
            new_path = path + [key]
            if not _fail(nb.b, nb.e):
                return new_path
            queue.append((nb.b, nb.e, new_path))
    return []


def _lcg_step(state: int) -> int:
    return (state * 1664525 + 1013904223) & 0xFFFF


# ── Task state ────────────────────────────────────────────────────────────────

class TaskState:
    __slots__ = (
        "task", "task_id", "workload_type",
        "b", "e",
        "opaque_state",
        "status",
        "steps", "wasted_steps",
        "deadline_tick", "deadline_missed",
        "failed_once", "recovered", "unrecoverable",
        "traps", "rik_attempts", "rik_successes", "recovery_attempts",
        "scheduler_latency_ns",
    )

    def __init__(self, task: dict, start_tick: int = 0):
        self.task = task
        self.task_id = task["task_id"]
        self.workload_type = task["workload_type"]
        self.status = "PENDING" if task["dependency_ids"] else "READY"
        self.b = task.get("initial_b", 1)
        self.e = task.get("initial_e", 1)
        self.opaque_state = task.get("opaque_initial", 0)
        self.steps = 0
        self.wasted_steps = 0
        self.deadline_tick = start_tick + task["deadline_steps"]
        self.deadline_missed = False
        self.failed_once = False
        self.recovered = False
        self.unrecoverable = False
        self.traps = 0
        self.rik_attempts = 0
        self.rik_successes = 0
        self.recovery_attempts = 0
        self.scheduler_latency_ns = 0

    def is_terminal(self) -> bool:
        return self.status in ("COMPLETED", "UNRECOVERABLE")

    def to_result(self, scheduler_name: str, workload_mode: str) -> TaskResult:
        return TaskResult(
            task_id=self.task_id,
            scheduler=scheduler_name,
            workload_mode=workload_mode,
            completed=self.status == "COMPLETED",
            failed_at_least_once=self.failed_once,
            recovered=self.recovered,
            unrecoverable=self.unrecoverable,
            steps_to_completion=self.steps,
            deadline_missed=self.deadline_missed,
            wasted_steps=self.wasted_steps,
            recovery_attempts=self.recovery_attempts,
            scheduler_latency_ns=self.scheduler_latency_ns,
            component_traps=self.traps,
            return_in_k_attempts=self.rik_attempts,
            return_in_k_successes=self.rik_successes,
        )


# ── Shared execution helpers ─────────────────────────────────────────────────

def _task_fail_state(state: TaskState) -> bool:
    task = state.task
    fail_orbit = task.get("fail_orbit_9")
    fail_cells: frozenset = task.get("fail_cells_frozen", frozenset())
    if state.workload_type in ("qa_lawful", "adversarial_trap"):
        if fail_orbit is not None and (state.b + state.e) % 9 == fail_orbit:
            return True
        return (state.b, state.e) in fail_cells
    else:
        return state.opaque_state in task.get("opaque_fail_set", set())


def _task_complete(state: TaskState) -> bool:
    task = state.task
    if state.workload_type in ("qa_lawful", "adversarial_trap"):
        return (state.b + state.e) % 9 == task["target_orbit_9"]
    else:
        return state.opaque_state == task.get("opaque_target", -1)


def execute_qa_step_random(
    state: TaskState, rng: random.Random, N: int
) -> None:
    """One QA step with RANDOM generator selection. Used by FIFO and priority."""
    task = state.task
    moves = _legal_moves(state.b, state.e, task["allowed_generators"], N)
    if not moves:
        state.status = "UNRECOVERABLE"
        state.unrecoverable = True
        return
    nb = rng.choice(moves)
    state.b, state.e = nb.b, nb.e
    state.steps += 1
    _post_qa_step(state, rng, N)


def execute_qa_step_informed(
    state: TaskState, rng: random.Random, N: int
) -> None:
    """One QA step with informed generator selection (minimize return_distance). QA scheduler."""
    task = state.task
    fail_orbit = task.get("fail_orbit_9")
    target_orbit = task["target_orbit_9"]
    k = task["recovery_window_k"]
    moves = _legal_moves(state.b, state.e, task["allowed_generators"], N)
    if not moves:
        state.status = "UNRECOVERABLE"
        state.unrecoverable = True
        return
    # Prefer moves that don't enter fail_orbit_9
    safe = [nb for nb in moves
            if fail_orbit is None or (nb.b + nb.e) % 9 != fail_orbit]
    if not safe:
        safe = moves
        state.traps += 1  # forced into fail territory
    # Among safe moves, pick one minimizing return_distance to target
    best = min(safe, key=lambda nb: _return_distance(nb.b, nb.e, target_orbit, k, N))
    state.b, state.e = best.b, best.e
    state.steps += 1
    _post_qa_step(state, rng, N)


def _post_qa_step(state: TaskState, rng: random.Random, N: int) -> None:
    """Check failure/completion after a QA move."""
    if _task_fail_state(state):
        state.failed_once = True
        state.wasted_steps += 1
        task = state.task
        fail_orbit = task.get("fail_orbit_9")
        fail_cells: frozenset = task.get("fail_cells_frozen", frozenset())
        k = task["recovery_window_k"]
        escape = _bfs_escape_path(state.b, state.e, fail_orbit, fail_cells, k, N)
        state.recovery_attempts += 1
        state.rik_attempts += 1
        if escape:
            for (pb, pe) in escape:
                state.b, state.e = pb, pe
                state.wasted_steps += 1
                state.steps += 1
            state.rik_successes += 1
            state.recovered = True
            state.status = "READY"
        else:
            state.status = "UNRECOVERABLE"
            state.unrecoverable = True
    elif _task_complete(state):
        state.status = "COMPLETED"


def execute_opaque_step(state: TaskState, rng: random.Random, N: int) -> None:
    """One LCG step for random_opaque / deadline_only tasks. Same for all schedulers."""
    state.opaque_state = _lcg_step(state.opaque_state)
    state.steps += 1
    task = state.task
    if state.opaque_state in task.get("opaque_fail_set", set()):
        state.failed_once = True
        state.wasted_steps += 1
        k = task["recovery_window_k"]
        escaped = False
        for _ in range(k):
            state.opaque_state = _lcg_step(state.opaque_state)
            state.steps += 1
            state.wasted_steps += 1
            if state.opaque_state not in task.get("opaque_fail_set", set()):
                escaped = True
                break
        state.recovery_attempts += 1
        state.rik_attempts += 1
        if escaped:
            state.rik_successes += 1
            state.recovered = True
            state.status = "READY"
        else:
            state.status = "UNRECOVERABLE"
            state.unrecoverable = True
    elif _task_complete(state):
        state.status = "COMPLETED"


# ── Simulation engine ─────────────────────────────────────────────────────────

def run_simulation(
    tasks: list[dict],
    scheduler_name: str,
    workload_mode: str,
    select_fn,        # (ready: list[TaskState], tick: int, rng) -> TaskState
    execute_fn,       # (state: TaskState, rng: Random, N: int) -> None
    N: int,
    seed: int,
    max_ticks: int,
) -> list[TaskResult]:
    rng = random.Random(seed)
    states: dict[str, TaskState] = {
        t["task_id"]: TaskState(t, start_tick=0) for t in tasks
    }
    completed_ids: set[str] = set()

    for tick in range(max_ticks):
        # Release dependencies
        for s in states.values():
            if s.status == "PENDING" and all(
                d in completed_ids for d in s.task["dependency_ids"]
            ):
                s.status = "READY"

        ready = [s for s in states.values() if s.status == "READY"]
        if not ready:
            if all(s.is_terminal() for s in states.values()):
                break
            continue

        # Select
        t0 = time.perf_counter_ns()
        selected = select_fn(ready, tick, rng)
        t1 = time.perf_counter_ns()
        selected.scheduler_latency_ns += t1 - t0

        # Execute
        execute_fn(selected, rng, N)

        # Deadline check
        if not selected.deadline_missed and tick >= selected.deadline_tick:
            selected.deadline_missed = True

        if selected.status == "COMPLETED":
            completed_ids.add(selected.task_id)

        # Hard timeout per task
        if selected.steps >= selected.task["deadline_steps"] * 3:
            if not selected.is_terminal():
                selected.status = "UNRECOVERABLE"
                selected.unrecoverable = True

    # Any still READY at end of simulation → timeout (unrecoverable)
    for s in states.values():
        if not s.is_terminal():
            s.status = "UNRECOVERABLE"
            s.unrecoverable = True

    return [s.to_result(scheduler_name, workload_mode) for s in states.values()]
