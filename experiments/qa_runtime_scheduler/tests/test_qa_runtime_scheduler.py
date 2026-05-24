"""
Tests for QA Runtime/Scheduler benchmark.
Run: pytest experiments/qa_runtime_scheduler/tests/test_qa_runtime_scheduler.py -q
"""
from __future__ import annotations
import sys
import os
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scheduler_core import (
    QAPacket, _legal_moves, _return_distance, _bfs_escape_path, _lcg_step,
)
from workload_builder import build_workload, VALID_MODES, FALSIFIER_MODES
from metrics import aggregate, TaskResult
import fifo_scheduler
import priority_scheduler
import qa_scheduler


# ── QA packet arithmetic ──────────────────────────────────────────────────────

class TestQAPacket:
    def test_derived_coords(self):
        p = QAPacket(3, 1)
        assert p.d == 4 and p.a == 5

    def test_C_F_G(self):
        p = QAPacket(3, 1)
        assert p.C == 8    # 2*1*4
        assert p.F == 15   # 5*3
        assert p.G == 17   # 16+1

    def test_J_X_K(self):
        p = QAPacket(3, 1)
        assert p.J == 12   # 3*4
        assert p.X == 4    # 1*4
        assert p.K == 20   # 4*5

    def test_D(self):
        p = QAPacket(3, 1)
        assert p.D == 16   # 4*4

    def test_I(self):
        p = QAPacket(3, 1)
        assert p.I == abs(p.C - p.F) == 7

    def test_orbit_9(self):
        assert QAPacket(3, 1).orbit_9 == 4
        assert QAPacket(9, 9).orbit_9 == 0

    def test_orbit_24(self):
        assert QAPacket(3, 1).orbit_24 == 4

    def test_pyth_identity(self):
        # C² + F² = G² for all (b,e) in [1,12]
        for b in range(1, 13):
            for e in range(1, 13):
                p = QAPacket(b, e)
                assert p.C * p.C + p.F * p.F == p.G * p.G

    def test_axis_identity(self):
        # J + K = 2D
        for b in range(1, 13):
            for e in range(1, 13):
                p = QAPacket(b, e)
                assert p.J + p.K == 2 * p.D


# ── Generators ────────────────────────────────────────────────────────────────

class TestGenerators:
    def test_sigma(self):
        moves = _legal_moves(3, 1, ["sigma"], 10)
        assert QAPacket(3, 2) in moves

    def test_sigma_boundary(self):
        moves = _legal_moves(3, 10, ["sigma"], 10)
        assert not any(nb.e > 10 for nb in moves)

    def test_mu(self):
        moves = _legal_moves(3, 5, ["mu"], 10)
        assert QAPacket(5, 3) in moves

    def test_mu_symmetric_skipped(self):
        # (k, k) → mu gives (k, k) = self; should be skipped
        moves = _legal_moves(4, 4, ["mu"], 10)
        assert QAPacket(4, 4) not in moves

    def test_lambda2(self):
        moves = _legal_moves(2, 3, ["lambda2"], 10)
        assert QAPacket(4, 6) in moves

    def test_lambda2_boundary(self):
        moves = _legal_moves(6, 3, ["lambda2"], 10)
        assert not any(nb.b > 10 or nb.e > 10 for nb in moves)

    def test_nu(self):
        moves = _legal_moves(4, 2, ["nu"], 10)
        assert QAPacket(2, 1) in moves

    def test_nu_invalid(self):
        # (3,1): both odd → nu produces nothing
        moves = _legal_moves(3, 1, ["nu"], 10)
        assert len(moves) == 0

    def test_all_generators_combined(self):
        moves = _legal_moves(3, 1, ["sigma", "mu", "lambda2", "nu"], 10)
        assert len(moves) >= 2


# ── Return distance ───────────────────────────────────────────────────────────

class TestReturnDistance:
    def test_already_at_target(self):
        # orbit_9 of (3,1) = 4; target = 4
        assert _return_distance(3, 1, 4, 5, 20) == 0

    def test_one_step(self):
        # (3,1) orbit=4; sigma gives (3,2) orbit=5; target=5 → dist=1
        assert _return_distance(3, 1, 5, 5, 20) == 1

    def test_unreachable_within_k(self):
        # Very tight k; some targets unreachable
        d = _return_distance(1, 1, 8, 1, 5)
        assert d <= 2  # either 1 or unreachable=2 for k=1

    def test_returns_k_plus_1_when_unreachable(self):
        d = _return_distance(1, 1, 8, 1, 100)
        # (1+1)%9=2; with k=1, reachable orbits from (1,1): sigma→(1,2) orbit=3, mu→same, lambda→(2,2) orbit=4
        # orbit=8 not reachable in 1 step → d = 2 = k+1
        assert d == 2


# ── BFS escape ───────────────────────────────────────────────────────────────

class TestBFSEscape:
    def test_already_safe(self):
        # Not in fail orbit, not in fail cells → empty path
        path = _bfs_escape_path(3, 1, 5, frozenset(), 5, 20)
        assert path == []

    def test_escape_from_fail_orbit(self):
        # Put (3,1) in fail orbit_9=4. Sigma→(3,2) orbit=5 (safe).
        path = _bfs_escape_path(3, 1, 4, frozenset(), 3, 20)
        assert len(path) > 0
        # Final state should not be in orbit 4
        b_final, e_final = path[-1]
        assert (b_final + e_final) % 9 != 4

    def test_escape_from_cell_trap(self):
        trap = frozenset([(3, 1)])
        path = _bfs_escape_path(3, 1, None, trap, 3, 20)
        assert len(path) > 0
        assert path[-1] not in trap

    def test_no_escape_tight_k(self):
        # Surrounded by fail orbit in a tiny domain
        # (1,1) orbit=2; all neighbors in small domain might also be orbit=2... unlikely but test k=0
        path = _bfs_escape_path(1, 1, 2, frozenset(), 0, 5)
        assert path == []


# ── LCG ──────────────────────────────────────────────────────────────────────

class TestLCG:
    def test_deterministic(self):
        assert _lcg_step(0) == _lcg_step(0)

    def test_range(self):
        s = 1234
        for _ in range(100):
            s = _lcg_step(s)
            assert 0 <= s <= 0xFFFF

    def test_not_constant(self):
        s0 = 100
        s1 = _lcg_step(s0)
        assert s0 != s1


# ── Workload builder ──────────────────────────────────────────────────────────

class TestWorkloadBuilder:
    def test_all_modes_build(self):
        for mode in VALID_MODES:
            tasks = build_workload(N=20, n_tasks=10, seed=7, workload_mode=mode)
            assert len(tasks) > 0

    def test_task_count(self):
        tasks = build_workload(N=20, n_tasks=30, seed=7, workload_mode="qa_lawful")
        assert len(tasks) == 30

    def test_qa_lawful_fields(self):
        tasks = build_workload(N=20, n_tasks=10, seed=7, workload_mode="qa_lawful")
        for t in tasks:
            assert t["workload_type"] == "qa_lawful"
            assert 1 <= t["initial_b"] <= 20
            assert 1 <= t["initial_e"] <= 20
            assert t["target_orbit_9"] in range(9)
            assert t["fail_orbit_9"] in range(9)
            assert t["target_orbit_9"] != t["fail_orbit_9"]

    def test_random_opaque_fields(self):
        tasks = build_workload(N=20, n_tasks=10, seed=7, workload_mode="random_opaque")
        for t in tasks:
            assert t["workload_type"] == "random_opaque"
            assert t["opaque_target"] >= 0
            assert len(t["opaque_fail_set"]) == 6

    def test_adversarial_trap_no_orbit_fail(self):
        tasks = build_workload(N=20, n_tasks=10, seed=7, workload_mode="adversarial_trap")
        for t in tasks:
            assert t["fail_orbit_9"] is None
            assert len(t["fail_cells"]) > 0

    def test_seed_determinism(self):
        t1 = build_workload(N=20, n_tasks=10, seed=99, workload_mode="qa_lawful")
        t2 = build_workload(N=20, n_tasks=10, seed=99, workload_mode="qa_lawful")
        assert [t["task_id"] for t in t1] == [t["task_id"] for t in t2]
        assert [t["initial_b"] for t in t1] == [t["initial_b"] for t in t2]

    def test_mixed_has_both_types(self):
        tasks = build_workload(N=20, n_tasks=20, seed=7, workload_mode="mixed_runtime")
        types = {t["workload_type"] for t in tasks}
        assert "qa_lawful" in types
        assert "random_opaque" in types

    def test_dependency_ids_valid(self):
        tasks = build_workload(N=20, n_tasks=30, seed=5, workload_mode="qa_lawful")
        task_ids = {t["task_id"] for t in tasks}
        for t in tasks:
            for dep in t["dependency_ids"]:
                assert dep in task_ids


# ── Scheduler runs ────────────────────────────────────────────────────────────

class TestSchedulerRuns:
    N = 15

    @pytest.fixture(scope="class")
    def qa_tasks(self):
        return build_workload(N=self.N, n_tasks=20, seed=7, workload_mode="qa_lawful")

    def test_fifo_runs(self, qa_tasks):
        results = fifo_scheduler.run(qa_tasks, N=self.N, seed=7, workload_mode="qa_lawful")
        assert len(results) == len(qa_tasks)
        for r in results:
            assert r.scheduler == "fifo"
            assert r.steps_to_completion >= 0

    def test_priority_runs(self, qa_tasks):
        results = priority_scheduler.run(qa_tasks, N=self.N, seed=7, workload_mode="qa_lawful")
        assert len(results) == len(qa_tasks)
        for r in results:
            assert r.scheduler == "priority"

    def test_qa_scheduler_runs(self, qa_tasks):
        results = qa_scheduler.run(qa_tasks, N=self.N, seed=7, workload_mode="qa_lawful")
        assert len(results) == len(qa_tasks)
        for r in results:
            assert r.scheduler == "qa_scheduler"

    def test_all_schedulers_all_modes(self):
        for mode in VALID_MODES:
            tasks = build_workload(N=self.N, n_tasks=15, seed=11, workload_mode=mode)
            for fn in [fifo_scheduler.run, priority_scheduler.run, qa_scheduler.run]:
                results = fn(tasks, N=self.N, seed=11, workload_mode=mode)
                assert len(results) == len(tasks)

    def test_latency_positive(self, qa_tasks):
        results = qa_scheduler.run(qa_tasks, N=self.N, seed=7, workload_mode="qa_lawful")
        # At least some tasks should have non-zero scheduler latency
        assert any(r.scheduler_latency_ns > 0 for r in results)

    def test_tasks_terminal(self, qa_tasks):
        for fn in [fifo_scheduler.run, priority_scheduler.run, qa_scheduler.run]:
            results = fn(qa_tasks, N=self.N, seed=7, workload_mode="qa_lawful")
            for r in results:
                assert r.completed or r.unrecoverable, (
                    f"{r.scheduler} task {r.task_id} is neither completed nor unrecoverable"
                )

    def test_wasted_steps_non_negative(self, qa_tasks):
        results = qa_scheduler.run(qa_tasks, N=self.N, seed=7, workload_mode="qa_lawful")
        assert all(r.wasted_steps >= 0 for r in results)


# ── H1: QA fewer unrecoverable on qa_lawful ──────────────────────────────────

class TestH1QALawful:
    N = 20

    @pytest.fixture(scope="class")
    def results(self):
        tasks = build_workload(N=self.N, n_tasks=50, seed=13, workload_mode="qa_lawful")
        return {
            "fifo": fifo_scheduler.run(tasks, N=self.N, seed=13, workload_mode="qa_lawful"),
            "priority": priority_scheduler.run(tasks, N=self.N, seed=13, workload_mode="qa_lawful"),
            "qa": qa_scheduler.run(tasks, N=self.N, seed=13, workload_mode="qa_lawful"),
        }

    def test_qa_fail_rate_lower_than_fifo(self, results):
        # QA avoids fail_orbit by move selection; FIFO random-walks into it.
        # Key metric: fail_at_least_once rate, not unrecoverable rate.
        qa_fail = sum(1 for r in results["qa"] if r.failed_at_least_once) / len(results["qa"])
        fi_fail = sum(1 for r in results["fifo"] if r.failed_at_least_once) / len(results["fifo"])
        assert qa_fail <= fi_fail, (
            f"QA fail rate {qa_fail:.3f} >= FIFO {fi_fail:.3f}"
        )

    def test_qa_failure_rate_not_higher_than_fifo(self, results):
        # QA avoids fail states proactively (rik_attempts may be 0 = never failed at all).
        # Either QA fails less often, or it never enters fail states (rik_attempts=0).
        qa_fail = sum(1 for r in results["qa"] if r.failed_at_least_once) / len(results["qa"])
        fi_fail = sum(1 for r in results["fifo"] if r.failed_at_least_once) / len(results["fifo"])
        assert qa_fail <= fi_fail, (
            f"QA fail rate {qa_fail:.3f} > FIFO {fi_fail:.3f}"
        )


# ── H2: QA fewer wasted steps ────────────────────────────────────────────────

class TestH2WastedSteps:
    N = 20

    @pytest.fixture(scope="class")
    def results(self):
        tasks = build_workload(N=self.N, n_tasks=50, seed=17, workload_mode="qa_lawful")
        return {
            "fifo": fifo_scheduler.run(tasks, N=self.N, seed=17, workload_mode="qa_lawful"),
            "qa": qa_scheduler.run(tasks, N=self.N, seed=17, workload_mode="qa_lawful"),
        }

    def test_qa_wasted_steps_not_greater(self, results):
        qa_w = sum(r.wasted_steps for r in results["qa"])
        fi_w = sum(r.wasted_steps for r in results["fifo"])
        assert qa_w <= fi_w, f"QA total wasted {qa_w} > FIFO {fi_w}"


# ── H3: QA no advantage on random_opaque ─────────────────────────────────────

class TestH3RandomOpaque:
    N = 20

    @pytest.fixture(scope="class")
    def results(self):
        tasks = build_workload(N=self.N, n_tasks=50, seed=19, workload_mode="random_opaque")
        return {
            "fifo": fifo_scheduler.run(tasks, N=self.N, seed=19, workload_mode="random_opaque"),
            "qa": qa_scheduler.run(tasks, N=self.N, seed=19, workload_mode="random_opaque"),
        }

    def test_unrecoverable_rates_similar(self, results):
        qa_r = sum(1 for r in results["qa"] if r.unrecoverable) / len(results["qa"])
        fi_r = sum(1 for r in results["fifo"] if r.unrecoverable) / len(results["fifo"])
        # QA should not significantly outperform FIFO (gap < 15%)
        assert abs(qa_r - fi_r) < 0.15, (
            f"Unexpected QA advantage on random_opaque: qa={qa_r:.3f} fi={fi_r:.3f}"
        )


# ── H4: QA advantage disappears on adversarial_trap ──────────────────────────

class TestH4AdversarialTrap:
    N = 20

    @pytest.fixture(scope="class")
    def results(self):
        tasks = build_workload(N=self.N, n_tasks=50, seed=23, workload_mode="adversarial_trap")
        return {
            "fifo": fifo_scheduler.run(tasks, N=self.N, seed=23, workload_mode="adversarial_trap"),
            "qa": qa_scheduler.run(tasks, N=self.N, seed=23, workload_mode="adversarial_trap"),
        }

    def test_qa_advantage_small(self, results):
        qa_r = sum(1 for r in results["qa"] if r.unrecoverable) / len(results["qa"])
        fi_r = sum(1 for r in results["fifo"] if r.unrecoverable) / len(results["fifo"])
        # QA advantage < 12% (orbit avoidance useless for cell-based traps)
        assert abs(qa_r - fi_r) < 0.12, (
            f"QA should not dominate on adversarial_trap: qa={qa_r:.3f} fi={fi_r:.3f}"
        )


# ── Metrics aggregation ───────────────────────────────────────────────────────

class TestMetrics:
    def test_aggregate_basic(self):
        tasks = build_workload(N=15, n_tasks=10, seed=3, workload_mode="qa_lawful")
        results = fifo_scheduler.run(tasks, N=15, seed=3, workload_mode="qa_lawful")
        s = aggregate(results)
        assert s.total_tasks == len(tasks)
        assert s.completed_tasks + s.unrecoverable_tasks == s.total_tasks
        assert 0.0 <= s.deadline_miss_rate <= 1.0
        assert s.mean_wasted_steps >= 0

    def test_aggregate_repair_rate_bounds(self):
        tasks = build_workload(N=15, n_tasks=10, seed=5, workload_mode="random_opaque")
        results = priority_scheduler.run(tasks, N=15, seed=5, workload_mode="random_opaque")
        s = aggregate(results)
        assert 0.0 <= s.recovery_success_rate <= 1.0
        assert 0.0 <= s.return_in_k_success_rate <= 1.0
