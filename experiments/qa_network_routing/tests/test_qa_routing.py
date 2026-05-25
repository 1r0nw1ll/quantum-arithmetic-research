"""
Tests for QA Network/Routing benchmark.
Run: pytest experiments/qa_network_routing/tests/test_qa_routing.py -q
"""
from __future__ import annotations
import sys
import os
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from network_core import (
    legal_neighbors, orbit_9, precompute_orbit_distances,
    build_reverse_graph, Packet, run_simulation,
)
from workload_builder import build_workload, VALID_MODES
from metrics import aggregate, PacketResult
import random_router
import greedy_router
import qa_router


# ── Legal neighbors ───────────────────────────────────────────────────────────

class TestLegalNeighbors:
    def test_sigma(self):
        assert (3, 2) in legal_neighbors(3, 1, 10)

    def test_sigma_boundary(self):
        nbrs = legal_neighbors(3, 10, 10)
        assert not any(e > 10 for _, e in nbrs)

    def test_mu(self):
        assert (5, 3) in legal_neighbors(3, 5, 10)

    def test_mu_skipped_when_symmetric(self):
        # (k,k) → mu produces same state → must be excluded
        assert (4, 4) not in legal_neighbors(4, 4, 10)

    def test_lambda2(self):
        assert (4, 6) in legal_neighbors(2, 3, 10)

    def test_lambda2_boundary(self):
        nbrs = legal_neighbors(6, 3, 10)
        assert not any(b > 10 or e > 10 for b, e in nbrs)

    def test_nu(self):
        assert (2, 1) in legal_neighbors(4, 2, 10)

    def test_nu_odd_skipped(self):
        nbrs = legal_neighbors(3, 1, 10)
        assert (1, 0) not in nbrs  # (3/2, 1/2) out of range anyway; 3 is odd → nu skipped

    def test_no_self_loop(self):
        for b in range(1, 6):
            for e in range(1, 6):
                nbrs = legal_neighbors(b, e, 5)
                assert (b, e) not in nbrs, f"self-loop at ({b},{e})"

    def test_no_duplicates(self):
        for b in range(1, 6):
            for e in range(1, 6):
                nbrs = legal_neighbors(b, e, 5)
                assert len(nbrs) == len(set(nbrs))


# ── Orbit ─────────────────────────────────────────────────────────────────────

class TestOrbit9:
    def test_basic(self):
        assert orbit_9(3, 1) == 4   # (3+1)%9
        assert orbit_9(9, 9) == 0   # (9+9)=18, 18%9=0

    def test_range(self):
        for b in range(1, 10):
            for e in range(1, 10):
                assert 0 <= orbit_9(b, e) <= 8


# ── Precomputed distances ─────────────────────────────────────────────────────

class TestPrecomputeDistances:
    N = 10

    @pytest.fixture(scope="class")
    def dist_table(self):
        return precompute_orbit_distances(self.N)

    def test_shape(self, dist_table):
        assert len(dist_table) == self.N
        assert len(dist_table[0]) == self.N
        assert len(dist_table[0][0]) == 9

    def test_self_distance_zero(self, dist_table):
        # (b,e) with (b+e)%9 == t → dist_table[b-1][e-1][t] == 0
        for b in range(1, self.N + 1):
            for e in range(1, self.N + 1):
                t = orbit_9(b, e)
                assert dist_table[b - 1][e - 1][t] == 0, (
                    f"({b},{e}) orbit {t}: dist should be 0"
                )

    def test_sigma_neighbor_distance(self, dist_table):
        # (3,1) has orbit=4; sigma gives (3,2) orbit=5.
        # dist_table[2][0][5] = dist from (3,2) to orbit 5, but (3,2) IS orbit 5 → 0.
        # Instead: (3,1) orbit=4, dist to orbit=5 should be 1 (one sigma step).
        d = dist_table[2][0][5]  # (b=3, e=1) → dist to orbit 5; sigma gives orbit 5
        assert d == 1, f"Expected 1 hop from (3,1) to orbit 5, got {d}"

    def test_reachable_from_all(self, dist_table):
        INF = self.N * self.N + 1
        # From any cell, orbit=0 should be reachable (via repeated sigma)
        unreachable = [
            (b, e) for b in range(1, self.N + 1) for e in range(1, self.N + 1)
            if dist_table[b - 1][e - 1][0] >= INF
        ]
        # Allow a small number of stuck cells (none expected for N=10)
        assert len(unreachable) == 0, f"Cells unreachable to orbit 0: {unreachable[:5]}"

    def test_non_negative(self, dist_table):
        for row in dist_table:
            for col in row:
                for d in col:
                    assert d >= 0


# ── Packet ────────────────────────────────────────────────────────────────────

class TestPacket:
    def test_fields(self):
        pkt = Packet("p0", (3, 1), 7, "qa_lawful")
        assert pkt.packet_id == "p0"
        assert pkt.src == (3, 1)
        assert pkt.dst_orbit == 7
        assert pkt.workload_type == "qa_lawful"

    def test_frozen(self):
        pkt = Packet("p0", (3, 1), 7, "qa_lawful")
        with pytest.raises((AttributeError, TypeError)):
            pkt.dst_orbit = 3  # frozen dataclass, should raise

    def test_orbit_at_src(self):
        pkt = Packet("p0", (3, 1), 4, "qa_lawful")
        # Packet is an immutable descriptor; delivery state lives in _PacketState
        assert orbit_9(*pkt.src) == pkt.dst_orbit  # src IS at dst_orbit


# ── Workload builder ──────────────────────────────────────────────────────────

class TestWorkloadBuilder:
    N = 10

    def test_all_modes_build(self):
        for mode in VALID_MODES:
            pkts = build_workload(N=self.N, n_packets=10, seed=7, workload_mode=mode)
            assert len(pkts) == 10

    def test_qa_lawful_dst_orbit_differs(self):
        pkts = build_workload(N=self.N, n_packets=20, seed=7, workload_mode="qa_lawful")
        for pkt in pkts:
            assert orbit_9(*pkt.src) != pkt.dst_orbit

    def test_adversarial_congestion_all_same_src(self):
        pkts = build_workload(N=self.N, n_packets=20, seed=7,
                              workload_mode="adversarial_congestion")
        srcs = {pkt.src for pkt in pkts}
        assert len(srcs) == 1, "All packets should share the same source cell"

    def test_adversarial_congestion_all_same_dst_orbit(self):
        pkts = build_workload(N=self.N, n_packets=20, seed=7,
                              workload_mode="adversarial_congestion")
        dst_orbits = {pkt.dst_orbit for pkt in pkts}
        assert len(dst_orbits) == 1

    def test_seed_determinism(self):
        p1 = build_workload(N=self.N, n_packets=10, seed=99, workload_mode="qa_lawful")
        p2 = build_workload(N=self.N, n_packets=10, seed=99, workload_mode="qa_lawful")
        assert [p.src for p in p1] == [p.src for p in p2]
        assert [p.dst_orbit for p in p1] == [p.dst_orbit for p in p2]

    def test_mixed_has_both_types(self):
        pkts = build_workload(N=self.N, n_packets=20, seed=7, workload_mode="mixed")
        types = {pkt.workload_type for pkt in pkts}
        assert "qa_lawful" in types
        assert "random_opaque" in types


# ── Router runs ───────────────────────────────────────────────────────────────

class TestRouterRuns:
    N = 10

    @pytest.fixture(scope="class")
    def qa_pkts(self):
        return build_workload(N=self.N, n_packets=20, seed=7, workload_mode="qa_lawful")

    def test_random_runs(self, qa_pkts):
        results = random_router.run(qa_pkts, N=self.N, seed=7, workload_mode="qa_lawful")
        assert len(results) == len(qa_pkts)
        for r in results:
            assert r.router == "random"
            assert r.steps >= 0

    def test_greedy_runs(self, qa_pkts):
        results = greedy_router.run(qa_pkts, N=self.N, seed=7, workload_mode="qa_lawful")
        assert len(results) == len(qa_pkts)
        for r in results:
            assert r.router == "greedy"

    def test_qa_router_runs(self, qa_pkts):
        results = qa_router.run(qa_pkts, N=self.N, seed=7, workload_mode="qa_lawful")
        assert len(results) == len(qa_pkts)
        for r in results:
            assert r.router == "qa_router"

    def test_all_routers_all_modes(self):
        for mode in VALID_MODES:
            pkts = build_workload(N=self.N, n_packets=15, seed=11, workload_mode=mode)
            for fn in [random_router.run, greedy_router.run, qa_router.run]:
                results = fn(pkts, N=self.N, seed=11, workload_mode=mode)
                assert len(results) == len(pkts)

    def test_congestion_events_non_negative(self, qa_pkts):
        results = qa_router.run(qa_pkts, N=self.N, seed=7, workload_mode="qa_lawful")
        assert all(r.congestion_events >= 0 for r in results)

    def test_peak_node_load_positive(self, qa_pkts):
        results = random_router.run(qa_pkts, N=self.N, seed=7, workload_mode="qa_lawful")
        assert any(r.peak_node_load >= 1 for r in results)

    def test_orbit_saturation_bounded(self, qa_pkts):
        results = qa_router.run(qa_pkts, N=self.N, seed=7, workload_mode="qa_lawful")
        for r in results:
            assert 0.0 <= r.mean_orbit_saturation <= 1.0
            assert 0.0 <= r.peak_orbit_saturation <= 1.0


# ── H1: greedy/QA fewer hops than random ─────────────────────────────────────

class TestH1GreedyFasterThanRandom:
    N = 15

    @pytest.fixture(scope="class")
    def results(self):
        pkts = build_workload(N=self.N, n_packets=50, seed=13, workload_mode="qa_lawful")
        return {
            "random": random_router.run(pkts, N=self.N, seed=13, workload_mode="qa_lawful"),
            "greedy": greedy_router.run(pkts, N=self.N, seed=13, workload_mode="qa_lawful"),
            "qa":     qa_router.run(pkts, N=self.N, seed=13, workload_mode="qa_lawful"),
        }

    def test_greedy_fewer_mean_hops_than_random(self, results):
        rnd = aggregate(results["random"]).mean_steps
        gr = aggregate(results["greedy"]).mean_steps
        assert gr < rnd, f"Greedy ({gr:.1f}) should use fewer hops than random ({rnd:.1f})"

    def test_qa_fewer_mean_hops_than_random(self, results):
        rnd = aggregate(results["random"]).mean_steps
        qa = aggregate(results["qa"]).mean_steps
        assert qa < rnd, f"QA ({qa:.1f}) should use fewer hops than random ({rnd:.1f})"


# ── H2: QA orbit tiebreaker is inert on qa_lawful (confirmed null) ───────────

class TestH2QAOrbitTiebreakerInert:
    N = 15

    @pytest.fixture(scope="class")
    def results(self):
        pkts = build_workload(N=self.N, n_packets=50, seed=17, workload_mode="qa_lawful")
        return {
            "greedy": greedy_router.run(pkts, N=self.N, seed=17, workload_mode="qa_lawful"),
            "qa":     qa_router.run(pkts, N=self.N, seed=17, workload_mode="qa_lawful"),
        }

    def test_qa_orbit_saturation_same_as_greedy(self, results):
        qa_sat = aggregate(results["qa"]).mean_orbit_saturation
        gr_sat = aggregate(results["greedy"]).mean_orbit_saturation
        assert abs(qa_sat - gr_sat) < 0.01, (
            f"QA and greedy should have equal orbit saturation on qa_lawful "
            f"(tiebreaker is inert): qa={qa_sat:.4f}  greedy={gr_sat:.4f}"
        )


# ── H3: QA no extra orbit-sat benefit vs greedy on random_opaque ─────────────

class TestH3RandomOpaqueNoBenefit:
    N = 15

    @pytest.fixture(scope="class")
    def results(self):
        pkts = build_workload(N=self.N, n_packets=50, seed=19, workload_mode="random_opaque")
        return {
            "greedy": greedy_router.run(pkts, N=self.N, seed=19, workload_mode="random_opaque"),
            "qa":     qa_router.run(pkts, N=self.N, seed=19, workload_mode="random_opaque"),
        }

    def test_orbit_saturation_similar(self, results):
        qa_sat = aggregate(results["qa"]).peak_orbit_saturation
        gr_sat = aggregate(results["greedy"]).peak_orbit_saturation
        assert abs(qa_sat - gr_sat) < 0.06, (
            f"QA and greedy orbit saturation should be similar on random_opaque: "
            f"qa={qa_sat:.3f}  greedy={gr_sat:.3f}"
        )


# ── H4: adversarial_congestion — random disperses, greedy/QA cluster ─────────

class TestH4AdversarialCongestion:
    N = 10

    @pytest.fixture(scope="class")
    def results(self):
        pkts = build_workload(N=self.N, n_packets=40, seed=23,
                              workload_mode="adversarial_congestion")
        return {
            "random": random_router.run(pkts, N=self.N, seed=23,
                                        workload_mode="adversarial_congestion"),
            "greedy": greedy_router.run(pkts, N=self.N, seed=23,
                                        workload_mode="adversarial_congestion"),
            "qa":     qa_router.run(pkts, N=self.N, seed=23,
                                    workload_mode="adversarial_congestion"),
        }

    def test_qa_identical_mean_steps_to_greedy(self, results):
        # Identical starting position → identical orbit_load → QA degenerates to greedy
        qa_steps = aggregate(results["qa"]).mean_steps
        gr_steps = aggregate(results["greedy"]).mean_steps
        assert abs(qa_steps - gr_steps) < 0.5, (
            f"QA and greedy should route identically on adversarial_congestion: "
            f"qa={qa_steps:.2f}  greedy={gr_steps:.2f}"
        )

    def test_random_more_steps_than_greedy(self, results):
        # Random wanders; greedy/QA take the minimum-distance path
        rnd_steps = aggregate(results["random"]).mean_steps
        gr_steps  = aggregate(results["greedy"]).mean_steps
        assert rnd_steps > gr_steps + 1.0, (
            f"Random should take more steps than greedy on adversarial_congestion: "
            f"random={rnd_steps:.2f}  greedy={gr_steps:.2f}"
        )

    def test_greedy_and_qa_similar_peak_node(self, results):
        gr_pk = aggregate(results["greedy"]).peak_node_load
        qa_pk = aggregate(results["qa"]).peak_node_load
        # Both make identical deterministic choices → same clustering
        assert gr_pk == qa_pk, (
            f"Greedy and QA should cluster identically on adversarial_congestion: "
            f"greedy={gr_pk}  qa={qa_pk}"
        )


# ── Metrics aggregation ───────────────────────────────────────────────────────

class TestMetrics:
    N = 10

    def test_aggregate_basic(self):
        pkts = build_workload(N=self.N, n_packets=10, seed=3, workload_mode="qa_lawful")
        results = greedy_router.run(pkts, N=self.N, seed=3, workload_mode="qa_lawful")
        s = aggregate(results)
        assert s.total_packets == len(pkts)
        assert 0.0 <= s.delivery_rate <= 1.0
        assert s.mean_steps >= 0
        assert 0.0 <= s.peak_orbit_saturation <= 1.0

    def test_aggregate_all_delivered(self):
        pkts = build_workload(N=self.N, n_packets=10, seed=5, workload_mode="qa_lawful")
        results = greedy_router.run(pkts, N=self.N, seed=5, workload_mode="qa_lawful")
        s = aggregate(results)
        # Greedy routing should deliver all qa_lawful packets (orbit always reachable)
        assert s.delivery_rate == 1.0

    def test_peak_orbit_saturation_in_range(self):
        pkts = build_workload(N=self.N, n_packets=10, seed=7, workload_mode="random_opaque")
        results = qa_router.run(pkts, N=self.N, seed=7, workload_mode="random_opaque")
        s = aggregate(results)
        assert 0.0 <= s.mean_orbit_saturation <= 1.0
        assert 0.0 <= s.peak_orbit_saturation <= 1.0
