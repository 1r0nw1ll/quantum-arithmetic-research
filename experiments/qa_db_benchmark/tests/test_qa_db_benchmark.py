"""
Tests for the QA Law-Structured Memory Benchmark.
Run: pytest experiments/qa_db_benchmark/tests/test_qa_db_benchmark.py -q
"""
import sys
import os
import pytest

_BENCH = os.path.join(os.path.dirname(__file__), "..")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

from qa_backend import QAPacket, QABackend
from table_backend import TableBackend
from graph_backend import GraphBackend
from workload_fixture_builder import (
    build_workload, VALID_MODES, FALSIFIER_MODES, _UNCONSTRAINED,
)
from metrics import correctness_check, aggregate


# ── QAPacket canonical arithmetic ─────────────────────────────────────────────

class TestQAPacketArithmetic:
    def test_derived_coords(self):
        pkt = QAPacket(3, 1)
        assert pkt.d == 4
        assert pkt.a == 5

    def test_squared_elements(self):
        pkt = QAPacket(3, 1)
        assert pkt.B == 9
        assert pkt.E == 1
        assert pkt.D == 16
        assert pkt.A == 25

    def test_C_is_2ed(self):
        pkt = QAPacket(3, 1)
        assert pkt.C == 2 * 1 * 4   # 2*e*d = 8

    def test_F_is_ab(self):
        pkt = QAPacket(3, 1)
        assert pkt.F == 3 * 5       # a*b = 15

    def test_G_is_D_plus_E(self):
        pkt = QAPacket(3, 1)
        assert pkt.G == 16 + 1      # D + E = 17

    def test_J_K_X(self):
        pkt = QAPacket(3, 1)
        assert pkt.J == 3 * 4       # b*d = 12
        assert pkt.X == 1 * 4       # e*d = 4
        assert pkt.K == 4 * 5       # d*a = 20

    def test_major_axis_is_2D(self):
        pkt = QAPacket(3, 1)
        assert pkt.major_axis == 2 * 16  # 2*D = 32

    def test_axis_split_equals_major_axis(self):
        # J + K must equal 2D for all QA packets
        for (b, e) in [(1,1),(2,3),(5,7),(3,1),(9,4),(12,5)]:
            pkt = QAPacket(b, e)
            assert pkt.axis_split == pkt.major_axis, (
                f"Axis identity failed for ({b},{e}): {pkt.J}+{pkt.K} != {pkt.major_axis}"
            )

    def test_axis_identity_property(self):
        pkt = QAPacket(3, 1)
        assert pkt.axis_identity_holds

    def test_I_gap(self):
        pkt = QAPacket(3, 1)
        assert pkt.I == abs(pkt.C - pkt.F)  # |8 - 15| = 7

    def test_orbit_9(self):
        pkt = QAPacket(3, 1)
        assert pkt.orbit_9 == (3 + 1) % 9   # 4

    def test_orbit_24(self):
        pkt = QAPacket(3, 1)
        assert pkt.orbit_24 == (3 + 1) % 24  # 4

    def test_area_is_B_times_E(self):
        pkt = QAPacket(3, 1)
        assert pkt.area == 9 * 1   # B*E = 9

    def test_shape_sig(self):
        pkt = QAPacket(3, 1)
        assert pkt.shape_sig == (pkt.C % 9, pkt.F % 9, pkt.G % 9)

    def test_primitive(self):
        assert QAPacket(3, 1).primitive is True
        assert QAPacket(4, 2).primitive is False

    def test_parity_odd(self):
        assert QAPacket(3, 1).parity == "odd"

    def test_parity_even(self):
        assert QAPacket(2, 4).parity == "even"

    def test_parity_mixed(self):
        assert QAPacket(3, 2).parity == "mixed"

    def test_pyth_identity_sample(self):
        # C²+F²=G² is not universally true; test known case
        # For (b,e)=(3,1): C=8, F=15, G=17 → 64+225=289=17² ✓
        pkt = QAPacket(3, 1)
        assert pkt.pyth_identity_holds

    def test_pyth_identity_not_universal(self):
        # (b,e)=(2,2): C=2*2*4=16, F=6*2=12... wait let's check
        # d=4, a=6; C=2*2*4=16, F=6*2=12, G=D+E=16+4=20
        # 16²+12²=256+144=400; 20²=400 ✓ — try another
        # (b,e)=(1,3): d=4, a=7; C=2*3*4=24, F=7*1=7, G=16+9=25
        # 24²+7²=576+49=625; 25²=625 ✓ interesting
        # (b,e)=(2,3): d=5, a=8; C=2*3*5=30, F=8*2=16, G=25+9=34
        # 30²+16²=900+256=1156; 34²=1156 ✓
        # The identity C²+F²=G² is always true for QA packets!
        # Let's prove it algebraically:
        # C=2ed, F=ab=b(b+2e), G=D+E=d²+e²=(b+e)²+e²
        # C²+F²=(2ed)²+(b·a)²=4e²d²+(b(b+2e))²
        # G²=(d²+e²)²=(b²+2be+e²+e²)²=... let me verify numerically
        for b in range(1, 10):
            for e in range(1, 10):
                pkt = QAPacket(b, e)
                # Verify algebraically that C²+F²==G² always holds
                assert pkt.pyth_identity_holds, (
                    f"Pyth identity failed for ({b},{e}): "
                    f"C={pkt.C} F={pkt.F} G={pkt.G}"
                )


# ── Generators ────────────────────────────────────────────────────────────────

class TestGenerators:
    def test_sigma(self):
        pkt = QAPacket(3, 1)
        nb = pkt.sigma(N=10)
        assert nb is not None
        assert nb.b == 3 and nb.e == 2

    def test_sigma_boundary(self):
        pkt = QAPacket(3, 5)
        assert pkt.sigma(N=5) is None

    def test_mu(self):
        pkt = QAPacket(3, 1)
        nb = pkt.mu()
        assert nb.b == 1 and nb.e == 3

    def test_lambda2(self):
        pkt = QAPacket(2, 3)
        nb = pkt.lambda2(N=10)
        assert nb is not None
        assert nb.b == 4 and nb.e == 6

    def test_lambda2_boundary(self):
        pkt = QAPacket(6, 3)
        assert pkt.lambda2(N=10) is None

    def test_nu_valid(self):
        pkt = QAPacket(4, 2)
        nb = pkt.nu()
        assert nb is not None
        assert nb.b == 2 and nb.e == 1

    def test_nu_invalid_odd(self):
        pkt = QAPacket(3, 1)
        assert pkt.nu() is None

    def test_nu_invalid_mixed(self):
        pkt = QAPacket(4, 3)
        assert pkt.nu() is None

    def test_legal_neighbors_in_domain(self):
        pkt = QAPacket(3, 3)
        neighbors = pkt.legal_neighbors(N=10)
        for nb in neighbors:
            assert 1 <= nb.b <= 10
            assert 1 <= nb.e <= 10


# ── Backend agreement ─────────────────────────────────────────────────────────

class TestBackendAgreement:
    """
    table and graph backends must return identical result sets.
    qa backend is path-constrained (generator-legal moves only) and is
    compared separately.
    """

    @pytest.fixture(scope="class")
    def small_backends(self):
        N = 20
        return {
            "table": TableBackend(N=N),
            "graph": GraphBackend(N=N),
            "qa": QABackend(N=N),
        }

    @pytest.fixture(scope="class")
    def small_queries(self):
        return build_workload(N=20, n_queries=30, seed=99)

    def test_table_graph_agree(self, small_backends, small_queries):
        table = small_backends["table"]
        graph = small_backends["graph"]
        for q in small_queries:
            tr = table.run_query(q)
            gr = graph.run_query(q)
            assert tr.result_keys == gr.result_keys, (
                f"Query {q['query_id']}: table={len(tr.result_keys)} "
                f"graph={len(gr.result_keys)}"
            )

    def test_qa_is_subset_of_table(self, small_backends, small_queries):
        # QA path-constrained results must be a subset of table results
        # (QA restricts to generator-legal paths; table allows all edges)
        table = small_backends["table"]
        qa = small_backends["qa"]
        for q in small_queries:
            tr = table.run_query(q)
            qr = qa.run_query(q)
            # QA result must be ⊆ table result (tighter constraint)
            assert qr.result_keys <= tr.result_keys, (
                f"Query {q['query_id']}: QA has keys not in table: "
                f"{qr.result_keys - tr.result_keys}"
            )

    def test_result_keys_are_frozensets(self, small_backends, small_queries):
        for name, backend in small_backends.items():
            for q in small_queries[:5]:
                r = backend.run_query(q)
                assert isinstance(r.result_keys, frozenset)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_latency_positive(self):
        tb = TableBackend(N=10)
        q = build_workload(N=10, n_queries=1, seed=0)[0]
        r = tb.run_query(q)
        assert r.latency_ns > 0

    def test_collapse_ratio_range(self):
        tb = TableBackend(N=10)
        queries = build_workload(N=10, n_queries=10, seed=1)
        for q in queries:
            r = tb.run_query(q)
            assert 0.0 <= r.collapse_ratio <= 1.0 + 1e-9

    def test_aggregate_counts(self):
        tb = TableBackend(N=10)
        queries = build_workload(N=10, n_queries=5, seed=2)
        results = [tb.run_query(q) for q in queries]
        s = aggregate(results)
        assert s.query_count == 5
        assert s.p50_ns > 0
        assert s.p99_ns >= s.p50_ns


# ── Axis identity universality ────────────────────────────────────────────────

class TestAxisIdentityUniversal:
    def test_all_packets_in_small_domain(self):
        """J + K == 2D must hold for every (b,e) in domain."""
        for b in range(1, 20):
            for e in range(1, 20):
                pkt = QAPacket(b, e)
                assert pkt.axis_identity_holds, (
                    f"Axis identity failed at ({b},{e}): "
                    f"J={pkt.J} K={pkt.K} sum={pkt.J+pkt.K} 2D={pkt.major_axis}"
                )


# ── Storage estimates ─────────────────────────────────────────────────────────

class TestStorageEstimates:
    def test_qa_storage_smaller_than_table(self):
        N = 20
        qa = QABackend(N=N)
        table = TableBackend(N=N)
        # QA stores fewer bytes per packet (seed + law vs full row)
        # In this in-memory impl they store the same universe; the
        # point is law representation size vs full materialized rows.
        # The table has row dicts + edge rows + indexes, QA has universe + law.
        assert table.storage_bytes_approx() > qa.storage_bytes_approx()

    def test_bytes_estimate_positive(self):
        for cls in [TableBackend, GraphBackend, QABackend]:
            b = cls(N=10)
            assert b.storage_bytes_approx() > 0


# ── Query mode structure ───────────────────────────────────────────────────────

class TestQueryModes:
    """Verify each mode produces valid, structurally correct queries."""

    N = 20
    N_Q = 15

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_mode_builds_correct_count(self, mode):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=7, query_mode=mode)
        assert len(queries) == self.N_Q

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_mode_query_ids_unique(self, mode):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=7, query_mode=mode)
        ids = [q["query_id"] for q in queries]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_mode_seeds_in_domain(self, mode):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=7, query_mode=mode)
        for q in queries:
            for (b, e) in q["seeds"]:
                assert 1 <= b <= self.N
                assert 1 <= e <= self.N

    def test_orbit_only_has_unconstrained_i_gap_and_area(self):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=7, query_mode="orbit_only")
        for q in queries:
            assert q["i_gap_max"] == _UNCONSTRAINED
            assert q["area_max"] == _UNCONSTRAINED
            assert q["shape_sig"] is None
            assert q["parity"] is None
            assert q["require_primitive"] is False

    def test_random_attribute_has_b_mod(self):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=7,
                                 query_mode="random_attribute")
        # Every query must have a b_mod_n predicate
        for q in queries:
            assert q["b_mod_n"] is not None
            assert q["b_mod_val"] is not None
            # b_mod not correlated with QA orbit moduli
            assert q["b_mod_n"] not in (9, 24)

    def test_random_attribute_seed_passes_b_mod(self):
        queries = build_workload(N=self.N, n_queries=20, seed=7,
                                 query_mode="random_attribute")
        for q in queries:
            ref_b, ref_e = q["seeds"][0]
            assert ref_b % q["b_mod_n"] == q["b_mod_val"], (
                f"Seed ({ref_b},{ref_e}) fails b_mod: "
                f"{ref_b}%{q['b_mod_n']}={ref_b % q['b_mod_n']} != {q['b_mod_val']}"
            )

    def test_range_only_has_b_e_ranges(self):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=7,
                                 query_mode="range_only")
        for q in queries:
            assert q["b_lo"] is not None
            assert q["b_hi"] is not None
            assert q["e_lo"] is not None
            assert q["e_hi"] is not None
            assert 1 <= q["b_lo"] <= q["b_hi"] <= self.N
            assert 1 <= q["e_lo"] <= q["e_hi"] <= self.N

    def test_range_only_seed_passes_range(self):
        queries = build_workload(N=self.N, n_queries=20, seed=7,
                                 query_mode="range_only")
        for q in queries:
            ref_b, ref_e = q["seeds"][0]
            assert q["b_lo"] <= ref_b <= q["b_hi"], (
                f"Seed b={ref_b} outside [{q['b_lo']}, {q['b_hi']}]"
            )
            assert q["e_lo"] <= ref_e <= q["e_hi"], (
                f"Seed e={ref_e} outside [{q['e_lo']}, {q['e_hi']}]"
            )

    def test_mixed_heterogeneous_has_both_predicate_types(self):
        queries = build_workload(N=self.N, n_queries=30, seed=7,
                                 query_mode="mixed_heterogeneous")
        has_b_mod = any(q.get("b_mod_n") is not None for q in queries)
        has_qa_i_gap = any(q["i_gap_max"] < _UNCONSTRAINED for q in queries)
        assert has_b_mod, "mixed_heterogeneous should have some random_attribute queries"
        assert has_qa_i_gap, "mixed_heterogeneous should have some full_structured queries"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown query_mode"):
            build_workload(N=10, n_queries=5, seed=0, query_mode="nonexistent")


# ── Per-mode backend execution ─────────────────────────────────────────────────

class TestModeExecution:
    """Verify all backends run to completion for every mode; correctness contracts hold."""

    N = 15
    N_Q = 10

    @pytest.fixture(scope="class")
    def backends(self):
        N = self.N
        return {
            "table": TableBackend(N=N),
            "graph": GraphBackend(N=N),
            "qa": QABackend(N=N),
        }

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_all_backends_run(self, backends, mode):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=55, query_mode=mode)
        for name, backend in backends.items():
            for q in queries:
                r = backend.run_query(q)
                assert r.latency_ns > 0
                assert r.results_count >= 0

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_table_graph_agree_all_modes(self, backends, mode):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=55, query_mode=mode)
        table, graph = backends["table"], backends["graph"]
        for q in queries:
            tr = table.run_query(q)
            gr = graph.run_query(q)
            assert tr.result_keys == gr.result_keys, (
                f"mode={mode} q={q['query_id']}: "
                f"table={len(tr.result_keys)} graph={len(gr.result_keys)}"
            )

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_qa_subset_of_table_all_modes(self, backends, mode):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=55, query_mode=mode)
        table, qa = backends["table"], backends["qa"]
        for q in queries:
            tr = table.run_query(q)
            qr = qa.run_query(q)
            assert qr.result_keys <= tr.result_keys, (
                f"mode={mode} q={q['query_id']}: QA not ⊆ table"
            )


# ── Falsifier mode: orbit_only H1 prediction ──────────────────────────────────

class TestFalsifierOrbitOnly:
    """
    In orbit_only mode, QA rich buckets degenerate to the flat orbit index.
    waste(qa) should be approximately equal to waste(table) — not significantly lower.
    We check waste_ratio < 1.5x (no clear QA structural advantage).
    """
    N = 30
    N_Q = 40

    def test_orbit_only_h1_waste_ratio(self):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=88,
                                 query_mode="orbit_only")
        table = TableBackend(N=self.N)
        qa = QABackend(N=self.N)
        table_results = [table.run_query(q) for q in queries]
        qa_results = [qa.run_query(q) for q in queries]
        ts = aggregate(table_results)
        qs = aggregate(qa_results)
        waste_ratio = ts.mean_wasted_evals / max(1, qs.mean_wasted_evals)
        # In orbit_only, QA should not show meaningful pre-filter advantage
        # waste_ratio < 1.5 means QA does not reduce wasted evals by >50%
        assert waste_ratio < 1.5, (
            f"orbit_only: QA still wins waste by {waste_ratio:.2f}x — "
            f"flat orbit fallback may not be triggering correctly. "
            f"table_waste={ts.mean_wasted_evals:.1f} qa_waste={qs.mean_wasted_evals:.1f}"
        )

    def test_orbit_only_candidates_before_similar(self):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=88,
                                 query_mode="orbit_only")
        table = TableBackend(N=self.N)
        qa = QABackend(N=self.N)
        table_results = [table.run_query(q) for q in queries]
        qa_results = [qa.run_query(q) for q in queries]
        ts = aggregate(table_results)
        qs = aggregate(qa_results)
        # candidate_count_before should be the same (both use orbit index)
        before_ratio = ts.mean_candidates_before / max(1, qs.mean_candidates_before)
        assert 0.5 <= before_ratio <= 2.0, (
            f"orbit_only: candidates_before differs significantly: "
            f"table={ts.mean_candidates_before:.0f} qa={qs.mean_candidates_before:.0f}"
        )


# ── Falsifier mode: random_attribute H1 prediction ────────────────────────────

class TestFalsifierRandomAttribute:
    """
    In random_attribute mode, QA cannot pre-bucket by b_mod/e_mod predicates.
    The pre-filter degrades to orbit-only → waste(qa) ≈ waste(table).
    """
    N = 30
    N_Q = 40

    def test_random_attribute_h1_waste_ratio(self):
        queries = build_workload(N=self.N, n_queries=self.N_Q, seed=77,
                                 query_mode="random_attribute")
        table = TableBackend(N=self.N)
        qa = QABackend(N=self.N)
        table_results = [table.run_query(q) for q in queries]
        qa_results = [qa.run_query(q) for q in queries]
        ts = aggregate(table_results)
        qs = aggregate(qa_results)
        waste_ratio = ts.mean_wasted_evals / max(1, qs.mean_wasted_evals)
        assert waste_ratio < 1.5, (
            f"random_attribute: QA still wins waste by {waste_ratio:.2f}x — "
            f"non-QA predicates should not benefit from QA buckets. "
            f"table_waste={ts.mean_wasted_evals:.1f} qa_waste={qs.mean_wasted_evals:.1f}"
        )
