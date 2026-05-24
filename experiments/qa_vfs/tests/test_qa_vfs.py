"""
Tests for the QA-VFS benchmark.
Run: pytest experiments/qa_vfs/tests/test_qa_vfs.py -q
"""
import sys
import os
import pytest

_VFS = os.path.join(os.path.dirname(__file__), "..")
if _VFS not in sys.path:
    sys.path.insert(0, _VFS)

from vfs_core import (
    QAPacket, VFSFile, derive_chunk_sequence,
    chunk_content_val, chunk_content_hash,
    size_class, lineage_class, _log2_class,
)
from fixture_builder import (
    build_file_universe, build_lookup_workload,
    build_append_ops, build_mutation_ops, build_corruption_ops,
)
from qa_vfs_backend import QAVFSBackend
from sqlite_backend import SQLiteBackend
from graph_vfs_backend import GraphVFSBackend
from vfs_metrics import aggregate


# ── Packet arithmetic ─────────────────────────────────────────────────────────

class TestPacketArithmetic:
    def test_derived_coords(self):
        p = QAPacket(3, 1)
        assert p.d == 4 and p.a == 5

    def test_squared_elements(self):
        p = QAPacket(3, 1)
        assert p.B == 9 and p.E == 1 and p.D == 16 and p.A == 25

    def test_C_F_G(self):
        p = QAPacket(3, 1)
        assert p.C == 8   # 2*1*4
        assert p.F == 15  # 5*3
        assert p.G == 17  # 16+1

    def test_J_X_K(self):
        p = QAPacket(3, 1)
        assert p.J == 12  # 3*4
        assert p.X == 4   # 1*4
        assert p.K == 20  # 4*5

    def test_axis_identity_universal(self):
        for b in range(1, 15):
            for e in range(1, 15):
                p = QAPacket(b, e)
                assert p.J + p.K == 2 * p.D, f"axis identity failed at ({b},{e})"

    def test_pyth_identity_universal(self):
        for b in range(1, 10):
            for e in range(1, 10):
                p = QAPacket(b, e)
                assert p.C * p.C + p.F * p.F == p.G * p.G

    def test_I_gap(self):
        p = QAPacket(3, 1)
        assert p.I == abs(p.C - p.F) == 7

    def test_orbit_9(self):
        assert QAPacket(3, 1).orbit_9 == 4

    def test_orbit_24(self):
        assert QAPacket(3, 1).orbit_24 == 4


# ── Generators ────────────────────────────────────────────────────────────────

class TestGenerators:
    def test_sigma_in_domain(self):
        nbs = QAPacket(3, 5).legal_neighbors(10)
        assert QAPacket(3, 6) in nbs

    def test_sigma_at_boundary(self):
        nbs = QAPacket(3, 10).legal_neighbors(10)
        assert not any(nb.e > 10 for nb in nbs)

    def test_mu(self):
        nbs = QAPacket(3, 1).legal_neighbors(10)
        assert QAPacket(1, 3) in nbs

    def test_lambda2(self):
        nbs = QAPacket(2, 3).legal_neighbors(10)
        assert QAPacket(4, 6) in nbs

    def test_lambda2_boundary(self):
        nbs = QAPacket(6, 3).legal_neighbors(10)
        assert not any(nb.b > 10 or nb.e > 10 for nb in nbs)

    def test_nu_valid(self):
        nbs = QAPacket(4, 2).legal_neighbors(10)
        assert QAPacket(2, 1) in nbs

    def test_nu_invalid(self):
        nbs = QAPacket(3, 1).legal_neighbors(10)
        assert QAPacket(1, 0) not in nbs and not any(nb.b < 1 or nb.e < 1 for nb in nbs)


# ── Chunk derivation ──────────────────────────────────────────────────────────

class TestChunkDerivation:
    def test_first_chunk_is_root(self):
        seq = derive_chunk_sequence(3, 1, 1, 10)
        assert seq == [(3, 1)]

    def test_chunk_count(self):
        seq = derive_chunk_sequence(3, 1, 5, 20)
        assert len(seq) <= 5

    def test_chunks_in_domain(self):
        seq = derive_chunk_sequence(3, 2, 8, 10)
        for (b, e) in seq:
            assert 1 <= b <= 10 and 1 <= e <= 10

    def test_chunk_uniqueness(self):
        seq = derive_chunk_sequence(5, 3, 10, 20)
        assert len(seq) == len(set(seq))

    def test_content_deterministic(self):
        v1 = chunk_content_val(3, 1)
        v2 = chunk_content_val(3, 1)
        assert v1 == v2

    def test_content_range(self):
        for b in range(1, 10):
            for e in range(1, 10):
                v = chunk_content_val(b, e)
                assert 0 <= v < 2 ** 31

    def test_hash_hex(self):
        h = chunk_content_hash(3, 1)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# ── Fixture builder ───────────────────────────────────────────────────────────

class TestFixtureBuilder:
    def test_file_universe_count(self):
        files = build_file_universe(N=20, n_files=30, seed=7)
        assert len(files) <= 30
        assert len(files) > 0

    def test_file_roots_in_domain(self):
        files = build_file_universe(N=20, n_files=30, seed=7)
        for f in files:
            assert 1 <= f.root_b <= 20
            assert 1 <= f.root_e <= 20

    def test_file_root_uniqueness(self):
        files = build_file_universe(N=20, n_files=30, seed=7)
        ids = [f.file_id for f in files]
        assert len(ids) == len(set(ids))

    def test_lookup_workload_count(self):
        files = build_file_universe(N=20, n_files=30, seed=7)
        q = build_lookup_workload(files, N=20, n_queries=10, seed=7)
        assert len(q) == 10

    def test_structured_queries_have_tight_bounds(self):
        files = build_file_universe(N=20, n_files=30, seed=7)
        q = build_lookup_workload(files, N=20, n_queries=20, seed=7, query_mode="structured")
        for qi in q:
            assert qi["i_gap_max"] >= 0
            assert qi["size_class_max"] >= 0

    def test_unstructured_queries_have_b_mod(self):
        files = build_file_universe(N=20, n_files=30, seed=7)
        q = build_lookup_workload(files, N=20, n_queries=20, seed=7, query_mode="unstructured")
        for qi in q:
            assert qi.get("b_mod_n") is not None

    def test_seed_passes_structured_query(self):
        files = build_file_universe(N=30, n_files=50, seed=9)
        q = build_lookup_workload(files, N=30, n_queries=30, seed=9, query_mode="structured")
        for qi in q:
            seed_b, seed_e = qi["seeds"][0]
            p = QAPacket(seed_b, seed_e)
            orbit_ok = (
                (qi["orbit_mod"] == 9 and p.orbit_9 == qi["orbit_val"] % 9) or
                (qi["orbit_mod"] == 24 and p.orbit_24 == qi["orbit_val"] % 24)
            )
            assert orbit_ok, f"Seed fails orbit filter: {qi['query_id']}"
            assert p.I <= qi["i_gap_max"]
            assert p.area <= qi["size_class_max"]

    def test_seed_passes_unstructured_query(self):
        files = build_file_universe(N=30, n_files=50, seed=9)
        q = build_lookup_workload(files, N=30, n_queries=30, seed=9, query_mode="unstructured")
        for qi in q:
            seed_b, _ = qi["seeds"][0]
            assert seed_b % qi["b_mod_n"] == qi["b_mod_val"]


# ── Backend operations ────────────────────────────────────────────────────────

class TestBackendOperations:
    N = 20

    @pytest.fixture(scope="class")
    def files(self):
        return build_file_universe(N=self.N, n_files=30, seed=7)

    @pytest.fixture(scope="class")
    def backends(self, files):
        bs = {
            "qa_vfs": QAVFSBackend(N=self.N),
            "sqlite": SQLiteBackend(N=self.N),
            "graph_vfs": GraphVFSBackend(N=self.N),
        }
        for b in bs.values():
            b.load_files(files)
        return bs

    def test_lookup_runs_all_backends(self, backends, files):
        q = build_lookup_workload(files, N=self.N, n_queries=5, seed=11)
        for name, backend in backends.items():
            for qi in q:
                r = backend.run_lookup(qi)
                assert r.latency_ns > 0, f"{name} lookup latency <= 0"

    def test_append_runs_all_backends(self, backends, files):
        ops = build_append_ops(files, n_ops=5, seed=11)
        for name, backend in backends.items():
            for op in ops:
                r = backend.run_append(op)
                assert r.latency_ns > 0

    def test_mutation_runs_all_backends(self, backends, files):
        ops = build_mutation_ops(files, n_ops=6, seed=11)
        for name, backend in backends.items():
            for op in ops:
                r = backend.run_mutation(op)
                assert r.latency_ns > 0

    def test_corruption_recovery_runs_all_backends(self, backends, files):
        ops = build_corruption_ops(files, n_ops=5, seed=11)
        for name, backend in backends.items():
            for op in ops:
                r = backend.run_corruption_recovery(op)
                assert r.latency_ns > 0

    def test_qa_repairs_type_b_sqlite_does_not(self, backends, files):
        # Force all ops to type_b (stored record destroyed) — QA re-derives by law,
        # SQLite/graph have no law and cannot repair.
        ops = build_corruption_ops(files, n_ops=20, seed=11)
        type_b_ops = [{**op, "corrupt_type": "type_b"} for op in ops]
        qa_rate = sum(
            1 for op in type_b_ops if backends["qa_vfs"].run_corruption_recovery(op).repaired
        ) / len(type_b_ops)
        sql_rate = sum(
            1 for op in type_b_ops if backends["sqlite"].run_corruption_recovery(op).repaired
        ) / len(type_b_ops)
        assert qa_rate >= 0.85, f"QA type_b repair_rate={qa_rate:.2f}, expected >=0.85"
        assert sql_rate < 0.15, f"SQLite type_b repair_rate={sql_rate:.2f}, expected ~0"
        assert qa_rate > sql_rate + 0.7, f"QA advantage gap too small: qa={qa_rate:.2f} sql={sql_rate:.2f}"

    def test_storage_bytes_positive(self, backends):
        for name, backend in backends.items():
            assert backend.storage_bytes_approx() > 0, f"{name}: zero storage"

    def test_qa_storage_less_than_sqlite(self, backends):
        qa = backends["qa_vfs"].storage_bytes_approx()
        sql = backends["sqlite"].storage_bytes_approx()
        assert qa < sql, f"QA storage {qa} >= SQLite {sql}"


# ── Lookup result consistency ─────────────────────────────────────────────────

class TestLookupConsistency:
    """
    SQLite and graph_vfs must return equivalent lookup results
    (same predicate semantics, no path constraint).
    QA-VFS is path-constrained (generator-legal moves only).
    """
    N = 20

    @pytest.fixture(scope="class")
    def setup(self):
        files = build_file_universe(N=self.N, n_files=40, seed=13)
        backends = {
            "qa_vfs": QAVFSBackend(N=self.N),
            "sqlite": SQLiteBackend(N=self.N),
            "graph_vfs": GraphVFSBackend(N=self.N),
        }
        for b in backends.values():
            b.load_files(files)
        queries = build_lookup_workload(files, N=self.N, n_queries=15, seed=13)
        return backends, queries

    def test_sqlite_graph_agree_on_structured(self, setup):
        backends, queries = setup
        for q in queries:
            sr = backends["sqlite"].run_lookup(q)
            gr = backends["graph_vfs"].run_lookup(q)
            assert sr.result_count == gr.result_count, (
                f"q={q['query_id']}: sqlite={sr.result_count} graph={gr.result_count}"
            )
            assert sr.candidate_before == gr.candidate_before, (
                f"q={q['query_id']}: before differs"
            )


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_aggregate_lookup(self):
        files = build_file_universe(N=15, n_files=20, seed=3)
        b = QAVFSBackend(N=15)
        b.load_files(files)
        q = build_lookup_workload(files, N=15, n_queries=5, seed=3)
        results = [b.run_lookup(qi) for qi in q]
        s = aggregate(results)
        assert s.query_count == 5
        assert s.mean_ns > 0
        assert s.p50_ns > 0

    def test_aggregate_corruption(self):
        files = build_file_universe(N=15, n_files=20, seed=3)
        b = SQLiteBackend(N=15)
        b.load_files(files)
        ops = build_corruption_ops(files, n_ops=5, seed=3)
        results = [b.run_corruption_recovery(op) for op in ops]
        s = aggregate(results)
        assert 0.0 <= s.repair_rate <= 1.0
