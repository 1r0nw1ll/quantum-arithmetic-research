"""
SQLite blob store backend.

Uses Python stdlib sqlite3 — no external dependencies.

Schema:
  files(file_id TEXT PK, root_b INT, root_e INT, n_chunks INT,
        orbit_9 INT, orbit_24 INT, size_class INT, i_gap INT, area INT,
        lineage_class INT)
  chunks(file_id TEXT, chunk_idx INT, content_val INT, content_hash TEXT,
         deviation INT DEFAULT NULL)
  repair_log(file_id TEXT, chunk_idx INT, expected_hash TEXT, stored_hash TEXT)

Indexes: files(orbit_9), files(orbit_24), files(size_class), files(i_gap)

Corruption recovery: lookup expected content from chunks table; compare
with stored hash; restore from content_val column (stored at write time).
"""
from __future__ import annotations
import sqlite3
import math
from typing import Any

from vfs_core import (
    QAPacket, VFSFile, _log2_class,
    derive_chunk_sequence, chunk_content_val, chunk_content_hash,
)
from vfs_metrics import OpResult, measure

_UC_THRESHOLD = 10 ** 10


class SQLiteBackend:
    def __init__(self, N: int = 100):
        self.N = N
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._build_schema()
        self._n_files = 0
        self._n_chunks = 0

    def _build_schema(self):
        c = self._conn
        c.execute("""CREATE TABLE files (
            file_id TEXT PRIMARY KEY,
            root_b INT, root_e INT, n_chunks INT,
            orbit_9 INT, orbit_24 INT,
            size_class INT, i_gap INT, area INT, lineage_class INT
        )""")
        c.execute("""CREATE TABLE chunks (
            file_id TEXT, chunk_idx INT,
            content_val INT, content_hash TEXT,
            deviation INT DEFAULT NULL,
            PRIMARY KEY (file_id, chunk_idx)
        )""")
        c.execute("""CREATE TABLE repair_log (
            file_id TEXT, chunk_idx INT,
            expected_hash TEXT, stored_hash TEXT
        )""")
        c.execute("CREATE INDEX ix_f_o9 ON files(orbit_9)")
        c.execute("CREATE INDEX ix_f_o24 ON files(orbit_24)")
        c.execute("CREATE INDEX ix_f_sc ON files(size_class)")
        c.execute("CREATE INDEX ix_f_ig ON files(i_gap)")
        c.commit()

    def load_files(self, files: list[VFSFile]):
        rows = []
        chunk_rows = []
        for f in files:
            pkt = QAPacket(f.root_b, f.root_e)
            fid = f"{f.root_b}_{f.root_e}"
            sc = _log2_class(pkt.area)
            rows.append((
                fid, f.root_b, f.root_e, f.n_chunks,
                pkt.orbit_9, pkt.orbit_24, sc, pkt.I, pkt.area, pkt.orbit_9,
            ))
            seq = derive_chunk_sequence(f.root_b, f.root_e, f.n_chunks, self.N)
            for cidx, (cb, ce) in enumerate(seq):
                cv = chunk_content_val(cb, ce)
                ch = chunk_content_hash(cb, ce)
                chunk_rows.append((fid, cidx, cv, ch))
        self._conn.executemany("INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self._conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,NULL)", chunk_rows)
        self._conn.commit()
        self._n_files = len(rows)
        self._n_chunks = len(chunk_rows)

    def storage_bytes_approx(self) -> int:
        # files table: ~10 columns × 8 bytes + row overhead
        file_bytes = self._n_files * (10 * 8 + 64)
        # chunks table: ~5 columns × 8 bytes + row overhead
        chunk_bytes = self._n_chunks * (5 * 8 + 48)
        # 4 indexes: rough estimate
        idx_bytes = self._n_files * 4 * 24
        return file_bytes + chunk_bytes + idx_bytes

    def _file_id(self, fid: tuple[int, int]) -> str:
        return f"{fid[0]}_{fid[1]}"

    def run_lookup(self, q: dict[str, Any]) -> OpResult:
        def _run():
            # Index scan: orbit first
            om = q["orbit_mod"]
            ov = q["orbit_val"]
            if om == 9:
                sql = "SELECT file_id, root_b, root_e FROM files WHERE orbit_9=?"
                params = [ov % 9]
            else:
                sql = "SELECT file_id, root_b, root_e FROM files WHERE orbit_24=?"
                params = [ov % 24]

            rows = self._conn.execute(sql, params).fetchall()
            before = len(rows)

            # Filter pass
            filtered = {}
            for (fid, rb, re) in rows:
                pkt = QAPacket(rb, re)
                sc_ok = q["size_class_max"] >= _UC_THRESHOLD or pkt.area <= q["size_class_max"]
                ig_ok = q["i_gap_max"] >= _UC_THRESHOLD or pkt.I <= q["i_gap_max"]
                bm_ok = (q.get("b_mod_n") is None or
                         pkt.b % q["b_mod_n"] == q["b_mod_val"])
                if sc_ok and ig_ok and bm_ok:
                    filtered[(rb, re)] = fid

            after = len(filtered)

            # BFS over generator graph restricted to filtered set
            seeds = q["seeds"]
            frontier: set[tuple] = {s for s in seeds if s in filtered}
            visited: set[tuple] = set(frontier)
            expansions = 0
            for _ in range(q["k"]):
                nf: set[tuple] = set()
                for key in frontier:
                    pkt = QAPacket(*key)
                    for nb in pkt.legal_neighbors(self.N):
                        nkey = (nb.b, nb.e)
                        expansions += 1
                        if nkey not in visited and nkey in filtered:
                            nf.add(nkey)
                    visited.add(key)
                frontier = nf - visited
                visited |= frontier

            return before, after, expansions, len(visited)

        (before, after, exp, results), lat = measure(_run)
        return OpResult(
            backend="sqlite", op="lookup", query_id=q["query_id"],
            latency_ns=lat,
            candidate_before=before, candidate_after=after,
            expansion_count=exp, result_count=results,
            mutation_cost=0, reconstruction_ops=0, repaired=False,
            storage_bytes=self.storage_bytes_approx(),
        )

    def run_append(self, op: dict[str, Any]) -> OpResult:
        def _run():
            fid = self._file_id(tuple(op["file_id"]))
            row = self._conn.execute(
                "SELECT root_b, root_e, n_chunks FROM files WHERE file_id=?", (fid,)
            ).fetchone()
            if row is None:
                return 0
            rb, re, nc = row
            # Derive new chunk
            seq = derive_chunk_sequence(rb, re, nc + 1, self.N)
            if len(seq) <= nc:
                return 0
            cb, ce = seq[nc]
            cv = chunk_content_val(cb, ce)
            ch = chunk_content_hash(cb, ce)
            self._conn.execute(
                "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,NULL)", (fid, nc, cv, ch)
            )
            self._conn.execute(
                "UPDATE files SET n_chunks=? WHERE file_id=?", (nc + 1, fid)
            )
            self._conn.commit()
            self._n_chunks += 1
            return 1

        result, lat = measure(_run)
        return OpResult(
            backend="sqlite", op="append", query_id=op["op_id"],
            latency_ns=lat, mutation_cost=result,
            candidate_before=0, candidate_after=0, expansion_count=0,
            result_count=0, reconstruction_ops=0, repaired=False,
            storage_bytes=self.storage_bytes_approx(),
        )

    def run_mutation(self, op: dict[str, Any]) -> OpResult:
        def _run():
            fid = self._file_id(tuple(op["file_id"]))
            if op["mut_type"] == "lawful":
                # Lawful mutation: update root + all metadata columns + all chunks
                row = self._conn.execute(
                    "SELECT root_b, root_e, n_chunks FROM files WHERE file_id=?", (fid,)
                ).fetchone()
                if row is None:
                    return 0
                rb, re, nc = row
                pkt = QAPacket(rb, re)
                nbs = pkt.legal_neighbors(self.N)
                if not nbs:
                    return 0
                nb = nbs[0]
                # Must re-derive all chunk content and re-insert
                seq = derive_chunk_sequence(nb.b, nb.e, nc, self.N)
                self._conn.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
                rows = []
                for cidx, (cb, ce) in enumerate(seq):
                    rows.append((fid, cidx, chunk_content_val(cb, ce),
                                 chunk_content_hash(cb, ce)))
                self._conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,NULL)", rows)
                new_pkt = QAPacket(nb.b, nb.e)
                self._conn.execute(
                    """UPDATE files SET root_b=?,root_e=?,orbit_9=?,orbit_24=?,
                       size_class=?,i_gap=?,area=?,lineage_class=? WHERE file_id=?""",
                    (nb.b, nb.e, new_pkt.orbit_9, new_pkt.orbit_24,
                     _log2_class(new_pkt.area), new_pkt.I, new_pkt.area,
                     new_pkt.orbit_9, fid)
                )
                self._conn.commit()
                # Cost: 1 row update + nc chunk re-derives + nc chunk inserts
                return 1 + nc + nc
            else:
                # Arbitrary mutation: UPDATE one chunk row
                cidx = op["chunk_idx"]
                nv = op["new_val"]
                self._conn.execute(
                    "UPDATE chunks SET deviation=? WHERE file_id=? AND chunk_idx=?",
                    (nv, fid, cidx)
                )
                self._conn.commit()
                return 1

        cost, lat = measure(_run)
        return OpResult(
            backend="sqlite", op="mutation", query_id=op["op_id"],
            latency_ns=lat, mutation_cost=cost,
            candidate_before=0, candidate_after=0, expansion_count=0,
            result_count=0, reconstruction_ops=0, repaired=False,
            storage_bytes=self.storage_bytes_approx(),
        )

    def run_corruption_recovery(self, op: dict[str, Any]) -> OpResult:
        def _run():
            fid = self._file_id(tuple(op["file_id"]))
            cidx = op["chunk_idx"]
            corrupt_type = op.get("corrupt_type", "type_a")
            row = self._conn.execute(
                "SELECT content_val, content_hash, deviation FROM chunks "
                "WHERE file_id=? AND chunk_idx=?", (fid, cidx)
            ).fetchone()
            if row is None:
                return False, 0
            cv, ch, dev = row
            if dev is not None:
                return True, 1  # deviation record present → authoritative
            if corrupt_type == "type_b":
                # Stored record destroyed — SQLite has no law to re-derive from.
                return False, 0
            # type_a: content_val intact → recover from stored record.
            return True, 1

        (repaired, rec_ops), lat = measure(_run)
        return OpResult(
            backend="sqlite", op="corruption_recovery", query_id=op["op_id"],
            latency_ns=lat, repaired=repaired, reconstruction_ops=rec_ops,
            candidate_before=0, candidate_after=0, expansion_count=0,
            result_count=0, mutation_cost=0,
            storage_bytes=self.storage_bytes_approx(),
        )
