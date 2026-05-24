"""
Relational table simulation backend.

Simulates:
- packet rows (one row per (b,e) pair with all derived columns)
- edge rows (one row per legal generator move)
- dictionary indexes on selected columns
- recursive reachability over edge table (Python BFS, no actual SQL)

All filters applied as sequential row scans or index lookups.
Storage estimate includes row dicts + index dicts.
"""
from __future__ import annotations
import math
import sys
from collections import deque
from typing import Any

from qa_backend import QAPacket, _passes_filter
from metrics import QueryResult, measure


def _row(pkt: QAPacket) -> dict:
    return {
        "b": pkt.b, "e": pkt.e,
        "d": pkt.d, "a": pkt.a,
        "B": pkt.B, "E": pkt.E, "D": pkt.D, "A": pkt.A,
        "C": pkt.C, "F": pkt.F, "G": pkt.G,
        "J": pkt.J, "X": pkt.X, "K": pkt.K,
        "major_axis": pkt.major_axis,
        "axis_split": pkt.axis_split,
        "axis_identity": pkt.axis_identity_holds,
        "pyth_identity": pkt.pyth_identity_holds,
        "I": pkt.I,
        "orbit_9": pkt.orbit_9,
        "orbit_24": pkt.orbit_24,
        "area": pkt.area,
        "shape_sig_c9": pkt.shape_sig[0],
        "shape_sig_f9": pkt.shape_sig[1],
        "shape_sig_g9": pkt.shape_sig[2],
        "parity": pkt.parity,
        "primitive": pkt.primitive,
    }


class TableBackend:
    def __init__(self, N: int = 250):
        self.N = N
        self._rows: dict[tuple[int, int], dict] = {}
        self._edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        # Indexes: orbit_9 → set of keys, orbit_24 → set of keys, parity → set
        self._idx_orbit9: dict[int, set[tuple[int, int]]] = {}
        self._idx_orbit24: dict[int, set[tuple[int, int]]] = {}
        self._idx_parity: dict[str, set[tuple[int, int]]] = {}
        self._build()

    def _build(self):
        N = self.N
        for b in range(1, N + 1):
            for e in range(1, N + 1):
                pkt = QAPacket(b, e)
                key = (b, e)
                self._rows[key] = _row(pkt)
                # Build edge table
                self._edges[key] = [
                    (nb.b, nb.e) for nb in pkt.legal_neighbors(N)
                ]
                # Build indexes
                self._idx_orbit9.setdefault(pkt.orbit_9, set()).add(key)
                self._idx_orbit24.setdefault(pkt.orbit_24, set()).add(key)
                self._idx_parity.setdefault(pkt.parity, set()).add(key)

    def storage_bytes_approx(self) -> int:
        # Row size: ~30 columns × 8 bytes + dict overhead (~240 bytes)
        row_bytes = len(self._rows) * (30 * 8 + 240)
        # Edge rows: each edge is a tuple ~56 bytes
        edge_bytes = sum(len(v) for v in self._edges.values()) * 56
        # Index overhead: 3 indexes × keys
        idx_bytes = len(self._rows) * 3 * 56
        return row_bytes + edge_bytes + idx_bytes

    def run_query(self, q: dict[str, Any]) -> QueryResult:
        def _run():
            # Step 1: use index for orbit + parity pre-filter
            orbit_val = q["orbit_val"]
            if q["orbit_mod"] == 9:
                candidate_keys = set(self._idx_orbit9.get(orbit_val % 9, set()))
            else:
                candidate_keys = set(self._idx_orbit24.get(orbit_val % 24, set()))

            if q.get("parity") is not None:
                parity_set = self._idx_parity.get(q["parity"], set())
                candidate_keys &= parity_set

            candidate_count_before = len(candidate_keys)

            # Step 2: full row scan for remaining predicates
            filtered: set[tuple[int, int]] = set()
            for key in candidate_keys:
                row = self._rows[key]
                pkt = QAPacket(row["b"], row["e"])
                if _passes_filter(pkt, q):
                    filtered.add(key)

            candidate_count_after = len(filtered)

            # Step 3: BFS over edge table from seeds, no path-legality constraint
            seeds = q["seeds"]
            frontier: set[tuple[int, int]] = set()
            for (b, e) in seeds:
                if (b, e) in filtered:
                    frontier.add((b, e))

            visited: set[tuple[int, int]] = set(frontier)
            expansion_count = 0

            for _step in range(q["k"]):
                next_frontier: set[tuple[int, int]] = set()
                for key in frontier:
                    for nkey in self._edges.get(key, []):
                        expansion_count += 1
                        if nkey not in visited and nkey in filtered:
                            next_frontier.add(nkey)
                    visited.add(key)
                frontier = next_frontier - visited
                visited |= frontier

            results = frozenset(visited)
            collapse_ratio = (candidate_count_after / candidate_count_before
                              if candidate_count_before > 0 else 0.0)
            return candidate_count_before, candidate_count_after, expansion_count, results, collapse_ratio

        (cb, ca, exp, results, cr), lat = measure(_run)
        return QueryResult(
            backend_name="table",
            query_id=q["query_id"],
            latency_ns=lat,
            candidate_count_before=cb,
            candidate_count_after=ca,
            expansion_count=exp,
            results_count=len(results),
            collapse_ratio=cr,
            bytes_estimate=self.storage_bytes_approx(),
            result_keys=results,
            path_constrained=False,
        )
