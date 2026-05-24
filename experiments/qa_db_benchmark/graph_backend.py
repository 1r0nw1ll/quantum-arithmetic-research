"""
Generic property graph simulation backend.

Simulates:
- adjacency list (node → neighbor list)
- node property dictionary (all 14+ QA fields stored as generic properties)
- property indexes (orbit_9, orbit_24, parity)
- traversal: expand neighbors, then filter by property predicates

No arithmetic structure is encoded — the graph engine treats every field
as an opaque property value.  Retrieval cannot exploit generator algebra
or identity relationships; it must traverse and test.

Storage estimate: adjacency list + per-node property dicts + indexes.
"""
from __future__ import annotations
import math
from typing import Any

from qa_backend import QAPacket, _passes_filter
from metrics import QueryResult, measure


class GraphBackend:
    def __init__(self, N: int = 250):
        self.N = N
        self._adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._props: dict[tuple[int, int], dict] = {}
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
                # All fields stored as opaque properties — no law encoding
                self._props[key] = {
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
                    "shape_sig": pkt.shape_sig,
                    "parity": pkt.parity,
                    "primitive": pkt.primitive,
                }
                self._adj[key] = [(nb.b, nb.e) for nb in pkt.legal_neighbors(N)]
                self._idx_orbit9.setdefault(pkt.orbit_9, set()).add(key)
                self._idx_orbit24.setdefault(pkt.orbit_24, set()).add(key)
                self._idx_parity.setdefault(pkt.parity, set()).add(key)

    def storage_bytes_approx(self) -> int:
        # Node property dicts: ~30 entries × (8+8 key+val) + 240 dict overhead
        prop_bytes = len(self._props) * (30 * 16 + 240)
        # Adjacency: each neighbor ref is a tuple ~56 bytes
        adj_bytes = sum(len(v) for v in self._adj.values()) * 56
        # Indexes: same as table
        idx_bytes = len(self._props) * 3 * 56
        return prop_bytes + adj_bytes + idx_bytes

    def run_query(self, q: dict[str, Any]) -> QueryResult:
        def _run():
            orbit_val = q["orbit_val"]
            if q["orbit_mod"] == 9:
                candidate_keys = set(self._idx_orbit9.get(orbit_val % 9, set()))
            else:
                candidate_keys = set(self._idx_orbit24.get(orbit_val % 24, set()))

            if q.get("parity") is not None:
                candidate_keys &= self._idx_parity.get(q["parity"], set())

            candidate_count_before = len(candidate_keys)

            # Filter by property predicates (opaque — no algebraic shortcut)
            filtered: set[tuple[int, int]] = set()
            for key in candidate_keys:
                p = self._props[key]
                pkt = QAPacket(p["b"], p["e"])
                if _passes_filter(pkt, q):
                    filtered.add(key)

            candidate_count_after = len(filtered)

            # BFS: expand adjacency, test membership in filtered set
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
                    for nkey in self._adj.get(key, []):
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
            backend_name="graph",
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
