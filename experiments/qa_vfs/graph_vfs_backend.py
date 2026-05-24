"""
Filesystem metadata graph backend.

Nodes: file nodes (one per VFSFile root packet)
Node properties: all metadata fields stored as opaque dict values
Edges: generator-legal moves between file roots
Indexes: orbit_9, orbit_24 property indexes
Chunks: stored as node sub-properties

Retrieval: property-filter on orbit index → BFS → predicate check.
Corruption recovery: explicit repair record per chunk (not law-derived).
Mutation: update node properties + re-derive chunk sub-properties if needed.
"""
from __future__ import annotations
from typing import Any

from vfs_core import (
    QAPacket, VFSFile, _log2_class,
    derive_chunk_sequence, chunk_content_val, chunk_content_hash,
)
from vfs_metrics import OpResult, measure

_UC_THRESHOLD = 10 ** 10


class GraphVFSBackend:
    def __init__(self, N: int = 100):
        self.N = N
        # Node: file_id → property dict (all metadata + chunk list)
        self._nodes: dict[tuple, dict] = {}
        # Edges: file_id → set of neighbor file_ids (generator-legal)
        self._adj: dict[tuple, set[tuple]] = {}
        # Indexes
        self._idx_orbit9: dict[int, set[tuple]] = {}
        self._idx_orbit24: dict[int, set[tuple]] = {}
        # Repair records: (file_id, chunk_idx) → repair_val
        self._repair_records: dict[tuple, int] = {}

    def load_files(self, files: list[VFSFile]):
        # Build nodes
        for f in files:
            fid = f.file_id
            pkt = QAPacket(f.root_b, f.root_e)
            seq = derive_chunk_sequence(f.root_b, f.root_e, f.n_chunks, self.N)
            chunks = {
                cidx: {
                    "content_val": chunk_content_val(cb, ce),
                    "content_hash": chunk_content_hash(cb, ce),
                    "deviation": None,
                }
                for cidx, (cb, ce) in enumerate(seq)
            }
            self._nodes[fid] = {
                "root_b": f.root_b, "root_e": f.root_e,
                "n_chunks": f.n_chunks,
                "orbit_9": pkt.orbit_9, "orbit_24": pkt.orbit_24,
                "size_class": _log2_class(pkt.area),
                "i_gap": pkt.I, "area": pkt.area,
                "lineage_class": pkt.orbit_9,
                "chunks": chunks,
            }
            self._idx_orbit9.setdefault(pkt.orbit_9, set()).add(fid)
            self._idx_orbit24.setdefault(pkt.orbit_24, set()).add(fid)

        # Build edges
        for fid in self._nodes:
            b, e = fid
            pkt = QAPacket(b, e)
            neighbors = set()
            for nb in pkt.legal_neighbors(self.N):
                nkey = (nb.b, nb.e)
                if nkey in self._nodes:
                    neighbors.add(nkey)
            self._adj[fid] = neighbors

    def storage_bytes_approx(self) -> int:
        # Node property dicts: ~10 properties × 16 bytes + chunk dicts
        node_bytes = 0
        for fid, props in self._nodes.items():
            node_bytes += 10 * 16 + 64  # base properties
            node_bytes += len(props.get("chunks", {})) * (3 * 16 + 48)  # chunk props
        # Adjacency lists
        adj_bytes = sum(len(v) * 48 for v in self._adj.values())
        # Indexes
        idx_bytes = len(self._nodes) * 2 * 48
        return node_bytes + adj_bytes + idx_bytes

    def run_lookup(self, q: dict[str, Any]) -> OpResult:
        def _run():
            # Index pre-filter: orbit
            o9 = q["orbit_val"] % 9
            o24 = q["orbit_val"] % 24
            if q["orbit_mod"] == 9:
                candidates: set[tuple] = set(self._idx_orbit9.get(o9, set()))
            else:
                candidates: set[tuple] = set(self._idx_orbit24.get(o24, set()))

            before = len(candidates)

            # Property filter (opaque — no algebraic shortcut)
            filtered: set[tuple] = set()
            for fid in candidates:
                p = self._nodes[fid]
                pkt = QAPacket(p["root_b"], p["root_e"])
                sc_ok = q["size_class_max"] >= _UC_THRESHOLD or pkt.area <= q["size_class_max"]
                ig_ok = q["i_gap_max"] >= _UC_THRESHOLD or pkt.I <= q["i_gap_max"]
                bm_ok = (q.get("b_mod_n") is None or
                         pkt.b % q["b_mod_n"] == q["b_mod_val"])
                if sc_ok and ig_ok and bm_ok:
                    filtered.add(fid)

            after = len(filtered)

            # BFS over graph edges
            seeds = q["seeds"]
            frontier: set[tuple] = {s for s in seeds if s in filtered}
            visited: set[tuple] = set(frontier)
            expansions = 0
            for _ in range(q["k"]):
                nf: set[tuple] = set()
                for fid in frontier:
                    for nkey in self._adj.get(fid, set()):
                        expansions += 1
                        if nkey not in visited and nkey in filtered:
                            nf.add(nkey)
                    visited.add(fid)
                frontier = nf - visited
                visited |= frontier

            return before, after, expansions, len(visited)

        (before, after, exp, results), lat = measure(_run)
        return OpResult(
            backend="graph_vfs", op="lookup", query_id=q["query_id"],
            latency_ns=lat,
            candidate_before=before, candidate_after=after,
            expansion_count=exp, result_count=results,
            mutation_cost=0, reconstruction_ops=0, repaired=False,
            storage_bytes=self.storage_bytes_approx(),
        )

    def run_append(self, op: dict[str, Any]) -> OpResult:
        def _run():
            fid = tuple(op["file_id"])
            node = self._nodes.get(fid)
            if node is None:
                return 0
            nc = node["n_chunks"]
            seq = derive_chunk_sequence(node["root_b"], node["root_e"], nc + 1, self.N)
            if len(seq) > nc:
                cb, ce = seq[nc]
                node["chunks"][nc] = {
                    "content_val": chunk_content_val(cb, ce),
                    "content_hash": chunk_content_hash(cb, ce),
                    "deviation": None,
                }
                node["n_chunks"] += 1
            return 1

        result, lat = measure(_run)
        return OpResult(
            backend="graph_vfs", op="append", query_id=op["op_id"],
            latency_ns=lat, mutation_cost=result,
            candidate_before=0, candidate_after=0, expansion_count=0,
            result_count=0, reconstruction_ops=0, repaired=False,
            storage_bytes=self.storage_bytes_approx(),
        )

    def run_mutation(self, op: dict[str, Any]) -> OpResult:
        def _run():
            fid = tuple(op["file_id"])
            node = self._nodes.get(fid)
            if node is None:
                return 0
            if op["mut_type"] == "lawful":
                pkt = QAPacket(node["root_b"], node["root_e"])
                nbs = pkt.legal_neighbors(self.N)
                if not nbs:
                    return 0
                nb = nbs[0]
                nc = node["n_chunks"]
                # Must re-derive all chunk content (graph stores explicitly)
                seq = derive_chunk_sequence(nb.b, nb.e, nc, self.N)
                node["chunks"] = {
                    cidx: {
                        "content_val": chunk_content_val(cb, ce),
                        "content_hash": chunk_content_hash(cb, ce),
                        "deviation": None,
                    }
                    for cidx, (cb, ce) in enumerate(seq)
                }
                # Update indexes
                old_o9 = node["orbit_9"]
                old_o24 = node["orbit_24"]
                self._idx_orbit9.get(old_o9, set()).discard(fid)
                self._idx_orbit24.get(old_o24, set()).discard(fid)
                new_pkt = QAPacket(nb.b, nb.e)
                node.update({
                    "root_b": nb.b, "root_e": nb.e,
                    "orbit_9": new_pkt.orbit_9, "orbit_24": new_pkt.orbit_24,
                    "size_class": _log2_class(new_pkt.area),
                    "i_gap": new_pkt.I, "area": new_pkt.area,
                    "lineage_class": new_pkt.orbit_9,
                })
                self._idx_orbit9.setdefault(new_pkt.orbit_9, set()).add(fid)
                self._idx_orbit24.setdefault(new_pkt.orbit_24, set()).add(fid)
                return 1 + nc  # property update + nc chunk re-derives
            else:
                # Arbitrary: update one chunk property
                cidx = op["chunk_idx"]
                nv = op["new_val"]
                if cidx in node["chunks"]:
                    node["chunks"][cidx]["deviation"] = nv
                return 1

        cost, lat = measure(_run)
        return OpResult(
            backend="graph_vfs", op="mutation", query_id=op["op_id"],
            latency_ns=lat, mutation_cost=cost,
            candidate_before=0, candidate_after=0, expansion_count=0,
            result_count=0, reconstruction_ops=0, repaired=False,
            storage_bytes=self.storage_bytes_approx(),
        )

    def run_corruption_recovery(self, op: dict[str, Any]) -> OpResult:
        def _run():
            fid = tuple(op["file_id"])
            node = self._nodes.get(fid)
            if node is None:
                return False, 0
            cidx = op["chunk_idx"]
            chunk = node["chunks"].get(cidx)
            if chunk is None:
                return False, 0
            corrupt_type = op.get("corrupt_type", "type_a")
            if chunk["deviation"] is not None:
                return True, 1  # deviation record present → authoritative
            if corrupt_type == "type_b":
                # Stored node property destroyed — graph has no law to re-derive.
                return False, 0
            # type_a: content_val intact in node property.
            return True, 1

        (repaired, rec_ops), lat = measure(_run)
        return OpResult(
            backend="graph_vfs", op="corruption_recovery", query_id=op["op_id"],
            latency_ns=lat, repaired=repaired, reconstruction_ops=rec_ops,
            candidate_before=0, candidate_after=0, expansion_count=0,
            result_count=0, mutation_cost=0,
            storage_bytes=self.storage_bytes_approx(),
        )
