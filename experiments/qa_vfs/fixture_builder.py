"""
VFS workload fixture builder.

Produces:
  - A file universe: list[VFSFile], deterministic for given (N, n_files, seed)
  - Lookup queries: structured vs unstructured predicates
  - Append/mutation/recovery/retrieval operations

All seeds are calibrated so the root packet actually satisfies the query predicates.
"""
from __future__ import annotations
import random
from typing import Any

from vfs_core import (
    QAPacket, VFSFile, make_lookup_query,
    chunk_content_val, _log2_class,
)

_UNCONSTRAINED = 10 ** 15


def build_file_universe(N: int = 100, n_files: int = 200, seed: int = 42) -> list[VFSFile]:
    rng = random.Random(seed)
    files = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(files) < n_files and attempts < n_files * 10:
        attempts += 1
        b = rng.randint(1, N)
        e = rng.randint(1, N)
        if (b, e) in seen:
            continue
        seen.add((b, e))
        n_chunks = rng.randint(4, 16)
        files.append(VFSFile(root_b=b, root_e=e, n_chunks=n_chunks))
    return files


def build_lookup_workload(
    files: list[VFSFile],
    N: int = 100,
    n_queries: int = 100,
    seed: int = 42,
    query_mode: str = "structured",
) -> list[dict[str, Any]]:
    """
    query_mode:
      structured   — orbit + size_class + i_gap (QA-law predicates)
      unstructured — b_mod predicates (non-QA-law)
    """
    rng = random.Random(seed)
    queries = []
    tightness = [1.0, 2.0, 4.0, 16.0, 64.0]

    for qi in range(n_queries):
        ref_file = rng.choice(files)
        ref = QAPacket(ref_file.root_b, ref_file.root_e)
        k = rng.choice([2, 3, 4])
        orbit_mod = rng.choice([9, 24])
        orbit_val = ref.orbit_9 if orbit_mod == 9 else ref.orbit_24

        if query_mode == "structured":
            tight = rng.choice(tightness)
            sc_max = max(ref.area, int(ref.area * tight) + 1)
            i_gap_max = max(ref.I, int(ref.I * tight) + 1)
            q = make_lookup_query(
                qi, [(ref.b, ref.e)], k, orbit_mod, orbit_val,
                size_class_max=sc_max, i_gap_max=i_gap_max,
                query_mode="structured",
            )
        else:  # unstructured
            non_qa_mod = rng.choice([5, 7, 11, 13, 17, 19])
            b_mod_val = ref.b % non_qa_mod
            q = make_lookup_query(
                qi, [(ref.b, ref.e)], k, orbit_mod, orbit_val,
                size_class_max=_UNCONSTRAINED, i_gap_max=_UNCONSTRAINED,
                query_mode="unstructured",
                b_mod_n=non_qa_mod, b_mod_val=b_mod_val,
            )
        queries.append(q)
    return queries


def build_append_ops(
    files: list[VFSFile], n_ops: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """Append a new chunk (generator move) to a file."""
    rng = random.Random(seed)
    ops = []
    for i in range(n_ops):
        f = rng.choice(files)
        ops.append({
            "op_id": f"app{i:04d}",
            "file_id": f.file_id,
            "op": "append",
        })
    return ops


def build_mutation_ops(
    files: list[VFSFile], n_ops: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Two mutation types:
      lawful   — apply generator to root; all chunks shift
      arbitrary — change one chunk to a random value (requires deviation record)
    """
    rng = random.Random(seed)
    ops = []
    for i in range(n_ops):
        f = rng.choice(files)
        mut_type = rng.choice(["lawful", "arbitrary"])
        chunk_idx = rng.randint(0, max(0, f.n_chunks - 1)) if mut_type == "arbitrary" else None
        new_val = rng.randint(0, 2 ** 31 - 1) if mut_type == "arbitrary" else None
        ops.append({
            "op_id": f"mut{i:04d}",
            "file_id": f.file_id,
            "op": "mutation",
            "mut_type": mut_type,
            "chunk_idx": chunk_idx,
            "new_val": new_val,
        })
    return ops


def build_corruption_ops(
    files: list[VFSFile], n_ops: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """Mark a chunk corrupted; then attempt recovery."""
    rng = random.Random(seed)
    ops = []
    for i in range(n_ops):
        f = rng.choice(files)
        chunk_idx = rng.randint(0, max(0, f.n_chunks - 1))
        ops.append({
            "op_id": f"cor{i:04d}",
            "file_id": f.file_id,
            "op": "corruption_recovery",
            "chunk_idx": chunk_idx,
        })
    return ops
