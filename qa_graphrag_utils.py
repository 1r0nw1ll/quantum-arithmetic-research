"""
qa_graphrag_utils.py

Shared utilities for QA-GraphRAG prototype.

Provides:
- E8 root generation (lightweight, no side effects)
- QA tuple embedding into R^8
- E8 alignment and Harmonic Index computations
- Deterministic hash-based QA encoding
- Tuple similarity utility for retrieval

This is imported by:
- qa_entity_encoder.py
- qa_knowledge_graph.py
- qa_graph_query.py

Python 3.10+
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

MODULUS_DEFAULT = 24

# Cache for E8 roots
_E8_ROOTS_CACHE: np.ndarray | None = None


# --- E8 Root System ---
def generate_e8_root_system() -> np.ndarray:
    """Generate the 240 E8 root vectors in R^8.

    Two types (standard construction):
    - 112 roots: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    - 128 roots: vectors (±1/2, …, ±1/2) with even number of minus signs
    """
    roots = []
    # Type 1
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(8, dtype=float)
                    v[i] = s1
                    v[j] = s2
                    roots.append(v)
    # Type 2
    for bits in range(1 << 8):
        # even number of minus signs => parity of set bits is even
        if bin(bits).count("1") % 2 == 0:
            v = np.array([
                -0.5 if (bits >> k) & 1 else 0.5 for k in range(8)
            ], dtype=float)
            roots.append(v)
    roots_arr = np.unique(np.array(roots), axis=0)
    # Expect 240 unique roots
    return roots_arr


def get_e8_roots_cached() -> np.ndarray:
    """Return cached E8 roots array to avoid regeneration overhead."""
    global _E8_ROOTS_CACHE
    if _E8_ROOTS_CACHE is None:
        _E8_ROOTS_CACHE = generate_e8_root_system()
    return _E8_ROOTS_CACHE


# --- QA Embedding + Metrics ---
IDEAL_E8_ROOT = np.array([1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def embed_qa_tuple_to_r8(qa: Sequence[int]) -> np.ndarray:
    """Embed (b,e,d,a) into R^8.

    Current scheme mirrors qa_core.QASystem: place (b,e,d,a) in first 4 coords,
    zeros elsewhere. This keeps alignment comparable to prior experiments.
    """
    b, e, d, a = qa
    v = np.zeros(8, dtype=float)
    v[:4] = [float(b), float(e), float(d), float(a)]
    return v


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def compute_e8_alignment(qa: Sequence[int], use_full_roots: bool = False) -> float:
    """Compute E8 alignment for a QA tuple.

    - If use_full_roots is False (default), align to the ideal root used in
      qa_core (single-direction proxy), returning |cos(theta)|.
    - If True, compute the maximum |cos| against the 240 E8 roots.
    """
    vec = embed_qa_tuple_to_r8(qa)
    if not use_full_roots:
        return abs(_cosine_similarity(vec, IDEAL_E8_ROOT))
    roots = get_e8_roots_cached()
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0.0:
        return 0.0
    # Normalize once
    v = vec / vec_norm
    roots_norm = roots / np.linalg.norm(roots, axis=1, keepdims=True)
    sims = roots_norm @ v
    return float(np.max(np.abs(sims)))


def compute_harmonic_index(qa: Sequence[int], loss: float = 0.0, k: float = 0.1) -> float:
    """Harmonic Index proxy: HI = E8_alignment × exp(-k × loss)."""
    return compute_e8_alignment(qa) * math.exp(-k * float(loss))


def is_valid_qa_tuple(b: int, e: int, d: int, a: int, modulus: int = MODULUS_DEFAULT) -> bool:
    return (d % modulus) == ((b + e) % modulus) and (a % modulus) == ((b + 2 * e) % modulus)


def hash_to_qa_tuple(entity_name: str, modulus: int = MODULUS_DEFAULT) -> Tuple[int, int, int, int]:
    """Deterministic hash-based mapping from name → (b,e,d,a) with constraints.

    Uses SHA-256; derives b and e from disjoint 4-byte slices; computes d, a.
    """
    h = hashlib.sha256(entity_name.encode("utf-8")).digest()
    b = int.from_bytes(h[0:4], "big") % modulus
    e = int.from_bytes(h[4:8], "big") % modulus
    d = (b + e) % modulus
    a = (b + 2 * e) % modulus
    return b, e, d, a


def torus_distance(u: int, v: int, modulus: int = MODULUS_DEFAULT) -> int:
    """Shortest circular distance on Z_modulus between two residues."""
    diff = abs((u - v) % modulus)
    return min(diff, modulus - diff)


def compute_tuple_similarity(t1: Sequence[int], t2: Sequence[int], modulus: int = MODULUS_DEFAULT) -> float:
    """Similarity in [0,1] based on torus distances of (b,e) components.

    1.0 means identical (b,e); 0.0 means maximally distant on both axes.
    """
    b1, e1 = int(t1[0]), int(t1[1])
    b2, e2 = int(t2[0]), int(t2[1])
    db = torus_distance(b1, b2, modulus)
    de = torus_distance(e1, e2, modulus)
    # Max distance on each axis is modulus/2
    max_axis = modulus / 2.0
    sim = 1.0 - ((db / max_axis) + (de / max_axis)) / 2.0
    return float(max(0.0, min(1.0, sim)))


def transition_tuple(src: Sequence[int], tgt: Sequence[int], modulus: int = MODULUS_DEFAULT) -> Tuple[int, int, int, int]:
    """Compute (tgt - src) mod modulus component-wise for QA tuples."""
    return tuple((int(tgt[i]) - int(src[i])) % modulus for i in range(4))
