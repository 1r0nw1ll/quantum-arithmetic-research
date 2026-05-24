"""
QA-VFS core: shared packet arithmetic and chunk derivation.

Imports no external dependencies.  All backends share this module.

Canonical QA fields (integers only, no float):
  d = b + e          a = b + 2e
  B = b*b            E = e*e        D = d*d        A = a*a
  C = 2*e*d          F = a*b        G = D + E
  J = b*d            X = e*d        K = d*a
  I = |C - F|        area = B*E
  orbit_9 = (b+e)%9  orbit_24 = (b+e)%24
  size_class = floor(log2(area+1))
  lineage_class = orbit_9
"""
from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass, field
from typing import Sequence


# ── Packet arithmetic ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QAPacket:
    b: int
    e: int

    @property
    def d(self) -> int: return self.b + self.e
    @property
    def a(self) -> int: return self.e + self.d
    @property
    def B(self) -> int: return self.b * self.b
    @property
    def E(self) -> int: return self.e * self.e
    @property
    def D(self) -> int: d = self.d; return d * d
    @property
    def A(self) -> int: a = self.a; return a * a
    @property
    def C(self) -> int: return 2 * self.e * self.d
    @property
    def F(self) -> int: return self.a * self.b
    @property
    def G(self) -> int: return self.D + self.E
    @property
    def J(self) -> int: return self.b * self.d
    @property
    def X(self) -> int: return self.e * self.d
    @property
    def K(self) -> int: return self.d * self.a
    @property
    def I(self) -> int: return abs(self.C - self.F)
    @property
    def area(self) -> int: return self.B * self.E
    @property
    def orbit_9(self) -> int: return (self.b + self.e) % 9
    @property
    def orbit_24(self) -> int: return (self.b + self.e) % 24

    def legal_neighbors(self, N: int) -> list["QAPacket"]:
        out = []
        # sigma: (b, e+1)
        if self.e + 1 <= N:
            out.append(QAPacket(self.b, self.e + 1))
        # mu: (e, b)
        nb = QAPacket(self.e, self.b)
        if 1 <= nb.b <= N and 1 <= nb.e <= N:
            out.append(nb)
        # lambda2: (2b, 2e)
        if 2 * self.b <= N and 2 * self.e <= N:
            out.append(QAPacket(2 * self.b, 2 * self.e))
        # nu: (b/2, e/2) only when both even
        if self.b % 2 == 0 and self.e % 2 == 0:
            out.append(QAPacket(self.b // 2, self.e // 2))
        return out


_UNCONSTRAINED = 10 ** 15


def _log2_class(x: int) -> int:
    if x <= 0:
        return 0
    cls = 0
    while (1 << (cls + 1)) <= x:
        cls += 1
    return cls


def size_class(pkt: QAPacket) -> int:
    return _log2_class(pkt.area)


def lineage_class(pkt: QAPacket) -> int:
    return pkt.orbit_9


# ── Chunk derivation ──────────────────────────────────────────────────────────

def derive_chunk_sequence(root_b: int, root_e: int, n_chunks: int, N: int) -> list[tuple[int, int]]:
    """BFS from root packet; return first n_chunks (b,e) keys in discovery order."""
    root = QAPacket(root_b, root_e)
    visited: set[tuple[int, int]] = {(root_b, root_e)}
    order: list[tuple[int, int]] = [(root_b, root_e)]
    frontier: list[QAPacket] = [root]
    while len(order) < n_chunks and frontier:
        next_f: list[QAPacket] = []
        for pkt in frontier:
            for nb in pkt.legal_neighbors(N):
                key = (nb.b, nb.e)
                if key not in visited:
                    visited.add(key)
                    order.append(key)
                    next_f.append(nb)
                if len(order) >= n_chunks:
                    break
            if len(order) >= n_chunks:
                break
        frontier = next_f
    return order[:n_chunks]


def chunk_content_val(b: int, e: int) -> int:
    """Deterministic chunk content derived from packet arithmetic. Integer, no float."""
    pkt = QAPacket(b, e)
    return (pkt.C * pkt.D + pkt.F * pkt.J) % (2 ** 31)


def chunk_content_hash(b: int, e: int) -> str:
    """Hex digest of content value — used for corruption detection."""
    val = chunk_content_val(b, e)
    return hashlib.sha256(val.to_bytes(8, "big")).hexdigest()[:16]


# ── File model ────────────────────────────────────────────────────────────────

@dataclass
class VFSFile:
    root_b: int
    root_e: int
    n_chunks: int
    corrupted_chunks: set[int] = field(default_factory=set)
    # Arbitrary mutation deviations: chunk_idx → deviation_content_val
    # Present only when content was changed outside QA law.
    deviations: dict[int, int] = field(default_factory=dict)

    @property
    def file_id(self) -> tuple[int, int]:
        return (self.root_b, self.root_e)

    @property
    def root_packet(self) -> QAPacket:
        return QAPacket(self.root_b, self.root_e)

    def size_class(self) -> int:
        return _log2_class(self.root_packet.area)

    def lineage_class(self) -> int:
        return self.root_packet.orbit_9

    def i_gap(self) -> int:
        return self.root_packet.I

    def orbit_9(self) -> int:
        return self.root_packet.orbit_9

    def orbit_24(self) -> int:
        return self.root_packet.orbit_24


# ── Workload query format ─────────────────────────────────────────────────────

def make_lookup_query(
    qi: int,
    seeds: list[tuple[int, int]],
    k: int,
    orbit_mod: int,
    orbit_val: int,
    size_class_max: int,
    i_gap_max: int,
    query_mode: str = "structured",
    b_mod_n: int | None = None,
    b_mod_val: int | None = None,
) -> dict:
    return {
        "query_id": f"q{qi:04d}",
        "seeds": seeds,
        "k": k,
        "orbit_mod": orbit_mod,
        "orbit_val": orbit_val,
        "size_class_max": size_class_max,
        "i_gap_max": i_gap_max,
        "query_mode": query_mode,
        "b_mod_n": b_mod_n,
        "b_mod_val": b_mod_val,
    }
