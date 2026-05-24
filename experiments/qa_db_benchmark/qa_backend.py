"""
QA-native backend.

Canonical arithmetic-geometric derived structure (integer/Fraction only):
  d = b + e
  a = e + d = b + 2e

  B = b*b
  E = e*e
  D = d*d
  A = a*a

  C = 2*e*d          (loci/focus-distance analogue)
  F = a*b            (product of outer elements)
  G = D + E          (sum of squares = d²+e²)
  J = b*d
  X = e*d
  K = d*a

  major_axis   = 2*D        (quantum ellipse major axis = 2d²)
  axis_split   = J + K      (must equal 2*D)
  pyth_ident   = C*C + F*F  (must equal G*G when G = D + E)
                             NOTE: C²+F²==G² iff (2ed)²+(ab)²==(d²+e²)²
                             This is a QA structural identity, not always true for
                             arbitrary (b,e); it is an invariant-check, not a
                             universal law.
  I            = abs(C - F)

  orbit_9  = (b + e) % 9
  orbit_24 = (b + e) % 24
  area     = B * E          (proxy for ellipse area)
  shape_sig = (C % 9, F % 9, G % 9)

  primitive = (gcd(b, e) == 1)
  parity    = 'odd' if b % 2 == 1 and e % 2 == 1 else 'even' if b % 2 == 0 and e % 2 == 0 else 'mixed'

Generators (all return new (b,e) pairs):
  sigma:   (b, e) -> (b, e+1)
  mu:      (b, e) -> (e, b)
  lambda2: (b, e) -> (2b, 2e)
  nu:      (b, e) -> (b//2, e//2)  only when b%2==0 and e%2==0

QA-native retrieval prunes the candidate set by invariant buckets BEFORE
expanding the reachability graph, exploiting the fact that legal generator
moves preserve or shift invariant signatures predictably.

Storage estimate: one QAPacket is represented as (b, e) seed + the law that
generates all derived fields — O(2 integers) per packet, not O(14 fields).
"""
from __future__ import annotations
import math
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

from metrics import QueryResult, measure


@dataclass(frozen=True)
class QAPacket:
    b: int
    e: int

    @property
    def d(self) -> int:
        return self.b + self.e

    @property
    def a(self) -> int:
        return self.e + self.d   # = b + 2e

    @property
    def B(self) -> int:
        return self.b * self.b

    @property
    def E(self) -> int:
        return self.e * self.e

    @property
    def D(self) -> int:
        d = self.d
        return d * d

    @property
    def A(self) -> int:
        a = self.a
        return a * a

    @property
    def C(self) -> int:
        return 2 * self.e * self.d

    @property
    def F(self) -> int:
        return self.a * self.b

    @property
    def G(self) -> int:
        return self.D + self.E

    @property
    def J(self) -> int:
        return self.b * self.d

    @property
    def X(self) -> int:
        return self.e * self.d

    @property
    def K(self) -> int:
        return self.d * self.a

    @property
    def major_axis(self) -> int:
        return 2 * self.D

    @property
    def axis_split(self) -> int:
        return self.J + self.K

    @property
    def axis_identity_holds(self) -> bool:
        return self.axis_split == self.major_axis

    @property
    def pyth_identity_holds(self) -> bool:
        C, F, G = self.C, self.F, self.G
        return C * C + F * F == G * G

    @property
    def I(self) -> int:
        return abs(self.C - self.F)

    @property
    def orbit_9(self) -> int:
        return (self.b + self.e) % 9

    @property
    def orbit_24(self) -> int:
        return (self.b + self.e) % 24

    @property
    def area(self) -> int:
        return self.B * self.E

    @property
    def shape_sig(self) -> tuple[int, int, int]:
        return (self.C % 9, self.F % 9, self.G % 9)

    @property
    def primitive(self) -> bool:
        return math.gcd(self.b, self.e) == 1

    @property
    def parity(self) -> str:
        if self.b % 2 == 1 and self.e % 2 == 1:
            return "odd"
        if self.b % 2 == 0 and self.e % 2 == 0:
            return "even"
        return "mixed"

    def storage_bytes_approx(self) -> int:
        # Two ints (b, e) + law pointer — approximate Python object size
        return 2 * 8 + 56  # 2 int64 + dataclass overhead

    def sigma(self, N: int) -> "QAPacket | None":
        ne = self.e + 1
        if ne > N:
            return None
        return QAPacket(self.b, ne)

    def mu(self) -> "QAPacket":
        return QAPacket(self.e, self.b)

    def lambda2(self, N: int) -> "QAPacket | None":
        nb, ne = 2 * self.b, 2 * self.e
        if nb > N or ne > N:
            return None
        return QAPacket(nb, ne)

    def nu(self) -> "QAPacket | None":
        if self.b % 2 != 0 or self.e % 2 != 0:
            return None
        return QAPacket(self.b // 2, self.e // 2)

    def legal_neighbors(self, N: int) -> list["QAPacket"]:
        out = []
        for m in [self.sigma(N), self.mu(), self.lambda2(N), self.nu()]:
            if m is not None and 1 <= m.b <= N and 1 <= m.e <= N:
                out.append(m)
        return out


def _passes_filter(pkt: QAPacket, q: dict[str, Any]) -> bool:
    # ── QA-structured predicates ──────────────────────────────────────────────
    if q["orbit_mod"] == 9:
        if pkt.orbit_9 != q["orbit_val"] % 9:
            return False
    else:
        if pkt.orbit_24 != q["orbit_val"] % 24:
            return False
    if pkt.I > q["i_gap_max"]:
        return False
    if pkt.area > q["area_max"]:
        return False
    if q["shape_sig"] is not None and pkt.shape_sig != q["shape_sig"]:
        return False
    if q["parity"] is not None and pkt.parity != q["parity"]:
        return False
    if q["require_primitive"] and not pkt.primitive:
        return False
    if q["axis_check"] and not pkt.axis_identity_holds:
        return False
    if q["pyth_check"] and not pkt.pyth_identity_holds:
        return False
    # ── Non-QA predicates (random_attribute, range_only, mixed modes) ─────────
    if q.get("b_mod_n") is not None:
        if pkt.b % q["b_mod_n"] != q["b_mod_val"]:
            return False
    if q.get("e_mod_n") is not None:
        if pkt.e % q["e_mod_n"] != q["e_mod_val"]:
            return False
    if q.get("b_lo") is not None:
        if not (q["b_lo"] <= pkt.b <= q["b_hi"]):
            return False
    if q.get("e_lo") is not None:
        if not (q["e_lo"] <= pkt.e <= q["e_hi"]):
            return False
    return True


def _log2_class(x: int) -> int:
    """Logarithmic bucket class: floor(log2(x+1)). x=0→0, 1→0, 2-3→1, 4-7→2, ..."""
    if x <= 0:
        return 0
    cls = 0
    while (1 << (cls + 1)) <= x:
        cls += 1
    return cls


# Sentinel: i_gap_max or area_max above this value → treat as unconstrained
# (no QA advantage from those dimensions; fall back to flat orbit index)
_UNCONSTRAINED_THRESHOLD: int = 10 ** 10


class QABackend:
    """
    QA-native backend with richer invariant buckets than table/graph.

    Rich bucket key: (orbit_9, orbit_24, parity, i_gap_class, area_class)
    where i_gap_class = floor(log2(I+1)) and area_class = floor(log2(area+1)).

    Flat fallback indexes (same as table/graph):
      _flat_orbit9[orbit_9_val] → set of keys
      _flat_orbit24[orbit_24_val] → set of keys
      _flat_parity[parity] → set of keys

    Phase 1 strategy:
      If i_gap_max AND area_max are both above _UNCONSTRAINED_THRESHOLD:
        → use flat orbit index (same as table; no QA structural advantage)
      Else:
        → use rich 5-dim compound bucket (QA structural advantage)

    This means:
      orbit_only / random_attribute / range_only modes degrade cleanly to the
      flat index, making waste(qa) ≈ waste(table) as the falsifier predicts.
      full_structured mode uses rich buckets → waste(qa) < waste(table).
    """
    def __init__(self, N: int = 250):
        self.N = N
        self._universe: dict[tuple[int, int], QAPacket] = {}
        # Rich 5-dim compound buckets
        self._invariant_buckets: dict[tuple, list[tuple[int, int]]] = {}
        # Flat orbit/parity indexes (fallback for unconstrained queries)
        self._flat_orbit9: dict[int, set[tuple[int, int]]] = {}
        self._flat_orbit24: dict[int, set[tuple[int, int]]] = {}
        self._flat_parity: dict[str, set[tuple[int, int]]] = {}
        self._build()

    def _build(self):
        N = self.N
        for b in range(1, N + 1):
            for e in range(1, N + 1):
                pkt = QAPacket(b, e)
                key = (b, e)
                self._universe[key] = pkt
                # Rich compound bucket
                bk = (
                    pkt.orbit_9,
                    pkt.orbit_24,
                    pkt.parity,
                    _log2_class(pkt.I),
                    _log2_class(pkt.area),
                )
                self._invariant_buckets.setdefault(bk, []).append(key)
                # Flat indexes
                self._flat_orbit9.setdefault(pkt.orbit_9, set()).add(key)
                self._flat_orbit24.setdefault(pkt.orbit_24, set()).add(key)
                self._flat_parity.setdefault(pkt.parity, set()).add(key)

    def storage_bytes_approx(self) -> int:
        universe_bytes = len(self._universe) * (2 * 8 + 56)
        return universe_bytes

    def run_query(self, q: dict[str, Any]) -> QueryResult:
        def _run():
            N = self.N
            k = q["k"]

            orbit_9_target = q["orbit_val"] % 9
            orbit_24_target = q["orbit_val"] % 24
            parity_target = q.get("parity")

            i_constrained = q["i_gap_max"] < _UNCONSTRAINED_THRESHOLD
            a_constrained = q["area_max"] < _UNCONSTRAINED_THRESHOLD

            # Phase 1: choose pre-filter strategy
            if i_constrained or a_constrained:
                # Rich compound bucket — QA structural advantage
                max_i_gap_cls = _log2_class(q["i_gap_max"])
                max_area_cls = _log2_class(q["area_max"])
                candidate_keys: set[tuple[int, int]] = set()
                for (o9, o24, par, igc, ac), keys in self._invariant_buckets.items():
                    orbit_match = (
                        (q["orbit_mod"] == 9 and o9 == orbit_9_target) or
                        (q["orbit_mod"] == 24 and o24 == orbit_24_target)
                    )
                    parity_match = parity_target is None or par == parity_target
                    gap_ok = igc <= max_i_gap_cls
                    area_ok = ac <= max_area_cls
                    if orbit_match and parity_match and gap_ok and area_ok:
                        candidate_keys.update(keys)
            else:
                # Flat orbit index — same as table/graph; no QA pre-filter advantage
                if q["orbit_mod"] == 9:
                    candidate_keys = set(self._flat_orbit9.get(orbit_9_target, set()))
                else:
                    candidate_keys = set(self._flat_orbit24.get(orbit_24_target, set()))
                if parity_target is not None:
                    candidate_keys &= self._flat_parity.get(parity_target, set())

            candidate_count_before = len(candidate_keys)

            # Phase 2: filter by remaining predicates (I-gap, area, shape_sig, etc.)
            filtered: set[tuple[int, int]] = set()
            for key in candidate_keys:
                pkt = self._universe[key]
                if _passes_filter(pkt, q):
                    filtered.add(key)

            candidate_count_after = len(filtered)

            # Phase 3: BFS reachability from seeds, constrained to filtered set
            seeds = q["seeds"]
            frontier: set[tuple[int, int]] = set()
            for (b, e) in seeds:
                if (b, e) in filtered:
                    frontier.add((b, e))

            visited: set[tuple[int, int]] = set(frontier)
            expansion_count = 0

            for _step in range(k):
                next_frontier: set[tuple[int, int]] = set()
                for key in frontier:
                    pkt = self._universe[key]
                    for nb in pkt.legal_neighbors(N):
                        nkey = (nb.b, nb.e)
                        if nkey not in visited and nkey in filtered:
                            next_frontier.add(nkey)
                            expansion_count += 1
                    visited.add(key)
                frontier = next_frontier - visited
                visited |= frontier

            results = frozenset(visited)
            collapse_ratio = (candidate_count_after / candidate_count_before
                              if candidate_count_before > 0 else 0.0)
            return candidate_count_before, candidate_count_after, expansion_count, results, collapse_ratio

        (cb, ca, exp, results, cr), lat = measure(_run)
        return QueryResult(
            backend_name="qa",
            query_id=q["query_id"],
            latency_ns=lat,
            candidate_count_before=cb,
            candidate_count_after=ca,
            expansion_count=exp,
            results_count=len(results),
            collapse_ratio=cr,
            bytes_estimate=self.storage_bytes_approx(),
            result_keys=results,
            path_constrained=True,  # QA enforces generator-legal moves
        )
