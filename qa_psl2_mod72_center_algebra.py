#!/usr/bin/env python3
"""
PSL2(Z/72Z) Center Algebra Engine (internal CRT route, exact)

Entry points:
  - Build caches:
      python qa_psl2_mod72_center_algebra.py --build
  - Self-test (sanity + deterministic hashes):
      python qa_psl2_mod72_center_algebra.py --self_test

What this is
------------
An exact, deterministic engine to support the next QA certificate families:

  1) Build exact center-algebra data for SL2(Z/8Z) and SL2(Z/9Z)
  2) Combine by CRT into SL2(Z/72Z) ≅ SL2(Z/8Z) × SL2(Z/9Z)
  3) Pass to PSL2(Z/72Z) by quotienting by the diagonal sign subgroup
     H = {(I,I),(-I,-I)} (central).

The engine provides:
  - conjugacy classes for SL2 mod 8 and mod 9 (deterministic order)
  - class-sum multiplication constants for each factor center Z(Q[G_n])
  - the sign-action maps on conjugacy classes (g -> -g)
  - orbit data giving PSL2(Z/72Z) conjugacy classes as sign-orbits of product classes
  - exact multiplication for central elements in PSL2(Z/72Z) *in orbit-basis*,
    without enumerating the 72-level group elements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# --- Section: paths / IO ---


CACHE_DIR = Path("qa_center_algebra_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _read_json(path: Path) -> object:
    with path.open() as f:
        return json.load(f)


# --- Section: arithmetic / group basics ---


Mat = Tuple[int, int, int, int]


def mul_mod(x: Mat, y: Mat, n: int) -> Mat:
    ax, bx, cx, dx = x
    ay, by, cy, dy = y
    return (
        (ax * ay + bx * cy) % n,
        (ax * by + bx * dy) % n,
        (cx * ay + dx * cy) % n,
        (cx * by + dx * dy) % n,
    )


def inv_mod(x: Mat, n: int) -> Mat:
    a, b, c, d = x
    return (d % n, (-b) % n, (-c) % n, a % n)


def conj_mod(g: Mat, x: Mat, n: int) -> Mat:
    return mul_mod(mul_mod(g, x, n), inv_mod(g, n), n)


def det_mod(x: Mat, n: int) -> int:
    a, b, c, d = x
    return (a * d - b * c) % n


def neg_mod(x: Mat, n: int) -> Mat:
    a, b, c, d = x
    return ((-a) % n, (-b) % n, (-c) % n, (-d) % n)


def identity(n: int) -> Mat:
    return (1 % n, 0, 0, 1 % n)


def expected_sl2_order_prime_power(p: int, k: int) -> int:
    if p == 2:
        return 6 * (2 ** (3 * k - 3))
    return (p ** (3 * k - 2)) * (p * p - 1)


def factorize(n: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            out[d] = out.get(d, 0) + 1
            x //= d
        d = 3 if d == 2 else d + 2
    if x > 1:
        out[x] = out.get(x, 0) + 1
    return out


def expected_sl2_order(n: int) -> int:
    f = factorize(n)
    out = 1
    for p, k in f.items():
        out *= expected_sl2_order_prime_power(p, k)
    return out


# --- Section: SL2(Z/nZ) construction ---


def enumerate_sl2_mod_n(n: int) -> List[Mat]:
    els: List[Mat] = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    if (a * d - b * c) % n == 1 % n:
                        els.append((a, b, c, d))
    els.sort()
    return els


def sl2_generators_mod_n(n: int) -> Tuple[Mat, Mat, Mat]:
    # Reduce standard S,T,T^{-1} to mod n.
    s = (0, (-1) % n, 1 % n, 0)
    t = (1 % n, 1 % n, 0, 1 % n)
    u = (1 % n, (-1) % n, 0, 1 % n)
    return s, t, u


def conjugacy_classes_sl2_mod_n(elements: List[Mat], n: int) -> Tuple[List[List[Mat]], Dict[Mat, int]]:
    s, t, u = sl2_generators_mod_n(n)
    conj_gens = (s, t, u)

    unassigned = set(elements)
    classes: List[List[Mat]] = []

    while unassigned:
        start = min(unassigned)
        q = deque([start])
        unassigned.remove(start)
        cls = []
        while q:
            x = q.popleft()
            cls.append(x)
            for g in conj_gens:
                y = conj_mod(g, x, n)
                if y in unassigned:
                    unassigned.remove(y)
                    q.append(y)
        cls.sort()
        classes.append(cls)

    # Deterministic ordering: by smallest element.
    classes.sort(key=lambda c: c[0])

    elem_to_class: Dict[Mat, int] = {}
    for idx, cls in enumerate(classes):
        for x in cls:
            elem_to_class[x] = idx
    return classes, elem_to_class


def class_sum_structure_constants(
    elements: List[Mat],
    elem_to_class: Dict[Mat, int],
    classes: List[List[Mat]],
    n: int,
) -> List[List[List[int]]]:
    """
    Compute integers m[i][j][k] such that for class sums K_i = sum_{x in C_i} x,
    we have K_i K_j = sum_k m[i][j][k] K_k in Q[SL2(Z/nZ)].
    """
    r = len(classes)
    sizes = [len(c) for c in classes]

    total = [[[0 for _ in range(r)] for _ in range(r)] for __ in range(r)]

    # Accumulate counts_total[i][j][k] = #{(x,y) in C_i x C_j : xy in C_k}.
    for x in elements:
        i = elem_to_class[x]
        for y in elements:
            j = elem_to_class[y]
            xy = mul_mod(x, y, n)
            k = elem_to_class[xy]
            total[i][j][k] += 1

    m = [[[0 for _ in range(r)] for _ in range(r)] for __ in range(r)]
    for i in range(r):
        for j in range(r):
            for k in range(r):
                cnt = total[i][j][k]
                sz = sizes[k]
                if cnt % sz != 0:
                    raise RuntimeError(f"non-integer structure constant at (i,j,k)=({i},{j},{k}): {cnt}/{sz}")
                m[i][j][k] = cnt // sz
    return m


@dataclass(frozen=True, slots=True)
class FactorCenterCache:
    modulus: int
    group_order: int
    num_classes: int
    identity_class: int
    neg_identity_class: int
    class_reps: List[Mat]
    class_sizes: List[int]
    inv_class: List[int]
    neg_class: List[int]
    mult: List[List[List[int]]]
    reps_hash: str

    def to_json(self) -> object:
        return {
            "modulus": self.modulus,
            "group_order": self.group_order,
            "num_classes": self.num_classes,
            "identity_class": self.identity_class,
            "neg_identity_class": self.neg_identity_class,
            "class_reps": [list(x) for x in self.class_reps],
            "class_sizes": self.class_sizes,
            "inv_class": self.inv_class,
            "neg_class": self.neg_class,
            "mult": self.mult,
            "reps_hash": self.reps_hash,
        }

    @staticmethod
    def from_json(obj: object) -> "FactorCenterCache":
        d = obj  # type: ignore[assignment]
        class_reps = [tuple(x) for x in d["class_reps"]]
        return FactorCenterCache(
            modulus=int(d["modulus"]),
            group_order=int(d["group_order"]),
            num_classes=int(d["num_classes"]),
            identity_class=int(d["identity_class"]),
            neg_identity_class=int(d["neg_identity_class"]),
            class_reps=class_reps,  # type: ignore[arg-type]
            class_sizes=list(d["class_sizes"]),
            inv_class=list(d["inv_class"]),
            neg_class=list(d["neg_class"]),
            mult=d["mult"],
            reps_hash=str(d["reps_hash"]),
        )


def build_factor_cache(n: int) -> FactorCenterCache:
    elements = enumerate_sl2_mod_n(n)
    exp = expected_sl2_order(n)
    if len(elements) != exp:
        raise RuntimeError(f"SL2(Z/{n}Z) enumeration size mismatch: got {len(elements)} expected {exp}")

    classes, elem_to_class = conjugacy_classes_sl2_mod_n(elements, n)
    reps = [c[0] for c in classes]
    sizes = [len(c) for c in classes]
    reps_hash = _sha256_json([list(r) for r in reps])

    id_cls = elem_to_class[identity(n)]
    neg_id_cls = elem_to_class[neg_mod(identity(n), n)]

    inv_class = []
    neg_class = []
    for i, rep in enumerate(reps):
        inv_rep = inv_mod(rep, n)
        inv_class.append(elem_to_class[inv_rep])

        neg_rep = neg_mod(rep, n)
        neg_class.append(elem_to_class[neg_rep])

    mult = class_sum_structure_constants(elements, elem_to_class, classes, n)

    return FactorCenterCache(
        modulus=n,
        group_order=len(elements),
        num_classes=len(classes),
        identity_class=id_cls,
        neg_identity_class=neg_id_cls,
        class_reps=reps,
        class_sizes=sizes,
        inv_class=inv_class,
        neg_class=neg_class,
        mult=mult,
        reps_hash=reps_hash,
    )


def load_or_build_factor_cache(n: int) -> FactorCenterCache:
    path = CACHE_DIR / f"sl2_mod{n}_center_cache.json"
    if path.exists():
        try:
            cache = FactorCenterCache.from_json(_read_json(path))
            if cache.modulus != n:
                raise ValueError("cache modulus mismatch")
            return cache
        except (KeyError, ValueError, TypeError):
            # Cache schema or content mismatch; rebuild deterministically.
            pass
    cache = build_factor_cache(n)
    _write_json(path, cache.to_json())
    return cache


# --- Section: CRT combine for mod 72 ---


def crt_combine_mod72(x8: int, x9: int) -> int:
    """
    Solve x ≡ x8 (mod 8), x ≡ x9 (mod 9), returning x in {0..71}.

    Using:
      inv(9 mod 8) = 1
      inv(8 mod 9) = 8 (since 8*8=64≡1 mod9)
    So x = x8*9*1 + x9*8*8 (mod 72).
    """
    return (x8 * 9 + x9 * 64) % 72


def crt_combine_mat_mod72(m8: Mat, m9: Mat) -> Mat:
    a8, b8, c8, d8 = m8
    a9, b9, c9, d9 = m9
    return (
        crt_combine_mod72(a8, a9),
        crt_combine_mod72(b8, b9),
        crt_combine_mod72(c8, c9),
        crt_combine_mod72(d8, d9),
    )


def psl_normal_form_mod_n(x: Mat, n: int) -> Mat:
    neg = neg_mod(x, n)
    return x if x <= neg else neg


# --- Section: PSL2(Z/72Z) conjugacy classes via factor orbits ---


@dataclass(frozen=True, slots=True)
class OrbitClass72:
    orbit_index: int
    i8: int
    i9: int
    neg_i8: int
    neg_i9: int
    size_psl: int
    rep72: Mat

    def to_json(self) -> object:
        return {
            "orbit_index": self.orbit_index,
            "i8": self.i8,
            "i9": self.i9,
            "neg_i8": self.neg_i8,
            "neg_i9": self.neg_i9,
            "size_psl": self.size_psl,
            "rep72": list(self.rep72),
        }


@dataclass(frozen=True, slots=True)
class PSL2Mod72OrbitCache:
    modulus: int
    r8: int
    r9: int
    num_orbits: int
    factor_hash8: str
    factor_hash9: str
    orbits: List[OrbitClass72]
    pair_to_orbit: List[int]  # length r8*r9

    def to_json(self) -> object:
        return {
            "modulus": self.modulus,
            "r8": self.r8,
            "r9": self.r9,
            "num_orbits": self.num_orbits,
            "factor_hash8": self.factor_hash8,
            "factor_hash9": self.factor_hash9,
            "orbits": [o.to_json() for o in self.orbits],
            "pair_to_orbit": self.pair_to_orbit,
        }

    @staticmethod
    def from_json(obj: object) -> "PSL2Mod72OrbitCache":
        d = obj  # type: ignore[assignment]
        orbits = [
            OrbitClass72(
                orbit_index=int(o["orbit_index"]),
                i8=int(o["i8"]),
                i9=int(o["i9"]),
                neg_i8=int(o["neg_i8"]),
                neg_i9=int(o["neg_i9"]),
                size_psl=int(o["size_psl"]),
                rep72=tuple(o["rep72"]),
            )
            for o in d["orbits"]
        ]
        return PSL2Mod72OrbitCache(
            modulus=int(d["modulus"]),
            r8=int(d["r8"]),
            r9=int(d["r9"]),
            num_orbits=int(d["num_orbits"]),
            factor_hash8=str(d["factor_hash8"]),
            factor_hash9=str(d["factor_hash9"]),
            orbits=orbits,
            pair_to_orbit=list(d["pair_to_orbit"]),
        )


def build_psl2_mod72_orbit_cache(cache8: FactorCenterCache, cache9: FactorCenterCache) -> PSL2Mod72OrbitCache:
    r8, r9 = cache8.num_classes, cache9.num_classes
    neg8, neg9 = cache8.neg_class, cache9.neg_class

    pair_to_orbit = [-1 for _ in range(r8 * r9)]
    orbits: List[OrbitClass72] = []

    def key_pair(i: int, j: int) -> Tuple[int, int]:
        return (i, j)

    for i in range(r8):
        for j in range(r9):
            idx = i * r9 + j
            if pair_to_orbit[idx] != -1:
                continue
            ni, nj = neg8[i], neg9[j]
            # Canonicalize orbit representative among (i,j) and (ni,nj).
            a = key_pair(i, j)
            b = key_pair(ni, nj)
            ci, cj = (a if a <= b else b)
            cni, cnj = (b if a <= b else a)

            orbit_index = len(orbits)

            # Mark both pairs as belonging to this orbit.
            pair_to_orbit[ci * r9 + cj] = orbit_index
            pair_to_orbit[cni * r9 + cnj] = orbit_index

            # PSL class size equals |C8_ci| * |C9_cj| (since union has size 2*, quotient by ± halves).
            size_psl = cache8.class_sizes[ci] * cache9.class_sizes[cj]

            rep72 = psl_normal_form_mod_n(crt_combine_mat_mod72(cache8.class_reps[ci], cache9.class_reps[cj]), 72)

            orbits.append(
                OrbitClass72(
                    orbit_index=orbit_index,
                    i8=ci,
                    i9=cj,
                    neg_i8=cni,
                    neg_i9=cnj,
                    size_psl=size_psl,
                    rep72=rep72,
                )
            )

    if any(x == -1 for x in pair_to_orbit):
        raise RuntimeError("pair_to_orbit incomplete")

    return PSL2Mod72OrbitCache(
        modulus=72,
        r8=r8,
        r9=r9,
        num_orbits=len(orbits),
        factor_hash8=cache8.reps_hash,
        factor_hash9=cache9.reps_hash,
        orbits=orbits,
        pair_to_orbit=pair_to_orbit,
    )


def load_or_build_psl2_mod72_orbit_cache(cache8: FactorCenterCache, cache9: FactorCenterCache) -> PSL2Mod72OrbitCache:
    path = CACHE_DIR / "psl2_mod72_orbit_cache.json"
    if path.exists():
        try:
            cache = PSL2Mod72OrbitCache.from_json(_read_json(path))
            # Basic compatibility check: factor hashes must match.
            if cache.factor_hash8 == cache8.reps_hash and cache.factor_hash9 == cache9.reps_hash:
                return cache
        except (KeyError, ValueError, TypeError):
            pass
    cache = build_psl2_mod72_orbit_cache(cache8, cache9)
    _write_json(path, cache.to_json())
    return cache


# --- Section: exact center multiplication (orbit basis) ---


def mul_center_vec(mult: List[List[List[int]]], a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
    r = len(a)
    out = [Fraction(0, 1) for _ in range(r)]
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            cij = ai * bj
            row = mult[i][j]
            for k in range(r):
                m = row[k]
                if m:
                    out[k] += cij * m
    return out


def mul_tensor_center(
    cache8: FactorCenterCache,
    cache9: FactorCenterCache,
    a_mat: List[List[Fraction]],  # r8 x r9
    b_mat: List[List[Fraction]],  # r8 x r9
) -> List[List[Fraction]]:
    """
    Multiply central elements in Z(Q[G8×G9]) expressed in the tensor class-sum basis:
      E = Σ_{i,j} a[i][j] (K8_i ⊗ K9_j)
    returning the same shape matrix for the product.
    """
    r8, r9 = cache8.num_classes, cache9.num_classes

    # Precompute 9-center products of each row-pair: P[i][j] = (a_row_i * b_row_j) in mod9 center.
    products_9: List[List[List[Fraction]]] = [[[] for _ in range(r8)] for __ in range(r8)]
    for i in range(r8):
        for j in range(r8):
            products_9[i][j] = mul_center_vec(cache9.mult, a_mat[i], b_mat[j])

    out = [[Fraction(0, 1) for _ in range(r9)] for __ in range(r8)]
    for i in range(r8):
        for j in range(r8):
            p9 = products_9[i][j]
            row = cache8.mult[i][j]  # length r8
            for k in range(r8):
                m = row[k]
                if not m:
                    continue
                for l in range(r9):
                    if p9[l]:
                        out[k][l] += p9[l] * m
    return out


def lift_orbit_vec_to_pair_mat(orbit_cache: PSL2Mod72OrbitCache, vec: List[Fraction]) -> List[List[Fraction]]:
    """
    Lift coefficients in PSL class-sum basis:

      K̄_orbit := (1/2)(K_(i,j) + K_(neg(i),neg(j)))

    to coefficients in the product SL class-sum tensor basis K_(i,j).
    Therefore each pair entry receives vec[orbit]/2.
    """
    r8, r9 = orbit_cache.r8, orbit_cache.r9
    out = [[Fraction(0, 1) for _ in range(r9)] for __ in range(r8)]
    for i in range(r8):
        for j in range(r9):
            o = orbit_cache.pair_to_orbit[i * r9 + j]
            out[i][j] = vec[o] / 2
    return out


def project_pair_mat_to_orbit_vec(orbit_cache: PSL2Mod72OrbitCache, mat: List[List[Fraction]]) -> List[Fraction]:
    """
    Project coefficients in product SL tensor basis K_(i,j) back to PSL class-sum basis K̄_orbit.
    Since K̄_orbit has coefficient 1/2 on K_(i,j), we multiply by 2.
    """
    out = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    for o in orbit_cache.orbits:
        out[o.orbit_index] = 2 * mat[o.i8][o.i9]
    return out


def mul_psl2_mod72_center_orbits(
    cache8: FactorCenterCache,
    cache9: FactorCenterCache,
    orbit_cache: PSL2Mod72OrbitCache,
    a_orbit: List[Fraction],
    b_orbit: List[Fraction],
) -> List[Fraction]:
    a_mat = lift_orbit_vec_to_pair_mat(orbit_cache, a_orbit)
    b_mat = lift_orbit_vec_to_pair_mat(orbit_cache, b_orbit)
    prod_mat = mul_tensor_center(cache8, cache9, a_mat, b_mat)
    # Project back to orbit basis.
    return project_pair_mat_to_orbit_vec(orbit_cache, prod_mat)


# --- Section: CLI / self-tests ---


def _fraction_to_json(x: Fraction) -> Dict[str, int]:
    return {"num": x.numerator, "den": x.denominator}


def self_test() -> None:
    cache8 = load_or_build_factor_cache(8)
    cache9 = load_or_build_factor_cache(9)
    orbit_cache = load_or_build_psl2_mod72_orbit_cache(cache8, cache9)

    assert orbit_cache.num_orbits == 375, f"expected 375 orbits, got {orbit_cache.num_orbits}"
    assert cache8.num_classes == 30, f"expected 30 classes for mod8, got {cache8.num_classes}"
    assert cache9.num_classes == 25, f"expected 25 classes for mod9, got {cache9.num_classes}"

    # Check that sign-orbits are all size-2 in the product class index set.
    seen = set()
    for o in orbit_cache.orbits:
        a = (o.i8, o.i9)
        b = (o.neg_i8, o.neg_i9)
        if a == b:
            raise RuntimeError("found fixed sign-orbit; expected size-2 orbits for 72")
        seen.add(a)
        seen.add(b)
    assert len(seen) == cache8.num_classes * cache9.num_classes, "orbit coverage mismatch"

    # Identity orbit should be (id_class8, id_class9). Given class ordering by minimal element,
    # identity is not necessarily class index 0, so locate via cached indices.
    id_pair = (cache8.identity_class, cache9.identity_class)
    id_orbit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    # Multiplicative identity in orbit basis should satisfy e*x = x where e has coefficient 1 on id_orbit.
    e = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    e[id_orbit_idx] = Fraction(1, 1)

    # Test associativity on a few sparse central elements.
    a = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    b = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    c = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    a[1] = Fraction(2, 1)
    a[17] = Fraction(1, 3)
    b[5] = Fraction(7, 2)
    c[12] = Fraction(5, 1)
    c[6] = Fraction(-1, 4)

    ab = mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, b)
    abc1 = mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, ab, c)

    bc = mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, b, c)
    abc2 = mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, bc)

    if abc1 != abc2:
        raise RuntimeError("associativity check failed for sample central elements")

    # Identity check.
    ea = mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, e, a)
    if ea != a:
        raise RuntimeError("identity check failed: e*a != a")

    print("✓ SELF-TEST OK")
    print(f"  SL2 mod8:  order={cache8.group_order} classes={cache8.num_classes} reps_hash={cache8.reps_hash[:12]}")
    print(f"  SL2 mod9:  order={cache9.group_order} classes={cache9.num_classes} reps_hash={cache9.reps_hash[:12]}")
    print(f"  PSL2 mod72: orbits={orbit_cache.num_orbits} (product classes={cache8.num_classes*cache9.num_classes})")


def build() -> None:
    cache8 = load_or_build_factor_cache(8)
    cache9 = load_or_build_factor_cache(9)
    orbit_cache = load_or_build_psl2_mod72_orbit_cache(cache8, cache9)

    print("Caches ready:")
    print(f"  {CACHE_DIR / 'sl2_mod8_center_cache.json'}")
    print(f"  {CACHE_DIR / 'sl2_mod9_center_cache.json'}")
    print(f"  {CACHE_DIR / 'psl2_mod72_orbit_cache.json'}")
    print()
    print("Summary:")
    print(f"  |SL2(Z/8Z)|  = {cache8.group_order}, #classes={cache8.num_classes}")
    print(f"  |SL2(Z/9Z)|  = {cache9.group_order}, #classes={cache9.num_classes}")
    print(f"  #product classes = {cache8.num_classes * cache9.num_classes}")
    print(f"  #PSL2(Z/72Z) orbits (classes) = {orbit_cache.num_orbits}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.build:
        build()
        return 0
    if args.self_test:
        self_test()
        return 0
    raise SystemExit("usage: --build or --self_test")


if __name__ == "__main__":
    raise SystemExit(main())
