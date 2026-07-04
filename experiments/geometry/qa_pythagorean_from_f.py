#!/usr/bin/env python3
"""
Derive QA Pythagorean right triangles from a supplied F leg.

Canonical QA direction variables:
    b = d - e
    a = d + e
    F = b*a = d*d - e*e
    C = 2*d*e
    G = d*d + e*e

Two engines are included:
    tree:   QA direction-tree reachability from root (d,e)=(2,1), using the
            certified moves M_A/M_B/M_C. This is the genuine QA engine.
    factor: divisor-pair enumeration of F=b*a. This is only a baseline.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


QA_COMPLIANCE = "standalone_experiment, canonical_direction_derivation, integer_substrate"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def divisors_up_to_sqrt(n: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive")
    out = []
    limit = math.isqrt(n)
    for candidate in range(1, limit + 1):
        if n % candidate == 0:
            out.append(candidate)
    return out


def digital_root(n: int) -> int:
    if n == 0:
        return 0
    residue = n % 9
    return 9 if residue == 0 else residue


def classify_mod24(value: int) -> int:
    return value % 24


def is_square(n: int) -> tuple[bool, int]:
    if n < 0:
        return False, 0
    root = math.isqrt(n)
    return root*root == n, root


def square_residues(modulus: int) -> set[int]:
    if modulus <= 1:
        raise ValueError("sieve moduli must be greater than 1")
    return {(value*value) % modulus for value in range(modulus)}


def parse_moduli(text: str) -> list[int]:
    moduli = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 1:
            raise ValueError("sieve moduli must be greater than 1")
        moduli.append(value)
    return moduli


def primes_up_to(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = False
    sieve[1] = False
    for value in range(2, math.isqrt(limit) + 1):
        if sieve[value]:
            start = value * value
            for composite in range(start, limit + 1, value):
                sieve[composite] = False
    return [value for value, is_prime in enumerate(sieve) if is_prime]


def modular_roots_square(target: int, modulus: int) -> list[int]:
    target %= modulus
    return [value for value in range(modulus) if value*value % modulus == target]


def lcm_many(values: list[int]) -> int:
    out = 1
    for value in values:
        out = out * value // math.gcd(out, value)
    return out


def crt_pair(a: int, m: int, b: int, n: int) -> tuple[int, int]:
    if math.gcd(m, n) != 1:
        raise ValueError("CRT moduli must be coprime")
    inv = pow(m, -1, n)
    k = ((b - a) * inv) % n
    modulus = m * n
    return (a + m * k) % modulus, modulus


def combine_roots_crt(root_sets: list[tuple[int, list[int]]], limit: int) -> list[int]:
    combined = [(0, 1)]
    for modulus, roots in root_sets:
        next_combined = []
        for value, current_modulus in combined:
            for root in roots:
                merged, merged_modulus = crt_pair(value, current_modulus, root, modulus)
                next_combined.append((merged, merged_modulus))
                if len(next_combined) >= limit:
                    break
            if len(next_combined) >= limit:
                break
        combined = next_combined
        if not combined:
            break
    return [value for value, _modulus in combined]


def candidate_row(F: int, d: int, e: int, path: str, method: str) -> dict[str, Any]:
    b = d - e
    a = d + e
    C = 2 * d * e
    G = d*d + e*e
    area = (C * F) // 2

    gcd_all = math.gcd(math.gcd(C, F), G)
    gcd_de = math.gcd(d, e)
    opposite_parity = (d - e) % 2 == 1
    is_primitive = gcd_all == 1 and gcd_de == 1 and opposite_parity

    return {
        "F": F,
        "C": C,
        "G": G,
        "area": area,
        "a": a,
        "b": b,
        "d": d,
        "e": e,
        "checks": {
            "area_is_6n": area % 6 == 0,
            "C_is_4n": C % 4 == 0,
            "gcd_CFG": gcd_all,
            "gcd_de": gcd_de,
            "is_primitive": is_primitive,
            "is_pythagorean": F*F + C*C == G*G,
            "is_whole_number": d > e > 0 and C > 0 and G > 0,
            "opposite_parity": opposite_parity,
        },
        "qa": {
            "method": method,
            "path": path,
            "root": [2, 1],
        },
        "accelerators": {
            "C_mod24": classify_mod24(C),
            "F_mod24": classify_mod24(F),
            "G_mod24": classify_mod24(G),
            "digital_root_a": digital_root(a),
            "digital_root_b": digital_root(b),
            "direction_mod24": [classify_mod24(d), classify_mod24(e)],
            "divisor_pair": [b, a],
        },
    }


def qa_children(d: int, e: int) -> list[tuple[str, int, int]]:
    return [
        ("A", 2*d - e, d),
        ("B", 2*d + e, d),
        ("C", d + 2*e, e),
    ]


def qa_f(d: int, e: int) -> int:
    return d*d - e*e


def max_remaining_a_depth(d: int, e: int, target_F: int) -> int:
    depth = 0
    while True:
        child_d, child_e = 2*d - e, d
        if qa_f(child_d, child_e) > target_F:
            return depth
        d, e = child_d, child_e
        depth += 1


def productive_residue_levels(
    modulus: int, target_residue: int, max_depth: int
) -> tuple[list[set[tuple[int, int]]], bool]:
    states = [(d, e) for d in range(modulus) for e in range(modulus)]
    productive = {
        (d, e)
        for d, e in states
        if (d*d - e*e) % modulus == target_residue
    }
    levels = [set(productive)]
    stable = False

    for _depth in range(1, max_depth + 1):
        previous = levels[-1]
        current = set(previous)
        for d, e in states:
            if (d, e) in current:
                continue
            for _move, child_d, child_e in qa_children(d, e):
                if (child_d % modulus, child_e % modulus) in previous:
                    current.add((d, e))
                    break
        levels.append(current)
        if current == previous:
            stable = True
            break

    return levels, stable


def residue_gate_allows(
    d: int,
    e: int,
    target_F: int,
    modulus: int,
    levels: list[set[tuple[int, int]]],
    stable: bool,
) -> bool:
    remaining = max_remaining_a_depth(d, e, target_F)
    if remaining < len(levels):
        return (d % modulus, e % modulus) in levels[remaining]
    if stable:
        return (d % modulus, e % modulus) in levels[-1]
    return True


def derive_from_f_tree(
    F: int,
    use_heap: bool = False,
    residue_modulus: int | None = None,
    residue_depth_cap: int = 512,
) -> dict[str, Any]:
    if F <= 0:
        raise ValueError("F must be a positive integer")

    candidates = []
    visited = set()
    expanded = 0
    pruned = 0
    residue_pruned = 0
    method = "qa_tree"

    levels = []
    stable = False
    if residue_modulus is not None:
        method = "qa_tree_accelerated" if use_heap else "qa_tree_residue"
        max_depth = min(max_remaining_a_depth(2, 1, F), residue_depth_cap)
        levels, stable = productive_residue_levels(
            residue_modulus, F % residue_modulus, max_depth
        )
    elif use_heap:
        method = "qa_tree_best_first"

    if use_heap:
        frontier = [(3, 2, 1, "")]
    else:
        frontier = [(2, 1, "")]

    while frontier:
        if use_heap:
            current_F, d, e, path = heapq.heappop(frontier)
            if current_F > F:
                break
        else:
            d, e, path = frontier.pop()
            current_F = qa_f(d, e)
        if (d, e) in visited:
            continue
        visited.add((d, e))
        expanded += 1

        if current_F == F:
            candidates.append(candidate_row(F, d, e, path or "ROOT", method))

        for move, child_d, child_e in qa_children(d, e):
            child_F = qa_f(child_d, child_e)
            if child_F <= F:
                if residue_modulus is not None and not residue_gate_allows(
                    child_d,
                    child_e,
                    F,
                    residue_modulus,
                    levels,
                    stable,
                ):
                    residue_pruned += 1
                    continue
                if use_heap:
                    heapq.heappush(frontier, (child_F, child_d, child_e, f"{path}{move}"))
                else:
                    frontier.append((child_d, child_e, f"{path}{move}"))
            else:
                pruned += 1

    candidates.sort(key=lambda item: (item["G"], item["C"], item["qa"]["path"]))
    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": method,
        "qa_engine": {
            "generator_set": {
                "M_A": "(d,e)->(2*d-e,d)",
                "M_B": "(d,e)->(2*d+e,d)",
                "M_C": "(d,e)->(d+2*e,e)",
            },
            "monotone_prune": "discard child when child_F > target F",
            "root": [2, 1],
            "expanded_nodes": expanded,
            "pruned_children": pruned,
            "residue_gate": None
            if residue_modulus is None
            else {
                "modulus": residue_modulus,
                "target_residue": F % residue_modulus,
                "levels_computed": len(levels),
                "stable": stable,
                "residue_pruned_children": residue_pruned,
            },
        },
    }


def derive_from_f_factor(F: int, primitive_only: bool = False) -> dict[str, Any]:
    if F <= 0:
        raise ValueError("F must be a positive integer")

    quick_reject_reason = None
    if primitive_only and F % 2 == 0:
        quick_reject_reason = "primitive canonical F must be odd"
    elif F % 4 == 2:
        quick_reject_reason = "F is 2 mod 4, so it is not a difference of squares"

    candidates = []
    rejected = []

    for b in divisors_up_to_sqrt(F):
        a = F // b
        if b > a:
            continue
        if a == b:
            rejected.append(
                {
                    "a": a,
                    "b": b,
                    "reason": "degenerate_e_zero",
                }
            )
            continue
        if (a - b) % 2 != 0:
            rejected.append(
                {
                    "a": a,
                    "b": b,
                    "reason": "non_integer_direction",
                }
            )
            continue

        d = (a + b) // 2
        e = (a - b) // 2
        row = candidate_row(F, d, e, "factor_pair", "factor")

        if primitive_only and not row["checks"]["is_primitive"]:
            rejected.append(
                {
                    "a": a,
                    "b": b,
                    "reason": "not_primitive",
                }
            )
            continue

        candidates.append(row)

    candidates.sort(key=lambda item: (item["G"], item["C"], item["b"]))

    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": "factor",
        "filters": {
            "factorization": "F=b*a",
            "integer_direction": "a and b must have the same parity",
            "quick_reject_reason": quick_reject_reason,
            "primitive_condition": "gcd(d,e)=1 and d-e is odd",
            "C_4n_check": "C % 4 == 0",
            "area_6n_check": "(C*F/2) % 6 == 0",
        },
        "primitive_only": primitive_only,
        "rejected": rejected,
    }


def geometric_d_bounds(F: int) -> tuple[int, int, int, str | None]:
    if F % 4 == 2:
        return 0, -1, 1, "F is 2 mod 4, so it is not a difference of squares"
    start = math.isqrt(F)
    if start*start < F:
        start += 1
    if F % 2 == 1:
        return start, (F + 1) // 2, 1, None
    return start + (start % 2), F // 4 + 1, 2, None


def derive_from_f_geometric_sieve(
    F: int,
    primitive_only: bool = False,
    sieve_moduli: list[int] | None = None,
    use_wheel: bool = False,
) -> dict[str, Any]:
    if F <= 0:
        raise ValueError("F must be a positive integer")
    if sieve_moduli is None:
        sieve_moduli = [3, 5, 7, 11, 13, 16]
    method_name = "qa_geometric_wheel" if use_wheel else "qa_geometric_sieve"

    start, stop, step, quick_reject_reason = geometric_d_bounds(F)
    residue_sets = {modulus: square_residues(modulus) for modulus in sieve_moduli}
    candidates = []
    rejected = []
    scanned_d = 0
    residue_skipped = 0
    square_tests = 0
    wheel_modulus = None
    wheel_allowed_residues = None

    if use_wheel and quick_reject_reason is None:
        wheel_modulus = lcm_many(sieve_moduli)
        wheel_allowed_residues = []
        for residue in range(wheel_modulus):
            if step == 2 and residue % 2 != start % 2:
                continue
            if all(
                (residue*residue - F) % modulus in residues
                for modulus, residues in residue_sets.items()
            ):
                wheel_allowed_residues.append(residue)

    if quick_reject_reason is None:
        if wheel_allowed_residues is None:
            d_values = range(start, stop + 1, step)
        else:
            generated = []
            for residue in wheel_allowed_residues:
                if residue < start:
                    offset = ((start - residue + wheel_modulus - 1) // wheel_modulus)
                    d = residue + offset * wheel_modulus
                else:
                    d = residue
                while d <= stop:
                    generated.append(d)
                    d += wheel_modulus
            d_values = sorted(generated)

        for d in d_values:
            scanned_d += 1
            if wheel_allowed_residues is None:
                passes_residue = True
                failed_modulus = None
                for modulus, residues in residue_sets.items():
                    if (d*d - F) % modulus not in residues:
                        passes_residue = False
                        failed_modulus = modulus
                        break
                if not passes_residue:
                    residue_skipped += 1
                    rejected.append(
                        {
                            "d": d,
                            "reason": "non_square_residue",
                            "modulus": failed_modulus,
                        }
                    )
                    continue

            square_tests += 1
            ok, e = is_square(d*d - F)
            if not ok:
                rejected.append({"d": d, "reason": "non_square_integer"})
                continue
            if not (d > e > 0):
                rejected.append({"d": d, "e": e, "reason": "invalid_direction"})
                continue

            row = candidate_row(F, d, e, "d_square_sieve", method_name)
            if primitive_only and not row["checks"]["is_primitive"]:
                rejected.append(
                    {
                        "d": d,
                        "e": e,
                        "reason": "not_primitive",
                    }
                )
                continue
            candidates.append(row)

    candidates.sort(key=lambda item: (item["G"], item["C"], item["d"]))
    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": method_name,
        "qa_engine": {
            "identity": "F=d*d-e*e",
            "geometry": "scan d and require e*e=d*d-F",
            "d_range": [start, stop],
            "d_step": step,
            "sieve_moduli": sieve_moduli,
            "wheel_modulus": wheel_modulus,
            "wheel_allowed_residues": None
            if wheel_allowed_residues is None
            else len(wheel_allowed_residues),
            "scanned_d": scanned_d,
            "residue_skipped": residue_skipped,
            "square_tests": square_tests,
            "quick_reject_reason": quick_reject_reason,
        },
        "primitive_only": primitive_only,
        "rejected_count": len(rejected),
        "rejected": rejected[:200],
    }


# QA orbit pruning for Fermat sieve: which d%24 values can satisfy d²-e²≡F (mod 24)?
_VALID_D_RESIDUES: dict[int, frozenset[int]] = {
    f: frozenset(
        d % 24
        for d in range(24)
        for e in range(24)
        if (d * d - e * e) % 24 == f
    )
    for f in range(24)
}


def qa_fermat_factor(F: int, limit: int = 100000, use_pruning: bool = True) -> set[int]:
    """QA-Fermat: find factors of F = d²-e² by scanning d from ceil(sqrt(F)).

    Checks is_square(d²-F) for each candidate d. With QA orbit pruning, only
    d values in the valid residue class for F%24 are checked — 2-12× fewer
    candidates depending on F%24.
    """
    if F <= 1:
        return set()
    d = math.isqrt(F)
    if d * d < F:
        d += 1
    valid = _VALID_D_RESIDUES.get(F % 24, frozenset(range(24))) if use_pruning else frozenset(range(24))
    factors: set[int] = set()
    checked = 0
    while checked < limit:
        if d % 24 in valid or not use_pruning:
            v = d * d - F
            if v > 0:
                sq, e = is_square(v)
                if sq:
                    b, a = d - e, d + e
                    if 1 < b < F:
                        factors.add(b)
                        factors.add(a)
                        return factors
            checked += 1
        d += 1
    return factors


def qa_pollard_rho(F: int, limit: int = 50000, seed: int = 2, c: int = 1) -> set[int]:
    """Pollard's rho (Brent variant) — finds one factor of F up to ~10^9 in microseconds.

    Faster than MPQS for unbalanced F = p*q where p < 10^8 and p >> sqrt(F).
    QA context: F is a Pythagorean semiprime; this bridges trial division and MPQS.
    """
    if F <= 1:
        return set()
    if F % 2 == 0:
        return ({2, F // 2}) if F > 2 else set()

    y = seed % F or 1
    r = 1
    q = 1
    g = 1
    ys = y
    x = y
    steps = 0

    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % F
        k = 0
        while k < r and g == 1:
            ys = y
            m = min(128, r - k)
            for _ in range(m):
                y = (y * y + c) % F
                q = q * abs(x - y) % F
            g = math.gcd(q, F)
            k += m
            steps += m
            if steps >= limit:
                return set()
        r *= 2

    if g == F:
        # q collapsed to 0 (gcd shortcut hit F); backtrack one step at a time
        g = 1
        while g == 1:
            ys = (ys * ys + c) % F
            g = math.gcd(abs(x - ys), F)
            steps += 1
            if steps >= limit:
                return set()

    if 1 < g < F:
        return {g, F // g}
    return set()


def qa_pollard_factor(F: int, limit: int = 200000) -> set[int]:
    """Try multiple (seed, c) pairs; seeds drawn from QA orbit residues for F%24."""
    valid_residues = sorted(_VALID_D_RESIDUES.get(F % 24, frozenset(range(24))))
    seeds = [r for r in valid_residues if r > 1][:4] or [2, 3, 5, 7]
    per_attempt = max(1000, limit // (len(seeds) * 3))
    for c in (1, 2, 5):
        for seed in seeds:
            result = qa_pollard_rho(F, limit=per_attempt, seed=seed, c=c)
            if result:
                return result
    return set()


def _mod_inverse(a: int, n: int) -> tuple[int, int]:
    """Extended-Euclid inverse of a mod n. Returns (gcd(a,n), inverse); inverse
    is 0 when gcd>1 — the caller reads that gcd as an ECM factor discovery."""
    a %= n
    if a == 0:
        return n, 0
    old_r, r = a, n
    old_s, s = 1, 0
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    if old_r != 1:
        return old_r, 0
    return 1, old_s % n


def _ec_add(
    p1: tuple[int, int] | None,
    p2: tuple[int, int] | None,
    curve_a: int,
    n: int,
) -> tuple[tuple[int, int] | None, int]:
    """Point addition mod n on y*y = x*x*x + curve_a*x + curve_b.

    Returns (sum_point, factor). A nonzero factor means a modular inverse
    failed mid-addition — for prime n that only happens at the point at
    infinity (harmless), but for composite n it exposes a nontrivial divisor,
    which is the entire ECM discovery mechanism (Lenstra 1987).
    """
    if p1 is None:
        return p2, 0
    if p2 is None:
        return p1, 0
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % n == 0:
            return None, 0
        num = (3 * x1 * x1 + curve_a) % n
        den = (2 * y1) % n
    else:
        num = (y2 - y1) % n
        den = (x2 - x1) % n
    gcd_den, inv = _mod_inverse(den, n)
    if gcd_den != 1:
        return None, gcd_den
    lam = (num * inv) % n
    x3 = (lam * lam - x1 - x2) % n
    y3 = (lam * (x1 - x3) - y1) % n
    return (x3, y3), 0


def _ec_multiply(
    point: tuple[int, int],
    k: int,
    curve_a: int,
    n: int,
) -> tuple[tuple[int, int] | None, int]:
    """Scalar-multiply point by k mod n via double-and-add. Returns (result,
    factor); a nonzero factor short-circuits with the discovered divisor."""
    result: tuple[int, int] | None = None
    addend = point
    while k > 0:
        if k & 1:
            result, factor = _ec_add(result, addend, curve_a, n)
            if factor:
                return None, factor
        addend, factor = _ec_add(addend, addend, curve_a, n)
        if factor:
            return None, factor
        k >>= 1
    return result, 0


def qa_ecm_factor(
    F: int,
    b1: int = 2000,
    max_curves: int = 25,
    seed: int = 1,
) -> set[int]:
    """Lenstra's ECM, stage 1 only — bridges Pollard's rho and MPQS.

    Pollard's rho cost depends on F's smallest factor being small enough to
    hit via a birthday-style collision (practical up to ~10^9-10^10). ECM
    cost instead depends on that smallest factor p having a B1-smooth curve
    order, so a 14-15 digit p can still be found in milliseconds if it is
    B1-smooth — the gap the auto pipeline previously had no bridge for.
    """
    if F <= 1:
        return set()
    if F % 2 == 0:
        return ({2, F // 2}) if F > 2 else set()

    rng = random.Random(seed)
    stage1_k = 1
    for prime in primes_up_to(b1):
        power = prime
        while power * prime <= b1:
            power *= prime
        stage1_k *= power

    for _ in range(max_curves):
        x0 = rng.randrange(2, F)
        y0 = rng.randrange(2, F)
        curve_a = rng.randrange(1, F)
        curve_b = (y0 * y0 - x0 * x0 * x0 - curve_a * x0) % F

        discriminant = (4 * curve_a * curve_a * curve_a + 27 * curve_b * curve_b) % F
        disc_gcd = math.gcd(discriminant, F)
        if 1 < disc_gcd < F:
            return {disc_gcd, F // disc_gcd}
        if disc_gcd == F:
            continue

        _point, factor = _ec_multiply((x0, y0), stage1_k, curve_a, F)
        if 1 < factor < F:
            return {factor, F // factor}

    return set()


def factor_over_base(value: int, factor_base: list[int]) -> tuple[bool, list[int], int]:
    remaining = value
    exponents = []
    for prime in factor_base:
        count = 0
        while remaining % prime == 0:
            remaining //= prime
            count += 1
        exponents.append(count)
    return remaining == 1, exponents, remaining


def parity_mask(exponents: list[int]) -> int:
    mask = 0
    for index, exponent in enumerate(exponents):
        if exponent % 2 == 1:
            mask |= 1 << index
    return mask


def null_dependencies(vectors: list[int]) -> list[int]:
    basis: dict[int, tuple[int, int]] = {}
    dependencies = []
    for row_index, vector in enumerate(vectors):
        combo = 1 << row_index
        reduced = vector
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (reduced, combo)
                break
            basis_vector, basis_combo = basis[pivot]
            reduced ^= basis_vector
            combo ^= basis_combo
        if reduced == 0:
            dependencies.append(combo)
    return dependencies


def singleton_filter_vectors(vectors: list[int]) -> dict[str, Any]:
    active = set(range(len(vectors)))
    removed_rows = 0
    passes = 0

    while True:
        column_counts: dict[int, int] = {}
        for row_index in active:
            vector = vectors[row_index]
            while vector:
                low_bit = vector & -vector
                column = low_bit.bit_length() - 1
                column_counts[column] = column_counts.get(column, 0) + 1
                vector ^= low_bit

        singleton_columns = {
            column for column, count in column_counts.items() if count == 1
        }
        if not singleton_columns:
            break

        remove_now = set()
        for row_index in active:
            vector = vectors[row_index]
            while vector:
                low_bit = vector & -vector
                column = low_bit.bit_length() - 1
                if column in singleton_columns:
                    remove_now.add(row_index)
                    break
                vector ^= low_bit

        if not remove_now:
            break
        active.difference_update(remove_now)
        removed_rows += len(remove_now)
        passes += 1

    kept_indices = sorted(active)
    kept_vectors = [vectors[index] for index in kept_indices]
    active_columns = set()
    for vector in kept_vectors:
        while vector:
            low_bit = vector & -vector
            active_columns.add(low_bit.bit_length() - 1)
            vector ^= low_bit

    return {
        "kept_indices": kept_indices,
        "kept_vectors": kept_vectors,
        "stats": {
            "input_rows": len(vectors),
            "kept_rows": len(kept_vectors),
            "removed_rows": removed_rows,
            "passes": passes,
            "active_columns": len(active_columns),
        },
    }


def weight_two_merge(vecs: list[int]) -> dict[str, Any]:
    """SGE second pass: absorb weight-2 rows.
    Returns kept_vectors, combo_matrix (each entry is bitmask over the n input rows), extra_deps, stats.
    """
    n = len(vecs)
    current = list(vecs)
    combos = [1 << i for i in range(n)]
    active = set(range(n))
    extra_deps: list[int] = []
    merges = 0

    changed = True
    while changed:
        changed = False
        for i in sorted(active):
            v = current[i]
            if v.bit_count() != 2:
                continue
            tmp, bit, cols = v, 0, []
            while tmp:
                if tmp & 1:
                    cols.append(bit)
                tmp >>= 1
                bit += 1
            elim_mask = 1 << cols[1]
            for j in list(active):
                if j == i:
                    continue
                if current[j] & elim_mask:
                    current[j] ^= v
                    combos[j] ^= combos[i]
                    if current[j] == 0:
                        extra_deps.append(combos[j])
                        active.discard(j)
            active.discard(i)
            merges += 1
            changed = True
            break  # restart scan for new weight-2 rows

    survivors = sorted(active)
    return {
        "kept_vectors": [current[i] for i in survivors],
        "combo_matrix": [combos[i] for i in survivors],
        "extra_deps": extra_deps,
        "stats": {
            "input_rows": n,
            "kept_rows": len(survivors),
            "merges": merges,
            "extra_deps_found": len(extra_deps),
        },
    }


def _build_mat_cols(rows: list[int], n_cols: int) -> list[int]:
    """Build column vectors: mat_cols[j] = n-bit int where bit i = rows[i] has bit j set."""
    mat_cols: list[int] = [0] * n_cols
    for i, row in enumerate(rows):
        tmp, j = row, 0
        while tmp:
            if tmp & 1:
                mat_cols[j] |= 1 << i
            tmp >>= 1
            j += 1
    return mat_cols


def _apply_A_vec(rows: list[int], mat_cols: list[int], v: int) -> int:
    """Apply A = rows * rows^T to vector v (n-bit int). Uses precomputed column vectors."""
    # Pass 1: bt_j = (mat_cols[j] & v).bit_count() % 2  →  m-bit int
    bt = 0
    for j, col in enumerate(mat_cols):
        if (col & v).bit_count() & 1:
            bt |= 1 << j
    # Pass 2: result_i = (rows[i] & bt).bit_count() % 2  →  n-bit int
    result = 0
    for i, row in enumerate(rows):
        if (row & bt).bit_count() & 1:
            result |= 1 << i
    return result


def berlekamp_massey_gf2(seq: list[int]) -> list[int]:
    """Berlekamp-Massey over GF(2). Returns connection polynomial [1, c_1, ..., c_L]."""
    C = [1]
    B = [1]
    L, m = 0, 1
    for k, s_k in enumerate(seq):
        d = s_k
        for i in range(1, min(L + 1, len(C))):
            d ^= C[i] & seq[k - i]
        if d == 0:
            m += 1
        elif 2 * L <= k:
            T = list(C)
            while len(C) < len(B) + m:
                C.append(0)
            for i, b in enumerate(B):
                C[i + m] ^= b
            L, B, m = k + 1 - L, T, 1
        else:
            while len(C) < len(B) + m:
                C.append(0)
            for i, b in enumerate(B):
                C[i + m] ^= b
            m += 1
    return C


def block_lanczos_gf2(
    rows: list[int],
    seed: int = 0,
    n_probes: int | None = None,
) -> dict[str, Any]:
    """Wiedemann GF(2) null-space finder (Berlekamp-Massey on scalar Krylov projections).

    For each (u, v) probe: builds scalar sequence s_k = u^T (mat*mat^T)^k v,
    applies BM, checks if the reversed polynomial has a zero constant term
    (C[L] == 0), then reconstructs null_vec = g(A)*v via a second Krylov pass.

    Precomputes column vectors of mat for O(n_cols * n / 64) Pass-1 cost per
    application instead of O(n) big-integer bit-extractions.

    n_probes scales with matrix size: 64 for small (n<=200), 32 for medium, 8
    for large — Gauss is the primary solver; Wiedemann adds random combinations
    from a different part of the null space.
    """
    n = len(rows)
    n_cols = max((r.bit_length() for r in rows), default=0)
    if n_cols == 0 or n < 2:
        return {"null_vectors": [], "stats": {"n_rows": n, "n_cols": n_cols,
                "probes_run": 0, "probes_successful": 0, "null_vectors_found": 0,
                "method": "wiedemann_berlekamp_massey"}}

    # Precompute column vectors for fast Pass-1
    mat_cols = _build_mat_cols(rows, n_cols)

    if n_probes is None:
        n_probes = 64 if n <= 200 else (32 if n <= 500 else 8)

    rng = random.Random(seed)
    found: set[int] = set()
    probes_run = 0
    probes_successful = 0
    seq_len = 2 * n + 4

    for _ in range(n_probes):
        probes_run += 1
        v = rng.getrandbits(n) | 1
        u = rng.getrandbits(n) | 1

        # Pass 1: build scalar Krylov sequence and run BM
        current = v
        scalars: list[int] = []
        for _k in range(seq_len):
            scalars.append((u & current).bit_count() & 1)
            current = _apply_A_vec(rows, mat_cols, current)

        C = berlekamp_massey_gf2(scalars)
        L = len(C) - 1

        # Reversed polynomial p(x) = x^L * C(1/x); p_0 = C[L].
        # p_0 == 0  ↔  x | p(x)  ↔  null vector exists: g(A)*v where g = p/x.
        if L == 0 or C[L] != 0:
            continue

        # Pass 2: accumulate null_vec = sum_{i=0}^{L-1} C[L-1-i] * A^i * v
        null_vec = 0
        current = v
        for i in range(L):
            if C[L - 1 - i]:
                null_vec ^= current
            current = _apply_A_vec(rows, mat_cols, current)

        if null_vec and null_vec not in found:
            found.add(null_vec)
            probes_successful += 1

    return {
        "null_vectors": list(found),
        "stats": {
            "n_rows": n,
            "n_cols": n_cols,
            "probes_run": probes_run,
            "probes_successful": probes_successful,
            "null_vectors_found": len(found),
            "method": "wiedemann_berlekamp_massey",
        },
    }


def _expand_combo(nv: int, combo_matrix: list[int]) -> int:
    """Expand a null vector over post-merge rows to a bitmask over pre-merge rows."""
    result, tmp, i = 0, nv, 0
    while tmp:
        if tmp & 1:
            result ^= combo_matrix[i]
        tmp >>= 1
        i += 1
    return result


def prime_factors_trial(n: int) -> list[int]:
    """Return all prime factors of n via trial division (small n only)."""
    primes = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.append(d)
            n //= d
        d += 1
    if n > 1:
        primes.append(n)
    return primes


def _is_probable_prime(n: int, rounds: int = 20) -> bool:
    """Miller-Rabin — used to size up a cofactor before deciding how to
    finish factoring it, so callers never fall into unbounded trial division."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    rng = random.Random(n)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def factor_large_component(n: int, trial_limit: int = 10**6) -> list[int]:
    """Fully factor n without ever falling into unbounded trial division.

    Pollard's rho and ECM can each surface a cofactor F/p up to F itself —
    if that cofactor is a large prime, plain prime_factors_trial() would
    trial-divide up to its square root and hang. Here small factors are
    stripped up to trial_limit, then Miller-Rabin decides whether the
    residual is prime; if not, one more round of rho/ECM is tried before
    giving up and reporting the residual as a single unresolved component.
    """
    primes: list[int] = []
    remaining = n
    d = 2
    while d <= trial_limit and d * d <= remaining:
        while remaining % d == 0:
            primes.append(d)
            remaining //= d
        d += 1
    if remaining == 1:
        return primes
    if remaining <= trial_limit * trial_limit:
        primes.extend(prime_factors_trial(remaining))
        return primes
    if _is_probable_prime(remaining):
        primes.append(remaining)
        return primes
    for split in (qa_pollard_factor(remaining), qa_ecm_factor(remaining)):
        for factor in split:
            if 1 < factor < remaining:
                primes.extend(factor_large_component(factor, trial_limit))
                primes.extend(factor_large_component(remaining // factor, trial_limit))
                return primes
    primes.append(remaining)
    return primes


def divisors_from_known_factors(factors: list[int]) -> list[int]:
    divisors = {1}
    for factor in factors:
        divisors.update({item * factor for item in list(divisors)})
    return sorted(divisors)


def triples_from_factorization(
    F: int, factors: list[int], primitive_only: bool, method: str
) -> list[dict[str, Any]]:
    candidates = []
    for b in divisors_from_known_factors(factors):
        if F % b != 0:
            continue
        a = F // b
        if b > a or a == b or (a - b) % 2 != 0:
            continue
        d = (a + b) // 2
        e = (a - b) // 2
        row = candidate_row(F, d, e, "qs_factor_recovery", method)
        if primitive_only and not row["checks"]["is_primitive"]:
            continue
        candidates.append(row)
    candidates.sort(key=lambda item: (item["G"], item["C"], item["d"]))
    return candidates


def parse_external_factor_output(text: str, F: int) -> list[int]:
    factors = set()
    patterns = [
        r"(?i)\b(?:p|prp|c)\d+\s*[=:]\s*(\d+)\b",
        r"(?i)\bfactor\s*[=:]\s*(\d+)\b",
        r"(?i)\bprime factor\s*[=:]\s*(\d+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            factor = int(match.group(1))
            if 1 < factor < F and F % factor == 0:
                factors.add(factor)
                factors.add(F // factor)
    return sorted(factors)


def command_for_external_engine(engine: str, executable: str, F: int) -> list[str]:
    if engine == "msieve":
        return [executable, "-q", str(F)]
    if engine == "yafu":
        return [executable, f"factor({F})"]
    if engine == "cado-nfs":
        return [executable, str(F)]
    raise ValueError(f"unknown external engine: {engine}")


def resolve_external_engine(engine: str) -> tuple[str | None, str | None]:
    candidates = {
        "msieve": ["msieve"],
        "yafu": ["yafu"],
        "cado-nfs": ["cado-nfs.py", "cado-nfs"],
    }
    if engine != "auto":
        for name in candidates[engine]:
            path = shutil.which(name)
            if path:
                return engine, path
        return engine, None
    for candidate_engine in ("msieve", "yafu", "cado-nfs"):
        resolved_engine, path = resolve_external_engine(candidate_engine)
        if path:
            return resolved_engine, path
    return None, None


def derive_from_f_external_backend(
    F: int,
    primitive_only: bool = False,
    engine: str = "auto",
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    if F <= 1:
        raise ValueError("F must be an integer greater than 1")
    if engine not in ("auto", "msieve", "yafu", "cado-nfs"):
        raise ValueError("external engine must be auto, msieve, yafu, or cado-nfs")

    resolved_engine, executable = resolve_external_engine(engine)
    if executable is None or resolved_engine is None:
        return {
            "F": F,
            "candidate_count": 0,
            "candidates": [],
            "method": "external_factor_backend",
            "ok": False,
            "error": "no external SIQS/MPQS/NFS backend found",
            "requested_engine": engine,
            "searched_engines": ["msieve", "yafu", "cado-nfs.py", "cado-nfs"],
            "primitive_only": primitive_only,
        }

    command = command_for_external_engine(resolved_engine, executable, F)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    output = completed.stdout + "\n" + completed.stderr
    factors = parse_external_factor_output(output, F)
    candidates = triples_from_factorization(
        F, factors, primitive_only, f"external_{resolved_engine}"
    )
    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": "external_factor_backend",
        "ok": completed.returncode == 0 and bool(factors),
        "qa_engine": {
            "engine": resolved_engine,
            "executable": executable,
            "command": command,
            "returncode": completed.returncode,
            "factors_found": factors,
            "timeout_seconds": timeout_seconds,
            "output_excerpt": output[-4000:],
        },
        "primitive_only": primitive_only,
    }


def derive_from_f_production(
    F: int,
    primitive_only: bool = False,
    engine: str = "auto",
    timeout_seconds: int = 300,
    qs_factor_base_bound: int = 100,
    qs_interval: int = 5000,
    qs_log_tolerance: float = 1.5,
) -> dict[str, Any]:
    external = derive_from_f_external_backend(
        F,
        primitive_only=primitive_only,
        engine=engine,
        timeout_seconds=timeout_seconds,
    )
    if external.get("ok"):
        external["method"] = "production_factor_backend"
        external["qa_engine"]["strategy"] = "external_current_art"
        external["qa_engine"]["fallback_used"] = False
        return external

    fallback = derive_from_f_qa_qs(
        F,
        primitive_only=primitive_only,
        factor_base_bound=qs_factor_base_bound,
        interval=qs_interval,
        log_tolerance=qs_log_tolerance,
    )
    fallback["method"] = "production_factor_backend"
    fallback["qa_engine"]["strategy"] = "internal_qa_qs_fallback"
    fallback["qa_engine"]["fallback_used"] = True
    fallback["qa_engine"]["external_error"] = external.get("error")
    return fallback


def derive_from_f_qa_qs(
    F: int,
    primitive_only: bool = False,
    factor_base_bound: int = 50,
    interval: int = 2000,
    max_dependencies: int = 16,
    log_tolerance: float = 1.5,
) -> dict[str, Any]:
    if F <= 1:
        raise ValueError("F must be an integer greater than 1")

    # QA-Fermat fast path: d²-F = e² at d ≈ sqrt(F) solves Pythagorean problems
    # in O(e / pruning_factor) steps; works instantly when factors are balanced.
    fermat_factors = qa_fermat_factor(F, limit=interval)
    if fermat_factors:
        # Fully factor each discovered component so triples_from_factorization
        # gets all prime factors, not just the top-level divisor pair.
        all_primes: list[int] = []
        for p in fermat_factors:
            if 1 < p < F:
                all_primes.extend(prime_factors_trial(p))
        factors_sorted = sorted(set(all_primes))
        candidates = triples_from_factorization(
            F, factors_sorted, primitive_only, "qa_quadratic_sieve"
        )
        return {
            "F": F,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "method": "qa_quadratic_sieve",
            "qa_engine": {
                "polynomial": "Q(d)=d*d-F",
                "factor_base_bound": factor_base_bound,
                "factor_base": [],
                "direct_factors": factors_sorted,
                "interval": interval,
                "start_d": math.isqrt(F) + (0 if (math.isqrt(F) ** 2) >= F else 1),
                "log_tolerance": log_tolerance,
                "log_candidates": 0,
                "skipped_by_log": 0,
                "tested_after_sieve": 0,
                "smooth_relations": 0,
                "dependencies_checked": 0,
                "factors_found": factors_sorted,
                "fermat_fast_path": True,
                "relation_sets": [],
            },
            "primitive_only": primitive_only,
        }

    factor_base = []
    roots_by_prime = {}
    direct_factors = set()
    for prime in primes_up_to(factor_base_bound):
        if F % prime == 0:
            direct_factors.add(prime)
            direct_factors.add(F // prime)
            continue
        roots = modular_roots_square(F, prime)
        if roots:
            factor_base.append(prime)
            roots_by_prime[prime] = roots

    start = math.isqrt(F)
    if start*start < F:
        start += 1

    sieve_scores = [0.0] * interval
    sieve_hits = [0] * interval
    for prime in factor_base:
        roots = sorted(set(roots_by_prime[prime]))
        prime_log = math.log(prime)
        for root in roots:
            offset = (root - start) % prime
            for index in range(offset, interval, prime):
                sieve_scores[index] += prime_log
                sieve_hits[index] += 1

    relations = []
    parity_vectors = []
    tested_after_sieve = 0
    skipped_by_log = 0
    needed_relations = len(factor_base) + max_dependencies
    candidate_indices = []
    for index, score in enumerate(sieve_scores):
        d = start + index
        q_value = d*d - F
        if q_value <= 0:
            continue
        # Perfect squares (e.g. q=1 when d=ceil(sqrt(F))) have sieve_hits=0
        # but are trivially smooth — handle them before the log gate.
        sq, e_val = is_square(q_value)
        if sq and e_val > 0:
            b, a = d - e_val, d + e_val
            if 1 < b < F:
                direct_factors.add(b)
                direct_factors.add(a)
            continue
        if sieve_hits[index] == 0:
            skipped_by_log += 1
            continue
        residual_log = math.log(q_value) - score
        if residual_log > log_tolerance:
            skipped_by_log += 1
            continue
        candidate_indices.append((residual_log, index, d, q_value))

    candidate_indices.sort(key=lambda item: (item[0], item[2]))
    for _residual_log, _index, d, q_value in candidate_indices:
        tested_after_sieve += 1
        direct = math.gcd(d, F)
        if 1 < direct < F:
            direct_factors.add(direct)
            direct_factors.add(F // direct)
            continue
        is_smooth, exponents, remaining = factor_over_base(q_value, factor_base)
        if not is_smooth:
            continue
        relations.append(
            {
                "d": d,
                "q_value": q_value,
                "exponents": exponents,
            }
        )
        parity_vectors.append(parity_mask(exponents))
        if len(relations) >= needed_relations:
            break

    factors_found = set(direct_factors)
    dependencies_checked = 0
    used_relation_sets = []
    dependency_basis = null_dependencies(parity_vectors)
    dependency_combos = []
    for combo in dependency_basis:
        dependency_combos.append(combo)
    for left_index in range(len(dependency_basis)):
        for right_index in range(left_index + 1, len(dependency_basis)):
            dependency_combos.append(dependency_basis[left_index] ^ dependency_basis[right_index])
            if len(dependency_combos) >= max_dependencies:
                break
        if len(dependency_combos) >= max_dependencies:
            break

    seen_combos = set()
    for combo in dependency_combos:
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        if dependencies_checked >= max_dependencies:
            break
        dependencies_checked += 1
        relation_indices = [
            index for index in range(len(relations)) if combo & (1 << index)
        ]
        if not relation_indices:
            continue

        x = 1
        exponent_sums = [0] * len(factor_base)
        for index in relation_indices:
            relation = relations[index]
            x = (x * relation["d"]) % F
            for p_index, exponent in enumerate(relation["exponents"]):
                exponent_sums[p_index] += exponent

        y = 1
        for prime, exponent in zip(factor_base, exponent_sums):
            y = (y * pow(prime, exponent // 2, F)) % F

        g1 = math.gcd((x - y) % F, F)
        g2 = math.gcd((x + y) % F, F)
        used_relation_sets.append(
            {
                "relation_indices": relation_indices,
                "x_mod_F": x,
                "y_mod_F": y,
                "gcd_minus": g1,
                "gcd_plus": g2,
            }
        )
        for factor in (g1, g2):
            if 1 < factor < F:
                factors_found.add(factor)
                factors_found.add(F // factor)

    factors_sorted = sorted(factors_found)
    candidates = triples_from_factorization(
        F, factors_sorted, primitive_only, "qa_quadratic_sieve"
    )

    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": "qa_quadratic_sieve",
        "qa_engine": {
            "polynomial": "Q(d)=d*d-F",
            "factor_base_bound": factor_base_bound,
            "factor_base": factor_base,
            "direct_factors": sorted(direct_factors),
            "interval": interval,
            "start_d": start,
            "log_tolerance": log_tolerance,
            "log_candidates": len(candidate_indices),
            "skipped_by_log": skipped_by_log,
            "tested_after_sieve": tested_after_sieve,
            "smooth_relations": len(relations),
            "dependencies_checked": dependencies_checked,
            "factors_found": factors_sorted,
            "relation_sets": used_relation_sets,
        },
        "primitive_only": primitive_only,
    }


def select_mpqs_a_values(
    factor_base: list[int],
    roots_by_prime: dict[int, list[int]],
    F: int,
    limit: int,
    half_width: int = 128,
) -> list[tuple[int, list[int]]]:
    odd_primes = [
        prime for prime in factor_base if prime != 2 and prime in roots_by_prime
    ]
    # Silverman optimal: A ≈ sqrt(2F) / M so |q(x)| ≤ sqrt(F/2)*M over [-M, M]
    target = max(3, math.isqrt(2 * F // max(half_width, 1)))
    max_a = max(target * 8, 64)
    candidates: dict[int, list[int]] = {}

    def visit(start: int, depth: int, product: int, primes_used: list[int]) -> None:
        if primes_used:
            candidates[product] = list(primes_used)
        if depth == 3:
            return
        for index in range(start, len(odd_primes)):
            next_product = product * odd_primes[index]
            if next_product > max_a:
                continue
            visit(index + 1, depth + 1, next_product, primes_used + [odd_primes[index]])

    visit(0, 0, 1, [])
    ranked = sorted(candidates.items(), key=lambda item: (abs(item[0] - target), item[0]))
    return ranked[:limit]


def derive_from_f_qa_mpqs(
    F: int,
    primitive_only: bool = False,
    factor_base_bound: int = 100,
    half_width: int = 128,
    polynomial_count: int = 12,
    max_dependencies: int = 32,
    log_tolerance: float = 1.5,
    large_prime_bound: int = 0,
) -> dict[str, Any]:
    if F <= 1:
        raise ValueError("F must be an integer greater than 1")

    factor_base = []
    roots_by_prime = {}
    direct_factors = set()
    direct_prime_factors = []
    remaining_after_direct = F
    for prime in primes_up_to(factor_base_bound):
        if F % prime == 0:
            direct_factors.add(prime)
            direct_factors.add(F // prime)
            while remaining_after_direct % prime == 0:
                direct_prime_factors.append(prime)
                remaining_after_direct //= prime
            continue
        roots = modular_roots_square(F, prime)
        if roots:
            factor_base.append(prime)
            roots_by_prime[prime] = roots

    if remaining_after_direct == 1 and direct_prime_factors:
        factors_sorted = sorted(direct_factors)
        candidates = triples_from_factorization(
            F, direct_prime_factors, primitive_only, "qa_mpqs"
        )
        return {
            "F": F,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "method": "qa_mpqs",
            "qa_engine": {
                "chart": "d=A*x+B",
                "condition": "B*B == F mod A",
                "relation": "(A*x+B)^2-F = A*q(x)",
                "factor_base_bound": factor_base_bound,
                "factor_base": factor_base,
                "direct_factors": factors_sorted,
                "direct_prime_factors": direct_prime_factors,
                "remaining_after_direct": remaining_after_direct,
                "small_factor_frontier_complete": True,
                "half_width": half_width,
                "polynomial_count": polynomial_count,
                "large_prime_bound": large_prime_bound,
                "large_prime_partials": 0,
                "large_prime_pairs": 0,
                "polynomials_used": 0,
                "polynomial_summaries": [],
                "log_tolerance": log_tolerance,
                "skipped_by_log": 0,
                "tested_after_sieve": 0,
                "smooth_relations": 0,
                "total_usable_relations": 0,
                "dependencies_checked": 0,
                "factors_found": factors_sorted,
                "relation_sets": [],
            },
            "primitive_only": primitive_only,
        }

    sqrt_f = math.isqrt(F)
    if sqrt_f*sqrt_f < F:
        sqrt_f += 1

    relations = []
    parity_vectors = []
    partial_relations: dict[int, dict[str, Any]] = {}
    polynomial_summaries = []
    tested_after_sieve = 0
    skipped_by_log = 0
    smooth_relations_total = 0
    large_prime_partials = 0
    large_prime_pairs = 0
    needed_relations = len(factor_base) + max_dependencies
    residual_log_limit = log_tolerance
    if large_prime_bound > 1:
        residual_log_limit = math.log(large_prime_bound) + log_tolerance

    for A, a_primes in select_mpqs_a_values(
        factor_base, roots_by_prime, F, polynomial_count, half_width=half_width
    ):
        root_sets = [(prime, roots_by_prime[prime]) for prime in a_primes]
        b_roots = combine_roots_crt(root_sets, limit=64)
        is_a_smooth, a_exponents, _a_remaining = factor_over_base(A, factor_base)
        if not is_a_smooth:
            continue

        for B in b_roots:
            constant = (B*B - F) // A
            scores = [0.0] * (2 * half_width + 1)
            hits = [0] * (2 * half_width + 1)

            for prime in factor_base:
                if A % prime == 0:
                    continue
                inv_a = pow(A % prime, -1, prime)
                prime_log = math.log(prime)
                x_roots = {
                    ((root - B) * inv_a) % prime
                    for root in roots_by_prime[prime]
                }
                for root in x_roots:
                    first = root
                    while first < -half_width:
                        first += prime
                    while first > -half_width:
                        first -= prime
                    for x_value in range(first, half_width + 1, prime):
                        if x_value < -half_width:
                            continue
                        index = x_value + half_width
                        scores[index] += prime_log
                        hits[index] += 1

            candidates = []
            for x_value in range(-half_width, half_width + 1):
                d = A * x_value + B
                if d <= sqrt_f:
                    continue
                numerator = d*d - F
                if numerator % A != 0:
                    continue
                q_value = numerator // A
                if q_value <= 0:
                    continue
                index = x_value + half_width
                if hits[index] == 0:
                    skipped_by_log += 1
                    continue
                residual_log = math.log(q_value) - scores[index]
                if residual_log > residual_log_limit:
                    skipped_by_log += 1
                    continue
                candidates.append((residual_log, x_value, d, q_value))

            candidates.sort(key=lambda item: (item[0], abs(item[1]), item[2]))
            smooth_here = 0
            partial_here = 0
            paired_here = 0
            for _residual_log, x_value, d, q_value in candidates:
                tested_after_sieve += 1
                direct = math.gcd(d, F)
                if 1 < direct < F:
                    direct_factors.add(direct)
                    direct_factors.add(F // direct)
                    continue
                is_smooth, q_exponents, remaining = factor_over_base(q_value, factor_base)
                exponents = [
                    q_exponents[index] + a_exponents[index]
                    for index in range(len(factor_base))
                ]
                relation = {
                    "A": A,
                    "B": B,
                    "x": x_value,
                    "d": d,
                    "q_value": q_value,
                    "exponents": exponents,
                    "large_prime": 1,
                    "source": "smooth",
                }
                if is_smooth:
                    relations.append(relation)
                    parity_vectors.append(parity_mask(exponents))
                    smooth_relations_total += 1
                    smooth_here += 1
                elif 1 < remaining <= large_prime_bound:
                    large_prime_partials += 1
                    partial_here += 1
                    previous = partial_relations.pop(remaining, None)
                    if previous is None:
                        partial_relations[remaining] = relation
                        continue
                    combined_exponents = [
                        previous["exponents"][index] + exponents[index]
                        for index in range(len(factor_base))
                    ]
                    combined = {
                        "A": [previous["A"], A],
                        "B": [previous["B"], B],
                        "x": [previous["x"], x_value],
                        "d": (previous["d"] * d) % F,
                        "q_value": [previous["q_value"], q_value],
                        "exponents": combined_exponents,
                        "large_prime": remaining,
                        "source": "large_prime_pair",
                    }
                    relations.append(combined)
                    parity_vectors.append(parity_mask(combined_exponents))
                    large_prime_pairs += 1
                    paired_here += 1
                else:
                    continue
                if len(relations) >= needed_relations:
                    break

            polynomial_summaries.append(
                {
                    "A": A,
                    "B": B,
                    "candidate_count": len(candidates),
                    "smooth_relations": smooth_here,
                    "large_prime_partials": partial_here,
                    "large_prime_pairs": paired_here,
                }
            )
            if len(relations) >= needed_relations:
                break
        if len(relations) >= needed_relations:
            break

    factors_found = set(direct_factors)
    dependencies_checked = 0
    relation_sets = []

    # SGE stage 1: singleton filtering
    filtered = singleton_filter_vectors(parity_vectors)
    kept_relation_indices = filtered["kept_indices"]
    filtered_vectors = filtered["kept_vectors"]
    singleton_stats = filtered["stats"]
    if len(filtered_vectors) < 2:
        kept_relation_indices = list(range(len(parity_vectors)))
        filtered_vectors = parity_vectors
        singleton_stats = dict(singleton_stats)
        singleton_stats["fallback_unfiltered"] = True
    else:
        singleton_stats["fallback_unfiltered"] = False

    # SGE stage 2: weight-2 merge
    w2 = weight_two_merge(filtered_vectors)
    merged_vectors = w2["kept_vectors"]
    combo_matrix = w2["combo_matrix"]
    w2_extra_deps = w2["extra_deps"]  # already bitmasks over kept_relation_indices rows
    w2_stats = w2["stats"]

    # Gaussian elimination on the merged matrix (primary solver — always run)
    dep_basis = null_dependencies(merged_vectors)
    gauss_basis_size = len(dep_basis)
    # Expand Gauss combos through combo_matrix to get bitmasks over filtered rows
    gauss_raw: list[int] = list(dep_basis)
    for li in range(len(dep_basis)):
        for ri in range(li + 1, len(dep_basis)):
            gauss_raw.append(dep_basis[li] ^ dep_basis[ri])
            if len(gauss_raw) >= max_dependencies * 4:
                break
        if len(gauss_raw) >= max_dependencies * 4:
            break
    gauss_combos = [_expand_combo(c, combo_matrix) for c in gauss_raw]

    # Wiedemann (supplementary solver) — always run; n_probes auto-scales with matrix size
    blkl = block_lanczos_gf2(merged_vectors, seed=0)
    blkl_combos = [_expand_combo(nv, combo_matrix) for nv in blkl["null_vectors"]]
    blkl_stats = blkl["stats"]
    la_strategy = "weight_two_merge_plus_wiedemann_plus_gauss"

    # Merge all dependency sources; Gauss first (most comprehensive), Wiedemann + extras supplement
    all_combos = gauss_combos + w2_extra_deps + blkl_combos

    seen = set()
    for combo in all_combos:
        if combo in seen:
            continue
        seen.add(combo)
        if dependencies_checked >= max_dependencies:
            break
        relation_indices = [
            kept_relation_indices[index]
            for index in range(len(filtered_vectors))
            if combo & (1 << index)
        ]
        if not relation_indices:
            continue
        dependencies_checked += 1

        x_mod = 1
        exponent_sums = [0] * len(factor_base)
        square_extra = 1
        for index in relation_indices:
            relation = relations[index]
            x_mod = (x_mod * relation["d"]) % F
            square_extra = (square_extra * relation.get("large_prime", 1)) % F
            for p_index, exponent in enumerate(relation["exponents"]):
                exponent_sums[p_index] += exponent

        y_mod = 1
        for prime, exponent in zip(factor_base, exponent_sums):
            y_mod = (y_mod * pow(prime, exponent // 2, F)) % F
        y_mod = (y_mod * square_extra) % F

        g1 = math.gcd((x_mod - y_mod) % F, F)
        g2 = math.gcd((x_mod + y_mod) % F, F)
        relation_sets.append(
            {
                "relation_indices": relation_indices,
                "x_mod_F": x_mod,
                "y_mod_F": y_mod,
                "gcd_minus": g1,
                "gcd_plus": g2,
            }
        )
        for factor in (g1, g2):
            if 1 < factor < F:
                factors_found.add(factor)
                factors_found.add(F // factor)

    factors_sorted = sorted(factors_found)
    candidates = triples_from_factorization(
        F, factors_sorted, primitive_only, "qa_mpqs"
    )
    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": "qa_mpqs",
        "qa_engine": {
            "chart": "d=A*x+B",
            "condition": "B*B == F mod A",
            "relation": "(A*x+B)^2-F = A*q(x)",
            "factor_base_bound": factor_base_bound,
            "factor_base": factor_base,
            "direct_factors": sorted(direct_factors),
            "half_width": half_width,
            "polynomial_count": polynomial_count,
            "large_prime_bound": large_prime_bound,
            "large_prime_partials": large_prime_partials,
            "large_prime_pairs": large_prime_pairs,
            "unpaired_large_primes": len(partial_relations),
            "polynomials_used": len(polynomial_summaries),
            "polynomial_summaries": polynomial_summaries[:64],
            "log_tolerance": log_tolerance,
            "skipped_by_log": skipped_by_log,
            "tested_after_sieve": tested_after_sieve,
            "smooth_relations": smooth_relations_total,
            "total_usable_relations": len(relations),
            "linear_algebra": {
                "strategy": la_strategy,
                "singleton_filter": singleton_stats,
                "weight_two_merge": w2_stats,
                "block_lanczos": blkl_stats,
                "gauss_basis_size": gauss_basis_size,
                "total_combos_tried": len(all_combos),
            },
            "dependencies_checked": dependencies_checked,
            "factors_found": factors_sorted,
            "relation_sets": relation_sets,
        },
        "primitive_only": primitive_only,
    }


def mpqs_parameter_profiles(F: int) -> list[dict[str, int]]:
    digits = len(str(F))
    if digits <= 6:
        start = [
            {
                "factor_base_bound": 100,
                "half_width": 128,
                "polynomial_count": 12,
                "large_prime_bound": 5000,
            },
        ]
    elif digits <= 13:
        start = [
            {
                "factor_base_bound": 1000,
                "half_width": 512,
                "polynomial_count": 32,
                "large_prime_bound": 50000,
            },
        ]
    elif digits <= 15:
        start = [
            {
                "factor_base_bound": 5000,
                "half_width": 1024,
                "polynomial_count": 64,
                "large_prime_bound": 250000,
            },
        ]
    elif digits <= 19:
        start = [
            {
                "factor_base_bound": 10000,
                "half_width": 2048,
                "polynomial_count": 96,
                "large_prime_bound": 500000,
            },
        ]
    else:
        start = [
            {
                "factor_base_bound": 20000,
                "half_width": 4096,
                "polynomial_count": 128,
                "large_prime_bound": 1000000,
            },
        ]

    escalations = [
        {
            "factor_base_bound": 1000,
            "half_width": 512,
            "polynomial_count": 32,
            "large_prime_bound": 50000,
        },
        {
            "factor_base_bound": 5000,
            "half_width": 1024,
            "polynomial_count": 64,
            "large_prime_bound": 250000,
        },
        {
            "factor_base_bound": 10000,
            "half_width": 2048,
            "polynomial_count": 96,
            "large_prime_bound": 500000,
        },
        {
            "factor_base_bound": 20000,
            "half_width": 4096,
            "polynomial_count": 128,
            "large_prime_bound": 1000000,
        },
        {
            "factor_base_bound": 40000,
            "half_width": 8192,
            "polynomial_count": 192,
            "large_prime_bound": 2000000,
        },
    ]
    profiles = []
    seen = set()
    for profile in start + escalations:
        key = (
            profile["factor_base_bound"],
            profile["half_width"],
            profile["polynomial_count"],
            profile["large_prime_bound"],
        )
        if key not in seen:
            seen.add(key)
            profiles.append(profile)
    return profiles


def qa_mpqs_success(result: dict[str, Any]) -> bool:
    engine = result.get("qa_engine", {})
    if engine.get("small_factor_frontier_complete"):
        return result.get("candidate_count", 0) > 0
    factors = engine.get("factors_found", [])
    return result.get("candidate_count", 0) >= 2 and len(factors) >= 2


def derive_from_f_qa_mpqs_auto(
    F: int,
    primitive_only: bool = False,
    log_tolerance: float = 1.5,
    ecm_b1: int = 20000,
    ecm_max_curves: int = 50,
) -> dict[str, Any]:
    # QA-Fermat fast path: for Pythagorean F = d²-e² with balanced factors
    # (d ≈ sqrt(F)), this finds the answer in O(e / orbit_pruning_factor) steps.
    # Limit 20000 keeps overhead < 4ms for unbalanced cases that fall through to Pollard.
    fermat_factors = qa_fermat_factor(F, limit=20000)
    if fermat_factors:
        all_primes: list[int] = []
        for p in fermat_factors:
            if 1 < p < F:
                all_primes.extend(factor_large_component(p))
        factors_sorted = sorted(set(all_primes))
        candidates = triples_from_factorization(F, factors_sorted, primitive_only, "qa_mpqs_auto")
        return {
            "F": F,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "method": "qa_mpqs_auto",
            "qa_engine": {
                "factors_found": factors_sorted,
                "fermat_fast_path": True,
                "pollard_rho_fast_path": False,
                "ecm_fast_path": False,
                "auto_attempts": [],
                "auto_parameters": None,
                "auto_success": True,
            },
            "primitive_only": primitive_only,
        }

    # Pollard's rho: fast for unbalanced F = p*q where p < ~10^9 and p >> sqrt(F).
    # Fills the gap between trial division (≤50) and MPQS (all sizes, 0.3-1s+).
    rho_factors = qa_pollard_factor(F, limit=200000)
    if rho_factors:
        all_primes: list[int] = []
        for p in rho_factors:
            if 1 < p < F:
                all_primes.extend(factor_large_component(p))
        factors_sorted = sorted(set(all_primes))
        candidates = triples_from_factorization(F, factors_sorted, primitive_only, "qa_mpqs_auto")
        return {
            "F": F,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "method": "qa_mpqs_auto",
            "qa_engine": {
                "factors_found": factors_sorted,
                "pollard_rho_fast_path": True,
                "fermat_fast_path": False,
                "ecm_fast_path": False,
                "auto_attempts": [],
                "auto_parameters": None,
                "auto_success": True,
            },
            "primitive_only": primitive_only,
        }

    # ECM (Lenstra, stage 1): bridges Pollard's rho and MPQS for factors past
    # ~10^9-10^10 that are still B1-smooth — the gap noted after the 2026-05-30
    # stress run (factors > 10^13 as individual primes had no bridge). Defaults
    # tuned empirically (2026-07-04): b1=20000/curves=50 measured 92%/80%
    # success against random 13/15-digit target factors, vs 25% for the
    # original b1=2000/curves=25 — bounded worst case (genuinely hard F,
    # no smooth factor) is ~6-19s before falling through to MPQS.
    ecm_factors = qa_ecm_factor(F, b1=ecm_b1, max_curves=ecm_max_curves)
    if ecm_factors:
        all_primes: list[int] = []
        for p in ecm_factors:
            if 1 < p < F:
                all_primes.extend(factor_large_component(p))
        factors_sorted = sorted(set(all_primes))
        candidates = triples_from_factorization(F, factors_sorted, primitive_only, "qa_mpqs_auto")
        return {
            "F": F,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "method": "qa_mpqs_auto",
            "qa_engine": {
                "factors_found": factors_sorted,
                "ecm_fast_path": True,
                "pollard_rho_fast_path": False,
                "fermat_fast_path": False,
                "auto_attempts": [],
                "auto_parameters": None,
                "auto_success": True,
            },
            "primitive_only": primitive_only,
        }

    attempts = []
    last_result = None
    for profile in mpqs_parameter_profiles(F):
        result = derive_from_f_qa_mpqs(
            F,
            primitive_only=primitive_only,
            factor_base_bound=profile["factor_base_bound"],
            half_width=profile["half_width"],
            polynomial_count=profile["polynomial_count"],
            large_prime_bound=profile["large_prime_bound"],
            log_tolerance=log_tolerance,
        )
        engine = result["qa_engine"]
        attempts.append(
            {
                "parameters": profile,
                "candidate_count": result["candidate_count"],
                "factors_found": engine.get("factors_found", []),
                "polynomials_used": engine.get("polynomials_used", 0),
                "tested_after_sieve": engine.get("tested_after_sieve", 0),
                "smooth_relations": engine.get("smooth_relations", 0),
                "total_usable_relations": engine.get("total_usable_relations", 0),
                "large_prime_bound": engine.get("large_prime_bound", 0),
                "large_prime_partials": engine.get("large_prime_partials", 0),
                "large_prime_pairs": engine.get("large_prime_pairs", 0),
                "linear_algebra": engine.get("linear_algebra", {}),
                "dependencies_checked": engine.get("dependencies_checked", 0),
                "small_factor_frontier_complete": engine.get(
                    "small_factor_frontier_complete", False
                ),
            }
        )
        last_result = result
        if qa_mpqs_success(result):
            result["method"] = "qa_mpqs_auto"
            result["qa_engine"]["auto_parameters"] = profile
            result["qa_engine"]["auto_attempts"] = attempts
            result["qa_engine"]["auto_success"] = True
            return result

    assert last_result is not None
    last_result["method"] = "qa_mpqs_auto"
    last_result["qa_engine"]["auto_parameters"] = attempts[-1]["parameters"]
    last_result["qa_engine"]["auto_attempts"] = attempts
    last_result["qa_engine"]["auto_success"] = False
    return last_result


def derive_from_f_qa_ecm(
    F: int,
    primitive_only: bool = False,
    b1: int = 20000,
    max_curves: int = 50,
) -> dict[str, Any]:
    ecm_factors = qa_ecm_factor(F, b1=b1, max_curves=max_curves)
    all_primes: list[int] = []
    for p in ecm_factors:
        if 1 < p < F:
            all_primes.extend(factor_large_component(p))
    factors_sorted = sorted(set(all_primes))
    candidates = triples_from_factorization(F, factors_sorted, primitive_only, "qa_ecm")
    return {
        "F": F,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "method": "qa_ecm",
        "qa_engine": {
            "b1": b1,
            "max_curves": max_curves,
            "factors_found": factors_sorted,
            "ecm_success": bool(ecm_factors),
        },
        "primitive_only": primitive_only,
    }


def derive_from_f(
    F: int,
    method: str,
    primitive_only: bool = False,
    residue_modulus: int = 24,
    sieve_moduli: list[int] | None = None,
    qs_factor_base_bound: int = 100,
    qs_interval: int = 5000,
    qs_log_tolerance: float = 1.5,
    external_engine: str = "auto",
    external_timeout: int = 300,
    mpqs_half_width: int = 128,
    mpqs_polynomial_count: int = 12,
    mpqs_large_prime_bound: int = 0,
    ecm_b1: int = 20000,
    ecm_max_curves: int = 50,
) -> dict[str, Any]:
    if method == "tree":
        return derive_from_f_tree(F)
    if method == "tree-best":
        return derive_from_f_tree(F, use_heap=True)
    if method == "tree-residue":
        return derive_from_f_tree(F, residue_modulus=residue_modulus)
    if method == "tree-accelerated":
        return derive_from_f_tree(F, use_heap=True, residue_modulus=residue_modulus)
    if method == "geometric-sieve":
        return derive_from_f_geometric_sieve(
            F, primitive_only=primitive_only, sieve_moduli=sieve_moduli
        )
    if method == "geometric-wheel":
        return derive_from_f_geometric_sieve(
            F,
            primitive_only=primitive_only,
            sieve_moduli=sieve_moduli,
            use_wheel=True,
        )
    if method == "qa-qs":
        return derive_from_f_qa_qs(
            F,
            primitive_only=primitive_only,
            factor_base_bound=qs_factor_base_bound,
            interval=qs_interval,
            log_tolerance=qs_log_tolerance,
        )
    if method == "qa-mpqs":
        return derive_from_f_qa_mpqs(
            F,
            primitive_only=primitive_only,
            factor_base_bound=qs_factor_base_bound,
            half_width=mpqs_half_width,
            polynomial_count=mpqs_polynomial_count,
            large_prime_bound=mpqs_large_prime_bound,
            log_tolerance=qs_log_tolerance,
        )
    if method == "qa-mpqs-auto":
        return derive_from_f_qa_mpqs_auto(
            F,
            primitive_only=primitive_only,
            log_tolerance=qs_log_tolerance,
            ecm_b1=ecm_b1,
            ecm_max_curves=ecm_max_curves,
        )
    if method == "qa-ecm":
        return derive_from_f_qa_ecm(
            F, primitive_only=primitive_only, b1=ecm_b1, max_curves=ecm_max_curves
        )
    if method == "external":
        return derive_from_f_external_backend(
            F,
            primitive_only=primitive_only,
            engine=external_engine,
            timeout_seconds=external_timeout,
        )
    if method == "production":
        return derive_from_f_production(
            F,
            primitive_only=primitive_only,
            engine=external_engine,
            timeout_seconds=external_timeout,
            qs_factor_base_bound=qs_factor_base_bound,
            qs_interval=qs_interval,
            qs_log_tolerance=qs_log_tolerance,
        )
    if method == "factor":
        return derive_from_f_factor(F, primitive_only=primitive_only)
    if method == "both":
        return {
            "F": F,
            "method": "both",
            "tree": derive_from_f_tree(F),
            "tree_best": derive_from_f_tree(F, use_heap=True),
            "tree_residue": derive_from_f_tree(F, residue_modulus=residue_modulus),
            "tree_accelerated": derive_from_f_tree(
                F, use_heap=True, residue_modulus=residue_modulus
            ),
            "geometric_sieve": derive_from_f_geometric_sieve(
                F, primitive_only=primitive_only, sieve_moduli=sieve_moduli
            ),
            "geometric_wheel": derive_from_f_geometric_sieve(
                F,
                primitive_only=primitive_only,
                sieve_moduli=sieve_moduli,
                use_wheel=True,
            ),
            "qa_qs": derive_from_f_qa_qs(
                F,
                primitive_only=primitive_only,
                factor_base_bound=qs_factor_base_bound,
                interval=qs_interval,
                log_tolerance=qs_log_tolerance,
            ),
            "qa_mpqs": derive_from_f_qa_mpqs(
                F,
                primitive_only=primitive_only,
                factor_base_bound=qs_factor_base_bound,
                half_width=mpqs_half_width,
                polynomial_count=mpqs_polynomial_count,
                large_prime_bound=mpqs_large_prime_bound,
                log_tolerance=qs_log_tolerance,
            ),
            "qa_mpqs_auto": derive_from_f_qa_mpqs_auto(
                F,
                primitive_only=primitive_only,
                log_tolerance=qs_log_tolerance,
                ecm_b1=ecm_b1,
                ecm_max_curves=ecm_max_curves,
            ),
            "qa_ecm": derive_from_f_qa_ecm(
                F, primitive_only=primitive_only, b1=ecm_b1, max_curves=ecm_max_curves
            ),
            "external": derive_from_f_external_backend(
                F,
                primitive_only=primitive_only,
                engine=external_engine,
                timeout_seconds=external_timeout,
            ),
            "production": derive_from_f_production(
                F,
                primitive_only=primitive_only,
                engine=external_engine,
                timeout_seconds=external_timeout,
                qs_factor_base_bound=qs_factor_base_bound,
                qs_interval=qs_interval,
                qs_log_tolerance=qs_log_tolerance,
            ),
            "factor": derive_from_f_factor(F, primitive_only=primitive_only),
        }
    raise ValueError(f"unknown method: {method}")


def derive_range(
    max_f: int,
    method: str,
    primitive_only: bool = False,
    residue_modulus: int = 24,
    sieve_moduli: list[int] | None = None,
    qs_factor_base_bound: int = 100,
    qs_interval: int = 5000,
    qs_log_tolerance: float = 1.5,
    external_engine: str = "auto",
    external_timeout: int = 300,
    mpqs_half_width: int = 128,
    mpqs_polynomial_count: int = 12,
    mpqs_large_prime_bound: int = 0,
) -> dict[str, Any]:
    if max_f <= 0:
        raise ValueError("max_f must be positive")
    rows = []
    for F in range(1, max_f + 1):
        result = derive_from_f(
            F,
            method=method,
            primitive_only=primitive_only,
            residue_modulus=residue_modulus,
            sieve_moduli=sieve_moduli,
            qs_factor_base_bound=qs_factor_base_bound,
            qs_interval=qs_interval,
            qs_log_tolerance=qs_log_tolerance,
            external_engine=external_engine,
            external_timeout=external_timeout,
            mpqs_half_width=mpqs_half_width,
            mpqs_polynomial_count=mpqs_polynomial_count,
            mpqs_large_prime_bound=mpqs_large_prime_bound,
        )
        if result.get("candidate_count", 0):
            rows.append(result)
    return {
        "max_F": max_f,
        "F_with_candidates": len(rows),
        "total_candidates": sum(row["candidate_count"] for row in rows),
        "rows": rows,
        "method": method,
        "primitive_only": primitive_only,
    }


def self_test() -> dict[str, Any]:
    checks = []
    expected = {
        3: [(4, 3, 5)],
        5: [(12, 5, 13)],
        7: [(24, 7, 25)],
        15: [(8, 15, 17), (112, 15, 113)],
    }
    for F, triples in expected.items():
        got_factor = derive_from_f(F, method="factor", primitive_only=True)
        got_tree = derive_from_f(F, method="tree", primitive_only=True)
        got_tree_best = derive_from_f(F, method="tree-best", primitive_only=True)
        got_tree_residue = derive_from_f(F, method="tree-residue", primitive_only=True)
        got_tree_accelerated = derive_from_f(
            F, method="tree-accelerated", primitive_only=True
        )
        got_geometric_sieve = derive_from_f(
            F, method="geometric-sieve", primitive_only=True
        )
        got_geometric_wheel = derive_from_f(
            F, method="geometric-wheel", primitive_only=True
        )
        got_qa_qs = derive_from_f(F, method="qa-qs", primitive_only=True)
        got_qa_mpqs = derive_from_f(F, method="qa-mpqs", primitive_only=True)
        # ECM params here are deliberately tiny — self-test only needs
        # correctness (ECM finding nothing on a prime still falls through to
        # MPQS, which is guaranteed), not the production success-rate tuning
        # applied to the qa-mpqs-auto/qa-ecm defaults for real use.
        got_qa_mpqs_auto = derive_from_f(
            F, method="qa-mpqs-auto", primitive_only=True, ecm_b1=200, ecm_max_curves=5
        )
        got_qa_ecm = derive_from_f(
            F, method="qa-ecm", primitive_only=True, ecm_b1=200, ecm_max_curves=5
        )
        observed_factor = [(row["C"], row["F"], row["G"]) for row in got_factor["candidates"]]
        observed_tree = [(row["C"], row["F"], row["G"]) for row in got_tree["candidates"]]
        observed_tree_best = [
            (row["C"], row["F"], row["G"]) for row in got_tree_best["candidates"]
        ]
        observed_tree_residue = [
            (row["C"], row["F"], row["G"]) for row in got_tree_residue["candidates"]
        ]
        observed_tree_accelerated = [
            (row["C"], row["F"], row["G"]) for row in got_tree_accelerated["candidates"]
        ]
        observed_geometric_sieve = [
            (row["C"], row["F"], row["G"]) for row in got_geometric_sieve["candidates"]
        ]
        observed_geometric_wheel = [
            (row["C"], row["F"], row["G"]) for row in got_geometric_wheel["candidates"]
        ]
        observed_qa_qs = [(row["C"], row["F"], row["G"]) for row in got_qa_qs["candidates"]]
        observed_qa_mpqs = [
            (row["C"], row["F"], row["G"]) for row in got_qa_mpqs["candidates"]
        ]
        observed_qa_mpqs_auto = [
            (row["C"], row["F"], row["G"]) for row in got_qa_mpqs_auto["candidates"]
        ]
        observed_qa_ecm = [(row["C"], row["F"], row["G"]) for row in got_qa_ecm["candidates"]]
        checks.append(
            {
                "F": F,
                "expected": triples,
                "observed_factor": observed_factor,
                "observed_tree": observed_tree,
                "observed_tree_best": observed_tree_best,
                "observed_tree_residue": observed_tree_residue,
                "observed_tree_accelerated": observed_tree_accelerated,
                "observed_geometric_sieve": observed_geometric_sieve,
                "observed_geometric_wheel": observed_geometric_wheel,
                "observed_qa_qs": observed_qa_qs,
                "observed_qa_mpqs": observed_qa_mpqs,
                "observed_qa_mpqs_auto": observed_qa_mpqs_auto,
                "observed_qa_ecm": observed_qa_ecm,
                "ok": (
                    observed_factor == triples
                    and observed_tree == triples
                    and observed_tree_best == triples
                    and observed_tree_residue == triples
                    and observed_tree_accelerated == triples
                    and observed_geometric_sieve == triples
                    and observed_geometric_wheel == triples
                    and observed_qa_qs == triples
                    and observed_qa_mpqs == triples
                    and observed_qa_mpqs_auto == triples
                    and observed_qa_ecm == triples
                ),
            }
        )
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def triple_signature(result: dict[str, Any]) -> list[tuple[int, int, int]]:
    return sorted((row["C"], row["F"], row["G"]) for row in result["candidates"])


def stress_suite(
    primitive_only: bool = True,
    factor_base_bound: int = 1000,
    half_width: int = 512,
    polynomial_count: int = 32,
    large_prime_bound: int = 0,
    log_tolerance: float = 1.5,
    auto: bool = False,
) -> dict[str, Any]:
    cases = [
        {"F": 1001, "label": "small_squarefree_7_11_13"},
        {"F": 15015, "label": "five_small_primes"},
        {"F": 255255, "label": "six_small_primes"},
        {"F": 100160063, "label": "semiprime_10007_10009"},
        {"F": 100460333, "label": "semiprime_10009_10037"},
        {"F": 1000036000099, "label": "semiprime_1000003_1000033"},
    ]
    rows = []
    all_ok = True

    for case in cases:
        F = case["F"]
        baseline = derive_from_f_factor(F, primitive_only=primitive_only)
        if auto:
            result = derive_from_f_qa_mpqs_auto(
                F,
                primitive_only=primitive_only,
                log_tolerance=log_tolerance,
            )
        else:
            result = derive_from_f_qa_mpqs(
                F,
                primitive_only=primitive_only,
                factor_base_bound=factor_base_bound,
                half_width=half_width,
                polynomial_count=polynomial_count,
                large_prime_bound=large_prime_bound,
                log_tolerance=log_tolerance,
            )
        baseline_sig = triple_signature(baseline)
        result_sig = triple_signature(result)
        matches = baseline_sig == result_sig
        all_ok = all_ok and matches
        engine = result["qa_engine"]
        rows.append(
            {
                "F": F,
                "label": case["label"],
                "digits": len(str(F)),
                "matches_factor_baseline": matches,
                "baseline_candidate_count": baseline["candidate_count"],
                "qa_mpqs_candidate_count": result["candidate_count"],
                "factors_found": engine.get("factors_found", []),
                "small_factor_frontier_complete": engine.get(
                    "small_factor_frontier_complete", False
                ),
                "polynomials_used": engine.get("polynomials_used", 0),
                "tested_after_sieve": engine.get("tested_after_sieve", 0),
                "smooth_relations": engine.get("smooth_relations", 0),
                "total_usable_relations": engine.get("total_usable_relations", 0),
                "large_prime_bound": engine.get("large_prime_bound", 0),
                "large_prime_partials": engine.get("large_prime_partials", 0),
                "large_prime_pairs": engine.get("large_prime_pairs", 0),
                "linear_algebra": engine.get("linear_algebra", {}),
                "dependencies_checked": engine.get("dependencies_checked", 0),
                "skipped_by_log": engine.get("skipped_by_log", 0),
                "auto_parameters": engine.get("auto_parameters"),
                "auto_attempts": engine.get("auto_attempts", []),
                "auto_success": engine.get("auto_success"),
            }
        )

    return {
        "ok": all_ok,
        "method": "qa_mpqs_auto_stress" if auto else "qa_mpqs_stress",
        "primitive_only": primitive_only,
        "parameters": {
            "factor_base_bound": factor_base_bound,
            "half_width": half_width,
            "polynomial_count": polynomial_count,
            "large_prime_bound": large_prime_bound,
            "log_tolerance": log_tolerance,
            "auto": auto,
        },
        "case_count": len(rows),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive QA Pythagorean right triangles from F."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--F", type=int, help="single F leg to derive from")
    group.add_argument("--max-F", type=int, help="derive for every F in [1,max-F]")
    parser.add_argument(
        "--primitive-only",
        action="store_true",
        help="keep only primitive canonical QA triples in factor mode",
    )
    parser.add_argument(
        "--method",
        choices=(
            "tree",
            "tree-best",
            "tree-residue",
            "tree-accelerated",
            "geometric-sieve",
            "geometric-wheel",
            "qa-qs",
            "qa-mpqs",
            "qa-mpqs-auto",
            "qa-ecm",
            "external",
            "production",
            "factor",
            "both",
        ),
        default="tree",
        help="QA methods include tree and geometric-sieve; factor is the baseline",
    )
    parser.add_argument(
        "--residue-modulus",
        type=int,
        default=24,
        help="modulus for tree-residue QA reachability gate",
    )
    parser.add_argument(
        "--sieve-moduli",
        default="3,5,7,11,13,16",
        help="comma-separated square-residue moduli for geometric-sieve",
    )
    parser.add_argument(
        "--qs-factor-base-bound",
        type=int,
        default=100,
        help="prime bound for qa-qs factor base",
    )
    parser.add_argument(
        "--qs-interval",
        type=int,
        default=5000,
        help="number of d values to sieve in qa-qs mode",
    )
    parser.add_argument(
        "--qs-log-tolerance",
        type=float,
        default=1.5,
        help="max log deficit allowed before qa-qs trial division",
    )
    parser.add_argument(
        "--mpqs-half-width",
        type=int,
        default=128,
        help="x half-width for qa-mpqs charts d=A*x+B",
    )
    parser.add_argument(
        "--mpqs-polynomial-count",
        type=int,
        default=12,
        help="number of QA MPQS chart moduli A to try",
    )
    parser.add_argument(
        "--mpqs-large-prime-bound",
        type=int,
        default=0,
        help="single-large-prime residual bound for qa-mpqs; 0 disables it",
    )
    parser.add_argument(
        "--ecm-b1",
        type=int,
        default=20000,
        help="ECM stage-1 smoothness bound for qa-ecm / the qa-mpqs-auto bridge stage",
    )
    parser.add_argument(
        "--ecm-max-curves",
        type=int,
        default=50,
        help="number of random curves to try for qa-ecm / the qa-mpqs-auto bridge stage",
    )
    parser.add_argument(
        "--external-engine",
        choices=("auto", "msieve", "yafu", "cado-nfs"),
        default="auto",
        help="external SIQS/MPQS/NFS-grade backend for --method external",
    )
    parser.add_argument(
        "--external-timeout",
        type=int,
        default=300,
        help="timeout in seconds for --method external",
    )
    parser.add_argument("--json-out", type=Path, help="optional output JSON path")
    parser.add_argument("--self-test", action="store_true", help="run built-in checks")
    parser.add_argument("--stress", action="store_true", help="run QA-MPQS stress suite")
    parser.add_argument(
        "--stress-auto",
        action="store_true",
        help="use qa-mpqs-auto profiles inside the stress suite",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.self_test:
        payload = self_test()
        print(canonical_json(payload))
        return 0 if payload["ok"] else 1

    if args.stress:
        payload = stress_suite(
            primitive_only=args.primitive_only,
            factor_base_bound=args.qs_factor_base_bound,
            half_width=args.mpqs_half_width,
            polynomial_count=args.mpqs_polynomial_count,
            large_prime_bound=args.mpqs_large_prime_bound,
            log_tolerance=args.qs_log_tolerance,
            auto=args.stress_auto,
        )
        text = canonical_json(payload)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if payload["ok"] else 1

    if args.F is None and args.max_F is None:
        raise SystemExit("Provide --F, --max-F, --stress, or --self-test")

    if args.F is not None:
        sieve_moduli = parse_moduli(args.sieve_moduli)
        payload = derive_from_f(
            args.F,
            method=args.method,
            primitive_only=args.primitive_only,
            residue_modulus=args.residue_modulus,
            sieve_moduli=sieve_moduli,
            qs_factor_base_bound=args.qs_factor_base_bound,
            qs_interval=args.qs_interval,
            qs_log_tolerance=args.qs_log_tolerance,
            external_engine=args.external_engine,
            external_timeout=args.external_timeout,
            mpqs_half_width=args.mpqs_half_width,
            mpqs_polynomial_count=args.mpqs_polynomial_count,
            mpqs_large_prime_bound=args.mpqs_large_prime_bound,
            ecm_b1=args.ecm_b1,
            ecm_max_curves=args.ecm_max_curves,
        )
    else:
        sieve_moduli = parse_moduli(args.sieve_moduli)
        payload = derive_range(
            args.max_F,
            method=args.method,
            primitive_only=args.primitive_only,
            residue_modulus=args.residue_modulus,
            sieve_moduli=sieve_moduli,
            qs_factor_base_bound=args.qs_factor_base_bound,
            qs_interval=args.qs_interval,
            qs_log_tolerance=args.qs_log_tolerance,
            external_engine=args.external_engine,
            external_timeout=args.external_timeout,
            mpqs_half_width=args.mpqs_half_width,
            mpqs_polynomial_count=args.mpqs_polynomial_count,
            mpqs_large_prime_bound=args.mpqs_large_prime_bound,
        )

    text = canonical_json(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
