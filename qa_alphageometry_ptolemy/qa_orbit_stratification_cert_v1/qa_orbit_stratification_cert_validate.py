#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=validator derives from in-repo theorem docs (docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md, QA_ORBIT_THEOREM_SYNTHESIS.md, QA_GENERATOR_REACHABILITY.md); cert is pure recomputation of their propositions, not derivation from external literature -->
"""
qa_orbit_stratification_cert_validate.py

Validator for QA_ORBIT_STRATIFICATION_CERT.v1  [family 261, pending registration]

Certifies the QA Orbit Stratification Theorem (two-layer form, proven
2026-04-20) as a machine-checkable structure.

    Part I  — ⟨σ, μ⟩-orbits on (Z/mZ)² are content-ideal classes:
              L_j = { (b,e) : min(v_p(b), v_p(e)) = j } at each prime
              power factor p^k of m, Cartesian-producted via CRT.

    Part II — σ-only orbits refine by how x² − x − 1 factors mod p,
              with three cases by Legendre symbol (5|p):
                Case A (inert):      uniform orbit length π(p^n)
                Case B (split):      eigenspace + generic orbits
                Case C (ramified):   Jordan filtration on p=5

    Bridge   — μ is the collapse operator from Part II's Frobenius
               / Jordan refinement to Part I's content-ideal classes.

Companion files (primary source for this validator):
    - docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md  (full proofs)
    - docs/theory/QA_ORBIT_THEOREM_SYNTHESIS.md       (one-page form)
    - docs/theory/QA_GENERATOR_REACHABILITY.md        (audit trail)

Checks:
    QOS_1      — schema_version matches
    QOS_A      — J(b,e) := min(v_p(b), v_p(e)) invariant under σ and μ
    QOS_B_SZ   — per-level sizes match closed form p^{2(k-j-1)}·(p²-1)
    QOS_B_CC   — {σ,μ}-orbit components equal the level sets
    QOS_C      — CRT factorization on composite m
    QOS_II_A   — Part II Case A on inert p^n: uniform orbit length
    QOS_II_B   — Part II Case B on split p: eigenspace + generic orbits
    QOS_II_C   — Part II Case C on 5^n (n≥2): 5^{n-1}×π(5^n) + 5^{n-1}×π(5^{n-1})
    QOS_F      — fail_ledger well-formed

QA axiom compliance:
    A1 — witnesses in {1..m}. A2 — derived coords as expressions only.
    T1 — integer path time. T2 — pure-integer validator, no floats.
    S1 — x*x (not x**2). S2 — int throughout.
"""
from __future__ import annotations

QA_COMPLIANCE = {
    "observer": "theorem_validator",
    "state_alphabet": "(b,e) in {1..m}^2 for declared moduli m; all integer arithmetic; no observer projections; validator computes orbits + checks Props A/B/C against closed forms",
}

import json
import sys
from pathlib import Path


SCHEMA_VERSION = "QA_ORBIT_STRATIFICATION_CERT.v1"


# ── Canonical primitives (mirrors qa_orbit_rules.py and qa_core.js) ──

def qa_mod(x: int, m: int) -> int:
    """A1-compliant: result in {1..m}, never 0."""
    return ((x - 1) % m) + 1


def sigma(b: int, e: int, m: int) -> tuple[int, int]:
    return e, qa_mod(b + e, m)


def mu(b: int, e: int) -> tuple[int, int]:
    return e, b


def v_p(n: int, p: int) -> int:
    if n == 0:
        return 10**9
    v, x = 0, n
    while x % p == 0:
        x //= p
        v += 1
    return v


def J(b: int, e: int, p: int) -> int:
    return min(v_p(b, p), v_p(e, p))


def prime_power_factor(m: int) -> list[tuple[int, int]]:
    out = []
    n = m
    d = 2
    while d * d <= n:
        if n % d == 0:
            k = 0
            while n % d == 0:
                n //= d
                k += 1
            out.append((d, k))
        d += 1
    if n > 1:
        out.append((n, 1))
    return out


def pisano(m: int) -> int:
    a, b = 0, 1
    for i in range(1, 6 * m + 2):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return i
    return -1


def sigma_mu_components(m: int) -> list[list[tuple[int, int]]]:
    seen: dict[tuple[int, int], int] = {}
    comps: list[list[tuple[int, int]]] = []
    for b0 in range(1, m + 1):
        for e0 in range(1, m + 1):
            if (b0, e0) in seen:
                continue
            R = {(b0, e0)}
            stack = [(b0, e0)]
            while stack:
                b, e = stack.pop()
                for nb, ne in (sigma(b, e, m), mu(b, e)):
                    if (nb, ne) not in R:
                        R.add((nb, ne))
                        stack.append((nb, ne))
            cid = len(comps)
            for pt in R:
                seen[pt] = cid
            comps.append(sorted(R))
    return comps


def sigma_only_orbits(m: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out = []
    for b0 in range(1, m + 1):
        for e0 in range(1, m + 1):
            if (b0, e0) in seen:
                continue
            orb = []
            b, e = b0, e0
            while (b, e) not in seen:
                seen.add((b, e))
                orb.append((b, e))
                b, e = sigma(b, e, m)
            out.append(orb)
    return out


def level_size(p: int, k: int, j: int) -> int:
    if j == k:
        return 1
    return p ** (2 * (k - j - 1)) * (p * p - 1)


def legendre_5(p: int) -> int:
    if p == 2:
        return -1
    if p == 5:
        return 0
    r = pow(5, (p - 1) // 2, p)
    if r == 1:
        return 1
    if r == p - 1:
        return -1
    raise ValueError(f"unexpected legendre(5, {p}) = {r}")


# ── Check routines ──

def _check_schema(data: dict) -> list[str]:
    errs = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"QOS_1: schema_version={data.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")
    return errs


def _check_J_invariance(m: int) -> list[str]:
    factors = prime_power_factor(m)
    errs = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            nb_s, ne_s = sigma(b, e, m)
            nb_m, ne_m = mu(b, e)
            for (p, _k) in factors:
                if J(nb_s, ne_s, p) != J(b, e, p):
                    errs.append(f"QOS_A: σ breaks J at ({b},{e}) on p={p}: J={J(b,e,p)}, J∘σ={J(nb_s,ne_s,p)}")
                    return errs
                if J(nb_m, ne_m, p) != J(b, e, p):
                    errs.append(f"QOS_A: μ breaks J at ({b},{e}) on p={p}")
                    return errs
    return errs


def _check_level_sizes(m: int) -> list[str]:
    factors = prime_power_factor(m)
    if len(factors) != 1:
        return []
    p, k = factors[0]
    errs = []
    measured: dict[int, int] = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            j = min(v_p(b, p), v_p(e, p))
            measured[j] = measured.get(j, 0) + 1
    for j in range(k + 1):
        expected = level_size(p, k, j)
        got = measured.get(j, 0)
        if got != expected:
            errs.append(f"QOS_B_SZ: |L_{j}| on {p}^{k}={m}: measured {got}, expected {expected}")
    return errs


def _check_components_equal_levels(m: int) -> list[str]:
    factors = prime_power_factor(m)
    if len(factors) != 1:
        return []
    p, _k = factors[0]
    comps = sigma_mu_components(m)
    errs = []
    for cid, comp in enumerate(comps):
        levels = {J(b, e, p) for (b, e) in comp}
        if len(levels) != 1:
            errs.append(f"QOS_B_CC: component {cid} has mixed J values: {sorted(levels)}")
    by_level: dict[int, list[int]] = {}
    for cid, comp in enumerate(comps):
        j = J(comp[0][0], comp[0][1], p)
        by_level.setdefault(j, []).append(cid)
    for j, cids in by_level.items():
        if len(cids) != 1:
            errs.append(f"QOS_B_CC: level {j} split across components {cids}")
    return errs


def _check_crt_factorization(m: int) -> list[str]:
    factors = prime_power_factor(m)
    if len(factors) < 2:
        return []
    comps = sigma_mu_components(m)
    sizes_got = sorted([len(c) for c in comps], reverse=True)
    import itertools
    per_factor_sizes = []
    for (p, k) in factors:
        per_factor_sizes.append([level_size(p, k, j) for j in range(k + 1)])
    predicted = []
    for combo in itertools.product(*per_factor_sizes):
        prod = 1
        for x in combo:
            prod *= x
        predicted.append(prod)
    predicted.sort(reverse=True)
    errs = []
    if sizes_got != predicted:
        errs.append(f"QOS_C: CRT size mismatch on m={m}. got={sizes_got}, expected={predicted}")
    return errs


def _check_part_ii_A(p: int, n: int) -> list[str]:
    if legendre_5(p) != -1:
        return []
    m = p ** n
    expected = pisano(m)
    orbits = sigma_only_orbits(m)
    errs = []
    for orb in orbits:
        b0, e0 = orb[0]
        if min(v_p(b0, p), v_p(e0, p)) != 0:
            continue
        if len(orb) != expected:
            errs.append(f"QOS_II_A: on p^n={m}, L_0 orbit at ({b0},{e0}) has length {len(orb)}, expected π={expected}")
            return errs
    return errs


def _check_part_ii_B(p: int, n: int) -> list[str]:
    if legendre_5(p) != 1 or n != 1:
        return []
    m = p
    sqrt5 = None
    for x in range(1, p):
        if (x * x) % p == 5 % p:
            sqrt5 = x
            break
    if sqrt5 is None:
        return [f"QOS_II_B: no sqrt(5) mod {p} — contradicts split hypothesis"]
    half = pow(2, -1, p)
    phi_ = ((1 + sqrt5) * half) % p
    psi_ = ((1 - sqrt5 + 2 * p) * half) % p

    def ord_in(a: int) -> int:
        k, x = 1, a % p
        while x != 1:
            x = (x * a) % p
            k += 1
            if k > p:
                return -1
        return k

    op = ord_in(phi_)
    opsi = ord_in(psi_)
    pi = pisano(m)

    orbits = sigma_only_orbits(m)
    length_counts: dict[int, int] = {}
    for orb in orbits:
        b0, e0 = orb[0]
        if min(v_p(b0, p), v_p(e0, p)) != 0:
            continue
        length_counts[len(orb)] = length_counts.get(len(orb), 0) + 1

    n_phi = (p - 1) // op
    n_psi = (p - 1) // opsi
    L0 = p * p - 1
    n_generic = (L0 - (p - 1) * 2) // pi

    expected: dict[int, int] = {}
    expected[op] = expected.get(op, 0) + n_phi
    expected[opsi] = expected.get(opsi, 0) + n_psi
    expected[pi] = expected.get(pi, 0) + n_generic

    errs = []
    if length_counts != expected:
        errs.append(f"QOS_II_B: on p={p}, length-count mismatch. got={length_counts}, expected={expected}")
    return errs


def _check_part_ii_C(n: int) -> list[str]:
    if n < 2:
        return []
    p = 5
    m = p ** n
    pi_n = pisano(m)
    pi_n1 = pisano(p ** (n - 1))
    orbits = sigma_only_orbits(m)
    length_counts: dict[int, int] = {}
    for orb in orbits:
        b0, e0 = orb[0]
        if min(v_p(b0, p), v_p(e0, p)) != 0:
            continue
        length_counts[len(orb)] = length_counts.get(len(orb), 0) + 1
    expected = {pi_n: 5 ** (n - 1), pi_n1: 5 ** (n - 1)}
    errs = []
    if length_counts != expected:
        errs.append(f"QOS_II_C: on 5^{n}={m}, length-count mismatch. got={length_counts}, expected={expected}")
    return errs


def _check_fail_ledger(data: dict) -> list[str]:
    errs = []
    fl = data.get("fail_ledger")
    if fl is None:
        return []
    if not isinstance(fl, list):
        errs.append("QOS_F: fail_ledger must be a list")
        return errs
    for i, item in enumerate(fl):
        if not isinstance(item, dict):
            errs.append(f"QOS_F: entry {i} not a dict")
            continue
        for key in ("timestamp", "reason"):
            if key not in item:
                errs.append(f"QOS_F: entry {i} missing {key!r}")
    return errs


def validate(path: Path) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    errs: list[str] = []
    warns: list[str] = []

    errs += _check_schema(data)

    for m in data.get("prime_power_moduli", []):
        errs += _check_J_invariance(m)
        errs += _check_level_sizes(m)
        errs += _check_components_equal_levels(m)
        factors = prime_power_factor(m)
        if len(factors) == 1:
            p, k = factors[0]
            errs += _check_part_ii_A(p, k)
            errs += _check_part_ii_B(p, k)
            if p == 5:
                errs += _check_part_ii_C(k)

    for m in data.get("composite_moduli", []):
        errs += _check_crt_factorization(m)

    errs += _check_fail_ledger(data)

    return errs, warns


# ── Self-test ──

def _self_test() -> dict:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("qos_pass_core.json", True),
        ("qos_fail_wrong_component_count.json", False),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, _warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False, "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False, "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA Orbit Stratification Cert [261] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list((Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
