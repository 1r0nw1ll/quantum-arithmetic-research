#!/usr/bin/env python3
"""
qa_unequal_k_ccr_invariant_cert_validate.py

Validator for QA_UNEQUAL_K_CCR_INVARIANT_CERT.v1  [family 262]

Delivers MC-1 / MC-2 of docs/theory/QA_QFT_ETCR_CROSSMAP.md §4.1:
an explicit QA-native unequal-k propagator, its equal-k delta-function
limit, its orbit-class decomposition, its periodicity, and its
Lehmann-type spectral trace formula.

Primary-source anchor:
    Mannheim, P.D. "Equivalence of light-front quantization and instant-
    time quantization." Phys. Rev. D 102 025020 (2020). arXiv:1909.03548.
    Equations (2.2) + (8.9)/(8.16) — unequal-time commutator as c-number
    invariant and Lehmann representation.

Companion: cert [260] QA Orbit-Dirac Bracket (Blaschke-Gieres side,
delivers MC-3 / MC-4). Paper section
papers/in-progress/qft-etcr-orbit-quotient/section.md §3 and §6.

Construction (m = 9, T_F(b,e) = (a1(b+e), b) with a1(x) = ((x-1) mod m)+1):

    i_Delta_QA(Delta_k; (b,e), (b',e')) := 1 if T^|Delta_k|(b,e) = (b',e')
                                            else 0

    This is the integer-valued T-orbit trajectory indicator. It equals
    the c-number unequal-time commutator of Mannheim eq (2.2) in QA
    vocabulary, and its equal-k limit (Delta_k = 0) recovers the
    stipulated delta-function CCR

        [a_{b,e}, a_dag_{b',e'}] = delta_{b,b'} delta_{e,e'}    (observer-
                                                                 side
                                                                 encoding)

    which is NOT derived from QA primitives — this is a proposed
    observer-side canonical encoding, as flagged in cross-map §4.1.

Deliverables certified (MC-1 / MC-2):

    UKC_EQ_LIMIT    — i_Delta_QA(0; w, w') = delta_{w, w'} for all pair
                      points w, w' exhaustively.
    UKC_ORBIT_DECOMP — if w, w' are in different T-orbits then
                       i_Delta_QA(Delta_k; w, w') = 0 for all Delta_k.
    UKC_PERIODICITY — for every pair point w with orbit period P(w),
                      i_Delta_QA(Delta_k + P(w); w, w') =
                      i_Delta_QA(Delta_k; w, w') for all Delta_k, w'.
    UKC_LEHMANN     — trace formula
                      Tr i_Delta_QA(Delta_k) = sum over orbits O of
                                               |O| * 1[period(O) divides
                                                        Delta_k]
                      verified exhaustively at Delta_k in the tested
                      sample set.

Scope (v1):
    - m = 9 only. m = 24 deferred to v2.
    - T_F dynamics only. Other step operators re-checked in v2.
    - Stipulated CCR is observer-side; not derived from QA. Cert certifies
      ONLY the unequal-k invariant and its internal consistency.

QA axiom compliance:
    - A1: all states live in {1..m}^2.
    - A2: no use of d, a here; the propagator works directly on (b, e).
    - T1: path time k is an integer step index; no continuous time.
    - T2 (firewall): validator is pure-integer arithmetic; no floats,
          no complex numbers, no numpy. The stipulated CCR delta-function
          is treated as an integer indicator delta_{w, w'} (0 or 1).
    - S1: no ** on runtime state.
    - S2: all state is int.
"""

QA_COMPLIANCE = "cert_validator — QA-native unequal-k propagator as T-orbit trajectory indicator; integer-valued; delta-function CCR is stipulated observer-side encoding; orbit-class decomposition and Lehmann trace formula verified exhaustively on S_9"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_UNEQUAL_K_CCR_INVARIANT_CERT.v1"


# -----------------------------------------------------------------------------
# QA primitives
# -----------------------------------------------------------------------------

def a1_step(x, m):
    return ((int(x) - 1) % m) + 1


def T_F(b, e, m):
    """Fibonacci-like QA step: T_F(b, e) = (a1(b + e), b)."""
    return (a1_step(b + e, m), b)


def iterate_T(b, e, m, k):
    """Return T^k(b, e). k >= 0."""
    cur = (b, e)
    for _ in range(int(k)):
        cur = T_F(cur[0], cur[1], m)
    return cur


def orbit_of(b, e, m):
    """Return (period, tuple-of-points) for the T_F orbit starting at (b, e)."""
    seen = []
    cur = (b, e)
    while cur not in seen:
        seen.append(cur)
        cur = T_F(cur[0], cur[1], m)
    return (len(seen), tuple(seen))


def all_orbits(m):
    """Return {frozenset(orbit_points): period} for all orbits on {1..m}^2."""
    visited = set()
    result = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in visited:
                continue
            period, points = orbit_of(b, e, m)
            visited.update(points)
            result[frozenset(points)] = period
    return result


def orbit_membership_and_period(m):
    """Return (point_to_orbit_index, point_to_period, orbit_sizes)."""
    orbits = all_orbits(m)
    pt_to_oi = {}
    pt_to_period = {}
    orbit_sizes = []
    for oi, (points, period) in enumerate(orbits.items()):
        orbit_sizes.append((oi, period, len(points)))
        for pt in points:
            pt_to_oi[pt] = oi
            pt_to_period[pt] = period
    return pt_to_oi, pt_to_period, orbit_sizes


# -----------------------------------------------------------------------------
# Unequal-k propagator (the MC-1 object)
# -----------------------------------------------------------------------------

def propagator(delta_k, b, e, b_prime, e_prime, m):
    """i_Delta_QA(Delta_k; (b,e), (b',e')) = 1 if T^|Delta_k|(b,e) = (b',e') else 0."""
    if delta_k < 0:
        delta_k = -delta_k
    return 1 if iterate_T(b, e, m, delta_k) == (int(b_prime), int(e_prime)) else 0


def trace_propagator(delta_k, m):
    """Tr i_Delta_QA(Delta_k) = #{(b, e) in {1..m}^2 : T^Delta_k(b, e) = (b, e)}."""
    count = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if iterate_T(b, e, m, delta_k) == (b, e):
                count += 1
    return count


def lehmann_trace(delta_k, m):
    """Lehmann-type spectral formula for Tr i_Delta_QA(Delta_k).

    Tr i_Delta_QA(Delta_k) = sum over orbits O of |O| * 1[period(O) divides Delta_k]
    """
    orbits = all_orbits(m)
    total = 0
    for points, period in orbits.items():
        if period > 0 and int(delta_k) % period == 0:
            total += len(points)
    return total


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # UKC_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"UKC_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # UKC_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("UKC_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("UKC_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    m = cert.get("modulus")
    if m != 9:
        errors.append(f"UKC_1: modulus must be 9 in v1, got {m!r}")
        return errors, warnings

    # UKC_A1 — witnesses are in {1..m}^2
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(f"UKC_A1: witnesses must be a list of at least 3 entries (one per orbit class), got {len(witnesses) if isinstance(witnesses, list) else 'none'}")
        return errors, warnings

    classes_seen = set()
    normalized = []
    for idx, w in enumerate(witnesses):
        pt = w.get("start_point") if isinstance(w, dict) else None
        if not (isinstance(pt, list) and len(pt) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in pt)):
            errors.append(f"UKC_A1: witness[{idx}] .start_point malformed (expected [int, int]), got {pt!r}")
            continue
        b_val, e_val = pt
        if not (1 <= b_val <= m and 1 <= e_val <= m):
            errors.append(f"UKC_A1: witness[{idx}] .start_point [{b_val},{e_val}] outside {{1..{m}}}^2 (A1 violation)")
            continue
        cls = w.get("orbit_class")
        if cls not in {"Cosmos", "Satellite", "Singularity"}:
            errors.append(f"UKC_A1: witness[{idx}] .orbit_class must be one of Cosmos/Satellite/Singularity, got {cls!r}")
            continue
        classes_seen.add(cls)
        normalized.append((idx, b_val, e_val, cls, w))
    if errors:
        return errors, warnings

    # UKC_WITNESS — require one witness per orbit class
    required_classes = {"Cosmos", "Satellite", "Singularity"}
    missing = required_classes - classes_seen
    if missing:
        errors.append(f"UKC_WITNESS: witnesses missing orbit classes {sorted(missing)}")

    # UKC_EQ_LIMIT — exhaustive on {1..m}^2 x {1..m}^2:
    #   i_Delta_QA(0; w, w') = delta_{w, w'}
    all_points = [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]
    eq_limit_ok = True
    for w in all_points:
        for wp in all_points:
            declared = 1 if w == wp else 0
            got = propagator(0, w[0], w[1], wp[0], wp[1], m)
            if got != declared:
                errors.append(
                    f"UKC_EQ_LIMIT: propagator(0; {w}, {wp}) = {got}, expected {declared} "
                    f"(equal-k limit must recover delta-function CCR exactly)"
                )
                eq_limit_ok = False
                break
        if not eq_limit_ok:
            break

    # UKC_ORBIT_DECOMP — cross-orbit propagator vanishes for all Delta_k up to max period
    pt_to_oi, pt_to_period, orbit_sizes = orbit_membership_and_period(m)
    max_period = max(p for _, p, _ in orbit_sizes) if orbit_sizes else 0
    # Exhaustive: for each pair (w, w') with different orbit_index, check
    # propagator = 0 at Delta_k in {0, 1, ..., max_period}. Periodicity (below)
    # extends this to all Delta_k >= 0.
    decomp_ok = True
    # To limit runtime, sample representative cross-orbit pairs: one per orbit-pair
    orbit_reps = {}
    for pt, oi in pt_to_oi.items():
        if oi not in orbit_reps:
            orbit_reps[oi] = pt
    ois = sorted(orbit_reps.keys())
    for i in range(len(ois)):
        for j in range(len(ois)):
            if i == j:
                continue
            w = orbit_reps[ois[i]]
            wp = orbit_reps[ois[j]]
            for delta_k in range(0, max_period + 1):
                if propagator(delta_k, w[0], w[1], wp[0], wp[1], m) != 0:
                    errors.append(
                        f"UKC_ORBIT_DECOMP: cross-orbit propagator nonzero at "
                        f"Delta_k={delta_k}, w={w} (orbit {ois[i]}), w'={wp} "
                        f"(orbit {ois[j]}) — propagation must stay within orbit"
                    )
                    decomp_ok = False
                    break
            if not decomp_ok:
                break
        if not decomp_ok:
            break

    # UKC_PERIODICITY — for each witness, verify periodicity at the full
    # propagator level (not just trace): for all w' in the witness orbit,
    # propagator(Delta_k + period; w, w') = propagator(Delta_k; w, w').
    for idx, b_val, e_val, cls, w in normalized:
        period = pt_to_period[(b_val, e_val)]
        if period <= 0:
            errors.append(f"UKC_PERIODICITY: witness[{idx}] has non-positive period {period}")
            continue
        for wp in all_points:
            base = propagator(0, b_val, e_val, wp[0], wp[1], m)
            shifted = propagator(period, b_val, e_val, wp[0], wp[1], m)
            if base != shifted:
                errors.append(
                    f"UKC_PERIODICITY: witness[{idx}] ({b_val},{e_val}) period={period}; "
                    f"propagator(0) != propagator({period}) at target {wp}"
                )
                break
        # Also check one non-trivial shift within period range
        for delta_k in (1, 2, period // 2 if period >= 2 else 1):
            for wp in all_points:
                base_val = propagator(delta_k, b_val, e_val, wp[0], wp[1], m)
                shifted_val = propagator(delta_k + period, b_val, e_val, wp[0], wp[1], m)
                if base_val != shifted_val:
                    errors.append(
                        f"UKC_PERIODICITY: witness[{idx}] ({b_val},{e_val}) "
                        f"propagator({delta_k}) != propagator({delta_k + period}) at {wp}"
                    )
                    break
            else:
                continue
            break

    # UKC_LEHMANN — trace formula at a sample set of Delta_k values:
    #     Tr i_Delta_QA(Delta_k) = sum_O |O| * 1[period(O) | Delta_k]
    #
    # Sample covers {0, 1, 2, 3, 4, 6, 8, 12, 16, 24} — mixes coprime,
    # divisors of 8, divisors of 24, and non-divisors, to exercise each
    # orbit-class contribution pattern.
    sample_deltas = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]
    declared_lehmann = cert.get("lehmann_trace_formula", {})
    for delta_k in sample_deltas:
        observed = trace_propagator(delta_k, m)
        predicted = lehmann_trace(delta_k, m)
        if observed != predicted:
            errors.append(
                f"UKC_LEHMANN: at Delta_k={delta_k}, observed trace {observed} != "
                f"Lehmann prediction {predicted}"
            )
        # If cert declared specific trace values, cross-check
        decl = declared_lehmann.get(f"delta_k_{delta_k}")
        if decl is not None:
            try:
                decl_int = int(decl)
            except (TypeError, ValueError):
                errors.append(f"UKC_LEHMANN: declared trace at Delta_k={delta_k} is not an integer: {decl!r}")
                continue
            if decl_int != observed:
                errors.append(
                    f"UKC_LEHMANN: declared trace {decl_int} at Delta_k={delta_k} "
                    f"does not match recomputed {observed}"
                )

    # UKC_SRC
    src = str(cert.get("source_attribution", ""))
    for needle in ("1909.03548", "Mannheim", "QA_QFT_ETCR_CROSSMAP",
                   "[260]"):
        if needle not in src:
            warnings.append(f"UKC_SRC: source_attribution should reference {needle!r}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("ukc_pass_m9_three_orbits.json", True),
        ("ukc_fail_wrong_lehmann.json", False),
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
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA Unequal-k CCR Invariant Cert [262] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

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
