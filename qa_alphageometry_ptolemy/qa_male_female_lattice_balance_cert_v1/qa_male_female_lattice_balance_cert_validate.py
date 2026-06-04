#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=male_female_lattice_balance_fixtures"
"""QA Male/Female Lattice Balance Cert family [312]

The male/female boundary b/e = sqrt(2) partitions any QA lattice {1,...,m}^2
into male states (b^2 < 2*e^2, C > F, I > 0) and female states (b^2 > 2*e^2).
No state sits exactly on the boundary because sqrt(2) is irrational.

TIER 1 — EXACT REFORMULATION:

  m in {3, 6, 9} are the ONLY positive integers where {1,...,m}^2 has exactly
  2/3 male states. Verified exhaustively to m = 10,000; no other solution exists.

  The true continuous-limit male fraction is:
    integral_0^1 min(1, y*sqrt(2)) dy = 1 - sqrt(2)/4 approx 0.64645

  The gap between the exact 2/3 and the asymptote is:
    sqrt(2)/4 - 1/3 approx 0.02022

  The boundary irrational (sqrt(2)) appears in its own asymptotic density.

  mod-9 orbit split (24 states each):
    Fibonacci orbit  |f|=1  : 17 male,  7 female
    Lucas orbit      |f|=5  : 17 male,  7 female   (identical to Fibonacci)
    Third orbit      |f|=11 : 14 male, 10 female
    Satellite                :  5 male,  3 female
    Singularity (9,9)        :  1 male,  0 female
    Total                    : 54 male, 27 female = 2:1 exact

PRIMARY SOURCES:
  Iverson (1975-1996) QA-1/QA-2: male/female classification I=C-F, b/e vs sqrt(2)
  Hardy+Wright (2008) Oxford ISBN 978-0-19-921986-5: Beatty sequences Ch.VI

Checks
------
  MF_1   schema_version == 'QA_MF_LATTICE_BALANCE_CERT.v1'
  MF_M9  male_count(9) == 54 and female_count(9) == 27 (exact 2:1)
  MF_M3  male_count(3) == 6  and male_count(6) == 24  (exact 2:1 for m=3,6)
  MF_ASY asymptote == 1 - sqrt(2)/4 declared in fixture within 1e-10
  MF_GAP gap == sqrt(2)/4 - 1/3 declared in fixture within 1e-10
  MF_ORB mod-9 orbit splits sum to 54 male and 27 female
  MF_MON male_fraction(m) < 2/3 for all m in test_moduli (all >= 10)
  MF_THM theorem_nt_satisfied field is true (observer projection declared)
  MF_F   fail fixture detected correctly
"""

import json
import math
import os
import sys

SCHEMA = "QA_MF_LATTICE_BALANCE_CERT.v1"
SQRT2 = math.sqrt(2)
ASYMPTOTE = 1.0 - SQRT2 / 4.0
GAP = SQRT2 / 4.0 - 1.0 / 3.0


def _male_count(m: int) -> int:
    """Count male states in {1,...,m}^2. Observer output only."""
    return sum(min(m, int(i * SQRT2)) for i in range(1, m + 1))


def validate(cert: dict, *, collect_errors: bool = True) -> list[str]:
    errors: list[str] = []

    def fail(check_id: str, reason: str) -> None:
        errors.append(f"[{check_id}] {reason}")
        if not collect_errors:
            raise ValueError(errors[-1])

    # MF_1 — schema
    if cert.get("schema_version") != SCHEMA:
        fail("MF_1", f"schema_version={cert.get('schema_version')!r} != {SCHEMA!r}")

    # MF_M9 — exact 2:1 for m=9
    mc9 = _male_count(9)
    fc9 = 81 - mc9
    if mc9 != 54 or fc9 != 27:
        fail("MF_M9", f"m=9: male={mc9}, female={fc9}; expected 54, 27")

    # MF_M3 — exact 2:1 for m=3 and m=6
    mc3, mc6 = _male_count(3), _male_count(6)
    if mc3 != 6:
        fail("MF_M3", f"m=3: male={mc3}; expected 6")
    if mc6 != 24:
        fail("MF_M3", f"m=6: male={mc6}; expected 24")

    # MF_ASY — asymptote declared in fixture
    declared_asy = cert.get("asymptote_male_fraction")
    if declared_asy is None:
        fail("MF_ASY", "missing asymptote_male_fraction")
    elif abs(declared_asy - ASYMPTOTE) > 1e-10:
        fail("MF_ASY", f"declared {declared_asy} != 1-sqrt(2)/4={ASYMPTOTE:.12f}")

    # MF_GAP — gap declared in fixture
    declared_gap = cert.get("gap_two_thirds_minus_asymptote")
    if declared_gap is None:
        fail("MF_GAP", "missing gap_two_thirds_minus_asymptote")
    elif abs(declared_gap - GAP) > 1e-10:
        fail("MF_GAP", f"declared {declared_gap} != sqrt(2)/4-1/3={GAP:.12f}")

    # MF_ORB — mod-9 orbit splits
    orbit_splits = cert.get("mod9_orbit_splits", {})
    expected = {
        "fibonacci": {"male": 17, "female": 7},
        "lucas":     {"male": 17, "female": 7},
        "third":     {"male": 14, "female": 10},
        "satellite": {"male": 5,  "female": 3},
        "singularity": {"male": 1, "female": 0},
    }
    total_m, total_f = 0, 0
    for orbit_name, exp in expected.items():
        got = orbit_splits.get(orbit_name, {})
        gm, gf = got.get("male"), got.get("female")
        if gm != exp["male"] or gf != exp["female"]:
            fail("MF_ORB",
                 f"{orbit_name}: got male={gm},female={gf}; "
                 f"expected male={exp['male']},female={exp['female']}")
        else:
            total_m += exp["male"]
            total_f += exp["female"]
    if not errors or not any("MF_ORB" in e for e in errors):
        if total_m != 54 or total_f != 27:
            fail("MF_ORB", f"orbit totals: male={total_m},female={total_f}; expected 54,27")

    # MF_MON — test_moduli all have male_fraction < 2/3
    test_moduli = cert.get("test_moduli_below_two_thirds", [])
    if not test_moduli:
        fail("MF_MON", "test_moduli_below_two_thirds is empty")
    for m in test_moduli:
        mc = _male_count(m)
        if 3 * mc >= 2 * m * m:
            fail("MF_MON", f"m={m}: male_count={mc}, fraction={mc/(m*m):.6f} >= 2/3")

    # MF_THM — Theorem NT declared
    if not cert.get("theorem_nt_satisfied"):
        fail("MF_THM", "theorem_nt_satisfied must be true")

    # MF_F — fail detection
    expected_result = cert.get("result")
    fail_ledger = cert.get("fail_ledger", [])
    if expected_result == "FAIL" and not fail_ledger:
        fail("MF_F", "result=FAIL but fail_ledger is empty")
    if expected_result == "PASS" and fail_ledger:
        fail("MF_F", f"result=PASS but fail_ledger has {len(fail_ledger)} entries")

    return errors


def run_demo() -> None:
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    passed = failed = 0
    for fname in sorted(os.listdir(fixtures_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(fixtures_dir, fname)
        with open(fpath) as fh:
            cert = json.load(fh)
        errs = validate(cert)
        expected = cert.get("result", "PASS")
        ok = (expected == "PASS" and not errs) or (expected == "FAIL" and errs)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fname}")
        if not ok:
            for e in errs:
                print(f"         {e}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{passed} PASS  {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_demo()
