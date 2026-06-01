# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Hardy+Wright 2008 ISBN 978-0-19-921986-5; Stern 1858; Brocot 1861) -->
"""Cert [294]: QA SL(2,Z) Spine.

PRIMARY CLAIM:
  The two QA increment moves L and R are the generators of SL(2,Z).
  Every primitive QA state is a unique word in {L,R}* (the Stern-Brocot tree).
  The QA Fibonacci matrix M = [[0,1],[1,1]] has M^2 = L*R in SL(2,Z).
  This is the group-theoretic backbone of the Ford circle arc [289-293].

DEFINITIONS:
  L = [[1,0],[1,1]]:  (b,e) -> (b, b+e)   [add b to e]
  R = [[1,1],[0,1]]:  (b,e) -> (b+e, e)   [add e to b]
  M = [[0,1],[1,1]]:  (b,e) -> (e, b+e)   [Fibonacci / QA T-step]

  det(L) = 1, det(R) = 1  =>  L, R in SL(2,Z)
  det(M) = -1              =>  M in GL(2,Z) but NOT SL(2,Z)
  M^2 = [[1,1],[1,2]] = L*R  in SL(2,Z)

STERN-BROCOT THEOREM:
  For every (b,e) with b,e >= 1 and gcd(b,e) = 1, there exists a UNIQUE
  word W in {L,R}* such that applying W to (1,1) gives (b,e).
  W is computed by the subtractive Euclidean algorithm:
    while (b,e) != (1,1):
        if b > e: prepend R; b -= e
        else:     prepend L; e -= b
  len(W) = number of Euclidean subtraction steps.

PELL CHAIN WORDS (from cert [289]):
  (1,1):    ''
  (3,2):    'LR'             [len 2]
  (7,5):    'RLLR'           [len 4]
  (17,12):  'LRRLLR'         [len 6]
  (41,29):  'RLLRRLLR'       [len 8]
  (99,70):  'LRRLLRRLLR'     [len 10]
  Pattern: w_{n+1} = ('LR' if n odd else 'RL') + w_n
  Descending Euclidean steps follow period-4 pattern (R,L,L,R)^inf

CONNECTION TO FORD CIRCLE ARC:
  - cert [289]: M = swap*R, M^2 = L*R, Pell chain is the sqrt(2) geodesic
  - cert [291]: M acts on (Z/9Z)^2 with order 24 = pi(9)
  - cert [293]: every shell S_k = {I=k} is an orbit under L,R words
  The orbit structure of SL(2,Z) on (Z/mZ)^2 is the REASON orbits have
  periods 24/8/1 — it is the mod-m quotient of the Stern-Brocot tree.
"""

from __future__ import annotations

from math import gcd
from typing import List, Tuple

# SL(2,Z) generators (column-vector convention)
_L = [[1, 0], [1, 1]]   # (b,e) -> (b, b+e)
_R = [[1, 1], [0, 1]]   # (b,e) -> (b+e, e)
_M = [[0, 1], [1, 1]]   # (b,e) -> (e, b+e)  [det = -1]

Matrix = List[List[int]]


def _det(A: Matrix) -> int:
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def _apply_move(move: str, b: int, e: int) -> Tuple[int, int]:
    if move == 'L':
        return b, b + e
    return b + e, e


def apply_word(word: str, b: int = 1, e: int = 1) -> Tuple[int, int]:
    """Apply word W (sequence of 'L','R') to state (b,e)."""
    for c in word:
        b, e = _apply_move(c, b, e)
    return b, e


def sb_word(b: int, e: int) -> str:
    """Stern-Brocot word for primitive (b,e): unique W s.t. W*(1,1)=(b,e)."""
    steps: List[str] = []
    while not (b == 1 and e == 1):
        if b > e:
            steps.append('R')
            b -= e
        else:
            steps.append('L')
            e -= b
    return ''.join(reversed(steps))


def euclid_steps(b: int, e: int) -> int:
    """Number of subtractive Euclidean steps to reduce (b,e) to (1,1)."""
    count = 0
    while not (b == 1 and e == 1):
        if b > e:
            b -= e
        else:
            e -= b
        count += 1
    return count


# ---------------------------------------------------------------------------
# Checks
# DET_LR     — det(L)=det(R)=1; det(M)=-1
# M_SQ       — M^2 = L*R (the QA two-step lands in SL(2,Z))
# STERN_PRIM — gcd(b,e) = 1 (precondition for SB representation)
# WORD_APPLY — applying declared word W to (1,1) gives (b,e)
# WORD_UNIQ  — Euclidean algorithm gives the same word as declared
# EUCLID_LEN — len(W) = number of Euclidean subtraction steps
# ---------------------------------------------------------------------------

_GLOBAL_CHECKED = False
_GLOBAL_OK = False


def _check_global() -> bool:
    global _GLOBAL_CHECKED, _GLOBAL_OK
    if _GLOBAL_CHECKED:
        return _GLOBAL_OK
    _GLOBAL_CHECKED = True
    det_ok = (_det(_L) == 1 and _det(_R) == 1 and _det(_M) == -1)
    msq_ok = (_mat_mul(_M, _M) == _mat_mul(_L, _R))
    _GLOBAL_OK = det_ok and msq_ok
    return _GLOBAL_OK


def validate_fixture(fixture: dict) -> dict:
    b, e = fixture["state"]
    declared_word: str = fixture["word"]

    results: dict = {}

    # Global checks (same regardless of state)
    results["DET_LR"] = _check_global()

    # STERN_PRIM
    results["STERN_PRIM"] = (gcd(b, e) == 1)

    if not results["STERN_PRIM"]:
        results["WORD_APPLY"] = False
        results["WORD_UNIQ"] = False
        results["EUCLID_LEN"] = False
        return results

    # WORD_APPLY
    applied = apply_word(declared_word)
    results["WORD_APPLY"] = (applied == (b, e))

    # WORD_UNIQ + EUCLID_LEN
    computed_word = sb_word(b, e)
    results["WORD_UNIQ"] = (declared_word == computed_word)
    results["EUCLID_LEN"] = (len(declared_word) == euclid_steps(b, e))

    return results


def self_test() -> bool:
    failures = []

    # --- Global invariants ---
    if _det(_L) != 1:
        failures.append(f"DET_L: expected 1, got {_det(_L)}")
    if _det(_R) != 1:
        failures.append(f"DET_R: expected 1, got {_det(_R)}")
    if _det(_M) != -1:
        failures.append(f"DET_M: expected -1, got {_det(_M)}")
    if _mat_mul(_M, _M) != _mat_mul(_L, _R):
        failures.append(f"M^2 != L*R: M^2={_mat_mul(_M,_M)}, L*R={_mat_mul(_L,_R)}")

    # --- Pell chain words ---
    pell = [(1, 1)]
    for _ in range(11):
        b, e = pell[-1]
        pell.append((b + 2*e, b + e))

    expected_words = ['', 'LR', 'RLLR', 'LRRLLR', 'RLLRRLLR',
                      'LRRLLRRLLR', 'RLLRRLLRRLLR', 'LRRLLRRLLRRLLR']
    for i, (b, e) in enumerate(pell[:8]):
        w = sb_word(b, e)
        if w != expected_words[i]:
            failures.append(f"Pell word [{i}] ({b},{e}): expected {expected_words[i]!r}, got {w!r}")
        if apply_word(w) != (b, e):
            failures.append(f"Pell apply [{i}]: expected ({b},{e}), got {apply_word(w)}")

    # --- Pell word pattern: w_{n+1} = prefix + w_n ---
    for i in range(1, 7):
        w_prev = expected_words[i]
        w_next = expected_words[i + 1]
        prefix = 'LR' if (i % 2 == 0) else 'RL'
        if w_next != prefix + w_prev:
            failures.append(f"Pell word pattern broken at n={i}: {w_next!r} != {prefix!r}+{w_prev!r}")

    # --- Descending Euclidean pattern for Pell: (R,L,L,R) period ---
    # Descending steps = word reversed
    for i in range(1, 6):
        b, e = pell[i]
        w = sb_word(b, e)
        descending = w[::-1]   # reverse of ascending word = descending steps
        # Every 4-char block should match (R,L,L,R)
        period = 'RLLR'
        reps = len(descending) // 4
        for j in range(reps):
            block = descending[4*j:4*j+4]
            if block != period:
                failures.append(f"Pell descending pattern [{i}] block {j}: {block!r} != {period!r}")

    # --- Round-trip for a variety of primitive pairs ---
    test_pairs = [(b, e) for b in range(1, 20) for e in range(1, 20) if gcd(b, e) == 1]
    for b, e in test_pairs:
        w = sb_word(b, e)
        if apply_word(w) != (b, e):
            failures.append(f"Round-trip failed for ({b},{e}): got {apply_word(w)}")
        # Word uniqueness: Euclidean steps match
        if len(w) != euclid_steps(b, e):
            failures.append(f"Euclid length mismatch for ({b},{e})")

    # --- Fail fixture detection ---
    fail_cases = [
        {"state": [7, 5], "word": "LRRL", "expected": "FAIL"},
        {"state": [4, 6], "word": "", "expected": "FAIL"},
    ]
    for case in fail_cases:
        checks = validate_fixture(case)
        state_checks = {k: v for k, v in checks.items()
                        if k in ("WORD_APPLY", "WORD_UNIQ", "STERN_PRIM")}
        if all(state_checks.values()):
            failures.append(f"Expected FAIL case passed: {case['state']}")

    if failures:
        for f in failures[:10]:
            print("FAIL:", f)
    return len(failures) == 0


FAMILY_ID = 294
CERT_SLUG = "qa_sl2z_spine_cert_v1"


def validate_cert_family(cert_dir) -> Tuple[bool, List[str]]:
    import json
    from pathlib import Path

    errors: List[str] = []
    fixture_dir = Path(cert_dir) / "fixtures"
    if not fixture_dir.is_dir():
        errors.append("missing fixtures/ directory")
        return False, errors

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        if all_pass == expect_pass:
            pass_count += 1
        else:
            fail_count += 1
            errors.append(
                f"fixture {path.name}: expected={'PASS' if expect_pass else 'FAIL'} "
                f"got={'PASS' if all_pass else 'FAIL'} checks={checks}"
            )

    if fail_count:
        errors.append(f"{fail_count} fixture(s) had wrong outcome")
    return fail_count == 0, errors


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="QA SL(2,Z) Spine Cert validator [294]"
    )
    parser.add_argument("cert_dir", nargs="?", default=str(Path(__file__).parent))
    parser.add_argument("--self-test", action="store_true", dest="selftest")
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)
    fixture_dir = cert_dir / "fixtures"

    if args.selftest:
        st_ok = self_test()
        fam_ok, fam_errors = validate_cert_family(cert_dir)
        fix_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
        pass_files = [f for f in fix_files if "pass_" in f.name]
        fail_files = [f for f in fix_files if "fail_" in f.name]
        errors = ([] if st_ok else ["self_test FAIL"]) + fam_errors
        payload = {
            "ok": st_ok and fam_ok,
            "family_id": FAMILY_ID,
            "slug": CERT_SLUG,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": errors,
        }
        print(json.dumps(payload, sort_keys=True))
        sys.exit(0 if payload["ok"] else 1)

    if not self_test():
        print("SELF_TEST FAIL")
        sys.exit(1)
    print("SELF_TEST PASS")

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(checks.values())
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        print(f"{'PASS' if ok else 'FAIL'} {path.name}: {checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)
