# Primary source: Reyes Jiménez A.E. (2025) arXiv:2606.02621; Wall D.D. (1960) doi:10.2307/2309169; Wildberger N.J. (2005) ISBN 978-0-9757492-0-8
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Collatz-Fibonacci spectral: "
    "sigma(b,e)=(e,((b+e-1)%m)+1); Singularity (9,9) period-1 fixed point; "
    "non-Singularity period set {8,24}; F(k) mod 9 via integer recurrence; "
    "Pisano witness: F(24)=0,F(25)=1 mod 9, pi(9)=24 minimal; decoy k=12 rejected); "
    "Theorem NT: phi=lim F(n+1)/F(n) is a continuous observer projection — "
    "phi never enters QA dynamics; no float state, no continuous observer in QA layer"
)
"""
QA Collatz-Fibonacci Spectral Cert [502]

Claim: QA mod-9 exhibits the same three structural features that enable the
Collatz-Fibonacci spectral result (Reyes Jiménez arXiv:2606.02621):

  CF_1  QA Singularity (9,9) is the unique period-1 fixed point under sigma.
  CF_2  All 80 non-Singularity pairs have period in {8, 24} — the Pisano
        hierarchy for m=9 — the Fibonacci-attractor orbits.
  CF_3  Pisano integer witness: F(24) ≡ 0 (mod 9), F(25) ≡ 1 (mod 9), and
        the only k in {1,...,23} with F(k) ≡ 0 (mod 9) is k=12, whose
        successor F(13) ≡ 8 ≠ 1, proving pi(9) = 24 is minimal.

Collatz-QA structural analogy (Reyes Jiménez §2):
  - Vertex 4 in Collatz mod-6 graph = absorbing fixed point outside Fibonacci
    attractor (spectral radius phi when removed) <-> QA Singularity (9,9)
  - G'={1,2,4,5} absorbing SCC <-> QA Cosmos + Satellite orbits
  - phi = observer projection over discrete F(m+1) counts (Theorem NT: phi
    is NOT a QA state; the primary fact is pi(9) = 24, a pure integer)

Schema: QA_COLLATZ_FIBONACCI_CERT.v1
"""

import json
import sys
from pathlib import Path

SCHEMA = "QA_COLLATZ_FIBONACCI_CERT.v1"
M = 9
_EXPECTED_SINGULARITY = (9, 9)
_EXPECTED_NON_SING_PERIOD_SET = frozenset({8, 24})
_EXPECTED_PISANO = 24
_EXPECTED_DECOY_K = 12       # F(12) ≡ 0 mod 9, but F(13) ≡ 8 ≠ 1
_EXPECTED_DECOY_NEXT = 8     # F(13) mod 9


def _qa_step(b, e, m):
    return e, ((b + e - 1) % m) + 1


def _qa_period(b0, e0, m):
    b, e = b0, e0
    for k in range(1, m * m * m + 2):
        b, e = _qa_step(b, e, m)
        if b == b0 and e == e0:
            return k
    raise RuntimeError(f"No period found for ({b0},{e0}) mod {m}")


def _fib_mod(k, m):
    """F(k) mod m using int arithmetic (no float)."""
    if k == 0:
        return 0
    a, b = 0, 1
    for _ in range(k - 1):
        a, b = b, (a + b) % m
    return b


def _pisano_min(m):
    """Smallest k>0 where F(k)≡0 and F(k+1)≡1 (mod m)."""
    a, b = 0, 1
    for k in range(1, m * m * 6 + 2):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return k
    raise RuntimeError(f"Pisano period not found for m={m}")


def _check_fixture(data):
    errors = []

    # SRC
    if data.get("schema_version") != SCHEMA:
        errors.append(f"SRC: expected schema_version={SCHEMA!r}, got {data.get('schema_version')!r}")

    m = data.get("modulus", M)
    if m != M:
        errors.append(f"MOD: expected modulus={M}, got {m}")

    # CF_1 — Singularity is unique period-1 fixed point
    declared_sing_period = data.get("singularity_period")
    computed_sing_period = _qa_period(*_EXPECTED_SINGULARITY, M)
    if declared_sing_period != computed_sing_period:
        errors.append(
            f"CF_1: declared singularity_period={declared_sing_period} "
            f"but computed={computed_sing_period}"
        )

    # CF_2 — Non-Singularity period set
    declared_period_set = set(data.get("non_singularity_period_set", []))
    computed_period_set = set()
    for b in range(1, M + 1):
        for e in range(1, M + 1):
            if (b, e) == _EXPECTED_SINGULARITY:
                continue
            computed_period_set.add(_qa_period(b, e, M))
    if declared_period_set != computed_period_set:
        errors.append(
            f"CF_2: declared period_set={sorted(declared_period_set)} "
            f"but computed={sorted(computed_period_set)}"
        )
    elif declared_period_set != _EXPECTED_NON_SING_PERIOD_SET:
        errors.append(
            f"CF_2: computed period_set={sorted(declared_period_set)} "
            f"expected {sorted(_EXPECTED_NON_SING_PERIOD_SET)}"
        )

    # CF_3 — Pisano integer witness
    f24_mod9 = data.get("f24_mod9")
    f25_mod9 = data.get("f25_mod9")
    pisano_period = data.get("pisano_period")
    decoy_k = data.get("decoy_k")
    decoy_next_mod9 = data.get("decoy_next_mod9")

    computed_f24 = _fib_mod(24, M)
    computed_f25 = _fib_mod(25, M)
    computed_pisano = _pisano_min(M)
    computed_decoy_next = _fib_mod(13, M)

    if f24_mod9 != computed_f24:
        errors.append(f"CF_3a: declared f24_mod9={f24_mod9} but computed={computed_f24}")
    if f25_mod9 != computed_f25:
        errors.append(f"CF_3b: declared f25_mod9={f25_mod9} but computed={computed_f25}")
    if pisano_period != computed_pisano:
        errors.append(f"CF_3c: declared pisano_period={pisano_period} but computed={computed_pisano}")
    if decoy_k != _EXPECTED_DECOY_K:
        errors.append(f"CF_3d: declared decoy_k={decoy_k} expected {_EXPECTED_DECOY_K}")
    if decoy_next_mod9 != computed_decoy_next:
        errors.append(
            f"CF_3e: declared decoy_next_mod9={decoy_next_mod9} "
            f"but computed={computed_decoy_next}"
        )

    # CF_3 minimality: decoy_k must satisfy F(decoy_k)≡0 but F(decoy_k+1)≢1
    if decoy_k is not None:
        dk_val = _fib_mod(decoy_k, M)
        dk1_val = _fib_mod(decoy_k + 1, M)
        if dk_val != 0:
            errors.append(f"CF_3f: decoy_k={decoy_k} should have F(k)≡0 mod 9, got {dk_val}")
        if dk1_val == 1:
            errors.append(
                f"CF_3g: decoy_k={decoy_k} should have F(k+1)≢1 mod 9, got {dk1_val} "
                "(this would make it a valid Pisano period, not a decoy)"
            )

    # Fail-ledger check
    if "fail_ledger" in data:
        fl = data["fail_ledger"]
        if not isinstance(fl, list) or not all(isinstance(s, str) for s in fl):
            errors.append("F: fail_ledger must be a list of strings")

    return errors


def _run_self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    results = {}

    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        expected_pass = fixture_path.stem.startswith("pass_")
        with open(fixture_path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        passed = len(errs) == 0
        ok = passed == expected_pass
        results[fixture_path.name] = {
            "expected": "PASS" if expected_pass else "FAIL",
            "got": "PASS" if passed else "FAIL",
            "ok": ok,
            "errors": errs,
        }

    all_ok = all(v["ok"] for v in results.values())
    return {"ok": all_ok, "fixtures": results}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        result = _run_self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        if errs:
            print(json.dumps({"ok": False, "errors": errs}, indent=2))
            sys.exit(1)
        print(json.dumps({"ok": True}, indent=2))
        sys.exit(0)

    result = _run_self_test()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
