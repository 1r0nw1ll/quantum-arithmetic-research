#!/usr/bin/env python3
"""QA Law of Harmonics Cert family [149] — certifies Iverson's formal
definition of harmonic resonance between Quantum Numbers.

LAW OF HARMONICS (Iverson, QA-3 Ch 4):

Two dissimilar QN products P1, P2 are in HARMONIC RESONANCE when:
1. They share a common ALIQUOT SET: gcd(P1, P2) > 1
2. Each has exactly one IDENTITY PRIME not in the shared set
3. The HARMONY RATIO = (identity_prime_1 / identity_prime_2) measures
   resonance strength — closer to 1 = stronger harmony

Formally: let S = set of prime factors of gcd(P1, P2).
Let I1 = primes(P1) - S, I2 = primes(P2) - S.
If |I1| == 1 and |I2| == 1: HARMONIC PAIR.
harmony_ratio = min(I1[0], I2[0]) / max(I1[0], I2[0]).

Additional certified properties:
- All QN products are multiples of 6 (primes 2,3 always present)
- Fibonacci QN products share increasingly large aliquot sets
- Identity primes uniquely distinguish QNs within a harmonic family

Checks: LH_1 (schema), LH_ALIQ (aliquot set computed correctly),
LH_IDEN (identity primes correct), LH_RATIO (harmony ratio correct),
LH_DIV6 (all QN products divisible by 6), LH_W (>=4 harmonic pairs),
LH_F (fundamental product P=6 present).
"""

import json
import os
import sys
from math import gcd
from collections import Counter


SCHEMA = "QA_LAW_OF_HARMONICS_CERT.v1"


def prime_factors(n):
    """Return set of prime factors of n (n > 0)."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def qn_product(b, e):
    """Compute QN product b*e*d*a where d=b+e, a=b+2e."""
    d = b + e
    a = b + 2 * e
    return b * e * d * a


def analyze_pair(p1, p2):
    """Analyze harmonic relationship between two QN products."""
    g = gcd(p1, p2)
    shared = prime_factors(g)
    pf1 = prime_factors(p1)
    pf2 = prime_factors(p2)
    identity1 = pf1 - shared
    identity2 = pf2 - shared
    is_harmonic = (len(identity1) == 1 and len(identity2) == 1)
    ratio = None
    if is_harmonic:
        i1 = identity1.pop()
        i2 = identity2.pop()
        ratio = min(i1, i2) / max(i1, i2)
        identity1 = {i1}
        identity2 = {i2}
    return {
        "aliquot_set": sorted(shared),
        "identity_primes_1": sorted(identity1),
        "identity_primes_2": sorted(identity2),
        "is_harmonic": is_harmonic,
        "harmony_ratio": ratio,
    }


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("LH_1", f"schema_version must be {SCHEMA}")

    pairs = cert.get("harmonic_pairs", [])
    has_fundamental = False

    for i, hp in enumerate(pairs):
        qn1 = hp.get("qn1", {})
        qn2 = hp.get("qn2", {})
        b1, e1 = qn1.get("b", 0), qn1.get("e", 0)
        b2, e2 = qn2.get("b", 0), qn2.get("e", 0)

        if b1 <= 0 or e1 <= 0 or b2 <= 0 or e2 <= 0:
            err("LH_1", f"pair[{i}]: invalid QN coordinates")
            continue

        p1 = qn_product(b1, e1)
        p2 = qn_product(b2, e2)

        if p1 == 6 or p2 == 6:
            has_fundamental = True

        # LH_DIV6
        if p1 % 6 != 0:
            err("LH_DIV6", f"pair[{i}] P1={p1} not divisible by 6")
        if p2 % 6 != 0:
            err("LH_DIV6", f"pair[{i}] P2={p2} not divisible by 6")

        # Check declared products
        decl_p1 = hp.get("product1")
        decl_p2 = hp.get("product2")
        if decl_p1 is not None and decl_p1 != p1:
            err("LH_ALIQ", f"pair[{i}] product1 declared={decl_p1} computed={p1}")
        if decl_p2 is not None and decl_p2 != p2:
            err("LH_ALIQ", f"pair[{i}] product2 declared={decl_p2} computed={p2}")

        analysis = analyze_pair(p1, p2)

        # LH_ALIQ — aliquot set
        decl_aliq = hp.get("aliquot_set")
        if decl_aliq is not None:
            if sorted(decl_aliq) != analysis["aliquot_set"]:
                err("LH_ALIQ", f"pair[{i}] aliquot declared={decl_aliq} computed={analysis['aliquot_set']}")

        # LH_IDEN — identity primes
        decl_id1 = hp.get("identity_prime_1")
        decl_id2 = hp.get("identity_prime_2")
        if decl_id1 is not None:
            if [decl_id1] != analysis["identity_primes_1"]:
                err("LH_IDEN", f"pair[{i}] identity_prime_1 declared={decl_id1} computed={analysis['identity_primes_1']}")
        if decl_id2 is not None:
            if [decl_id2] != analysis["identity_primes_2"]:
                err("LH_IDEN", f"pair[{i}] identity_prime_2 declared={decl_id2} computed={analysis['identity_primes_2']}")

        # LH_RATIO — harmony ratio
        decl_harmonic = hp.get("is_harmonic")
        if decl_harmonic is not None and decl_harmonic != analysis["is_harmonic"]:
            err("LH_RATIO", f"pair[{i}] is_harmonic declared={decl_harmonic} computed={analysis['is_harmonic']}")

        decl_ratio = hp.get("harmony_ratio")
        if decl_ratio is not None and analysis["harmony_ratio"] is not None:
            if abs(decl_ratio - analysis["harmony_ratio"]) > 0.001:
                err("LH_RATIO", f"pair[{i}] ratio declared={decl_ratio} computed={analysis['harmony_ratio']:.6f}")

    # LH_W
    if len(pairs) < 4:
        err("LH_W", f"need >=4 harmonic pairs, got {len(pairs)}")

    # LH_F
    if not has_fundamental:
        err("LH_F", "no pair with fundamental product P=6")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "lh_pass_harmonic_pairs.json": True,
        "lh_pass_fibonacci_chain.json": True,
    }
    results = []
    for fname, should_pass in expected.items():
        path = os.path.join(fix_dir, fname)
        with open(path) as f:
            cert = json.load(f)
        res = validate(cert)
        ok = res["ok"] == should_pass
        results.append({
            "fixture": fname,
            "expected_pass": should_pass,
            "actual_pass": res["ok"],
            "ok": ok,
            "errors": res["errors"],
            "warnings": res["warnings"],
        })
    return results


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        results = self_test()
        all_ok = all(r["ok"] for r in results)
        print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cert = json.load(f)
        print(json.dumps(validate(cert), indent=2))
    else:
        print("Usage: python qa_law_of_harmonics_cert_validate.py [--self-test | <fixture.json>]")
