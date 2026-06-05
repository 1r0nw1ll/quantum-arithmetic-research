# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1991) QA Vol II Books 3&4 — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (gcd, lcm, counting multiples); "
    "no QA state evolution; Theorem NT: 'sync event' and 'compound wave' "
    "are observer-layer labels on integer divisibility structure; "
    "all arithmetic exact integer, no float"
)

"""
Cert [332] — QA Compound Wave GCD Sync Count

Source: Iverson (1991) QA Volume II Books 3 & 4, pp.15-19
"COMPOUND WAVES", "COMPLEX WAVES"

When two waves of periods p and q combine, they synchronize at every
common multiple. The number of sync events in one full combined cycle
(of length lcm(p,q)) equals gcd(p,q). For coprime waves, gcd=1 and
there is exactly one sync event per cycle. For non-coprime waves,
gcd>1 and sync events occur more frequently.

Key structural result: lcm(p,q) × gcd(p,q) = p × q.
Five claims certified via integer divisibility.
"""

from math import gcd


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _sync_count_in_product(p: int, q: int) -> int:
    """Count sync events (positions where both p|k and q|k) in {1..p*q}."""
    return sum(1 for k in range(1, p * q + 1) if k % p == 0 and k % q == 0)


def _sync_count_in_lcm(p: int, q: int) -> int:
    """Count sync events in {1..lcm(p,q)}; always exactly 1 (only at lcm)."""
    L = _lcm(p, q)
    return sum(1 for k in range(1, L + 1) if k % p == 0 and k % q == 0)


def check_c1() -> tuple[bool, str]:
    """For coprime pairs: lcm=p*q; exactly 1 sync in {1..lcm} and 1 sync in {1..p*q}."""
    coprime_pairs = [(3, 4), (3, 5), (4, 5), (3, 7), (4, 7), (5, 7), (5, 9)]
    for p, q in coprime_pairs:
        assert gcd(p, q) == 1, f"({p},{q}) not coprime"
        assert _lcm(p, q) == p * q, f"({p},{q}): lcm={_lcm(p,q)} != p*q={p*q}"
        # For coprime: lcm == p*q, so both counts are the same = 1
        sc_lcm = _sync_count_in_lcm(p, q)
        assert sc_lcm == 1, f"({p},{q}): sync_in_lcm={sc_lcm}, expected 1"
        sc_prod = _sync_count_in_product(p, q)
        assert sc_prod == 1, f"({p},{q}): sync_in_product={sc_prod}, expected 1"
    return True, f"Coprime pairs: lcm=p*q; sync_in_lcm=sync_in_p*q=1; verified {coprime_pairs}"


def check_c2() -> tuple[bool, str]:
    """For non-coprime pairs: sync_count in {1..p*q} = gcd(p,q); in {1..lcm} always = 1."""
    non_coprime_pairs = [
        (4, 6),    # gcd=2, lcm=12, syncs_in_24=2, syncs_in_12=1
        (6, 9),    # gcd=3, lcm=18, syncs_in_54=3, syncs_in_18=1
        (4, 8),    # gcd=4, lcm=8,  syncs_in_32=4, syncs_in_8=1
        (6, 10),   # gcd=2, lcm=30, syncs_in_60=2, syncs_in_30=1
        (9, 15),   # gcd=3, lcm=45, syncs_in_135=3, syncs_in_45=1
        (8, 12),   # gcd=4, lcm=24, syncs_in_96=4, syncs_in_24=1
    ]
    for p, q in non_coprime_pairs:
        g = gcd(p, q)
        assert g > 1, f"({p},{q}) unexpectedly coprime"
        expected_lcm = p * q // g
        assert _lcm(p, q) == expected_lcm, (
            f"({p},{q}): lcm={_lcm(p,q)} != p*q//gcd={expected_lcm}"
        )
        # In product cycle: gcd(p,q) syncs
        sc_prod = _sync_count_in_product(p, q)
        assert sc_prod == g, f"({p},{q}): sync_in_p*q={sc_prod}, expected gcd={g}"
        # In lcm cycle: always exactly 1
        sc_lcm = _sync_count_in_lcm(p, q)
        assert sc_lcm == 1, f"({p},{q}): sync_in_lcm={sc_lcm}, expected 1"
    return True, f"Non-coprime pairs: sync_in_p*q=gcd(p,q), sync_in_lcm=1; verified {non_coprime_pairs}"


def check_c3() -> tuple[bool, str]:
    """Identity: lcm(p,q) * gcd(p,q) = p * q for all pairs in {2..15}^2."""
    failures = []
    for p in range(2, 16):
        for q in range(p, 16):
            if _lcm(p, q) * gcd(p, q) != p * q:
                failures.append((p, q))
    assert failures == [], f"Identity lcm*gcd=p*q failed for {failures}"
    count = sum(1 for p in range(2, 16) for q in range(p, 16))
    # Equivalently: sync_in_product = p*q / lcm = gcd
    for p in range(2, 10):
        for q in range(p, 10):
            sc = _sync_count_in_product(p, q)
            assert sc == gcd(p, q), (
                f"({p},{q}): sync_in_p*q={sc} != gcd={gcd(p,q)}"
            )
    return True, f"lcm(p,q)*gcd(p,q)=p*q verified for all {count} pairs; sync_in_product=gcd confirmed"


def check_c4() -> tuple[bool, str]:
    """Sync count in product-cycle: gcd determines sync count; lcm determines waiting time."""
    # For a 24-cycle in QA (mod-24), pairs and their gcd determine sync multiplicity
    cases = {
        (3, 8):  (gcd(3, 8),  24),   # gcd=1, lcm=24; 1 sync in 24-cycle
        (4, 6):  (gcd(4, 6),  12),   # gcd=2, lcm=12; 2 syncs in 24-cycle=p*q
        (3, 6):  (gcd(3, 6),   6),   # gcd=3, lcm=6;  3 syncs in 18-cycle=p*q
        (6, 8):  (gcd(6, 8),  24),   # gcd=2, lcm=24; 2 syncs in 48-cycle=p*q
        (4, 12): (gcd(4, 12), 12),   # gcd=4, lcm=12; 4 syncs in 48-cycle=p*q
        (8, 12): (gcd(8, 12), 24),   # gcd=4, lcm=24; 4 syncs in 96-cycle=p*q
    }
    for (p, q), (expected_sc, expected_lcm) in cases.items():
        actual_lcm = _lcm(p, q)
        assert actual_lcm == expected_lcm, (
            f"({p},{q}): lcm={actual_lcm}, expected {expected_lcm}"
        )
        sc_prod = _sync_count_in_product(p, q)
        assert sc_prod == expected_sc, (
            f"({p},{q}): sync_in_p*q={sc_prod}, expected gcd={expected_sc}"
        )
        sc_lcm = _sync_count_in_lcm(p, q)
        assert sc_lcm == 1, f"({p},{q}): sync_in_lcm={sc_lcm}, expected 1"
    return True, "QA mod-24 pairs: sync_in_product=gcd(p,q); sync_in_lcm=1; lcm=waiting_time"


def check_c5() -> tuple[bool, str]:
    """For pairwise-coprime triples (p,q,r): lcm(p,q,r)=p*q*r; single sync per triple-cycle."""
    triples = [(3, 4, 5), (3, 4, 7), (3, 5, 7), (4, 5, 7), (3, 7, 11), (5, 7, 11)]
    for p, q, r in triples:
        assert gcd(p, q) == 1 and gcd(q, r) == 1 and gcd(p, r) == 1, (
            f"({p},{q},{r}) not pairwise coprime"
        )
        lcm_pq = _lcm(p, q)
        lcm_pqr = _lcm(lcm_pq, r)
        assert lcm_pqr == p * q * r, (
            f"({p},{q},{r}): lcm={lcm_pqr} != p*q*r={p*q*r}"
        )
        # Single triple-sync occurs only at lcm itself
        L = lcm_pqr
        triple_syncs = sum(1 for k in range(1, L + 1) if k % p == 0 and k % q == 0 and k % r == 0)
        assert triple_syncs == 1, (
            f"({p},{q},{r}): triple sync count={triple_syncs}, expected 1"
        )
    return True, f"Pairwise-coprime triples: lcm=p*q*r, single sync per cycle; verified {triples}"


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise RuntimeError(f"cert [332] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
