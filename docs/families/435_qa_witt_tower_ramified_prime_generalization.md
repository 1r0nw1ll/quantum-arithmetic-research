<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Wall (1960) doi:10.2307/2309169, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Washington (1997) doi:10.1007/978-1-4612-1934-7 -->
# [435] QA Witt Tower Ramified Prime Generalization

**Cert family**: `qa_witt_tower_ramified_prime_generalization_cert_v1`
**Claim**: [434] derived the ramified-prime closed form for exactly
**one** case — p=5, the unique ramified prime of the Fibonacci
recurrence x²−x−1 (ℚ(√5)). This cert tests whether that closed form
is a genuine law or a Fibonacci-specific accident, by re-deriving and
checking it on a structurally different recurrence with **two**
ramified primes, one of which (p=2) turns out to be a real exception.

## The gap

[434] explicitly scoped its claim to a single recurrence and a single
ramified prime. Nothing in [434] tells you whether `e` (the Hensel-
lifting eigenvalue order) and `p` (the ramified prime itself) are free
parameters of a general law, or whether 4 and 5 were doing structural
work specific to Fibonacci. It also can't say anything about how two
ramified primes of the *same* discriminant interact, because Fibonacci
only has one.

## Why D=12 (x²−4x+1, M=[[4,−1],[1,0]])

Discriminant D=12=2²·3, generating ℚ(√3) — a different field from
Fibonacci's ℚ(√5) — with **two** ramified primes, 2 and 3. p=3 gives a
second odd-ramified-prime data point (testing whether [434]'s formula
is really parametric in `e,p` and not just "5 and 4 in disguise"); p=2
is the classically exceptional prime for unit-group structure, and
turned out to genuinely break the naive scaling law rather than just
confirm it.

## The generalized closed form (odd ramified primes)

For any unimodular companion matrix M with an odd ramified prime p
(double root λ₀ of the char poly mod p, `e := ord(λ₀ mod p)`):

```
Periods(p^k)        = {1, e} ∪ { e·p^L : L = 1, ..., k }
count(e·p^L, k)      = 0            if k < L     (not yet born)
count(e·p^L, k)      = p^(L-1)       if k == L    (birth)
count(e·p^L, k)      = (p+1)·p^(L-1) if k > L     (one delayed jump, frozen forever)
```

This is [434]'s formula with `5→p` and `4→e` substituted literally —
verified on D12/p=3 (e=2) **and** on a fresh re-derivation of [434]'s
own Fibonacci/p=5 case (e=4) through the identical generic code path,
confirming both that the substitution is exact and that it isn't an
artifact of re-fitting to either single case.

## p=2: the flagged exception

For D12, λ₀ mod 2 = 1, so `e = ord(1 mod 2) = 1` — the fixed point and
the eigenline periods collapse into a single bucket. Worse, the order
of M itself stalls:

```
ord(M mod 2^k) = 2^k        for k = 1, 2
ord(M mod 2^k) = 2^(k-1)    for k ≥ 3
```

one tower level short of the naive `e·p^k` doubling, verified directly
by fast modular matrix exponentiation for k=1..30 (no brute force).

**Mechanism** (checked as exact integer matrix identities, not mod-p
approximations): writing `N = M − λ₀·I`,

```
N² = 3·I     (mod nothing — exact)   at p=3   — N decouples from M (clean scalar)
N² = 2·M     (mod nothing — exact)   at p=2   — N stays coupled to M (recursive, not scalar)
```

The p=3 identity gives `ℤ[N] ≅ ℤ[√3]` exactly and licenses the one-step
binomial collapse `(λ₀I+N)ⁿ = λ₀ⁿI + n·λ₀^(n−1)·N` that drives the
odd-prime closed form above. The p=2 identity does not collapse the
same way, because N² depends on M, not on a scalar. This matches the
classical fact that `(ℤ/2^k ℤ)×` is cyclic only for k≤2 and becomes
`ℤ/2 × ℤ/2^(k-2)` (non-cyclic) for k≥3 (Washington 1997, Ch.5) — 2 is
the one prime where the unit-group shape itself changes, and that
shows up here as a one-level stall before ×2-per-level growth resumes.

## CRT cross-prime independence

Despite p=2's anomalous internal law, the two ramified primes of D12
compose by ordinary CRT with zero interaction terms:

```
ord(M mod 2^j·3^k) = lcm( ord(M mod 2^j), ord(M mod 3^k) )
```

verified for j=1..8, k=1..5 (40 pairs), exactness and minimality both
checked, fast modular exponentiation throughout. This is the genuinely
novel question this discriminant was chosen to probe — a single-
ramified-prime recurrence like Fibonacci cannot test it at all.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 PERIOD_SET_LAW_GENERALIZED | `{1,e}∪{e·p^L:L≤k}` on D12/p=3 (e=2,k=1..6) AND Fibonacci/p=5 (e=4,k=1..5) via the same generic code | **PASS** |
| C2 BIRTH_JUMP_FREEZE_LAW_GENERALIZED | `count = 0 / p^(L-1) / (p+1)·p^(L-1)` on both (M,p) pairs | **PASS** |
| C3 EIGENLINE_PERSISTENCE_GENERALIZED | periods 1,e frozen at count 1, both pairs | **PASS** |
| C4 P2_STALL_EXCEPTION | N²=3I (scalar) vs N²=2M (recursive); ord(M mod 2^k) stall verified k=1..30 | **PASS** |
| C5 CRT_CROSS_PRIME_INDEPENDENCE | ord(M mod 2^j·3^k)=lcm(...) for 40 (j,k) pairs, j=1..8, k=1..5 | **PASS** |

## Why this isn't a duplicate of [434]

[434] proves the closed form once, for p=5/Fibonacci only, and never
touches p=2 or a second discriminant. This cert re-derives [434]'s own
numbers (e.g. count=150 at p=5,k=4,L=3) as a special case of a
parametrized formula, then shows that formula breaks at p=2 in a
specific, mechanistically explained way, and confirms cross-prime
composition by CRT — three claims [434] does not and cannot make with
a single-ramified-prime recurrence.

## What this cert does NOT claim

- Does not claim every ramified prime of every unimodular recurrence
  behaves like p=3 here — only that the odd-ramified-prime law
  generalizes with `e,p` as free parameters across the two cases
  tested, and that p=2 requires a distinct, derived correction.
- Does not attempt a general proof for arbitrary discriminants; this
  is a second concrete data point plus a mechanistic explanation for
  why p=2 is different, not an exhaustive classification.

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_ramified_prime_generalization_cert_v1
python3 qa_witt_tower_ramified_prime_generalization_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "7/7 passed"}`

## Lineage

- Generalizes **[434]**'s p=5/Fibonacci-only ramified closed form to
  arbitrary odd ramified primes, with p=2 flagged as a genuine
  exception
- Uses [432]'s embedding-isomorphism mechanism directly (Claim 3 /
  eigenline persistence)
- Consistent with [433]'s split/inert multiplicity law throughout
  (disjoint prime classes, no overlap in scope)

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi.org/10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — ramification theory
- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi.org/10.2307/2309169](https://doi.org/10.2307/2309169) — Pisano-style period tables
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7 — primitive roots, Hensel lifting
- Washington, L.C. (1997). *Introduction to Cyclotomic Fields*. [doi.org/10.1007/978-1-4612-1934-7](https://doi.org/10.1007/978-1-4612-1934-7) Ch.5 — structure of `(ℤ/2^k ℤ)×`, non-cyclic for k≥3

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-implemented D=12's companion
matrix M=[[4,-1],[1,0]] from scratch and verified: the odd-ramified-
prime law ord(M mod 3^k)=2·3^k for k=1..4, the p=2 stall exception
(ord=2,4,4,8,16,32 for k=1..6 — genuinely one level short at k=3
before doubling resumes), and CRT cross-prime independence
ord(M mod 2^j·3^k)=lcm(ord mod 2^j, ord mod 3^k) for all 15 (j,k)
pairs tested. All exact matches. This is careful, honestly-scoped
work that goes out of its way to find and explain a real exception
(p=2) rather than overclaiming a universal law — exactly the kind of
rigor this audit values.
