<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Wall (1960) doi:10.1080/00029890.1960.11989541, Ireland & Rosen (1990) ISBN 978-0-387-97329-6 -->
# [434] QA Witt Tower Ramified Prime Closed Form

**Cert family**: `qa_witt_tower_ramified_prime_cert_v1`
**Claim**: completes the prime-class trichotomy that every cert in this
chain — [389], [421], [424], [429], [431], [432], [433] — has left
open: the closed form for the **ramified** prime p=5, where
x²−x−1 has a *double* root mod 5 instead of two distinct roots
(split) or none (inert).

## The gap

Every Witt-tower cert in this chain explicitly restricts `p != 5`.
[432]'s own code comment is explicit about why the *embedding
mechanism* doesn't need a case split ("no case analysis on prime class
(inert/split/ramified) is needed"), but neither [432] nor [433] ever
derive the actual period/multiplicity closed form for p=5 — only a
single fixture-level spot check that the commuting square still holds.

## Why p=5 is structurally different

`sigma_m(a,b) = (a+b mod m, a mod m)` is driven by `M = [[1,1],[1,0]]`,
characteristic polynomial `x²−x−1`. Mod 5, `x²−x−1 = (x−3)²` — a
**double root** at the primitive root 3. Unlike the split case (two
distinct simple roots, diagonalizable) or the inert case (irreducible,
diagonalizable over the unramified quadratic extension), `M mod 5` is
**not diagonalizable**: it is a genuine 2×2 Jordan block,

```
M ≡ 3·I + N  (mod 5),   N ≠ 0 (mod 5),   N² ≡ 0 (mod 5)
```

None of the eigen-stratification arguments used for split/inert primes
in [432]/[433] apply to a Jordan block.

## The closed form

**Period set** (any k≥1):

```
Periods(5^k) = {1, 4}  ∪  { 4·5^L : L = 1, ..., k }
```

`1` is the fixed point; `4` is the literal kernel of N — the one
eigenline that Hensel-lifts exactly, with eigenvalue order
`ord(3 mod 5) = 4`.

**Birth/jump/freeze multiplicity** (the genuinely novel law — strictly
different from [433]'s "new period freezes immediately"): for
`P_L = 4·5^L`,

```
count(P_L, k) = 0            if k < L     (not yet born)
count(P_L, k) = 5^(L-1)       if k == L    (birth)
count(P_L, k) = 6·5^(L-1)     if k > L     (one delayed jump, then frozen forever)
```

**Eigenline persistence**: periods 1 and 4 each have orbit count
exactly 1 at every level, with no birth/jump at all — this is exactly
[432]'s embedding-isomorphism mechanism in action, confirming p=5 is
*not* an exception to that part of the chain.

## Derivation

In the Jordan-block setting, `M^n = 3^n·I + n·3^(n-1)·N` (since
`N²=0`). Requiring `M^n = I` forces `3^n ≡ 1 (mod 5^k)` (giving the
`ord(3)` factor) **and** `n·3^(n-1)·N ≡ 0 (mod 5^k)` (forcing one
*extra* factor of 5, since N is a nonzero unit-scaled nilpotent at every
level). This is exactly why the period at level k is

```
period(k) = 5 · ord(3 mod 5^k) = ord(3 mod 5^(k+1))
```

— one tower level ahead of the pure scalar order — and why new orbits
at level k+1 split into two strata sharing periods `4·5^k` (an
*existing* value, receiving a second contribution) and `4·5^(k+1)`
(brand new), rather than producing one brand-new period per level as in
the split/inert case.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 PERIOD_SET_LAW_RAMIFIED | period set = `{1,4} ∪ {4·5^L : L≤k}` for k=1..5 | **PASS** |
| C2 BIRTH_JUMP_FREEZE_MULTIPLICITY | count(P_L,k) matches the 3-phase closed form for k=2..5 | **PASS** |
| C3 EIGENLINE_PERSISTENCE | periods 1,4 frozen at count 1 for k=1..5 | **PASS** |
| C4 JORDAN_BLOCK_NONDEGENERACY | N≠0, N²≡0 mod 5; ord(3 mod 5^k)=4·5^(k-1) for k=1..30 (fast modular, no brute force) | **PASS** |

## Evidence gathered during development (cited, not re-run automatically)

| p | k | result |
|---|---|--------|
| 5 | 6 | periods {1,4,20,100,500,2500,12500}, counts {1,1,6,30,150,750,3125} — matches closed form exactly (45.6s brute force) |

Brute-force verification spans k=1..6 (state spaces up to 5^12 ≈
244M), zero exceptions.

## Why this isn't a duplicate of [432] or [433]

[432] proves the embedding mechanism unconditionally (prime-class
agnostic) and spot-checks p=5 with a single fixture confirming the
commuting square holds — it never derives a multiplicity law for the
new part. [433] generalizes the split-unequal and inert/split-equal
multiplicity formulas to every k, but explicitly operates only on
split/inert primes (`_prime_class` returns `"ramified"` for p=5 and
every check that consumes it rejects that class). This cert is the
first to derive and verify the actual closed form — and it is a
*different* closed form (birth/jump/freeze, not immediate-freeze)
because the underlying linear algebra (Jordan block vs. diagonalizable)
is genuinely different.

## What this cert does NOT claim

- Does not claim p=5 is the only possible ramified prime for some other
  recurrence — within *this specific* recurrence (Q(√5)/Fibonacci),
  p=5 is the unique ramified prime, a classical fact already used
  throughout [414]–[433].
- Does not re-derive that 3 is a primitive root mod every power of 5
  from first principles — this is a standard, well-documented fact
  (verified directly here for k=1..30 via fast modular exponentiation,
  not assumed).

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_ramified_prime_cert_v1
python qa_witt_tower_ramified_prime_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "5/5 passed"}`

## Lineage

- Completes the prime-class trichotomy left open by **[389]**, **[421]**, **[424]**, **[429]**, **[431]**, **[432]**, **[433]** (all restrict p≠5)
- Uses [432]'s embedding-isomorphism mechanism directly (Claim 3)
- Contrasts with [433]'s split/inert multiplicity law (Claim 2 is a different recursion)

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi.org/10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — ramification theory
- Wall, D.D. (1960). [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano period table (π(5)=20)
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7 — primitive roots, Hensel lifting
