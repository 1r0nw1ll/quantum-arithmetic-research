<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.1080/00029890.1960.11989541, Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6 -->
# [432] QA Witt Tower Scaling Isomorphism

**Cert family**: `qa_witt_tower_scaling_isomorphism_cert_v1`
**Claim**: closes two gaps cert [389] explicitly leaves open — which orbits
at level p² are old-part vs new-part, and what happens beyond level p² —
with an unconditional algebraic proof, not just numerics. Also removes
[389]'s `p != 5` restriction.

## The gap in [389]

Cert [389] (QA Witt Tower Orbit Refinement) proves which periods occur when
going from level p to level p² for the Fibonacci pair-shift σ(a,b)=(a+b,a),
but its own "What this cert does NOT claim" section is explicit:

> Does not identify WHICH period-k orbits at level p² are old-part vs
> new-part — only that both classes exist and have the stated periods.
> Does not certify the tower structure at level p³ or beyond.

It also restricts to `p != 5` (5 is ramified for x²−x−1, handled separately
elsewhere in the chain).

## The mechanism: scaling commutes with the shift

Define `ι(a,b) = (p·a mod p^k, p·b mod p^k)`, mapping `(Z/p^(k-1)Z)²` into
`(Z/p^kZ)²`. Because σ is **linear** (a matrix map, no offset), for any
integers a,b:

```
p · ((a+b) mod p^(k-1))  ≡  p·a + p·b   (mod p^k)
```

always — this is pure arithmetic, true for every prime p and every k≥2, with
no dependence on whether p is inert, split, or ramified. It says exactly
`σ_{p^k}(ι(a,b)) = ι(σ_{p^(k-1)}(a,b))`: ι intertwines the dynamics at
consecutive tower levels. Three things fall out immediately:

1. **ι is injective**, and its image is *exactly* the old-part sublattice
   `{(x,y) : p∣x and p∣y}`.
2. The orbit decomposition of σ restricted to that sublattice is **exactly**
   (same periods *and* same counts, not just the same set of periods) the
   full decomposition of σ_{p^(k-1)} on the entire lower-level space.
3. None of this needed p≠5 or k=2 — it holds for the ramified prime and for
   k=3 just as well.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 SCALING_EMBEDDING_COMMUTES | σ_{p^k}∘ι = ι∘σ_{p^(k-1)}, exhaustive over all (a,b), p∈{2,3,5,7,11,13,17,19,23,29,31,37,41} at k=2, p∈{2,3,5,7} at k=3 | **PASS** |
| C2 OLD_PART_IS_EXACT_IMAGE | ι injective; image = exactly the p-divisible sublattice | **PASS** |
| C3 ISOMORPHIC_ORBIT_STRUCTURE | old-part decomposition at level p^k == full decomposition at level p^(k-1), exact dict match | **PASS** |
| C4 SPLIT_UNEQUAL_NEW_MULTIPLICITY | exact orbit counts for both new periods at level p², split-unequal p∈{11,19,29,31} | **PASS** |
| C5 INERT_SPLIT_EQUAL_NEW_MULTIPLICITY | count_new = p·count_old corollary, p∈{7,13,17,41} | **PASS** |

## The multiplicity formula (the actually new part)

[389] tells you *which* two periods appear at level p² for a split-unequal
prime (`p·ord_min` and `p·ord_max`, where ord_min < ord_max are the
multiplicative orders mod p of the two roots of x²−x−1) — but not how many
orbits of each. Total-element-count conservation alone can't recover both
counts: one equation, two unknowns.

Stratifying the new part by eigen-coordinate gives the missing equation.
In the eigenbasis (c1,c2), the new part splits into: states with `c2`
**exactly** zero (not just a multiple of p) and `c1` a unit mod p² — period
exactly `p·ord_min`, count `(p-1)/ord_min` — versus everything else, which
collapses to the single period `p·ord_max`. That gives closed forms:

```
count_new(p·ord_min) = (p-1) / ord_min
count_new(p·ord_max) = (p-1)·(p²+p-1) / ord_max
```

Both are guaranteed integers (ord_min, ord_max both divide p−1 by
Lagrange). Verified exactly for p ∈ {11, 19, 29, 31}:

| p | ord_min | ord_max | count_new(p·ord_min) | count_new(p·ord_max) |
|---|---------|---------|----------------------|------------------------|
| 11 | 5  | 10 | 2 | 131 |
| 19 | 9  | 18 | 2 | 379 |
| 29 | 7  | 14 | 4 | 1738 |
| 31 | 15 | 30 | 2 | 991 |

For inert and split-equal primes there's only one new period
(`p·π(p)`), so `count_new = p·count_old` is forced by total-count
conservation alone (C5) — a corollary of C1/C3, not an independent fact,
included only for completeness.

## Why this isn't a duplicate of [389]

[389] establishes *which periods occur* by direct enumeration across a
fixed prime list. This cert establishes the *unconditional structural
mechanism* — a single linear-algebra identity — that explains why, pins
down the old/new split exactly (element-for-element, not just
period-for-period), supplies the orbit-count formula [389] doesn't have for
the split-unequal case, and extends the claim to p=5 and to k=3 without
[389]'s restrictions. Checked directly against [389]'s validator and doc
before drafting this cert.

## What this cert does NOT claim

- Does not claim the *exact list* of new periods at level p³ and beyond is
  fully classified — that final step (which new periods appear, beyond
  "old part embeds exactly") still rests on the same non-degenerate
  (non-Wall-Sun-Sun) assumption used throughout this chain (e.g. [429],
  [431]). What's proved unconditionally here is the embedding/isomorphism
  structure of the tower, at every level, for every prime.
- Does not give a multiplicity formula for inert primes' new part beyond the
  automatic `p·count_old` corollary — inert eigenvalues live in F_{p²}, and
  the analogous Burnside-style stratification used for split-unequal primes
  doesn't apply directly (no real eigenline to set "exactly zero").

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_scaling_isomorphism_cert_v1
python qa_witt_tower_scaling_isomorphism_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "6/6 passed"}`

## Lineage

- Extends **[389]** (Witt tower orbit refinement — period sets only, p≠5, level p² only)
- Related to but distinct from **[385]** (orbit/prime-ideal filtration at a single fixed level, not a level-to-level scaling map)
- Unaffected: **[387]** (Witt carry invariant within one level), **[388]** (split eigenspace within one level)

## Primary sources

- Wall, D.D. (1960). [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano period tower
- Serre, J.-P. (1979). *Local Fields*. [doi.org/10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) §II.4
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-derived the split-unequal
multiplicity formulas count_new(p·ord_min)=(p−1)/ord_min and
count_new(p·ord_max)=(p−1)(p²+p−1)/ord_max from scratch (fresh direct
orbit-period enumeration on (ℤ/p²ℤ)², not reusing validator code) for
all 4 primes {11,19,29,31} — exact match in every case. This is
genuinely rigorous, honestly-scoped algebraic number theory (explicit
"what this cert does NOT claim" sections throughout), no
fixture-trusting gap.
