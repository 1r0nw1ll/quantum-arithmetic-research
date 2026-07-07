<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.1080/00029890.1960.11989541, Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6 -->
# [433] QA Witt Tower Recursive Refinement Law

**Cert family**: `qa_witt_tower_recursive_refinement_cert_v1`
**Claim**: closes the gap cert [432] explicitly leaves open — the exact
classification of new periods at level p³ and beyond — by generalizing
[432]'s level-p² multiplicity formula to a closed form valid at **every**
tower level, and [389]'s single level-p→p² refinement law to a recursive
law holding at **every** transition.

## The gap in [432]

Cert [432] (QA Witt Tower Scaling Isomorphism) proves the embedding
mechanism (old-part = exact image of `ι`, isomorphic to the full lower
level) unconditionally for every level k — but its own "What this cert
does NOT claim" section is explicit:

> Does not claim the exact list of new periods at level p³ and beyond is
> fully classified — that final step ... still rests on the same
> non-degenerate (non-Wall-Sun-Sun) assumption used throughout this chain.

And it only derives the split-unequal new-part multiplicity formula at
k=2.

## The generalization

**Recursive period-set law** (any k≥1, not just k=1→2 as [389] states):

```
Periods_nt(p^(k+1)) = Periods_nt(p^k)  ∪  p · Periods_nt(p^k)
```

with old-part orbit counts exactly frozen at every transition — a direct
consequence of [432]'s C1–C3 (proved unconditionally for all k) composed
with the classical fact that multiplicative order in `(Z/p^kZ)*` grows by
exactly one factor of p per tower level for a non-exceptional lift.

**Split-unequal multiplicity, every k≥2** (generalizes [432]'s k=2-only
C4):

```
count_new(p^(k-1)·ord_min) = (p-1) / ord_min                      [constant in k]
count_new(p^(k-1)·ord_max) = (p-1)·(p^k + p^(k-1) - 1) / ord_max
```

Derivation: in the eigenbasis, the `c2=0`-exactly stratum has size
`p^(k-1)·(p-1)`, and *every* unit `c1` in it shares the identical period
`p^(k-1)·ord_min` (multiplying any unit by a fixed-order element cycles
with that element's own order, independent of the unit) — so
`count = size/period = (p-1)/ord_min`, with the `p^(k-1)` factor
cancelling exactly. The complementary stratum's size follows from total
new-part size `p^(2k) - p^(2k-2)` minus the first stratum, giving the
second formula by the same division.

**Inert/split-equal closed form, every k≥2** (generalizes [432]'s C5 from
a one-step "p·count_old" corollary to an explicit formula at any level):

```
count_new(p^(k-1)·π(p)) = p^(k-1)·(p²-1) / π(p)
```

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 RECURSIVE_PERIOD_SET_LAW | period-set law holds at every transition (not just k=1→2), old counts frozen; p∈{7,11,13,17,19} at (1→2) and (2→3) | **PASS** |
| C2 SPLIT_UNEQUAL_GENERAL_K_MULTIPLICITY | closed forms hold at k=2 AND k=3 (constant-in-k confirmed); p∈{11,19} at k∈{2,3}, p∈{29,31} at k=2 | **PASS** |
| C3 INERT_SPLIT_EQUAL_GENERAL_K_MULTIPLICITY | closed form holds through k=4; p∈{7,13,17,23} at k∈{2,3}, p=7 at k=4 | **PASS** |
| C4 NONDEGENERACY_WIDE_SCAN | fast modular check (no brute force) that multiplicative order grows by exactly ×p per level — 78 split primes <1000 at k=2..10 (702 checks), 31 inert primes <300 at k=2..8 (217 checks) | **PASS**, zero exceptions |

## Evidence gathered during development (cited, not re-run automatically)

Beyond the automated self-test (17.6s), additional brute-force ground
truth was confirmed while developing this cert, at scales too slow for
routine runs:

| p | class | k | result |
|---|-------|---|--------|
| 11 | split_unequal | 4 | orbits_min=2, orbits_max=15971 — both match closed form exactly |
| 29 | split_unequal | 3 | orbits_min=4, orbits_max=50458 — match (99s) |
| 7 | inert | 5 | orbits=7203 — matches `p^4·48/16` (46.7s), one level beyond any prior check in the chain |
| 23 | inert | 3 | orbits=5819 — match (23.8s) |

Zero exceptions across every (p,k) pair tested — brute force or fast
algebraic — roughly 1,000 individual checks in total.

## Why this isn't a duplicate of [389] or [432]

[389] states the period-set law for a single transition (k=1→2) only,
without orbit counts. [432] proves the structural embedding mechanism
unconditionally for all k, and derives the multiplicity formula at k=2
only, explicitly flagging k≥3 as unclassified. This cert is the first to
state and verify: (a) the period-set law recursively at *every*
transition, (b) the multiplicity closed forms at *every* k (not just k=2),
and (c) an independent fast non-degeneracy check spanning far more
primes/levels than any brute-force enumeration in the chain could reach.
Checked directly against both [389]'s and [432]'s validators and docs
before drafting this cert.

## What this cert does NOT claim

- Does not re-derive the non-degenerate (non-Wall-Sun-Sun) Hensel-lifting
  assumption from first principles — it is the same standing hypothesis
  used throughout this chain (e.g. [429], [431]), here stress-tested far
  more widely (78 + 31 primes, up to k=10) than before, with zero
  counterexamples, but not proven impossible in general.
- Does not extend the split-unequal/inert distinction itself — prime
  classification (inert/split_equal/split_unequal/ramified) is unchanged
  from [421]/[424]/[432].

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_recursive_refinement_cert_v1
python qa_witt_tower_recursive_refinement_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "5/5 passed"}`

## Lineage

- Generalizes **[432]** (Witt tower scaling isomorphism — embedding mechanism for all k, multiplicity formula at k=2 only)
- Generalizes **[389]** (Witt tower orbit refinement — period sets at k=1→2 only)
- Unaffected: **[385]**, **[387]**, **[388]** (single-level structure, not the level-to-level scaling map)

## Primary sources

- Wall, D.D. (1960). [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano period tower
- Serre, J.-P. (1979). *Local Fields*. [doi.org/10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) §II.4
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified in fresh, separate
scripts (not reusing validator code): the recursive period-set law
Periods_nt(p^(k+1))=Periods_nt(p^k)∪p·Periods_nt(p^k) for p∈{7,11,13};
the split-unequal multiplicity formulas at k=3 for p=11 (n_min=2,
n_max=1451, both matching the closed form exactly); and the
inert/split-equal closed form at k=2,3 for p=7 (21, 147, both exact).
Genuine, carefully-scoped algebraic number theory with an honest
"standing hypothesis, not proven impossible" caveat on the underlying
non-Wall-Sun-Sun assumption. No fixture-trusting gap.
