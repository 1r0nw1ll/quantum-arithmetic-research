# [496] QA-E8 Satellite Chamber Theorem

**Cert slug**: `qa_e8_satellite_chamber_cert_v1`
**Family ID**: 496
**Derived**: 2026-06-21
**Downgraded**: 2026-07-04 (3 of 11 checks retracted — see below)

## Retraction Note (2026-07-04)

`branch_and_distances()` returns an index into the *filtered list of simple
roots* — the order depends on which of the 240 globally-enumerated E8 roots
happen to be positive/simple for a given height vector, and where they fall
in that fixed enumeration — **not** a Satellite-axis index. The original
`ESC_BRANCH`, `ESC_GRANT`, and `ESC_ELEM_UNIQUE` checks all conflated
"position 0 in that filtered list" with "Satellite axis 0 = (6,3)".

Verified directly: for `h=G`, the actual branch (degree-3) simple root is
`(0,0,2,0,-2,0,0,0)` — a Type-1 root touching **axes 2 and 4** (`SAT[2]=(9,3)`,
`SAT[4]=(3,6)`), not axis 0. This is not fixable by correcting the index:
7 of the 8 simple roots in this chamber are Type-2 (nonzero at all 8
coordinates simultaneously), so there is no well-defined sense in which a
single Satellite axis "is" the branch node or sits "at distance k" from it.
`ESC_BRANCH`, `ESC_GRANT`, and `ESC_ELEM_UNIQUE` are **retracted**, not
corrected, because their premise doesn't hold.

The chamber/wall-selection results below do **not** depend on this
axis-to-branch mapping (they compare root-projection signs or full
simple-root sets directly) and remain valid, independently re-verified
2026-07-04.

## Theorem Statement (narrowed)

For QA mod m=9, the Satellite orbit has 8 states with period 8, canonically
anchored at (6,3) — independently verified as the unique orbit member whose
reduced triple is the fundamental (3,4,5) (not merely one of several
satisfying C<F alone — three of the eight members do that).

For height function h = α·d² + β·e² (α,β > 0), h lies in the **same E₈
Weyl chamber as h=G** if and only if:

> **7/12 < α/β < 13/12**

The **unique isotropic** choice α = β gives α/β = 1 ∈ (7/12, 13/12).
Normalizing α = β = 1: **h = G = d² + e²** (Wildberger quadrance).

## Sub-Claims

| Check | Claim | Status |
|-------|-------|--------|
| ESC_PYTH | C² + F² = G² for all 576 QA pairs (b,e) ∈ {1..9}² | valid |
| ESC_PARITY | b+e+d+a+C+F+G ≡ b (mod 2) for all 576 pairs | valid |
| ESC_ROOTS | E₈ root system: 112 Type-1 (±2 entries) + 128 Type-2 (±1, even −) = 240 | valid |
| ESC_GRAM | G-chamber simple system Cartan matrix det = 1 | valid |
| ESC_WALL_LOWER | WALL_LOWER·G = +45 > 0 → lower bound α/β > 7/12 | valid |
| ESC_WALL_UPPER | WALL_UPPER·G = −9 < 0 → upper bound α/β < 13/12 | valid |
| ESC_ISO_INTERVAL | α/β = 1 ∈ (7/12, 13/12) strictly | valid |
| ESC_G2_EXITS | G² has opposite sign on Type-2 wall roots (G-chamber ≠ G²-chamber despite same axis ordering) | valid |
| ESC_CLOSURE_NO_RESCUE | no elementary QA invariant reaches the clean per-axis chamber (see Closure section below) | valid |
| ~~ESC_BRANCH~~ | ~~h = G places (6,3) at E₈ branch node~~ | **retracted** |
| ~~ESC_GRANT~~ | ~~(3,6) [Grant LRT] is at distance 4 from branch~~ | **retracted** |
| ~~ESC_ELEM_UNIQUE~~ | ~~G is unique in {b,e,d,a,C,F,G} with branch=(6,3)~~ | **retracted** |

## Wall Roots

**Lower wall** (root `[-1,-1,+1,-1,+1,-1,+1,+1]`):
A = 108·α − 63·β > 0 → α/β > 63/108 = **7/12**

**Upper wall** (root `[+1,-1,+1,-1,−1,−1,+1,+1]`):
A = 108·α − 117·β < 0 → α/β < 117/108 = **13/12**

Both binding walls have the same A-coefficient (108), with B-values −63 (lower) and −117 (upper).
The center of the interval (in A-B space) is at −(−63−117)/(2·108) = 90/108 = 5/6 ≈ 0.833.
The isotropic point (ratio = 1) lies at 0.833 + 0.167, interior to the interval.

## G vs G² (Metric Sensitivity)

G = (90,225,153,45,117,306,261,180) and G² = (8100,50625,23409,2025,13689,93636,68121,32400) give **identical Satellite axis orderings** but are in **different Weyl chambers**: 3 Type-2 wall roots change sign:

| Root | G·r | G²·r |
|------|-----|------|
| `[-1,-1,+1,-1,+1,-1,+1,+1]` | +45 | −16767 |
| `[-1,-1,+1,+1,-1,+1,-1,+1]` | −9 | +10935 |
| `[+1,+1,+1,+1,+1,-1,-1,+1]` | +243 | −31509 |

This proves chamber selection depends on metric values, not axis order alone — making d²+e² (Wildberger quadrance) geometrically canonical.

## What Is Retracted, and Why

The following sections from the original 2026-06-21 derivation are **withdrawn** as of 2026-07-04, since they depend on the same axis-to-branch conflation described above:

- The E₈ Dynkin diagram walk through named Satellite steps ("(6,3) = branch node", "(3,6) = Grant LRT, distance 4 from branch") — the diagram shape (240 roots, Cartan determinant 1, wall bounds) is real, but attaching individual axis labels to individual diagram positions is not well-defined here.
- The "Elementary Uniqueness Selection Chain" ({b,e,d,a,C,F,G} → {C,F,G} → {F,G} → {G}) — the first two narrowing steps (strict ordering, genericity) are fine, but the final step ("correct branch") relied on the retracted branch computation.

What remains genuinely established: `G=d²+e²` selects a *specific, non-arbitrary* E8 Weyl chamber (bounded by `7/12 < α/β < 13/12`), the isotropic choice `α=β` lands inside that interval, and `G` and `G²` — despite preserving the same axis ordering — land in different chambers. That's a real, verified fact about chamber selection; it just doesn't decompose into a story about individual Satellite axes sitting at individual Dynkin-diagram positions.

## Closure of the Retracted Claim (2026-07-04)

The retraction above raised an open question: is the axis-to-branch mapping
just *miscomputed* (fixable by re-deriving the right index), or is the
underlying premise itself unrecoverable — is there *any* natural QA-derived
axis assignment that gives a well-defined per-axis Dynkin structure? This
was checked directly, not just asserted.

**Finding: the "textbook" per-axis Dynkin diagram (exactly 1 Type-2 simple
root, 7 Type-1 roots each touching exactly 2 axes, forming the classical
D₈-extended-to-E₈ chain+fork) exists — but only for height vectors with
*super-increasing* spacing** (each axis value strictly exceeding the sum of
all values assigned to previous axes). Confirmed reproducibly across three
independent super-increasing test vectors — `(1,2,4,...,128)`,
`(1,3,7,...,255)`, `(5,11,23,...,905)` — all giving Type-1=7, Type-2=1, a
genuine E8 Cartan matrix (det=1), and a well-defined branch node with
distances `[1,1,0,1,2,3,4,2]` from it. This is the chamber where an
axis-based Dynkin story would actually be legitimate.

**None of QA's seven elementary invariants reach it.** Swept `{b,e,d,a,C,F,G}`
over the Satellite orbit, both in natural orbit-sequence order and sorted
numerically: `b,e,d,a` aren't even strictly distinct across the 8 states
(fails the ordering precondition outright); `C` is degenerate (zero
projection on some root) in both orderings; `F` and `G` (raw or sorted)
all land in messy, mostly-Type-2 chambers (`F`: 2 Type-1/6 Type-2; `G`
raw: 1/7; `G` sorted: 1/7 but a *different* chamber than raw G). Sorting
by the very invariant the original theorem was built around does not
rescue it — it just lands in another messy chamber.

**Why this is a structural dead end, not a harder-to-find fix**: QA's
Satellite invariants are all bounded within roughly the same order of
magnitude across the 8 states (b,e ≤ 9, so G,F ≤ ~9²-scale quantities all
within roughly a 7x range: G_VALS spans 45–306). Super-increasing spacing
requires *exponential* separation between consecutive axis values — a
structural property no natural polynomial function of bounded QA state can
produce over only 8 terms. The clean per-axis diagram isn't something the
original derivation merely failed to find the right index for; it requires
a numeric property (exponential separation) that is incompatible with how
QA invariants are built, closing the question rather than leaving it open.

## Primary Sources

- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8.
- Bourbaki (1968). *Groupes et algèbres de Lie*, Ch. 4–6.
- Humphreys, J.E. (1972). *Introduction to Lie Algebras and Representation Theory*. Springer. ISBN 978-0-387-90053-7.

## Parents

[244] (QA–E8 Orbit Embedding), [249] (E8 Embedding Orbit Classifier), [250] (ADE Mutation Game)
