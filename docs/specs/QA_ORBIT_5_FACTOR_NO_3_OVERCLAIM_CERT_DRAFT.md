# QA Orbit 5-Factor No-3 Overclaim Cert — Draft

> Status: **DESIGN DRAFT ONLY**. Not registered as a cert family.
> Will Dale directive 2026-05-09: capture the adjacent regime excluded
> from cert [277], do not register yet.

> Primary source: Wall, D. D. (1960). *Fibonacci series modulo m*.
> American Mathematical Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

> Secondary references: D. E. Knuth, *The Art of Computer Programming*
> Vol. 1 §1.2.8 (Fibonacci numbers); OEIS A001175 (Pisano periods).

## What this cert would certify

A boundary theorem about the divisor shortcut for QA orbit classification
in the **adjacent regime** to cert [277]: when the modulus has a 5-factor
but no 3-factor, the shortcut produces a fixed `3×3` grid of **false
satellite states** while the canonical period-based classifier finds
**zero** period-8 orbits. The cert documents this overclaim mode in a
form that an auditor can verify without re-running an empirical sweep.

## Relation to cert [277]

| Aspect | [277] (`m = 15k`) | [278] candidate (`5\|m ∧ 3∤m`) |
|---|---|---|
| Failure mode | shortcut **under-counts** satellites | shortcut **over-claims** satellites |
| Canonical satellites | 32 (period-8 orbits) | 0 (no period-8 orbits exist) |
| Shortcut satellites | 8 (the divisor cluster) | 9 (a 3×3 grid of false positives) |
| Missed (canonical − shortcut) | 32 | 0 |
| Overclaim (shortcut − canonical) | 0 | 9 |
| Structural cause | Pisano-mod-5 cycle lifts to extra period-8 orbits via mod-3 | `m // 3` is not a divisor of `m`, so the shortcut's divisor predicate picks out a misaligned 3×3 grid that has no orbit-period-8 meaning |

The two cert claims are **logically disjoint** — no modulus is in both,
since [277] requires `3 | m` and this candidate requires `3 ∤ m`. Together
they catalogue both failure modes of the divisor shortcut on 5-factor
moduli.

## Empirical sweep (2026-05-09)

### Will-specified scope

For `m ∈ M_will = {10, 20, 25, 35, 50, 100}`:

```text
m   m//3  canonical period-8  shortcut satellites  missed  overclaims
 10    3                  0                    9       0           9
 20    6                  0                    9       0           9
 25    8                  0                    9       0           9
 35   11                  0                    9       0           9
 50   16                  0                    9       0           9
100   33                  0                    9       0           9
```

### Extended sweep on 5|m ∧ 3∤m, m ≥ 10

For `m ∈ {10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85, 95, 100, 110, 115,
125, 140, 145, 155, 160, 175, 200, 205, 215, 250}` (25 verified moduli):
canonical = 0, shortcut = 9, missed = 0, overclaims = 9 in all cases.

### Surprising widening (NOT yet in cert scope)

The `5 | m` factor is **not load-bearing**. The same pattern holds for
**all** `3 ∤ m, m ≥ 7` with one exception. From the proof pass on
non-5-multiple `3∤m` moduli:

```text
m=7:   canon=0, shortcut=9,  overclaim=9
m=8:   canon=0, shortcut=15, overclaim=15   ← exception, see below
m=11:  canon=0, shortcut=9,  overclaim=9
m=13:  canon=0, shortcut=9,  overclaim=9
m=14:  canon=0, shortcut=9,  overclaim=9
m=16:  canon=0, shortcut=9,  overclaim=9
m=17:  canon=0, shortcut=9,  overclaim=9
m=19:  canon=0, shortcut=9,  overclaim=9
m=22:  canon=0, shortcut=9,  overclaim=9
m=23:  canon=0, shortcut=9,  overclaim=9
m=26:  canon=0, shortcut=9,  overclaim=9
m=28:  canon=0, shortcut=9,  overclaim=9
m=29:  canon=0, shortcut=9,  overclaim=9
...
```

So the `5 | m ∧ 3 ∤ m` framing is one slice of the broader `3 ∤ m, m ≥ 7`
family (with `m = 8` as an outlier). This raises the scope question: is
[278] the **5-specific slice** Will originally framed, or the **broader
`3 ∤ m` family**?

#### Why m=8 is an exception

When `m = 8`, `m // 3 = 2`. The shortcut predicate `2 | b ∧ 2 | e` picks
out **all even pairs** in `{1..8}^2`, which is `{2, 4, 6, 8}^2 = 16
pairs`. Singularity `(8, 8)` is in this grid (since `2 | 8`), so the
shortcut declares `16 − 1 = 15` false satellites. For larger `3 ∤ m`,
`m // 3 ≥ 3` and the grid `{m//3, 2(m//3), 3(m//3)}` has only 3 distinct
multiples in `{1..m}` (because `4(m//3) > m` when `m // 3 ≥ ⌈m/4⌉`,
which holds for all `m ≥ 6` except `m = 8`).

## Claim candidate (narrow, falsifiable)

The cert can be drafted at one of two scopes. **Scope A** is what Will
originally specified; **Scope B** is the empirically broader form.

### Scope A — Will's spec: `5 | m ∧ 3 ∤ m`

For `m ∈ M_5no3 = {10, 20, 25, 35, 50, 100}` (Will's tested set, plus
optionally 19 more verified moduli `{40, 55, 65, 70, 80, 85, 95, 110,
115, 125, 140, 145, 155, 160, 175, 200, 205, 215, 250}`):

```text
canonical_satellite_count(m)  = 0
shortcut_satellite_count(m)   = 9
missed(m)                     = 0
overclaim(m)                  = 9

The 9 overclaim pairs are exactly the 3×3 grid:
  { (a · m//3, b · m//3) : a, b ∈ {1, 2, 3} }
which is disjoint from the singularity (m, m) when 3 ∤ m.
```

### Scope B — broader `3 ∤ m, m ≥ 7, m ≠ 8`

For `m ∈ {7, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, ...}` (every
`3∤m` with `m ≥ 7` except `m = 8`):

```text
canonical_satellite_count(m)  = 0
shortcut_satellite_count(m)   = 9
overclaim(m)                  = 9
```

Plus a separate single-fixture sub-claim for `m = 8`:
```text
canonical_satellite_count(8)  = 0
shortcut_satellite_count(8)   = 15
overclaim(8)                  = 15
```

The 5-factor framing has no structural role here.

## Fixture plan (Scope A — Will's spec)

### PASS fixtures (6)

1. `pass_m10.json` — `m = 10`, expected_overclaims = 9, grid = `{3,6,9}^2`.
2. `pass_m20.json` — `m = 20`, expected_overclaims = 9, grid = `{6,12,18}^2`.
3. `pass_m25.json` — `m = 25`, expected_overclaims = 9, grid = `{8,16,24}^2`.
4. `pass_m35.json` — `m = 35`, expected_overclaims = 9, grid = `{11,22,33}^2`.
5. `pass_m50.json` — `m = 50`, expected_overclaims = 9, grid = `{16,32,48}^2`.
6. `pass_m100.json` — `m = 100`, expected_overclaims = 9, grid = `{33,66,99}^2`.

### FAIL fixtures (3)

1. `fail_wrong_overclaim.json` — declares `m = 10` overclaim = 4 instead of 9.
2. `fail_claims_canonical_sat.json` — fixture asserts canonical satellite
   count > 0 at a `5|m, 3∤m` modulus where reality is 0.
3. `fail_treats_overclaim_as_missed.json` — fixture transposes the failure
   mode (declares `expected_missed = 9, expected_overclaim = 0`).

## Validator outline

```python
from qa_orbit_rules import orbit_family, orbit_family_divisor_shortcut


def check_fixture(fixture):
    m = fixture["modulus"]
    expected_overclaims = fixture["expected_overclaims"]
    expected_canonical = fixture.get("expected_canonical_satellites", 0)

    canon_count = 0
    overclaims = 0
    overclaim_pairs = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            c = orbit_family(b, e, m)
            s = orbit_family_divisor_shortcut(b, e, m)
            if c == "satellite":
                canon_count += 1
            if s == "satellite" and c != "satellite":
                overclaims += 1
                overclaim_pairs.append((b, e))

    errors = []
    if canon_count != expected_canonical:
        errors.append(
            f"OVCL_1: m={m} canonical={canon_count}, expected={expected_canonical}"
        )
    if overclaims != expected_overclaims:
        errors.append(
            f"OVCL_2: m={m} overclaim={overclaims}, expected={expected_overclaims}"
        )
    expected_grid = fixture.get("expected_overclaim_grid")
    if expected_grid is not None:
        observed = sorted(overclaim_pairs)
        expected = sorted(tuple(p) for p in expected_grid)
        if observed != expected:
            errors.append(
                f"OVCL_3: m={m} overclaim grid {observed} != expected {expected}"
            )
    return errors
```

Checks: **OVCL_1** (canonical = 0), **OVCL_2** (overclaim count matches),
**OVCL_3** (overclaim pairs match the declared 3×3 grid), **SRC**
(`mapping_protocol_ref.json` present), **F** (every FAIL fixture declares
`expected_fail_type` and the validator observes that mode actually fires).

## Non-claims

- Does **not** claim the failure mode is 5-factor-specific. The empirical
  data shows the 9-overclaim pattern extends to `3 ∤ m, m ≥ 7, m ≠ 8`
  regardless of 5-factor. The Will-specified scope `5 | m ∧ 3 ∤ m` is
  one slice of a broader family.
- Does **not** cover `m = 8` (shortcut overclaim = 15, not 9). If [278]
  is broadened to `3 ∤ m, m ≥ 6`, `m = 8` should be a separate fixture
  with its own expected count.
- Does **not** cover `m ∈ {1, 2, 3, 4, 5}`. These moduli are degenerate
  for the divisor shortcut: `m//3 ∈ {0, 0, 1, 1, 1}`, so the shortcut
  classifies almost everything as satellite. The cert assumes `m ≥ 6`.
- Does **not** modify `qa_orbit_rules.py`. The canonical replacement is
  locked at commit `e7b2af0`; this cert documents one of the shortcut's
  failure modes.
- Does **not** prescribe a fix. The fix is already done at the code level.

## Recommended scope decision

**Three options** for [278] scope:

1. **Scope A (Will's spec)** — `5 | m ∧ 3 ∤ m, m ≥ 10`. Claim is exact;
   bounded to a sub-family that empirically has uniform behavior.
   Pro: matches the ask, separates cleanly from [277]. Con: framing
   suggests 5-factor is causal when it isn't.
2. **Scope B (broader `3 ∤ m`)** — `3 ∤ m, m ≥ 7, m ≠ 8`. Claim is
   structurally honest; covers the actual failure mode of the shortcut
   in this regime. Pro: tighter theorem. Con: requires a separate
   single-fixture sub-claim for `m = 8`.
3. **Scope A first, Scope B as v2** — register Scope A as [278] now (it
   matches Will's directive), then surface the broader form as a v2
   amendment after a wider sweep confirms `m ∉ {6, 8}` are the only
   exceptions.

If the goal is to close the failure surface of the divisor shortcut
cleanly, **Scope B** is the right cert claim and `5 | m` is incidental.
If the goal is to keep the cert chain readable as
`[277] = "undercount on 15k"` paired with `[278] = "overclaim on 5|m, 3∤m"`,
**Scope A** is the right framing.

## Recommended next proof step

Whichever scope is chosen, the structural derivation of the overclaim
count is the same:

1. Show that for `3 ∤ m` and `m ≥ 6`, the multiples of `m // 3` in
   `{1..m}` are exactly `{m // 3, 2(m // 3), 3(m // 3)}` (three elements,
   since `4(m // 3) > m`).
2. Show that the 3×3 grid `{m//3, 2(m//3), 3(m//3)}^2` contains 9 pairs
   distinct from the singularity `(m, m)` (since `3(m//3) < m` when
   `3 ∤ m`).
3. Show that none of these 9 pairs lies on a period-8 qa_step orbit
   (because qa_step's period-8 orbits at composite `m` arise from the
   mod-3 Pisano structure, which is absent when `3 ∤ m`).
4. The `m = 8` exception falls out of step 1: at `m = 8, m//3 = 2`, the
   multiples are `{2, 4, 6, 8} = 4` elements, not 3, so the grid is
   `4 × 4 - 1 = 15` (subtracting the singularity overlap at `(8, 8)`).

A successful step 3 promotes the cert from empirical to structural.

## Readiness assessment

The draft is **ready to promote** under either scope:

- Schema: 5 required fields (modulus, expected_canonical_satellites,
  expected_overclaims, optional expected_overclaim_grid, optional
  expected_fail_type).
- Validator: stdlib-only; recomputes both classifiers; cheap (max `m^2 ≤
  10000` iterations per PASS fixture for `m ≤ 100`).
- Mapping protocol: cite Wall 1960 DOI for Pisano framing context.
- Family ID: 278 is next free (276 and 277 registered).
- No README — the formal-publication gate triggers on README.md under
  `qa_alphageometry_ptolemy/`. SPEC.md only.

## References

- Wall, D. D. (1960). Fibonacci series modulo m. *American Mathematical
  Monthly* 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.
- D. E. Knuth, *The Art of Computer Programming* Vol. 1, §1.2.8.
- OEIS A001175 — Pisano periods `π(m)`.
- `qa_orbit_rules.py` (post-commit `e7b2af0`) — canonical orbit_family +
  orbit_family_divisor_shortcut.
- `experiments/qa_ml/03_gnn_modulus_sweep.py` — surfacing sweep that
  motivated the broader investigation.
- `docs/specs/QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT_DRAFT.md` — sister
  draft for the under-count regime.
- Cert family [277]
  (`qa_alphageometry_ptolemy/qa_orbit_pisano_5_factor_boundary_cert_v1/`)
  — the registered under-count cert this draft pairs with.
