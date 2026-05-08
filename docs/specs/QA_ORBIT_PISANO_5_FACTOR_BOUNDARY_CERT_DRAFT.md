# QA Orbit Pisano 5-Factor Boundary Cert — Draft

> Status: **DESIGN DRAFT ONLY**. Not registered as a cert family. Will Dale
> directive 2026-05-08: "Do not register the family yet."

> Primary source: Wall, D. D. (1960). *Fibonacci series modulo m*. American
> Mathematical Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.
> Establishes the Pisano period π(m) of Fibonacci-mod-m, including the
> multiplicative law π(mn) = lcm(π(m), π(n)) when gcd(m, n) = 1, and the
> values π(3) = 8 and π(5) = 20 used below.

> Secondary references: D. E. Knuth, *The Art of Computer Programming*
> Vol. 1 §1.2.8 (Fibonacci numbers); OEIS A001175 (Pisano periods).

## What this cert would certify

A boundary theorem about the algebraic divisor shortcut for QA orbit
classification, surfaced by the QA-ML v2 modulus sweep (cert [276],
`experiments/qa_ml/03_gnn_modulus_sweep.py`) and addressed in the canonical
classifier patch (commit `e7b2af0` 2026-05-08, which moved the shortcut
behind the named helper `orbit_family_divisor_shortcut`).

The cert would lock in **why** the shortcut fails when `5 | m`, in a form
that a future caller can audit against without re-running the empirical
sweep.

## Definitions (locked by the canonical patch)

```python
# qa_orbit_rules.py post-e7b2af0

def orbit_family(b, e, m):
    """Canonical, period-based:
        period 1 → singularity
        period 8 → satellite
        else     → cosmos"""
    period = orbit_period(b, e, m)
    if period == 1: return "singularity"
    if period == 8: return "satellite"
    return "cosmos"

def orbit_family_divisor_shortcut(b, e, m):
    """Algebraic shortcut, exact only on certain moduli:
        singularity : b == m AND e == m
        satellite   : (m//3)|b AND (m//3)|e (excluding singularity)
        cosmos      : everything else"""
```

The cert reasons about the **discrepancy** between these two functions on a
fixed sweep of moduli.

## Claim candidate (narrow, falsifiable)

The cert is restricted to `M_test = {9, 12, 15, 18, 21, 24, 27, 30, 33, 36,
39, 42, 45, 48, 51, 54, 60, 63, 75}` and, more sharply, to the 5-factor
subset `M_5 = {15, 30, 45, 60, 75} = {15k : k ∈ {1, 2, 3, 4, 5}}`.

```text
1. if gcd(m, 5) = 1 and m ∈ M_test:
       orbit_family_divisor_shortcut(b, e, m) = orbit_family(b, e, m)
       for every (b, e) ∈ {1..m}²        (shortcut is exact)

2. if m ∈ M_5 (i.e. m = 15k for k ∈ {1..5}):
       missed = { (b, e) ∈ {1..m}² : canonical = "satellite"
                                AND shortcut ≠ "satellite" }
       |missed| = 32

       and missed partitions exactly by (gcd(b, m), gcd(e, m)):

         (gcd(b, m), gcd(e, m)) = (k,  3k)   →  8 pairs
         (gcd(b, m), gcd(e, m)) = (k,  k )   → 16 pairs
         (gcd(b, m), gcd(e, m)) = (3k, k )   →  8 pairs

3. for all m ∈ M_test:
       orbit_family_divisor_shortcut never over-claims satellite.
```

### Why the gcd-signature decomposition is the right form

The earlier draft offered two weaker forms — the 4-residue enumeration
`{(1,3), (2,1), (3,4), (4,2)} mod 5` and its compact congruence
`e ≡ 3b (mod 5)`. A proof pass on the congruence (2026-05-08) showed it
is **necessary but not sufficient**: it captures the missed satellites
plus extras (a period-4 cosmos orbit and the singularity at m=15; many
more cosmos pairs at higher `m`).

The reason: `e ≡ 3b (mod 5)` is preserved by qa_step (it is an orbit
invariant under mod-5 reduction), and multiple orbits with different
periods live in this residue class. The 32 missed satellites are exactly
the period-8 orbits within the residue class, and the structural form
that picks them out is the gcd-signature decomposition above.

### Verbatim verification (2026-05-08)

Initial sweep, then extended sweep filling k = 6..12 plus k = 15 and k = 20:

```text
m= 15 (k= 1): {(1, 3): 8, (1, 1): 16, (3, 1): 8}             total 32
m= 30 (k= 2): {(2, 6): 8, (2, 2): 16, (6, 2): 8}             total 32
m= 45 (k= 3): {(3, 9): 8, (3, 3): 16, (9, 3): 8}             total 32
m= 60 (k= 4): {(4, 12): 8, (4, 4): 16, (12, 4): 8}           total 32
m= 75 (k= 5): {(5, 15): 8, (5, 5): 16, (15, 5): 8}           total 32
m= 90 (k= 6): {(6, 18): 8, (6, 6): 16, (18, 6): 8}           total 32
m=105 (k= 7): {(7, 21): 8, (7, 7): 16, (21, 7): 8}           total 32
m=120 (k= 8): {(8, 24): 8, (8, 8): 16, (24, 8): 8}           total 32
m=135 (k= 9): {(9, 27): 8, (9, 9): 16, (27, 9): 8}           total 32
m=150 (k=10): {(10, 30): 8, (10, 10): 16, (30, 10): 8}       total 32
m=165 (k=11): {(11, 33): 8, (11, 11): 16, (33, 11): 8}       total 32
m=180 (k=12): {(12, 36): 8, (12, 12): 16, (36, 12): 8}       total 32
m=225 (k=15): {(15, 45): 8, (15, 15): 16, (45, 15): 8}       total 32
m=300 (k=20): {(20, 60): 8, (20, 20): 16, (60, 20): 8}       total 32
```

Every signature is `(k·a, k·b)` with `(a, b) ∈ {(1, 3), (1, 1), (3, 1)}`.
All 14 tested values of `k` produce identical decomposition shape with 0
overclaims. The cert's bounded set is therefore `M_5 = {15k : k ∈ K_verified}`
with `K_verified = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20}`.

### Adjacent regime (5 | m, 3 ∤ m) — explicitly out of scope

Verified separately for context only:

```text
m=10  (m//3=3 ):  shortcut sat = 9, canonical sat = 0,  missed = 0,  overclaims = 9
m=20  (m//3=6 ):  shortcut sat = 9, canonical sat = 0,  missed = 0,  overclaims = 9
m=25  (m//3=8 ):  shortcut sat = 9, canonical sat = 0,  missed = 0,  overclaims = 9
m=35  (m//3=11):  shortcut sat = 9, canonical sat = 0,  missed = 0,  overclaims = 9
m=50  (m//3=16):  shortcut sat = 9, canonical sat = 0,  missed = 0,  overclaims = 9
m=100 (m//3=33):  shortcut sat = 9, canonical sat = 0,  missed = 0,  overclaims = 9
```

This regime is structurally different: when `3 ∤ m`, there are no period-8
orbits at all (canonical satellite count = 0), and the divisor shortcut
**over-claims** by 9 pairs at every tested modulus (the 3×3 grid of
`m//3`-multiples in `{1..m}`). This is a separate boundary worth its own
cert; this draft excludes it.

## Mechanism (Pisano-period decomposition, sketch only — proof deferred)

Under qa_step mod 5, the residue pair `(b, 3b)` cycles with period 4:

```text
(b, 3b) → (3b, 4b) → (4b, 2b) → (2b, b) → (b, 3b)   (mod 5)
```

This is a sub-cycle of the full Pisano period π(5) = 20 (Wall, 1960).

When this mod-5 cycle lifts to `m = 15 = 3 · 5`, it interacts with the
mod-3 structure (π(3) = 8) to produce an orbit of period
`lcm(period_5, period_3_complement) = 8` at the lifted level. The 4 mod-5
residue classes × 8 lifted pairs each = 32 missed satellites.

For `m = 5^a · 3^b · k` with `gcd(k, 30) = 1`, similar Pisano-multiplicative
structure is expected (see Wall 1960 §3 for π(5^a)), but the cert claim
restricts to the empirical `M_test` until the multiplicative law is
formalized.

## Fixture plan

### PASS fixtures (7)

1. `pass_m9_exact.json` — `m = 9`, gcd(m, 5) = 1, expected_undercount = 0.
2. `pass_m24_exact.json` — `m = 24`, gcd(m, 5) = 1, expected_undercount = 0.
3. `pass_m15_gcd_decomp.json` — `m = 15`, k = 1, expected_undercount = 32,
   expected_signatures = `{"1,3": 8, "1,1": 16, "3,1": 8}`,
   expected_overclaims = 0.
4. `pass_m30_gcd_decomp.json` — `m = 30`, k = 2, expected_signatures =
   `{"2,6": 8, "2,2": 16, "6,2": 8}`.
5. `pass_m45_gcd_decomp.json` — `m = 45`, k = 3, expected_signatures =
   `{"3,9": 8, "3,3": 16, "9,3": 8}`.
6. `pass_m60_gcd_decomp.json` — `m = 60`, k = 4, expected_signatures =
   `{"4,12": 8, "4,4": 16, "12,4": 8}`.
7. `pass_m75_gcd_decomp.json` — `m = 75`, k = 5, expected_signatures =
   `{"5,15": 8, "5,5": 16, "15,5": 8}`.

### FAIL fixtures (4)

1. `fail_wrong_undercount.json` — declares `m = 15` undercount = 16 instead
   of 32; validator should reject (PISANO_1).
2. `fail_overclaim.json` — declares an overclaim count > 0 at `m = 15`;
   validator should reject (PISANO_2).
3. `fail_wrong_signatures.json` — declares `m = 15` signatures =
   `{"2,6": 8, ...}` (m=30 pattern at m=15); validator should reject
   (PISANO_3).
4. `fail_treats_shortcut_as_canonical.json` — fixture asserts the divisor
   shortcut IS the canonical classifier; validator should reject because
   the canonical claim has been frozen as period-based by commit `e7b2af0`.

## Validator outline

```python
# qa_orbit_pisano_5_factor_boundary_cert_validate.py
from collections import Counter
from math import gcd
from qa_orbit_rules import orbit_family, orbit_family_divisor_shortcut


def check_fixture(fixture):
    m = fixture["modulus"]
    expected_undercount = fixture["expected_shortcut_undercount"]
    expected_overclaim = fixture.get("expected_overclaims", 0)

    misses = []
    overclaims = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            canonical = orbit_family(b, e, m)
            shortcut = orbit_family_divisor_shortcut(b, e, m)
            if canonical == "satellite" and shortcut != "satellite":
                misses.append((b, e))
            if shortcut == "satellite" and canonical != "satellite":
                overclaims += 1

    errors = []
    if len(misses) != expected_undercount:
        errors.append(f"PISANO_1: undercount={len(misses)}, expected={expected_undercount}")
    if overclaims != expected_overclaim:
        errors.append(f"PISANO_2: overclaim={overclaims}, expected={expected_overclaim}")

    # PISANO_3: gcd-signature decomposition (PASS-only when present).
    if "expected_signatures" in fixture:
        sig_counts = Counter((gcd(b, m), gcd(e, m)) for b, e in misses)
        observed = {f"{a},{b}": n for (a, b), n in sig_counts.items()}
        expected = fixture["expected_signatures"]
        if observed != expected:
            errors.append(f"PISANO_3: signatures {observed} != expected {expected}")

    return errors
```

Checks: **PISANO_1** (undercount matches), **PISANO_2** (no overclaim),
**PISANO_3** (gcd-signature decomposition for `m ∈ M_5`), **SRC**
(`mapping_protocol_ref` present), **F** (every FAIL fixture declares
`expected_fail_type`).

The validator no longer references the bare `e ≡ 3b (mod 5)` form;
PISANO_3 is the structural test. The bare congruence is preserved in the
docstring of this file and in `qa_orbit_rules.py` as orbit-invariant
context, but it is not the cert claim.

## Non-claims

- Does **not** claim `e ≡ 3b (mod 5)` is sufficient to characterize missed
  satellites. The proof pass (2026-05-08) showed it is necessary but not
  sufficient — multiple orbits of different periods live in this residue
  class. The exact cert claim is the gcd-signature decomposition above.
- Does **not** claim the gcd decomposition extends to `k` outside the
  verified set `K_verified = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20}`.
  The pattern is empirically extremely stable across 14 values, but
  promotion to "all `k ≥ 1`" requires a proof or a denser sweep over a
  declared upper bound.
- Does **not** claim anything about `m` with `5 | m` and `3 ∤ m` (e.g.
  m=10, 20, 25, 35, 50, 100). At those moduli the shortcut **over-claims**
  by 9 satellites with canonical = 0 (no period-8 orbits exist). That is a
  structurally distinct boundary worth a separate cert; this draft
  excludes it explicitly.
- Does **not** claim a universal theorem `5 | m ⇒ undercount = 32`. The
  cert is restricted to the empirical sweep `M_test` and the gcd-decomposition
  to `M_5`. Promotion to a universal claim requires the Pisano-multiplicative
  formalization (deferred — see Recommended next proof step).
- Does **not** claim the full Pisano-period structure of qa_step is solved.
  Pisano-period results (Wall 1960) describe the Fibonacci recurrence over
  `Z/mZ`; qa_step adds the A1 correction `((b+e-1) mod m) + 1`, which
  shifts the orbit space but preserves the cycle structure. The interplay
  between the A1 shift and the Pisano period for composite `m` is not
  formalized in this draft.
- Does **not** modify QA-ML cert family [276]. The boundary observation
  flagged in [276] scope_note is now grounded in `qa_orbit_rules.py`
  docstring + self_test (commit `e7b2af0`); this cert would be a separate,
  narrower formalization of the boundary itself.
- Does **not** prescribe a fix to the shortcut. The fix is already done at
  the code level (canonical replacement); this cert documents the shortcut's
  exact limit.

## Recommended next proof step (if the draft is promoted)

**Extend the sweep before promotion**, then characterize the gcd-signature
decomposition structurally.

### Sweep extensions (no proof, just data)

1. **k ≥ 6 frontier**: run the gcd-signature check at `m = 90, 105, 150`
   (k = 6, 7, 10). If the same `(k, 3k), (k, k), (3k, k)` decomposition
   with counts `8, 16, 8` holds, the cert claim graduates from `k ∈ {1..5}`
   to `k ∈ {1..7, 10}` (or wider).
2. **5 | m, 3 ∤ m frontier**: run the missed-count and overclaim checks at
   `m = 10, 20, 25, 35, 50, 100`. The shortcut at these moduli uses
   `m // 3 ∈ {3, 6, 8, 11, 16, 33}`, none of which divides `m`. Expected
   behaviors:
   - Shortcut may produce **0 satellites** (if `m // 3` doesn't divide any
     pair in {1..m}²) or a **different cluster** than `(m//3) | b ∧
     (m//3) | e` predicts.
   - Canonical satellites still defined by `orbit_period == 8`.
   - The undercount/overclaim relationship is qualitatively different from
     the `5 | m, 3 | m` regime.

### Structural characterization (after extended data is in)

3. Show that the gcd-signature classes `{(k, 3k), (k, k), (3k, k)}` are
   exactly the lifts to `m = 15k` of three sub-orbits under qa_step mod 5
   (period 4) interacting with mod-3 (period 8 from the qa_step + A1
   correction). The 8-pair orbits at signatures `(k, 3k)` and `(3k, k)`
   correspond to the period-8 cycle visiting non-trivial mod-3 residues;
   the 16-pair signature `(k, k)` corresponds to the same cycle visited
   from two distinct starting points modulo `(3, 3)`.
4. Generalize to `m = 5^a · 3^b · ℓ` with `gcd(ℓ, 30) = 1`: predict the
   gcd-signature decomposition as a function of `a, b, ℓ`, then verify
   against the extended sweep.

If steps 3–4 produce a closed-form prediction matching the extended sweep,
the cert claim can be promoted to `m ∈ {5^a · 3^b · ℓ : a ≥ 1, b ≥ 1,
gcd(ℓ, 30) = 1, m ≤ M_max}`. Until then, the cert stays bounded to the
empirically verified set.

## Readiness assessment

The draft is **ready to promote to a cert family** under the following
conditions:

- The bounded `M_test` claim is acceptable (yes — Will's "narrow first"
  guidance from 2026-05-08 explicitly endorses this).
- The validator is straightforward (yes — recomputes both classifiers and
  diffs them; no new TLA+ work, no Pisano machinery in the validator
  itself).
- The mapping_protocol_ref can cite Wall 1960 as the primary source for
  the Pisano-period framing (yes — Wall 1960 satisfies the
  `primary_source_gate` regex via the DOI marker).
- Family ID is available (yes — registry head was 276 as of 2026-05-08;
  next free is 277, and the cert(275) commit needs registration verification
  before claiming a number).

If promoted, the family directory would be:
`qa_alphageometry_ptolemy/qa_orbit_pisano_5_factor_boundary_cert_v1/`
with the standard layout (mapping_protocol_ref + schema + validator + 3
PASS + 3 FAIL fixtures + SPEC). No README is recommended (formal-publication
gate triggers on README.md under qa_alphageometry_ptolemy/, see the lesson
captured in `memory/feedback_isolate_push_blocker_first.md`).

## References

- Wall, D. D. (1960). Fibonacci series modulo m. *American Mathematical
  Monthly* 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.
- D. E. Knuth, *The Art of Computer Programming* Vol. 1, §1.2.8 Fibonacci
  numbers.
- OEIS A001175 — Pisano periods (`π(m)`).
- `qa_orbit_rules.py` (post-commit `e7b2af0`) — canonical orbit_family +
  orbit_family_divisor_shortcut.
- `experiments/qa_ml/03_gnn_modulus_sweep.py` — the empirical sweep that
  surfaced the boundary; cert [276] scope_note flags the observation.
