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

For `m ∈ M_test = {9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
51, 54, 60, 63, 75}`, the empirical sweep verifies:

```text
1. if gcd(m, 5) = 1:  orbit_family_divisor_shortcut(b, e, m)
                    = orbit_family(b, e, m)  for every (b, e) ∈ {1..m}²
                    (the shortcut is exact)

2. if 5 | m:          | { (b, e) ∈ {1..m}² : orbit_family(b, e, m) = "satellite"
                            AND orbit_family_divisor_shortcut(b, e, m) ≠ "satellite" } |
                      = 32  (the shortcut under-counts by exactly 32)

3. for all m ∈ M_test: orbit_family_divisor_shortcut never over-claims
                       satellite — i.e., the algebraic-satellite set is
                       always a SUBSET of the empirical-satellite set.
```

**Claim 2 refinement.** The 32 missed pairs at `m = 15` cluster into 4
residue classes modulo 5:

```text
(b mod 5, e mod 5) ∈ { (1, 3), (2, 1), (3, 4), (4, 2) }, 8 pairs each.
```

Equivalently: `e ≡ 3b (mod 5)` characterizes all 4 missed classes
(since 1·3=3, 2·3=6≡1, 3·3=9≡4, 4·3=12≡2 mod 5).

For `m = 30` the same pattern holds at scale 2: `(b mod 10, e mod 10) ∈
{ (2, 6), (4, 2), (6, 8), (8, 4) }` — the m=15 classes doubled.

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

### PASS fixtures (3)

1. `pass_m9_exact.json` — `m = 9`, asserts shortcut exactness across 81
   pairs; expected_undercount = 0.
2. `pass_m24_exact.json` — `m = 24`, asserts shortcut exactness across
   576 pairs; expected_undercount = 0.
3. `pass_m15_undercount_32.json` — `m = 15`, asserts shortcut
   under-counts by exactly 32 satellites; expected_residue_classes =
   `[[1,3],[2,1],[3,4],[4,2]]` mod 5; expected_overclaims = 0.

### Optional extended PASS fixtures (4, gated on test budget)

4. `pass_m30_undercount_32.json` — `m = 30`, undercount = 32, residue
   classes = m=15 doubled.
5. `pass_m45_undercount_32.json` — `m = 45 = 3^2 · 5`, undercount = 32.
6. `pass_m60_undercount_32.json` — `m = 60 = 2^2 · 3 · 5`, undercount = 32.
7. `pass_m75_undercount_32.json` — `m = 75 = 3 · 5^2`, undercount = 32.

### FAIL fixtures (3)

1. `fail_wrong_undercount.json` — declares `m = 15` undercount = 16
   instead of 32; validator should reject.
2. `fail_overclaim.json` — declares an overclaim case at `m = 15` (e.g.,
   asserts `(7, 11)` satisfies the shortcut as satellite when it does not);
   validator should reject.
3. `fail_treats_shortcut_as_canonical.json` — fixture asserts that
   `orbit_family_divisor_shortcut` IS the canonical classifier; validator
   should reject because the canonical claim has been frozen as
   period-based by commit e7b2af0.

## Validator outline

```python
# qa_orbit_pisano_5_factor_boundary_cert_validate.py
from qa_orbit_rules import (
    orbit_family,
    orbit_family_divisor_shortcut,
)

def check_fixture(fixture):
    m = fixture["modulus"]
    expected_undercount = fixture["expected_shortcut_undercount"]
    expected_overclaim = fixture.get("expected_overclaims", 0)

    misses = 0
    overclaims = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            canonical = orbit_family(b, e, m)
            shortcut = orbit_family_divisor_shortcut(b, e, m)
            if canonical == "satellite" and shortcut != "satellite":
                misses += 1
            if shortcut == "satellite" and canonical != "satellite":
                overclaims += 1

    errors = []
    if misses != expected_undercount:
        errors.append(f"PISANO_1: undercount={misses}, expected={expected_undercount}")
    if overclaims != expected_overclaim:
        errors.append(f"PISANO_2: overclaim={overclaims}, expected={expected_overclaim}")
    # PISANO_3 (PASS-only): residue-class characterization.
    if "expected_residue_classes" in fixture:
        observed = sorted({(b % 5, e % 5) for b in range(1, m + 1)
                           for e in range(1, m + 1)
                           if orbit_family(b, e, m) == "satellite"
                           and orbit_family_divisor_shortcut(b, e, m) != "satellite"})
        expected = sorted(tuple(c) for c in fixture["expected_residue_classes"])
        if observed != expected:
            errors.append(f"PISANO_3: residues {observed} != expected {expected}")
    return errors
```

Checks: PISANO_1 (undercount), PISANO_2 (no overclaim), PISANO_3 (residue
classes), SRC (mapping_protocol_ref present), F (every FAIL fixture
declares expected_fail_type).

## Non-claims

- Does **not** claim a universal theorem `5 | m ⇒ undercount = 32` for all
  `m`. The cert is restricted to the empirical sweep `M_test`. Promotion to
  a universal claim requires the Pisano-multiplicative formalization
  (deferred — see Recommended next proof step).
- Does **not** claim the full Pisano-period structure of qa_step is solved.
  Pisano-period results in the literature (Wall 1960) describe the
  Fibonacci recurrence over `Z/mZ`; qa_step adds the A1 correction
  `((b+e-1) mod m) + 1`, which shifts the orbit space but preserves the
  cycle structure. The interplay between the A1 shift and the Pisano
  period for composite `m` is not formalized in this draft.
- Does **not** modify QA-ML cert family [276]. The boundary observation
  flagged in [276] scope_note is now grounded in `qa_orbit_rules.py`
  docstring + self_test (commit e7b2af0); this cert would be a separate,
  narrower formalization of the boundary itself.
- Does **not** prescribe a fix to the shortcut. The fix is already done at
  the code level (canonical replacement); this cert documents the shortcut's
  exact limit.

## Recommended next proof step (if the draft is promoted)

**Characterize the 4 mod-5 residue classes producing the extra period-8
cycles**, structurally rather than empirically:

1. Show that the 4 residue classes `{(b, 3b mod 5) : b ∈ {1, 2, 3, 4}}` are
   exactly the orbit of `(1, 3)` under qa_step mod 5 (period 4).
2. Show the lift to mod-15 has period `lcm(qa_step period mod 3, qa_step
   period mod 5 on these residues) = lcm(8, 4) = 8`. (Note: qa_step is not
   pure Fibonacci because of the A1 correction `((b+e-1) mod m) + 1`; the
   periods quoted are for qa_step itself, which can be derived by direct
   computation over the small modulus.)
3. Generalize to `m = 5^a · 3^b · k` with `gcd(k, 30) = 1`: predict
   undercount as a function of `a, b, k`, then verify against an extended
   sweep up to e.g. `m = 150`.
4. If the multiplicative-Pisano law applies cleanly, the cert claim can be
   promoted from `m ∈ M_test` to `m = 5^a · 3^b · k`. If not, the cert stays
   bounded to the tested set with a note about which composite forms the
   theorem holds for.

A successful step 3 would let the cert claim graduate from "tested 19
moduli" to "all m of the form...", a substantial epistemic upgrade. Until
then, the cert claim is exactly the bounded statement on `M_test`.

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
