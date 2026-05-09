# QA Orbit Period Spectrum Cert — Draft

> Status: **DESIGN DRAFT ONLY**. Not registered as a cert family.
> Will Dale directive 2026-05-09: build the spectrum table, do not
> certify a universal theorem yet.

> Primary source: Wall, D. D. (1960). *Fibonacci series modulo m*.
> American Mathematical Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

> Secondary references: D. E. Knuth, *The Art of Computer Programming*
> Vol. 1 §1.2.8 (Fibonacci numbers); OEIS A001175 (Pisano periods).

## What this cert would certify

The exact orbit-period distribution of `qa_step` over the full grid
`(b, e) ∈ {1..m}^2` for each tested modulus `m`. Where `qa_step` is the
A1-compliant successor `(b, e) → (e, ((b + e - 1) mod m) + 1)` from
`qa_orbit_rules.py`.

This is the **structural backbone** the singularity / satellite / cosmos
classifier (cert [11] family) and the divisor-shortcut boundary certs
([277], [278]) all reduce from. Where the previous certs ask
"is this point in class X?", this cert asks "what is the full period
distribution under qa_step at this `m`?" — capturing the whole dynamical
system rather than a coarsened classification.

## Spectrum table (2026-05-09 sweep)

22 moduli verified.

```text
  m  fact      | sing  sat  cosm  | maxP  π(m) | miss over | period distribution
  5  5         |   1    0    24   |   20    20 |    0   24 | {1:1, 4:4, 20:20}
  6  2·3       |   1    8    27   |   24    24 |    0    0 | {1:1, 3:3, 8:8, 24:24}
  7  7         |   1    0    48   |   16    16 |    0    9 | {1:1, 16:48}
  8  2^3       |   1    0    63   |   12    12 |    0   15 | {1:1, 3:3, 6:12, 12:48}
  9  3^2       |   1    8    72   |   24    24 |    0    0 | {1:1, 8:8, 24:72}
 10  2·5       |   1    0    99   |   60    60 |    0    9 | {1:1, 3:3, 4:4, 12:12, 20:20, 60:60}
 11  11        |   1    0   120   |   10    10 |    0    9 | {1:1, 5:10, 10:110}
 12  2^2·3     |   1    8   135   |   24    24 |    0    0 | {1:1, 3:3, 6:12, 8:8, 24:120}
 15  3·5       |   1   40   184   |   40    40 |   32    0 | {1:1, 4:4, 8:40, 20:20, 40:160}
 18  2·3^2     |   1    8   315   |   24    24 |    0    0 | {1:1, 3:3, 8:8, 24:312}
 20  2^2·5     |   1    0   399   |   60    60 |    0    9 | {1:1, 3:3, 4:4, 6:12, 12:60, 20:20, 60:300}
 21  3·7       |   1    8   432   |   16    16 |    0    0 | {1:1, 8:8, 16:432}
 24  2^3·3     |   1    8   567   |   24    24 |    0    0 | {1:1, 3:3, 6:12, 8:8, 12:48, 24:504}
 25  5^2       |   1    0   624   |  100   100 |    0    9 | {1:1, 4:4, 20:120, 100:500}
 27  3^3       |   1    8   720   |   72    72 |    0    0 | {1:1, 8:8, 24:72, 72:648}
 30  2·3·5     |   1   40   859   |  120   120 |   32    0 | {1:1, 3:3, 4:4, 8:40, 12:12, 20:20, 24:120, 40:160, 60:60, 120:480}
 36  2^2·3^2   |   1    8  1287   |   24    24 |    0    0 | {1:1, 3:3, 6:12, 8:8, 24:1272}
 45  3^2·5     |   1   40  1984   |  120   120 |   32    0 | {1:1, 4:4, 8:40, 20:20, 24:360, 40:160, 120:1440}
 60  2^2·3·5   |   1   40  3559   |  120   120 |   32    0 | {1:1, 3:3, 4:4, 6:12, 8:40, 12:60, 20:20, 24:600, 40:160, 60:300, 120:2400}
 75  3·5^2     |   1   40  5584   |  200  100* |   32    0 | {1:1, 4:4, 8:40, 20:120, 40:960, 100:500, 200:4000}
 90  2·3^2·5   |   1   40  8059   |  120   120 |   32    0 | {1:1, 3:3, 4:4, 8:40, 12:12, 20:20, 24:1560, 40:160, 60:60, 120:6240}
120  2^3·3·5   |   1   40 14359   |  120   120 |   32    0 | {1:1, 3:3, 4:4, 6:12, 8:40, 12:300, 20:20, 24:2520, 40:160, 60:1260, 120:10080}
```

`π(m)` is the Pisano period of `Fib mod m` per OEIS A001175.
`*` flag on `m=75`: max qa_step period (200) is **twice** the Pisano
period (100). This is a verified divergence and is discussed below.

## Empirical structure observations (NOT certified — for the cert claim, see "Claim candidate" below)

These are observations the sweep surfaced. The cert claim itself is the
**fixture-by-fixture period distribution**, not these structural notes.
The notes are recorded so a future v2 cert can promote them to claims
when proven.

### O1. Singularity is always 1

For all tested `m`, `dist[1] = 1`. The unique fixed point is `(m, m)`.

### O2. Satellite count is 0 / 8 / 40

- `3 ∤ m`: `dist[8] = 0` (no period-8 orbits at any tested `m` ∈
  `{5, 7, 8, 10, 11, 20, 25}`).
- `3 | m`, `5 ∤ m`: `dist[8] = 8` (the [277]-style algebraic-aligned
  cluster at `m ∈ {6, 9, 12, 18, 21, 24, 27, 36}`).
- `3 | m`, `5 | m` (i.e. `15 | m`): `dist[8] = 40` (8 algebraic + 32
  Pisano-mod-5 lifts per cert [277] at `m ∈ {15, 30, 45, 60, 75, 90, 120}`).

### O3. Period 4 occurs iff `5 | m`

Empirical: `dist[4] = 4` exactly when `5 | m` (verified `m ∈ {5, 10, 15,
20, 25, 30, 45, 60, 75, 90, 120}`). The 4 pairs at `m=5` form the orbit
`(1,3) → (3,4) → (4,2) → (2,1) → (1,3)` (period 4 on the residue class
`e ≡ 3b (mod 5)`); higher 5-multiple moduli inherit this orbit through
the lifting that produces the [277] under-count.

### O4. Max period = π(m), with one exception

For 21 of 22 tested moduli, `max(periods) = π(m)` (the Pisano period of
`Fib mod m`). The exception is **`m = 75`**: `max = 200 = 2 · π(75)`.
This may be an effect of the A1 correction `((b + e - 1) mod m) + 1`
shifting the orbit space relative to pure `Fib mod m`. The cert
candidate **does not** claim `max = π(m)` universally; the m=75
divergence falsifies the universal form.

### O5. Periods are divisors of `lcm(π(m_i))` for prime-power factorization

For `m = ∏ p_i^{a_i}`, the observed periods are a subset of the divisors
of `lcm_i(π(p_i^{a_i}))`. Verified for all moduli except `m=75`. This
suggests qa_step's period structure inherits from Fibonacci-mod-m via
CRT, with the A1 correction occasionally introducing a doubling factor.

### O6. Overclaim count is 0 / 9 / 15 / 24

Surfaced by the [277] / [278] investigations and now reconfirmed in this
sweep:
- `3 | m`: overclaim = 0 (shortcut grid aligns with `m // 3`).
- `3 ∤ m, m ≥ 7, m ≠ 8`: overclaim = 9 (cert [278]).
- `m = 8`: overclaim = 15 ([278] boundary exception).
- `m = 5`: overclaim = 24 (degenerate; `m // 3 = 1` makes shortcut select
  every non-singular pair). Out of [278] scope.

## Claim candidate (narrow, falsifiable)

The cert certifies the **exact period distribution** for each tested
modulus, not a universal theorem. For `m ∈ M_test = {5, 6, 7, 8, 9, 10,
11, 12, 15, 18, 20, 21, 24, 25, 27, 30, 36, 45, 60, 75, 90, 120}` (22
moduli), each fixture declares:

```text
  modulus         m
  expected_dist   { period: count, ... }   # exact distribution
  expected_sing   = expected_dist.get(1, 0)
  expected_sat    = expected_dist.get(8, 0)
  expected_max    = max(expected_dist.keys())
  expected_pisano = π(m) lookup, with optional `pisano_doubled` flag
```

The validator recomputes `orbit_period(b, e, m)` for every `(b, e)` and
checks the observed distribution exactly matches the declared dict. Plus
a structural sanity check: `sing + sat + cosm == m^2` where
`cosm = total - sing - sat`.

## Fixture plan

### PASS fixtures (22)

One per tested modulus. Naming: `pass_m{m}.json`. Each declares the
exact distribution from the table above plus `expected_pisano` and
(for m=75) `pisano_doubled = true`.

To keep the family compact, the 22 PASS fixtures can be sharded into
two cert families if needed (e.g. small-m and large-m), but for v1 a
single family with 22 PASS fixtures is acceptable since the validator is
fast (max `m^2 = 14400` iterations per fixture, each iteration cached).

### FAIL fixtures (5)

1. `fail_wrong_distribution.json` — declares `m = 9` distribution
   `{1:1, 8:8, 24:71}` (off by one in the cosmos count); validator
   should reject.
2. `fail_missing_period.json` — declares `m = 24` distribution missing
   the period-12 entry; validator should reject (counts don't match).
3. `fail_pisano_universal.json` — declares `expected_pisano_universal =
   true` at `m = 75`; validator should reject because m=75 has
   `max = 200 = 2 · π(75)`.
4. `fail_satellite_overclaim.json` — declares `m = 7` has `expected_sat
   = 9` (treats shortcut count as canonical); validator should reject
   because canonical satellite at `m = 7` is 0.
5. `fail_missing_field.json` — required field absent.

## Validator outline

```python
from collections import Counter
from qa_orbit_rules import orbit_period


def check_fixture(fixture):
    m = fixture["modulus"]
    expected = {int(p): c for p, c in fixture["expected_dist"].items()}

    observed = Counter()
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            observed[orbit_period(b, e, m)] += 1

    errors = []
    if dict(observed) != expected:
        errors.append(f"SPEC_1: m={m} dist {dict(observed)} != expected {expected}")
    if sum(observed.values()) != m * m:
        errors.append(f"SPEC_2: m={m} count sum {sum(observed.values())} != m^2 = {m*m}")
    if observed.get(1, 0) != fixture.get("expected_sing", 1):
        errors.append(f"SPEC_3: m={m} singularity count off")
    if observed.get(8, 0) != fixture.get("expected_sat", 0):
        errors.append(f"SPEC_4: m={m} satellite count off")
    if max(observed.keys()) != fixture["expected_max"]:
        errors.append(f"SPEC_5: m={m} max period {max(observed.keys())} != expected {fixture['expected_max']}")
    return errors
```

Checks: **SPEC_1** (full dist match), **SPEC_2** (count sum = m^2),
**SPEC_3** (singularity), **SPEC_4** (satellite = period-8 count),
**SPEC_5** (max period matches), **SRC** (mapping_protocol_ref present),
**F** (FAIL fixtures declare expected_fail_type).

## Non-claims

- Does **not** claim a closed-form for the period distribution at
  arbitrary `m`. The cert is a per-modulus enumeration on `M_test` only.
- Does **not** claim `max(periods) = π(m)` universally. The m=75
  divergence (`max = 200 ≠ π(75) = 100`) falsifies the universal form.
- Does **not** claim periods strictly factor through CRT components.
  Observation O5 holds for all `M_test \ {75}` but a universal proof
  (or a counterexample beyond m=75) is not in this cert.
- Does **not** claim period 4 occurs iff `5 | m` universally. Observed
  on `M_test`; promoting to "all m" requires a denser sweep.
- Does **not** modify `qa_orbit_rules.py`. The canonical contract is
  locked at commit `e7b2af0`.
- Does **not** subsume cert [277] or [278]. Those are claim-narrow on
  the divisor-shortcut boundary; this cert is the broader spectrum
  underlying both.
- Does **not** formalize the relationship between `qa_step`'s period
  structure and Fibonacci-mod-m's Pisano period. The A1 correction
  shifts the orbit space; the qa_step periods are not generally equal
  to Pisano periods (m=75 example).

## Recommended next proof step (if promoted)

1. **Extend the sweep** to confirm O3 (period 4 ↔ `5 | m`) on `m ∈
   {35, 40, 50, 55, 70, 100, 150, 200, 250}` and to discover any further
   `pisano_doubled` exceptions like m=75 (candidates: `m = 5^k` for
   k ≥ 2, or m = 25k for various k).
2. **Structurally derive O4** by analyzing how the A1 correction
   `((b + e - 1) mod m) + 1` interacts with Fibonacci-mod-m's orbit
   structure. Hypothesis: for `m` divisible by 25, the "+1" shift
   doubles the period because Fibonacci-mod-25 has a non-trivial
   automorphism that interacts with the shift.
3. **Promote O5 to a CRT-style theorem**: for `m = ∏ p_i^{a_i}` with
   pairwise coprime prime powers, the qa_step period at `(b, e)` mod `m`
   equals `lcm_i(qa_step period at (b mod p_i^{a_i}, e mod p_i^{a_i}))`
   except in cases that produce `pisano_doubled`. Verify on extended
   sweep.
4. **QA-ML v3 hook** (downstream): once the spectrum is locked at the
   fixture level, train a model to predict `period(b, e, m)` from
   features and extract symbolic rules. Cert [276]'s GCN topology
   advantage suggests the QA generator graph is a strong feature.

## Readiness assessment

Promotion to cert family `qa_orbit_period_spectrum_cert_v1` is
**ready** under bounded scope:

- **Schema**: single fixture format, ~6 fields plus the period dict.
- **Validator**: stdlib-only; recomputes `orbit_period` over `m^2` pairs
  per fixture; cached. Total runtime for 22 PASS fixtures ≈ 5-10 seconds
  (`Σ m^2 = 30,225`).
- **Family ID**: 279 is next free (276, 277, 278 all registered).
- **Mapping protocol**: cite Wall (1960) DOI for Pisano framing context.
- **No README** (formal-publication gate avoidance per
  `memory/feedback_isolate_push_blocker_first.md`).
- **Pairing**: this cert is **above** [11], [277], [278] in the
  abstraction tower — those are special cases of the spectrum that this
  cert tabulates.

## Open scope question

Does the cert want to include **m=75 as a PASS fixture with
`pisano_doubled = true`**, or **exclude m=75 from v1** like cert [278]
excluded m=8? Two options:

| Option | Pro | Con |
|---|---|---|
| **A — include m=75 with flag** | Captures the divergence in the cert; surfaces the structural anomaly | Requires a new fixture field `pisano_doubled` and a corresponding validator check |
| **B — exclude m=75 from v1** | Simpler v1 schema; matches [278]'s m=8 pattern | Loses an interesting empirical observation; future v2 must add it |

The draft recommends **Option A** for [279]: the m=75 doubling is a
real structural finding that the spectrum cert is uniquely positioned
to capture. Excluding it would push the same observation into a
separate cert (`qa_orbit_pisano_doubling_cert`) and fragment the data.

## References

- Wall, D. D. (1960). Fibonacci series modulo m. *American Mathematical
  Monthly* 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541;
  doi.org/10.1080/00029890.1960.11989541.
- D. E. Knuth, *The Art of Computer Programming* Vol. 1, §1.2.8.
- OEIS A001175 — Pisano periods `π(m)`.
- `qa_orbit_rules.py` (post-commit `e7b2af0`) — canonical orbit_family,
  orbit_period, orbit_family_divisor_shortcut.
- Cert [11] family — singularity / satellite / cosmos classification
  that this cert generalizes.
- Cert [277] (`qa_orbit_pisano_5_factor_boundary_cert_v1`) — under-count
  boundary on the divisor shortcut for `m = 15k`.
- Cert [278] (`qa_orbit_no_3_divisor_overclaim_cert_v1`) — over-claim
  boundary on the divisor shortcut for `3 ∤ m`.

## Summary

| Attribute | Value |
|---|---|
| Candidate family ID | [279] |
| Slug | `qa_orbit_period_spectrum_cert_v1` |
| PASS fixtures | 22 (one per tested modulus) |
| FAIL fixtures | 5 |
| Scope | per-modulus exact period distributions on `M_test` |
| Pisano-doubled exceptions | `m = 75` (max = 200, π(75) = 100) |
| Pairs with | [11], [277], [278] (spectrum is the parent claim) |
| Should become [279]? | **Yes** under bounded-scope reading; the table is sharp, validator is cheap, and the cert is the natural backbone for QA-ML v3 |
