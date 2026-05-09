# [278] QA Orbit No-3-Divisor Overclaim Cert — Spec

> Glossary: "Theorem NT" — i.e. the Observer Projection Firewall axiom (an
> invariant that bars float values from re-entering the QA discrete layer;
> see `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`).

## Primary source

- Wall, D. D. (1960). *Fibonacci series modulo m*. American Mathematical
  Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541;
  doi.org/10.1080/00029890.1960.11989541.

Establishes the Pisano period `π(m)` of Fibonacci-mod-`m` and the
multiplicative law `π(mn) = lcm(π(m), π(n))` when `gcd(m, n) = 1`.

## Bounded claim

For every tested modulus `m` with `3 ∤ m` and `m ≥ 7`, **excluding `m = 8`
in v1**:

```text
canonical_satellite_count(m)  = 0     (no period-8 orbits exist)
shortcut_satellite_count(m)   = 9
overclaim(m)                  = 9
missed(m)                     = 0

The 9 over-claimed pairs are exactly the 3×3 grid:
  { (a · m//3, b · m//3) : a, b ∈ {1, 2, 3} }
which is disjoint from the singularity (m, m) when 3 ∤ m and m ≥ 7.
```

The boundary exception `m = 8` is excluded because `m // 3 = 2` makes the
shortcut's grid the entire even sub-lattice `{2, 4, 6, 8}^2` (16 pairs,
including the singularity `(8, 8)`), giving 15 overclaims rather than 9.

## Causal scope = `no_3_divisor`, NOT `5_factor`

The original investigation entered through the `5 | m ∧ 3 ∤ m` slice during
cert [277] adjacent-regime work. An empirical sweep on 25 such moduli (up
to `m = 250`) produced uniform 9-overclaim behavior, but a separate proof
pass on **non-5-multiple** `3 ∤ m` moduli (`m ∈ {7, 11, 13, 14, 16, 17,
19, 22, 23, 26, 28, 29, 31, 32, 34, 37, 38, ...}`) showed the same
9-overclaim pattern. The 5-factor framing is incidental; the structural
cause is `m // 3` not being a divisor of `m`.

## Schema (fixture)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | const | `"QA_ORBIT_NO_3_DIVISOR_OVERCLAIM_CERT.v1"` |
| `fixture_kind` | enum | `"pass"` or `"fail"` |
| `modulus` | int ≥ 2 | the test modulus `m` |
| `expected_canonical_satellites` | int ≥ 0 | expected canonical period-8 count |
| `expected_shortcut_satellites` | int ≥ 0 | expected shortcut satellite count |
| `expected_overclaims` | int ≥ 0 | expected overclaim count |
| `expected_missed` | int ≥ 0 | expected missed count |
| `expected_overclaim_grid` | array | optional explicit 3×3 grid of overclaim pairs |
| `causal_scope` | enum | `"no_3_divisor"` (cert claim) or `"5_factor"` (rejected by FIVE_FACTOR_CAUSAL FAIL fixture) |
| `expected_fail_type` | enum | declared failure mode (FAIL only) |
| `primary_source` | string | optional citation field |

## Checks

- **NO3_1** — canonical satellite count = expected (always 0 for v1 PASS).
- **NO3_2** — shortcut satellite count = expected (always 9 for v1 PASS).
- **NO3_3** — overclaim = expected, missed = expected (both 0 for PASS).
- **NO3_4** — `m = 8` boundary exception is rejected from v1
  (`fail_m8_boundary_exception.json` fixture must trigger an actual
  observed-vs-expected mismatch when the validator computes the m=8 counts).
- **SRC** — `mapping_protocol_ref.json` present with required fields.
- **F** — every FAIL fixture declares `expected_fail_type` from
  `{M8_BOUNDARY, WRONG_OVERCLAIM, FIVE_FACTOR_CAUSAL, SHORTCUT_AS_CANONICAL,
  MISSING_FIELD}`, and the validator observes that the declared mode
  actually fires.

## Failure modes

| `expected_fail_type` | Meaning |
|---|---|
| `M8_BOUNDARY` | Fixture targets `m = 8` and asserts the v1 contract (overclaim = 9) — validator observes actual 15 and reports NO3_4 mismatch. |
| `WRONG_OVERCLAIM` | Declared `expected_overclaims` differs from reality. |
| `FIVE_FACTOR_CAUSAL` | Fixture declares `causal_scope = "5_factor"` — rejected because the cert's claim is `causal_scope = "no_3_divisor"`. |
| `SHORTCUT_AS_CANONICAL` | Fixture asserts `expected_canonical_satellites > 0` (treats shortcut count as canonical). |
| `MISSING_FIELD` | Required schema field absent. |

## Pairing with cert [277]

| | [277] | [278] |
|---|---|---|
| Scope | `m = 15k`, `k ∈ {1..12, 15, 20}` | `3 ∤ m`, `m ≥ 7`, `m ≠ 8` |
| Failure mode | shortcut **under-counts** | shortcut **over-claims** |
| Canonical period-8 | 32 | 0 |
| Shortcut satellites | 8 | 9 |
| Missed | 32 | 0 |
| Overclaim | 0 | 9 |

Together [277] and [278] map both failure surfaces of the divisor
shortcut: false negatives on certain `3 | m` composites with extra
period-8 orbits (Pisano-mod-5 lifts), and false positives on `3 ∤ m` where
the divisor predicate `(m // 3) | b ∧ (m // 3) | e` selects a misaligned
3×3 grid.

## Out of scope (not certified)

- `m ∈ {1, 2, 3, 4, 5}` (degenerate: `m // 3 ∈ {0, 0, 1, 1, 1}`).
- `m = 6` (although `3 | m`, so already out of scope).
- `m = 8` (boundary exception; future v2 sub-claim with overclaim = 15).
- `3 | m` regime (covered by [277] or unaffected; not in this cert).

## Theorem NT compliance

Both classifiers operate on integer `(b, e, m)`. The validator computes
counts and grid coordinates as integers. No float arithmetic on the
QA-side state. Boundary crossings: integer fixture JSON → Python int →
integer arithmetic → fixture pass/fail outcome.

## Lineage

- Cert [277] adjacent-regime investigation surfaced the 9-overclaim
  pattern on `5 | m ∧ 3 ∤ m` moduli (`{10, 20, 25, 35, 50, 100}`) on
  2026-05-08.
- Extended sweep on 25 `5 | m ∧ 3 ∤ m` moduli (m up to 250) confirmed
  uniform behavior.
- Proof pass on non-5-multiple `3 ∤ m` moduli (2026-05-09) showed the
  same 9-overclaim pattern, demonstrating that `5 | m` is incidental and
  the structural cause is `m // 3` not being a divisor of `m`.
- Design draft: `docs/specs/QA_ORBIT_5_FACTOR_NO_3_OVERCLAIM_CERT_DRAFT.md`
  (commit `73916f8`) presented Scope A vs Scope B options.
- This cert: registers [278] under Scope B (broader `no_3_divisor`
  framing) per Will's 2026-05-09 directive.
