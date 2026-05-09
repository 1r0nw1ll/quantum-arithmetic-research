# [277] QA Orbit Pisano 5-Factor Boundary Cert — Spec

> Glossary: "Theorem NT" — i.e. the Observer Projection Firewall axiom (an
> invariant that bars float values from re-entering the QA discrete layer;
> see `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`).

## Primary source

- Wall, D. D. (1960). *Fibonacci series modulo m*. American Mathematical
  Monthly 67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

Establishes the Pisano period π(m) of Fibonacci-mod-m and the
multiplicative law π(mn) = lcm(π(m), π(n)) when gcd(m, n) = 1, used to
explain why a 5-factor in m introduces extra period-8 satellite orbits
that the algebraic divisor shortcut misses.

## Bounded claim

For `m = 15k` with `k ∈ K_verified = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
12, 15, 20}` (14 empirically verified values):

```text
1. orbit_family_divisor_shortcut(b, e, m) under-counts
   orbit_family(b, e, m) satellite class by exactly 32 pairs.

2. orbit_family_divisor_shortcut never over-claims satellite
   (overclaim count = 0).

3. The 32 missed pairs partition by (gcd(b, m), gcd(e, m)) into
   exactly three signatures:

      (k,  3k):  8 pairs
      (k,  k ): 16 pairs
      (3k, k ):  8 pairs
```

Both classifiers are taken from `qa_orbit_rules.py` (post-`e7b2af0`):
`orbit_family` is the canonical, period-based classifier (singularity =
period 1, satellite = period 8, cosmos else); `orbit_family_divisor_shortcut`
is the named helper preserving the algebraic rule
`(m // 3) | b ∧ (m // 3) | e`.

## Schema (fixture)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | const | `"QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT.v1"` |
| `fixture_kind` | enum | `"pass"` or `"fail"` |
| `modulus` | int ≥ 2 | the test modulus `m = 15k` |
| `k` | int ≥ 1 | optional helper field, equals `m / 15` |
| `expected_shortcut_undercount` | int ≥ 0 | declared undercount |
| `expected_overclaims` | int ≥ 0 | declared overclaim count |
| `expected_signatures` | object | gcd-signature → count, present on PASS |
| `expected_fail_type` | string | declared failure mode (FAIL only) |
| `primary_source` | string | optional citation field; satisfies primary_source_gate |

## Checks

- **PISANO_1** — undercount matches expected (re-runs both classifiers,
  counts pairs canonical-says-satellite ∧ shortcut-says-not).
- **PISANO_2** — overclaim count matches expected (re-runs both classifiers,
  counts pairs shortcut-says-satellite ∧ canonical-says-not).
- **PISANO_3** — gcd-signature decomposition matches expected (PASS-only;
  computes `(gcd(b, m), gcd(e, m))` distribution over the missed set).
- **SRC** — `mapping_protocol_ref.json` present with required fields.
- **F** — every FAIL fixture declares `expected_fail_type` from
  `{WRONG_UNDERCOUNT, OVERCLAIM_DECLARED, WRONG_SIGNATURES,
  TREATS_SHORTCUT_AS_CANONICAL, MISSING_FIELD}`, and the validator
  observes that the declared failure mode actually fires.

## Adjacent regime — explicitly NOT in this cert

When `5 | m` and `3 ∤ m` (verified for `m ∈ {10, 20, 25, 35, 50, 100}`)
the canonical satellite count is **0** (no period-8 orbits exist) and the
shortcut **over-claims** by exactly 9 pairs. This is a structurally
distinct boundary; a separate cert candidate
`qa_orbit_5_factor_no_3_overclaim_cert_v1` is the right home for it. This
cert excludes that regime explicitly.

## Non-claims

- Does **not** claim the gcd decomposition extends to `k` outside
  `K_verified`. Promotion to "all `k ≥ 1`" requires either a proof or a
  denser sweep up to a declared bound.
- Does **not** claim a universal theorem `5 | m ⇒ undercount = 32`. The
  cert is bounded to the empirically verified set.
- Does **not** modify `qa_orbit_rules.py`. The canonical replacement
  landed at commit `e7b2af0`; this cert documents the shortcut's exact
  limit relative to the canonical contract.
- Does **not** formalize the full Pisano-period structure of qa_step. The
  A1 correction `((b + e − 1) mod m) + 1` shifts the orbit space relative
  to pure Fibonacci-mod-m; the interplay with composite m is not formalized
  in this cert.
- Does **not** prescribe a fix. The fix is already done at the code level
  (canonical replacement at `e7b2af0`); this cert documents the shortcut's
  exact failure mode.

## Failure modes

| `expected_fail_type` | Meaning |
|---|---|
| `WRONG_UNDERCOUNT` | Declared `expected_shortcut_undercount` differs from reality. |
| `OVERCLAIM_DECLARED` | Declared `expected_overclaims > 0` for a regime where overclaims are 0. |
| `WRONG_SIGNATURES` | Declared `expected_signatures` differs from the observed gcd decomposition. |
| `TREATS_SHORTCUT_AS_CANONICAL` | Fixture asserts undercount = 0 on a `5 | m, 3 | m` modulus where it should be 32. |
| `MISSING_FIELD` | Required schema field absent. |

## Theorem NT compliance

Both classifiers operate on integer `(b, e, m)`. The validator computes
gcds and counts as integers. No float arithmetic anywhere on the QA-side
state. The orbit_period dependency in `orbit_family` uses the cached
integer `qa_step` simulation. Boundary crossings: integer fixture JSON
→ Python int → integer arithmetic → fixture pass/fail outcome.

## Lineage

- Surfaced by: QA-ML v2 modulus sweep (`experiments/qa_ml/03_gnn_modulus_sweep.py`,
  cert [276]). The sweep flagged a 32-pair undercount at m=15 and m=30 that
  was excluded from [276]'s scope_note as a follow-up.
- Investigated: 2026-05-08 sweep across 19 moduli {9..75}, then proof pass
  on the `e ≡ 3b (mod 5)` characterization (necessary but not sufficient),
  then gcd-signature investigation that produced the structurally exact
  rule.
- Code-level fix: commit `e7b2af0` (`fix(qa_orbit_rules): canonical
  orbit_family via orbit_period`) — the canonical replacement.
- Design draft: `docs/specs/QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT_DRAFT.md`
  (commits `deebcd8` + `fe1c0ba`).
- This cert: registers [277] with the bounded claim on `K_verified`.
