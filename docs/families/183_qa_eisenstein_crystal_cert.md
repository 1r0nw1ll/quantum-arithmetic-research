# Family [183] QA_EISENSTEIN_CRYSTAL_CERT.v1

## One-line summary

Three Eisenstein-lattice identities — Z−Y=J=bd, the factorization Z²−Y²=J·a·(a+e), and the norm relation F²−FW+W²=Z² — encode Ben Iverson's "four Forces" {3,5,7,8}={F,G,Z,W}, with the Unity Block giving cosmos period 2·3·4=24.

## Mathematical content

### New identity: Z − Y = J

From the 16 QA identities: Z = E + K = e² + ad, Y = A − D = a² − d². Their difference:

    Z − Y = (e² + ad) − (a² − d²) = e² + ad − a² + d² = bd = J

This connects the three "higher" identities (Z, Y) to the fundamental product J = bd.

### Factorization

    Z² − Y² = (Z − Y)(Z + Y) = J · (Z + Y)

Since Z + Y = (e² + ad) + (a² − d²) = a² + e² + ad − d² = a·(a + e) (after substitution), we get:

    Z² − Y² = J · a · (a + e) = bd · a · (a + e)

### Eisenstein norm relation

The identity F² − FW + W² = Z² (certified in [133]) means the triple (F, W, Z) lies on an Eisenstein integer norm. In Ben's numbering: F=3, W=8, Z=7, G=5 — the "four Forces" of QA arithmetic.

### Unity Block and period 24

The Unity Block QN (b,e,d,a)=(1,1,2,3) instantiates every identity above
concretely: F=ab=3, G=d²+e²=5, Z=e²+ad=7, W=d(e+a)=8 — exactly Ben
Iverson's four Forces {3,5,7,8}, derived here (not just asserted) from
the tuple's own element formulas. The general factorization
Z²−Y²=J·a·(a+e) evaluated at this tuple gives J=bd=2, a=3, a+e=4, so
J·a·(a+e) = 2·3·4 = **24** — matching the QA cosmos period exactly (and
independently, Z²−Y²=7²−5²=49−25=24, confirming the factorization
identity itself at this specific tuple).

## Checks

| ID | Description |
|----|-------------|
| EC_1 | schema_version == 'QA_EISENSTEIN_CRYSTAL_CERT.v1' |
| EC_ZYJ | Z − Y = J = bd for all 81 mod-9 states |
| EC_FACTOR | Z² − Y² = J · a · (a + e) verified |
| EC_EISEN | F² − FW + W² = Z² (Eisenstein norm) |
| EC_TUPLE | all tuples satisfy A1 (no-zero) and A2 (derived coords) |
| EC_UNITY | Unity Block (1,1,2,3) yields period structure |
| EC_W | ≥3 witnesses (distinct QA states) |
| EC_F | ≥1 falsifier (broken identity rejected) |

## Source grounding

- **Ben Iverson**: 16 QA identities, "four Forces" {F, G, Z, W} = {3, 5, 7, 8}
- **Eisenstein integers**: Z[ω] norm form x² − xy + y² (see [133])
- **Unity Block**: QN (1,1,2,3) as the fundamental QA unit

## Connection to other families

- **[133] Eisenstein Norm**: F² − FW + W² = Z² originally certified there
- **[182] Miller Orbit**: crystal reflections use the same Z, Y, J identities

## Fixture files

- `fixtures/ec_pass_identity_suite.json` — Z−Y=J, factorization, Eisenstein norm for all 81 states
- `fixtures/ec_fail_broken_zyj.json` — falsifier with incorrect Z−Y value

## Verification Note (2026-07-06)

Self-contained QA-internal algebra (no new external citation needed;
Ben Iverson's corpus already confirmed real in this session's audits of
[156]/[160]/[177], though this specific "QA-4 (1990)" quote wasn't
independently re-traced). Independently verified all three core
identities symbolically and against 20+ random (b,e) pairs: Z−Y=bd
exactly (derived symbolically: Z−Y=(e²+ad)−(a²−d²)=b²+be=b(b+e)=bd);
Z+Y=a(a+e) exactly; and therefore Z²−Y²=J·a·(a+e) follows. 0 mismatches.

**Found and fixed a real documentation bug** (not a validator/fixture
bug — the actual certified fixture was already correct): the doc's
"Unity Block" paragraph was self-contradictory prose ("4 = a = b+2e =
1+2 = 3... corrected") that doesn't actually parse — 1+2·1=3, not 4,
and the sentence never says what the real derivation is. Traced the
actual fixture (`ec_pass_identity_chain.json`) which has the correct,
clean version: at Unity Block (1,1,2,3), J=bd=2, a=3, a+e=4, so
J·a·(a+e)=2·3·4=24. Independently confirmed this exactly, and also
confirmed F=ab=3, G=d²+e²=5, Z=e²+ad=7, W=d(e+a)=8 (Ben's four Forces)
are genuinely *derived* from the tuple via the validator's own formulas,
not just asserted — and that Z²−Y²=49−25=24 independently confirms the
same 24 via the factorization identity. Rewrote the doc's paragraph to
match the fixture's actual (correct) reasoning.

Validator confirmed genuinely computing every identity from the raw
(b,e) witness at runtime (`J_exp`, `Z_exp`, `Y_exp`, `F_exp`, `W_exp`,
`G_exp` all derived, not read from fixture), including the Unity Block
check. `--self-test` passes on both fixtures.
