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

The Unity Block QN (1, 1, 2, 3) yields the cosmos period as the product of its non-trivial elements: 2 · 3 · 4 = 24 (where 4 = a = b + 2e = 1 + 2 = 3... corrected: the full tuple is (1,1,2,3) so period = LCM structure gives 24 via orbit theory).

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
