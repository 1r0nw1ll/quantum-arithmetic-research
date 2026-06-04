# [312] QA Male/Female Lattice Balance

**Status**: ACTIVE  
**Validator**: `qa_alphageometry_ptolemy/qa_male_female_lattice_balance_cert_v1/qa_male_female_lattice_balance_cert_validate.py`  
**Tier**: 1 — Exact Reformulation  
**Sources**: Iverson (1975–1996) QA-1/QA-2; Hardy+Wright (2008) ISBN 978-0-19-921986-5

---

## Claim

The male/female boundary `b/e = √2` (equivalently `b² < 2e²`, or `I = C−F > 0`) partitions
any QA lattice `{1,...,m}²` into male and female states. No state is exactly on the boundary
(√2 is irrational).

**C1 — Unique exact moduli**: `m ∈ {3, 6, 9}` are the *only* positive integers where
`{1,...,m}²` contains exactly 2/3 male states. Verified exhaustively to m = 10,000.

**C2 — Asymptote**: The continuous-limit male fraction is
`1 − √2/4 = ∫₀¹ min(1, y√2) dy ≈ 0.64645`.
All m ≥ 10 are strictly below 2/3, converging monotonically to this value.

**C3 — Gap formula**: The gap between the exact 2/3 and the asymptote is
`√2/4 − 1/3 ≈ 0.0202`. The boundary irrational (√2) appears in its own asymptotic
density formula.

**C4 — mod-9 orbit splits** (24 states each):

| Orbit | |f| (Z[φ] norm) | Male | Female | Ratio |
|-------|----------------|------|--------|-------|
| Fibonacci | 1 (unit) | 17 | 7 | 17:7 |
| Lucas | 5 (ramified prime) | 17 | 7 | 17:7 |
| Third | 11 (split prime) | 14 | 10 | 14:10 |
| Satellite | — | 5 | 3 | 5:3 |
| Singularity | — | 1 | 0 | 1:0 |
| **Total** | | **54** | **27** | **2:1** |

Fibonacci and Lucas share identical 17:7 splits — unit and ramified prime of Z[φ]
share the same male/female distribution. The split prime (|f|=11, Third orbit) breaks
the pattern.

**C5 — Theorem NT**: `b² < 2e²` is an observer projection. It is never used as a QA
state input or T-step operand.

---

## Significance

m = 9 is the last/largest modulus where the irrational √2 boundary "rounds" to a clean
rational (2/3) in the discrete lattice. This is a structural property of the theoretical
modulus complementing the dynamic property Π(9) = 24 (Pisano period = orbit period,
cert [291]).

---

## Checks

`MF_1` schema · `MF_M9` m=9 exact 54:27 · `MF_M3` m=3,6 exact · `MF_ASY` asymptote
· `MF_GAP` gap formula · `MF_ORB` orbit splits · `MF_MON` test moduli < 2/3
· `MF_THM` Theorem NT · `MF_F` fail detection
