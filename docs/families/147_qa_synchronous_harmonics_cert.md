# Family [147] QA_SYNCHRONOUS_HARMONICS_CERT.v1

## One-line summary

Coprime periods synchronize at their product (minimum time); same-par odd wavelets SUPPORT at quarter-points while cross-par wavelets OPPOSE; all QN products are divisible by 6.

## Mathematical content

### Synchronization theorem

Two periodic wavelets with periods m, n:
- If gcd(m,n) = 1 (coprime): first synchronization at time m×n (product). This is the **minimum** time both wavelets return to their initial phase together.
- If gcd(m,n) > 1: synchronization at LCM(m,n) < m×n. The shared factor creates an earlier meeting.

**Fundamental example**: periods 3 and 5 (coprime). First synchronization at 15 = 3×5.

### Par interference (quarter-point rule)

Iverson's "Double Parity" classifies odd integers by mod 4:
- **3-par** (4k+3): HIGH at 3/4 mark, LOW at 1/4 mark
- **5-par** (4k+1): HIGH at 1/4 mark, LOW at 3/4 mark

Phase sign at the 1/4-cycle point:
- 5-par wavelets: +1 (HIGH)
- 3-par wavelets: −1 (LOW)

**Same-par pairs** (both 3-par or both 5-par): signs agree → constructive interference → **SUPPORT**

**Cross-par pairs** (one 3-par, one 5-par): signs disagree → destructive interference → **OPPOSE**

| Pair | Par classes | Signs at 1/4 | Interference |
|------|------------|--------------|-------------|
| (3, 7) | 3-par, 3-par | −1, −1 | SUPPORT |
| (5, 13) | 5-par, 5-par | +1, +1 | SUPPORT |
| (3, 5) | 3-par, 5-par | −1, +1 | OPPOSE |
| (7, 13) | 3-par, 5-par | −1, +1 | OPPOSE |

### QN product divisibility by 6

For any Quantum Number (b, e, d, a) with d=b+e, a=b+2e:

Among {b, e, d}, at least one must be even (since d=b+e, if both b,e are odd then d is even). Among {b, e, d}, at least one must be divisible by 3 (pigeonhole on residues mod 3). Therefore b×e×d×a is always divisible by 2×3 = 6.

**Witnesses**: (1,1,2,3) product=6; (1,2,3,5) product=30; (2,1,3,4) product=24; (3,5,8,13) product=1560.

## Checks

| ID | Description |
|----|-------------|
| SH_1 | schema_version == 'QA_SYNCHRONOUS_HARMONICS_CERT.v1' |
| SH_SYNC | coprime pairs sync at product; non-coprime at LCM < product |
| SH_PAR | par classification correct; same-par SUPPORT, cross-par OPPOSE |
| SH_PROD6 | all QN products b×e×d×a divisible by 6 |
| SH_W | ≥5 total witnesses (sync + par pairs) |
| SH_F | fundamental pair (3,5) present |

## Source grounding

- **Ben Iverson, Pyth-2 Ch XIII**: "3-par wavelength: HIGH at 3/4, LOW at 1/4; 5-par: HIGH at 1/4, LOW at 3/4. Same-par SUPPORT; different-par OPPOSE."
- **Ben Iverson, QA-2 Ch 6**: coprime periods synchronize at their product; non-coprime at LCM
- **Ben Iverson, QA-3 Ch 4**: "all QNs are multiples of 6" (Theory of Harmony); minimum 5-7 prime-period wavelets per waveform
- **Ben Iverson, QA-4 Ch 4-9**: extended synchronous harmonics development

## Connection to other families

- **[128] Spread Period**: Cosmos period 24 = LCM of even wavelets {2,4,6,8}=LCM 24; satellite 8 = period of {2,4,8}
- **[144] Male/Female Octave**: female product = 4× male product; 24 = female of fundamental; 6 = male fundamental product (both divisible by 6)
- **[130] Origin of 24**: H²−G²=G²−I²=24 for 3-4-5; the "area quantum" = fundamental sync unit
- **[137] Koenig Twisted Squares**: 2CF=24L; L=CF/12 integer; 12=LCM(3,4)=sync time of par primes 3,4

## Fixture files

- `fixtures/sh_pass_sync_and_par.json` — 7 sync pairs (5 coprime, 2 non-coprime) + 7 par pairs (4 support, 3 oppose)
- `fixtures/sh_pass_qn_products.json` — 8 QN product witnesses (all ÷6) + 3 sync + 3 par pairs
