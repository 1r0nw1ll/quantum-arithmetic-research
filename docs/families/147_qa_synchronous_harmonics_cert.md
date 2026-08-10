# Family [147] QA_SYNCHRONOUS_HARMONICS_CERT.v1

## One-line summary

Coprime periods synchronize at their product (minimum time); same-par odd wavelets SUPPORT at quarter-points while cross-par wavelets OPPOSE; all QN products are divisible by 6.

## Scope boundary

This family certifies a narrow arithmetic slice of Iverson's Synchronous
Harmonics corpus: synchronization arithmetic, quarter-point par interference,
QN product divisibility by 6, and the source-stated 5/6/7-prime-factor
complete-wave rule with minimum 5-factor and 7-factor witnesses. It does not
certify the full QA-2 Ch.6 wave doctrine: elliptical male/female wave framing,
wave-packet/null-packet morphology, micro-waveform persistence, or
bonding/gear-wheel claims. Those remain source-grounded topics for separate
certs.

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

### Complete Quantum Wave factor rule

QA-2 Ch.6 states that a complete Quantum Wave needs 5, 6, or 7 prime
wavelets, always including factors 2 and 3 and usually including 5 and/or 7.
The validator recomputes the prime factorization of each complete-wave witness.
For eligibility it counts distinct prime bases, matching the source's own
`5046 = 2*3*29*29` discussion where the repeated `29` is still described as
"only three prime factors."

- Minimum 5-factor wave: 2*3*5*7*11 = 2310
- 6-factor witness: 2*3*5*7*11*13 = 30030
- Minimum 7-factor wave: 2*3*5*7*11*13*17 = 510510

The negative fixture uses 5046 = 2*3*29*29. It is divisible by 6, but it is
not accepted as a complete Quantum Wave because it has only three distinct
prime bases and no 5 or 7 wavelet.

## Checks

| ID | Description |
|----|-------------|
| SH_1 | schema_version == 'QA_SYNCHRONOUS_HARMONICS_CERT.v1' |
| SH_SYNC | coprime pairs sync at product; non-coprime at LCM < product |
| SH_PAR | par classification correct; same-par SUPPORT, cross-par OPPOSE |
| SH_PROD6 | all QN products b×e×d×a divisible by 6 |
| SH_QWAVE | complete Quantum Wave witnesses have 5/6/7 distinct prime bases, include 2 and 3, satisfy 5/7 expectation, and match 2310/510510 minima where declared |
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
- `fixtures/sh_pass_qn_products.json` — 8 QN product witnesses (all ÷6), 3 complete-wave witnesses (2310, 30030, 510510), 3 sync pairs, and 3 par pairs
- `fixtures/sh_fail_bad_sync.json` — Falsifier: sync_pair (3,5) with wrong sync_time=999, and par_pair (3,7) wrongly declared OPPOSE instead of SUPPORT (added 2026-07-07)
- `fixtures/sh_fail_bad_quantum_wave.json` — Falsifier: 5046=2*3*29*29 is divisible by 6 but fails the complete-wave 5/6/7 distinct-prime rule and lacks 5/7

## Verification Note (2026-07-06)

Confirmed clean, no bugs. The validator
(`qa_synchronous_harmonics_cert_validate.py`) already genuinely
recomputes everything live: `gcd`/`lcm` for every sync pair, `n mod 4`
par classification and `par_sign` for every par pair, and `b*e*(b+e)*(b+2e)`
for every QN witness — no fixture-trusting gap. Independently spot-checked
all 7 sync pairs (gcd/lcm arithmetic) and all 7 par pairs
(mod-4 classification and SUPPORT/OPPOSE logic) in both fixtures by hand
— every value matches the validator's own recomputation.

**Source-title check**: searched for independent corroboration of the
"Synchronous Harmonics" attribution to Ben Iverson — confirmed this is a
real, specific book title: "Quantum Arithmetic — Book 3 & 4 — New Wave
Theory — Synchronous Harmonics," documented on svpwiki.com's Iverson
page. This matches the cert's own title and its "QA-3 Ch 4" / "Pyth-2 Ch
XIII" source citations closely enough to corroborate the attribution is
real (not fabricated), though the specific page-level "3-par HIGH at
3/4" wording wasn't independently confirmed via search snippet — would
require access to the primary text itself.

**Follow-up (2026-07-07)**: this family had zero FAIL fixtures (part of
the 13-family zero-FAIL-fixture cluster). No `result=="FAIL"`
short-circuit exists (no print-corruption bug risk). Added
`fixtures/sh_fail_bad_sync.json` with two independent planted defects
(sync_pair (3,5) with wrong sync_time=999; par_pair (3,7) wrongly
declared OPPOSE) and wired it into `self_test()`; verified SH_SYNC and
SH_PAR both genuinely catch their respective defects.

**Source-scope tightening (2026-07-14)**: added `SH_QWAVE` so the cert no
longer stops at generic product divisibility. It now rejects the source's
explicit non-example 5046 and validates the stated 2310 and 510510 complete
wave minima by recomputed prime factorization.
