# [515] QA Orbit-Lattice Mod-3 Collapse

**Cert slug**: `qa_orbit_lattice_mod3_collapse_cert_v1`
**Family ID**: 515
**Derived**: 2026-07-04

## Claim

Deriving NTRU-style lattice-cryptography key material (small-coefficient ring
polynomials) from QA orbit sequences via `coefficient = (orbit b-value mod 3) - 1`
is **cryptographically unsafe whenever the QA modulus `m` is divisible by 3**, and
is **not measurably weaker than a properly random key when `gcd(m,3)=1`**.

This corrects a specific prior claim rather than introducing a new one: informal
chat-derived "QARSDC"/"QAFST" cryptography material asserted that QA "harmonic
class filtering" mitigates lattice attacks, but never specified an actual lattice
basis (literally `L=span{...}`, the ellipsis never filled in) and at one point
explicitly reported `fpylll` as unavailable. No real lattice attack had ever
actually been run against these constructions before this cert.

## Root Cause (proved, not observed)

`qa_step(b, e, m) = (e, ((b+e-1) mod m) + 1)`. Whenever `3 | m`, `(x mod m) ≡ x
(mod 3)` for all `x`, because `m` itself contributes 0 mod 3. Therefore the
mod-3 residue of the orbit state evolves **exactly** as if the recursion were
run directly mod 3 — independent of `m`'s own (possibly much larger) period.
The direct mod-3 recursion has intrinsic period at most 8. Any coefficient
sequence of length `N > 8` built this way is therefore periodic with period
`≤ 8`, regardless of how large `N` or `m` are chosen — a catastrophic entropy
loss for lattice-cryptography key material.

Verified exactly (not approximately, for a spread of `m` divisible by 3):
`orbit(b0, e0, m) mod 3 == orbit(b0 mod 3, e0 mod 3, 3)`, term for term.

## Empirical Record (2026-07-04, `fpylll` 0.6.4, real NTRU lattice, real LLL/BKZ)

Ring `Z[x]/(x^N-1)`, standard 2N-dimensional NTRU attack lattice `[I_N|H; 0|q·I_N]`.
Verified independently that the private key `(f,g)` genuinely lies in this
lattice for both key-generation methods before attacking either.

| N | q | key type | broken by LLL | avg(best/target norm²) |
|---|---|---|---|---|
| 83 | 256 | random baseline | 0/10 | ~594 (fully resists) |
| 83 | 256 | QA, m=9 (`3\|9`) | 6/10 | as low as 0.01-0.24 |
| 83 | 256 | QA, m=80 (`gcd=1`) | 0-1/12 | ~574-600 (matches random) |

Under BKZ (block_size 10/20/30 — a strictly stronger attack): **both** random
keys and `gcd(m,3)=1` QA keys break equally (8/8 each, avg ratio ~1.0). N=83
is simply too small for *any* NTRU instance to resist BKZ — a general
parameter-sizing fact, not a QA-specific weakness. This confirms the safe
construction fails the same way, at the same rate, as random keys under a
stronger attack too — not just resistant to the one attack it happened to be
tuned against.

### A confounder caught and corrected during derivation

An initial hypothesis — "short orbit period alone causes the weakness" —
appeared confirmed by one comparison (m=9 weak vs. m=50 strong), but a proper
2×2 design (period short/long × `gcd(m,3)` divisible/coprime) falsified it:
both "long period" cases (period=120, above N=83) split identically by
`gcd(m,3)` instead:

| period | `3\|m` | broken | avg ratio |
|---|---|---|---|
| 40 (short) | yes (m=15) | 6/12 | 1.56 |
| 40 (short) | no (m=41) | 2/12 | 463 |
| 120 (long) | yes (m=30) | 7/12 | 1.28 |
| 120 (long) | no (m=80) | 1/12 | 574 |

Period length does not predict the outcome; `gcd(m,3)` does, in both regimes.
The corrected mechanism (mod-3 CRT collapse) above is the one that survived
this check.

## Generalization (2026-07-04): m=24, QA's own "applied" modulus, is ALSO broken

The CRT-collapse identity `(x mod m) ≡ x (mod p)` whenever `p | m` is basic
modular arithmetic for **any** prime `p`, not special to 3 — verified
directly for `p∈{2,3,5,7,11}` against `m∈{9,24,27,81,80,35}`. It only
*matters* for this specific vulnerability because the ternary NTRU
coefficient map is fixed at mod 3 (`(v mod 3) - 1`); other prime factors of
`m` (e.g. the 2 in `m=24` or `m=80`) are irrelevant to that map. So the only
question that matters is whether `3 | m` — checked against 8 moduli, each
with a long-period seed pool so the effect can't be attributed to a merely
short overall orbit period (see `reproduce_fpylll_generalization.py`):

| m | 3∣m? | broken/10 | avg(best/target) |
|---|---|---|---|
| 9 | yes | 5 | 1.457 |
| **24** | **yes** | **7** | **0.888** |
| 27 | yes | 7 | 0.775 |
| 81 | yes | 8 | 0.614 |
| 80 | no | 1 | 538.522 |
| 35 | no | 0 | 598.255 |
| 25 | no | 0 | 610.998 |
| 49 | no | 0 | 617.411 |

**m=24 — the modulus CLAUDE.md documents as this project's "applied" QA
modulus (alongside m=9 "theoretical")** — breaks 7/10, more decisively than
the originally-tested m=9 (avg ratio 0.888 < 1: LLL finds a vector
*shorter* than the real private key, not merely comparable). Both of QA's
two standard moduli are divisible by 3. There is no existing QA convention
that is safe by accident — safety requires deliberately picking `gcd(m,3)=1`,
which nothing in the project currently does. Severity increases mildly with
higher powers of 3 (9→24→27→81 avg ratio decreases monotonically:
1.457→0.888→0.775→0.614), plausibly because longer orbit periods widen the
attack's structured search space — but the qualitative safe/unsafe split is
exactly `gcd(m,3)=1` vs `3|m` with zero exceptions across all 8 cases.

## Checks (OLC = Orbit-Lattice Collapse)

| Check | Claim |
|---|---|
| OLC_STEP | qa_step never produces a zero state (A1) |
| OLC_MOD3_PERIOD | direct mod-3 recursion has period 8 (non-fixed) / 1 (fixed=(3,3)) |
| OLC_CRT_COLLAPSE | orbit(m) mod 3 == direct mod-3 orbit, exact match for m∈{9,15,21,30,300} |
| OLC_NO_COLLAPSE | mod-3 reduced sequence does not collapse to period≤8 when gcd(m,3)=1 |
| OLC_LATTICE | NTRU lattice construction genuinely contains (f,g) for a worked N=7 example |
| OLC_TERNARY_RANGE | orbit-to-polynomial coefficients always in {-1,0,1} |
| OLC_EMPIRICAL_WITNESS | recorded fpylll parameters match the historical run (regression guard) |
| OLC_APPLIED_MODULUS_UNSAFE | m=24 CRT-collapses to the direct mod-3 orbit exactly like m=9 |
| OLC_GENERALIZATION_WITNESS | recorded 8-modulus sweep matches the historical run (regression guard) |

The validator's gating checks are all stdlib-only and deterministic (the
mod-3 collapse identity is a proof, reproducible instantly). The `fpylll`
LLL/BKZ experiments themselves are documented as historical, reproducible
records rather than re-run on every validator invocation, since `fpylll` is
a heavy non-stdlib dependency and the experiments take tens of seconds to
minutes per size.

## The Fix

Choose the QA modulus `m` coprime to 3. This is a one-line change (pick a
different modulus) that fully restores security to the random-key baseline,
confirmed under both LLL and BKZ. **Neither of QA's two standard moduli
(9, 24) qualifies** — this is not a hypothetical edge case to guard against,
it is the project's actual current default. Any future QA-orbit-derived
cryptographic key material must use a modulus explicitly chosen coprime to
3 (e.g. 25, 35, 49, 80 all tested safe here), not one of the two moduli QA
already uses elsewhere for unrelated (non-cryptographic) purposes.

## Primary Sources

- Hoffstein, J., Pipher, J., Silverman, J.H. (1998). "NTRU: A Ring-Based
  Public Key Cryptosystem." ANTS-III, LNCS 1423. DOI 10.1007/BFb0054868.
- Lenstra, A.K., Lenstra, H.W., Lovász, L. (1982). "Factoring polynomials
  with rational coefficients." Math. Annalen 261, 515–534.
  DOI 10.1007/BF01457454.
- fpylll development team (2024). fpylll v0.6.4. https://github.com/fplll/fpylll

## Parents

None — this is a new, standalone finding, not an extension of an existing
family. It was produced auditing informal prior "QA cryptography" chat
material rather than continuing any registered cert.
