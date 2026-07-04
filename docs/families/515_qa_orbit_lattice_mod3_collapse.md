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

The validator's gating checks are all stdlib-only and deterministic (the
mod-3 collapse identity is a proof, reproducible instantly). The `fpylll`
LLL/BKZ experiment itself is documented as a historical, reproducible record
rather than re-run on every validator invocation, since `fpylll` is a heavy
non-stdlib dependency and the experiment already takes tens of seconds to
minutes per size.

## The Fix

Choose the QA modulus `m` coprime to 3. This is a one-line change (pick a
different modulus) that fully restores security to the random-key baseline,
confirmed under both LLL and BKZ.

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
