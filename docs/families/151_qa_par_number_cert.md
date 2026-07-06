# Family [151] QA_PAR_NUMBER_CERT.v1

## One-line summary

Iverson's "Double Parity" classifies integers into four par classes (mod 4): 2-par, 3-par, 4-par, 5-par — with universal rules for squares, QA identities C and G, and a closed multiplication table.

## Mathematical content

### The par system (QA-2 Ch 3)

| Par | Form | First values | Gender | Euclid |
|-----|------|-------------|--------|--------|
| 2-par | 4k+2 | 2,6,10,14... | female | even-odd |
| 3-par | 4k+3 | 3,7,11,15... | male | odd-odd |
| 4-par | 4k | 4,8,12,16... | both | even-even |
| 5-par | 4k+1 | 5,9,13,17... | male | odd-even |

Etymology: "par" from Hindi "char" (four), not English "parity."

### Universal rules

**Male square rule**: The square of any male number (3-par or 5-par) is always 5-par.
- Proof: (4k+1)² = 16k²+8k+1 ≡ 1 mod 4 = 5-par
- Proof: (4k+3)² = 16k²+24k+9 ≡ 1 mod 4 = 5-par

**QA identity parity**:
- C = 2de is always **4-par** (C ≡ 0 mod 4). Proof: one of d,e is even for primitive directions; C = 2×even×odd = 4k.
- G = d²+e² is always **5-par** (G ≡ 1 mod 4). Proof: opposite parity squares; odd²+even² = 1+0 ≡ 1 mod 4.

**Multiplication table** (closed under par):

| × | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| 2 | 4 | 2 | 4 | 2 |
| 3 | 2 | 5 | 4 | 3 |
| 4 | 4 | 4 | 4 | 4 |
| 5 | 2 | 3 | 4 | 5 |

Key: 4-par absorbs everything (4×anything = 4-par). 5-par is the identity (5×n = n for odd n). 3×3 = 5 (male×male = male).

### Fibonacci zero-count connection (observed, not universal)

For primes p, the number of Fibonacci zeros mod p within one Pisano period relates to the quadratic character of 5 mod p. Observed: pi(9)=24 with 2 zeros; pi(11)=10 with 1 zero; pi(19)=18 with 1 zero.

## Checks

| ID | Description |
|----|-------------|
| PN_1 | schema_version == 'QA_PAR_NUMBER_CERT.v1' |
| PN_CLASS | par classification matches n mod 4 |
| PN_SQ | male squares always 5-par |
| PN_QA | C=4-par, G=5-par for all directions |
| PN_FIB | Fib_hits values match Pisano computation |
| PN_MULT | par multiplication table verified |
| PN_W | ≥8 total witnesses |
| PN_F | fundamental direction (2,1) present |

## Source grounding

- **Ben Iverson, QA-2 Ch 3**: par number system definition, gender classification, square rules
- **Cert [148]** Sixteen Identities: C=4-par, G=5-par already proved there; this cert formalizes the par system itself
- **Cert [147]** Synchronous Harmonics: 3-par LOW at 1/4, 5-par HIGH at 1/4 — uses par classification
- **Wall, D.D. (1960)**, "Fibonacci Series Modulo m," *American Mathematical Monthly* 67(6):525-532, DOI:10.1080/00029890.1960.11989541 — rank of apparition / Pisano period theory (also cited by cert [291])

## Connection to other families

- **[147] Synchronous Harmonics**: par interference (same-par SUPPORT, cross-par OPPOSE) — uses this classification
- **[148] Sixteen Identities**: C=4-par, G=5-par proved as part of the 16 identities
- **[128] Spread Period**: Pisano period pi(9)=24; Fibonacci zeros mod m connect to par class
- **[150] Septenary**: {1,2,4,5,7,8} mod 9 = all four par classes represented among units

## Fixture files

- `fixtures/pn_pass_classification.json` — 8 par witnesses + 6 male squares + 6 directions + multiplication table
- `fixtures/pn_pass_fib_hits.json` — 4 Fib_hits observations for par-classified integers

## Verification Note (2026-07-06)

Independently reconfirmed every claim from scratch: par classification for
all 8 witnesses, all 6 male-square results (mod 4), all 6 QA direction
witnesses (C=2de always 4-par, G=d²+e² always 5-par), the full 10-entry
multiplication table (recomputed a×b mod 4 and remapped to par labels for
every entry), and all 4 Fib_hits values (pi(9)=24/hits=2, pi(11)=10/hits=1,
pi(19)=18/hits=1, pi(29)=14/hits=1) — all correct, no arithmetic bugs. The
validator (`qa_par_number_cert_validate.py`) was already genuinely
recomputing everything live (`par_class`, `pisano_period`, `fib_hits` all
computed from scratch, not fixture-trusting) — no hardening needed, same
as [150].

**Found and fixed a real Legendre-symbol error**: the m=29 fib_witness's
explanatory note claimed `(5/29)=-1` (5 is a non-residue mod 29) as the
reason 29 breaks the naive "5-par → 2 hits" pattern. Independently
verified via Euler's criterion (`5^14 mod 29 = 1`) and exhaustive
quadratic-residue enumeration mod 29 (`11² = 121 ≡ 5 mod 29`) that
**5 IS a quadratic residue mod 29** — `(5/29) = +1`, not `-1`. This
doesn't affect the certified `fib_hits=1` value itself (independently
reconfirmed correct), only the free-text explanation for it, which was
factually backwards.

Went further and checked whether a bare Legendre(5,p) value determines
hits count at all: computed hits/Legendre for all primes 3–59 and found
a counterexample already within reach — **p=41 has Legendre(5,41)=+1 but
2 hits**, not 1. So even corrected, `(5/p)=+1 → 1 hit` is not a clean
rule; the fixture's own "observed, not universal" framing was the right
call, and the note now says so explicitly rather than implying a Legendre
symbol alone determines the count. The true determinant is the rank of
apparition α(p) (smallest k with p | F_k) relative to π(p) — a
finer-grained invariant this cert doesn't attempt to characterize, per
Wall (1960).
