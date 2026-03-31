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

## Connection to other families

- **[147] Synchronous Harmonics**: par interference (same-par SUPPORT, cross-par OPPOSE) — uses this classification
- **[148] Sixteen Identities**: C=4-par, G=5-par proved as part of the 16 identities
- **[128] Spread Period**: Pisano period pi(9)=24; Fibonacci zeros mod m connect to par class
- **[150] Septenary**: {1,2,4,5,7,8} mod 9 = all four par classes represented among units

## Fixture files

- `fixtures/pn_pass_classification.json` — 8 par witnesses + 6 male squares + 6 directions + multiplication table
- `fixtures/pn_pass_fib_hits.json` — 4 Fib_hits observations for par-classified integers
