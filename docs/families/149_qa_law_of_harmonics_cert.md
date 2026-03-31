# Family [149] QA_LAW_OF_HARMONICS_CERT.v1

## One-line summary

Two QN products sharing all prime factors except one each are in harmonic resonance; the ratio of identity primes measures resonance strength.

## Mathematical content

### Iverson's Law of Harmonics (QA-3 Ch 4)

Given two QN products P1 = b1×e1×d1×a1 and P2 = b2×e2×d2×a2:

1. Compute their **aliquot set** S = prime factors of gcd(P1, P2)
2. Compute **identity primes**: I1 = primes(P1) − S, I2 = primes(P2) − S
3. If |I1| = 1 and |I2| = 1: the pair is **HARMONIC**
4. **Harmony ratio** = min(i1, i2) / max(i1, i2) — closer to 1 = stronger resonance

### Fundamental examples

| QN1 | P1 | QN2 | P2 | Shared | Id1 | Id2 | Ratio |
|-----|----|----|-----|--------|-----|-----|-------|
| (1,2) | 30=2×3×5 | (1,3) | 84=2²×3×7 | {2,3} | 5 | 7 | 5/7=0.714 |
| (1,3) | 84=2²×3×7 | (2,3) | 240=2⁴×3×5 | {2,3} | 7 | 5 | 5/7=0.714 |
| (3,1) | 60=2²×3×5 | (1,3) | 84=2²×3×7 | {2,3} | 5 | 7 | 5/7=0.714 |

### Non-harmonic cases

- P=6 (fundamental) has factors {2,3} only — NO identity prime beyond any shared set
- P=30 vs P=240: shared set {2,3,5} fully contains P=30's primes — id1 is empty

### QN product divisibility by 6

All QN products b×e×(b+e)×(b+2e) are divisible by 6 — guaranteed by the structure of (b,e,d,a) tuples (primes 2 and 3 always present among the factors).

## Checks

| ID | Description |
|----|-------------|
| LH_1 | schema_version == 'QA_LAW_OF_HARMONICS_CERT.v1' |
| LH_ALIQ | aliquot set = prime factors of gcd(P1,P2) |
| LH_IDEN | identity primes = primes(P) − aliquot_set |
| LH_RATIO | harmony ratio correct for harmonic pairs |
| LH_DIV6 | all QN products divisible by 6 |
| LH_W | ≥4 harmonic pairs |
| LH_F | fundamental product P=6 present |

## Source grounding

- **Ben Iverson, QA-3 Ch 4**: "Harmonics occurs between two dissimilar cycles of energy when both can be divided into similar aliquot parts having the same magnitude but different multitudes."
- **Ben Iverson, QA-3 Ch 4**: each wave has identity prime NOT in the aliquot part; minimum 5 primes per wave
- **Memory/qa_source_texts.md**: "Two QNs sharing all but one prime factor are in harmonic resonance; lower ratio of excepted primes → stronger harmony."

## Connection to other families

- **[147] Synchronous Harmonics**: coprime sync + par interference — the dynamic counterpart; this cert is the structural definition
- **[148] Sixteen Identities**: prime factorization of identity values; L=CF/12 connects to QN product structure
- **[144] Male/Female Octave**: male P=6, female P=24; both {2,3} only — same aliquot class, not harmonic

## Fixture files

- `fixtures/lh_pass_harmonic_pairs.json` — 5 pairs (4 harmonic at 5:7 + 1 non-harmonic fundamental)
- `fixtures/lh_pass_fibonacci_chain.json` — Fibonacci QN chain showing adjacent harmonic pattern + non-harmonic contrasts
