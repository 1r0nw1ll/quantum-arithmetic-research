# Family [203] QA_SEFER_YETZIRAH_COMBINATORICS_CERT.v1

## One-line summary

The Sefer Yetzirah (Book of Formation, c. 2nd-6th century CE) contains the earliest known systematic treatment of combinatorics — 231 gates = C(22,2), factorial computation up to 7!=5040, and the 3-7-12 partition of 22 letters — all of which are discrete combinatorial structures compatible with QA's modular arithmetic framework.

## Background

**The Sefer Yetzirah** is the oldest Kabbalistic text, traditionally attributed to Abraham but dated by scholars to between the 2nd and 6th centuries CE. It describes creation through the manipulation of the 22 Hebrew letters and 10 Sefirot (numerical emanations), using explicitly combinatorial reasoning.

**Mathematical significance**: Gandz (1940s) identified the Sefer Yetzirah as containing the earliest known systematic use of factorial reasoning in Western intellectual history. Glaz (2021) provided a modern mathematical analysis, identifying the 231 gates as the complete graph K₂₂ and connecting the text to Pythagorean number theory via Iamblichus.

## Mathematical content

### GATES: 231 = C(22,2) = K₂₂

"Twenty-two Foundation Letters: He placed them in a circle like a wall with 231 Gates." (SY 2:4)

The 22 Hebrew letters arranged on a circle, with every pair connected by a line segment ("gate"), form the complete graph K₂₂ on 22 vertices. The number of edges is C(22,2) = 22×21/2 = **231**.

Factorization: 231 = 3 × 7 × 11 — factors through both elements of the 3-7-12 partition (3 and 7).

### FACT: Factorial computation (earliest known)

"Two stones build 2 houses / Three stones build 6 houses / Four stones build 24 houses / Five stones build 120 houses / Six stones build 720 houses / Seven stones build 5040 houses / From here on go out and calculate that which the mouth cannot speak and the ear cannot hear." (SY 4:16)

| n | n! | Digital root | mod 24 |
|---|-----|-------------|--------|
| 2 | 2 | 2 | 2 |
| 3 | 6 | 6 | 6 |
| 4 | **24** | 6 | **0** |
| 5 | 120 | 3 | 0 |
| 6 | 720 | **9** | 0 |
| 7 | 5040 | **9** | 0 |

Key observations:
- 4! = 24 = QA applied modulus = π(9) = min non-trivial Pisano FP
- dr(n!) converges to 9 for n ≥ 6 (since n! contains factors 3 and 6, hence divisible by 9)
- n! mod 24 = 0 for n ≥ 4 (since n! contains factors 3, 4, and 8, hence divisible by 24)
- Some manuscripts give 6! = 620 instead of 720 — scribal error across centuries of transmission

### PART: 3-7-12 partition

The 22 Hebrew letters partition into three classes:

| Class | Count | Letters | Domain |
|-------|-------|---------|--------|
| Mother (Imot) | 3 | Aleph, Mem, Shin | Elements/dimensions (fire, water, air) |
| Double (Kefulot) | 7 | Bet, Gimel, Dalet, Kaf, Pe, Resh, Tav | Planets/days of week |
| Simple (Peshutot) | 12 | He, Vav, Zayin, Chet, Tet, Yod, Lamed, Nun, Samekh, Ayin, Tsadi, Qof | Zodiac/months |

3 + 7 + 12 = 22. Mod-9 values: mothers=3, doubles=7, simples=3.

QA mapping candidates:
- 3 = triune structure (singularity residue, Keely [153])
- 7 = septenary (Iverson Iota interior: 3+5−1=7, [150])
- 12 = zodiac (mod-12 pitch class, half of mod-24)

### PATHS: 32 = 10 + 22 = 2⁵

"With 32 mystical paths of Wisdom..." (SY 1:1)

32 = 10 Sefirot + 22 letters = 2⁵. The Sefirot may have been represented as a Tetractys (1+2+3+4=10), shared with the Pythagorean tradition.

### CIRC: Oscillating circle

"The Circle oscillates back and forth" — cyclic structure on Z/22Z. Combined with "Their end is imbedded in their beginning and their beginning in their end" (SY 1:7), this establishes the cyclical topology natural to modular arithmetic.

### PYTH: Pythagorean transmission

Iamblichus records Pythagoras learning number theory in Canaan and meditating at Mount Carmel (a center for numerology since Elijah, 900-849 BCE). Both traditions share: abstract number manipulation, secret oral transmission, reverence for number properties, and the Tetractys/Sefirot structure (1+2+3+4=10).

### TZERUF: Permutation groups

Abraham Abulafia (13th century) systematized letter permutation (tzeruf) as a meditative and combinatorial practice. The Sefer Yetzirah's factorial passage (4:16) counts |Sₙ| = n! — the order of the symmetric group on n elements. This is a group action on discrete state space, analogous to QA's generator T acting on (b,e) pairs.

## Checks

| ID | Description |
|----|-------------|
| SYC_1 | schema_version == 'QA_SEFER_YETZIRAH_COMBINATORICS_CERT.v1' |
| SYC_GATES | C(22,2) = 231; factorization 3 × 7 × 11 |
| SYC_FACT | n! correct for n=2..7; dr convergence to 9 for n≥6 |
| SYC_PART | 3 + 7 + 12 = 22 |
| SYC_PATHS | 10 + 22 = 32 = 2⁵ |
| SYC_NUM | numerical verification of all combinatorial formulas |
| SYC_W | >= 5 witnesses |
| SYC_F | falsifier: wrong combination formula (C(22,3) instead of C(22,2)) |

## Source grounding

- **Glaz** (2021): *Bridges Conference Proceedings* pp. 39-46. Modern mathematical analysis.
- **Gandz** (1940s): *Proceedings of the AAJR*. Earliest academic analysis of SY combinatorics.
- **Hayman** (2004): *Sefer Yesira: Critical Edition*, Mohr Siebeck. 3-7-12 partition analysis.
- **Kaplan** (1991): *Sefer Yetzirah: The Book of Creation*, Weiser Books. Mathematical appendices.
- **Idel** (1989): *Language, Torah, and Hermeneutics in Abraham Abulafia*, SUNY Press. Tzeruf.
- **Eco** (1993): *The Search for the Perfect Language*, Blackwell. Lullian combinatorics from tzeruf.

## Connection to other families

- **[202] Hebrew Mod-9 Identity**: Aiq Bekar mod-9 structure; digital root homomorphism
- **[192] Dual Extremality**: 4!=24 as Pisano period; mod-24 bridge
- **[130] Origin of 24**: independent derivation of 24 as structural constant
- **[153] Keely Triune**: 3 mother letters = triune structure
- **[150] Septenary**: 7 double letters = septenary structure
- **[191] Bateson Learning Levels**: 3-7-12 as tiered classification

## Fixture files

- `fixtures/syc_pass_core.json` — 7 claims with full witnesses and sources
- `fixtures/syc_pass_numerical.json` — numerical verification of gates, factorials, partition, paths
- `fixtures/syc_fail_wrong_gates.json` — C(22,3)=1540 instead of C(22,2)=231 rejected
