# [383] QA Pyth-2 Basics Structural Cert

**Family**: `qa_pyth2_basics_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XI pp.1-27

> *(p.2)*: "Plato wrote of a civilization which existed '9,600 years' before his time, called Atlantis."

> *(p.2)*: "an inscribed counting bone of the Ishango people dated 7000 B.C. This latter showed all of the prime numbers up to 19."

> *(p.2)*: "a dropping of sea level up to 100 meters, and clouding of the atmosphere"

> *(p.3)*: "Pythagoras settled down... and established the first School of the Pythagoreans in 529 B.C."

> *(p.3)*: "About 505 B.C. the Pythagoreans were finally and permanently driven from Crotona... beginning with Plato all mathematical teachings were systematically destroyed for the next 600 years."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Plato's Atlantis 9600 years: 9600 mod 24=0=400×24; diff from Pyth-1's 9400 = 200; 200 mod 24=8=2³ | PASS |
| C2 | Ishango bone 7000 BC: 7000 mod 24=16=Myriad; 8 primes to 19 = 8=2³=φ(30) | PASS |
| C3 | Sea level 100 m drop: 100 mod 24=4=portal 4-par; 100=4×5²; gcd(100,24)=4 | PASS |
| C4 | School 529 BC to expulsion 505 BC: 529−505=24=QA modulus; both mod 24=1 (Singularity-class); 505=5×101 | PASS |
| C5 | 600 years of systematic destruction: 600 mod 24=0; four elements (earth/air/fire/water)=4=QA tuple; 600÷4=150 mod 24=6=seed product | PASS |

## Mathematical Details

### C1: Plato's Atlantis — 9600 vs 9400

Plato records Atlantis as existing "9,600 years" before his time. Pyth-1 Ch.I ([382]) records the sinking as "9,400 years before Pythagoras."

- **9600 mod 24 = 0** (9600 = 400 × 24 — exactly 400 complete QA cycles)
- **9400 mod 24 = 16** (Myriad residue, cert [382])
- Difference: 9600 − 9400 = **200**; 200 mod 24 = **8 = 2³** (same as 11000 BC residue in cert [378])

Plato's epoch (9600) hits the perfect zero, while Pythagoras's epoch (9400) hits the Myriad — the two ancient witnesses triangulate Atlantis with different mod-24 footprints.

### C2: Ishango Bone — 7000 BC and 8 Primes

"an inscribed counting bone of the Ishango people dated 7000 B.C. This latter showed all of the prime numbers up to 19."

- **7000 mod 24 = 16** = Myriad residue (7000 = 291×24 + 16)
- Primes up to 19: {2, 3, 5, 7, 11, 13, 17, 19} = **8 primes**
- **8 = 2³** = same as φ(30) (cert [367] C1: the 8-element coprime-to-30 bracket)

Both the artifact's date (7000 BC mod 24=16) and its mathematical content (8 primes) share residue classes with Myriad theory and the symmetry bracket.

### C3: Sea Level Drop — 100 Meters

"a dropping of sea level up to 100 meters"

- **100 mod 24 = 4** (portal 4-par class — same as Pythagoras 580 BC mod 24=4)
- **100 = 4 × 25 = 4 × 5²**
  - Factor 4 = QA tuple size
  - 25 = 5² (second power of 5-par generator)
- gcd(100, 24) = **4** (shared with QA tuple)

The 100-meter sea drop is in the portal class, matching the QA entry-point residue at each octave transition.

### C4: Pythagorean School Duration = 24 Years

School founded 529 BC; Pythagoreans expelled 505 BC.

- **529 − 505 = 24 = QA modulus** (the school spanned exactly one complete QA cycle)
- **529 mod 24 = 1** (Singularity-class, cert [382])
- **505 mod 24 = 1** (same Singularity-class: 505 = 21×24 + 1)
- **505 = 5 × 101**; 101 is prime

Both founding and dissolution years are in the Singularity class (mod 24=1), and the interval between them equals the QA modulus itself.

### C5: 600 Years of Destruction and Four Elements

"all mathematical teachings were systematically destroyed for the next 600 years" / "His four states of matter, earth, air, fire, and water"

- **600 mod 24 = 0** (600 = 25×24 — same as Christ Spirit gap in cert [379] C2)
- **4 elements** = QA 4-tuple size (b, e, diff=b+e, apex=b+2e)
- 4 mod 24 = **4** (portal class)
- 600 ÷ 4 = **150**; 150 mod 24 = **6** = male seed product (1×1×2×3=6, cert [374])

The 600-year destruction interval and the 4-element structure together encode: 600/4=150, and 150 mod 24=6=seed — the destruction epoch divided by the tuple size returns the seed.

## Theorem NT Note

Chapter XI is a historical introduction/review chapter covering Atlantis archaeological evidence, the Ishango counting bone, Pythagorean school history, and a brief review of Vol. I concepts. The QA structure certified here (Atlantis date, artifact date, sea level, school duration, destruction period, element count) is embedded in the chapter's historical narrative. All historical dates and physical measurements are observer-projection reports; the QA analysis concerns the integer arithmetic of the reported quantities.

**Depends on**: [354] Third Dimension (Myriad residue 16); [367] Prime Number Symmetry (φ(30)=8); [374] QA and Energy (seed product=6); [376] Metaphysics Myriad (portal octave mod24=4); [378] Billy's Story (residue 8=2³); [379] Human Spirit (600 mod24=0); [382] Pyth-1 Recovery of Knowledge (Pythagoras 580 BC, Atlantis 9400yr)
