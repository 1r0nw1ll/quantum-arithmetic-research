# Family [148] QA_SIXTEEN_IDENTITIES_CERT.v1

## One-line summary

The 16 named quantities (A through L, X, W, Y, Z) of a prime Pythagorean direction satisfy 9 universal algebraic relations including G+C=A, F²+C²=G², H²+I²=2G², and L=CF/12 is always a positive integer.

## Mathematical content

### The 16 identities

Given a primitive direction (d,e) with d>e>0, gcd(d,e)=1, opposite parity, define b=d−e, a=d+e:

| Symbol | Formula | Name/Meaning |
|--------|---------|-------------|
| A | a² | Apogee square |
| B | b² | Base square |
| C | 2de | Hypotenuse = green quadrance Qg(d,e) |
| D | d² | Direction square |
| E | e² | Eccentricity square |
| F | ba = d²−e² | Semi-latus = red quadrance Qr(d,e) |
| G | d²+e² | Blue quadrance Qb(d,e) |
| H | C+F | Outer Koenig circle |
| I | C−F | Inner Koenig circle / conic discriminant |
| J | bd | Perigee |
| K | ad | Apogee-direction |
| L | bade/6 = CF/12 | Quantum volume |
| X | de = C/2 | Half-hypotenuse |
| W | d(e+a) = X+K | |
| Y | a²−d² = A−D | |
| Z | E+K = e²+ad | |

### Algebraic relations (all universal — true for every primitive direction)

1. **G + C = A**: hypotenuse + base = apogee square
2. **G − C = B**: hypotenuse − base = base square
3. **G = (A+B)/2**: G is the arithmetic mean of A and B
4. **F² + C² = G²**: the Pythagorean/chromogeometry theorem
5. **H² + I² = 2G²**: the Koenig circle relation
6. **L = CF/12**: always a positive integer (quantum volume)
7. **W = X + K**
8. **Z = E + K**
9. **Y = A − D**

### Parity rules

- **C is always 4-par** (C ≡ 0 mod 4): C = 2de, and one of d,e is even (opposite parity), so C = 2×(even)×(odd) = 4k.
- **G is always 5-par** (G ≡ 1 mod 4): d²+e² with d,e of opposite parity. odd²+even² = 1+0 = 1 mod 4.

### L integrality proof

L = bade/6 = (d−e)(d+e)·d·e / 6 = F·C / 12. Among consecutive-like integers {b,e,d,a}, the product always contains factors of 2 and 3, so division by 6 is exact. Equivalently: C=2de is divisible by 4 (shown above), F=ba, so CF is divisible by 4·F, and CF/12 = (4k·F)/12 = kF/3, which is integer since among {b,d,a} one is divisible by 3.

## Checks

| ID | Description |
|----|-------------|
| SI_1 | schema_version == 'QA_SIXTEEN_IDENTITIES_CERT.v1' |
| SI_2 | all directions d>e>0, gcd=1, opposite parity |
| SI_IDEN | all 16 quantities recomputed correctly |
| SI_REL | all 9 algebraic relations verified |
| SI_PAR | C ≡ 0 mod 4 (4-par); G ≡ 1 mod 4 (5-par) |
| SI_L | L = CF/12 is a positive integer |
| SI_W | ≥3 direction witnesses |
| SI_F | fundamental (2,1) present |

## Source grounding

- **Ben Iverson, Pyth-1 Ch V**: "Sixteen Identities of a Prime Pythagorean Triangle" — original derivation
- **Ben Iverson, elements.txt**: canonical identity table (A through Z definitions)
- **Arto Heino (artoheino.com)**: independent verification; Y=A−D confirmed canonical
- **Cert [125]** QA_CHROMOGEOMETRY_CERT.v1: C=Qg, F=Qr, G=Qb; F²+C²=G² is Wildberger Thm 6
- **Cert [130]** QA_ORIGIN_OF_24_CERT.v1: H²−G²=G²−I²=2CF=24L for (2,1)

## Connection to other families

- **[125] Chromogeometry**: F,C,G = three chromogeometric quadrances; relation 4 = Wildberger Thm 6
- **[130] Origin of 24**: H²−G²=G²−I²=2CF=24L; derived from relations 4 and 5
- **[133] Eisenstein**: (F,Z,W) and (Y,Z,W) are Eisenstein triples — uses Z,W,Y from this cert
- **[137] Koenig Twisted Squares**: H,I,G from this cert form the arithmetic progression (I²,2CF,G²,H²)
- **[140] Conic Discriminant**: I=C−F as discriminant — identity 10 from this cert
- **[147] Synchronous Harmonics**: C=4-par, G=5-par parity rules connect to par interference

## Fixture files

- `fixtures/si_pass_fundamental.json` — 4 directions: (2,1),(3,2),(4,1),(4,3) with all 16 quantities + 9 relations
- `fixtures/si_pass_witnesses.json` — 6 directions spanning elliptic (I<0) and hyperbolic (I>0) regimes
