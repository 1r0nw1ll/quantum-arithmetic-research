# [324] QA Music of the Spheres Scale

**Family**: `qa_music_spheres_scale_cert_v1`  
**Depends on**: [322] Harmonic Aliquot Structure; [323] Harmonic Chemistry LCM; [314] QA Satellite 8-Cycle Orbit

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 8 keynotes = 8 Satellite states: mod-9 Satellite orbit has exactly 8 diagonal states (b=e, b∈{1..8}), corresponding to the 4 bulls + 4 cows (Archimedes Cattle Problem / I-Ching 8 trigrams) | PASS |
| C2 | 17 reduced fractions: exactly 17 fractions p/q with gcd(p,q)=1, 1≤p<q≤7 exist in (0,1); + keynote = 18 notes per key | PASS |
| C3 | 144 = 8 × 18: 8 keynotes × 18 notes = 144 total notes in all 8 keys (Iverson p.82) | PASS |
| C4 | 7-smooth denominators: all 17 scale fractions have denominator prime factors ≤ 7; the 4 prime denominators are {2,3,5,7} | PASS |
| C5 | Farey mediant property: all 16 adjacent fraction pairs in sorted scale satisfy \|p1·q2 − p2·q1\| = 1 (Farey neighbors) | PASS |

## Music of the Spheres — QA Structure

Iverson's Chapter 6 identifies the ancient "Music of the Spheres" tradition as a preservation of QA harmonic knowledge encoded in:
- The **Archimedes Cattle Problem** (4 bulls + 4 cows = 8 keynotes)
- The **I-Ching** (8 trigrams: 4 male "bull" + 4 female "cow")
- The **18-note scale** per keynote, totaling 144 notes in 8 keys

The QA identification is exact: 8 keynotes = 8 Satellite orbit states (the b=e diagonal). The Satellite orbit is the only orbit in mod-9 QA with exactly 8 distinct states (it excludes the Singularity at b=e=9). Iverson (p.81):

> "The curious feature of this book is that it contains eight persons being four males and four females."

## 18-Note Scale Construction

The 17 reduced fractions p/q with denominator ≤ 7 are:

```
1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 2/5, 3/7, 1/2,
4/7, 3/5, 2/3, 3/4, 4/5, 5/7, 5/6, 6/7
```

Adding the keynote itself (ratio 1/1 = the note) gives 18 notes per key.

**Why denominator ≤ 7?** Iverson (p.91): *"harmony depends on fractional relationships from halves to sevenths."* Fractions with denominator ≥ 8 produce non-harmonic (dissonant) intervals — they are not 7-smooth and cannot be part of the Music of the Spheres scale.

## Farey Structure

The 17 scale fractions are **Farey neighbors** in sequential order: for each consecutive pair (p1/q1, p2/q2), |p1·q2 - p2·q1| = 1. This is not a loose approximation — it is exact for all 16 adjacent pairs. Farey-neighbor fractions are the "simplest" fractions separating two scale degrees, with no fraction of smaller denominator lying between them.

This confirms Iverson's claim that the Music of the Spheres scale is maximally harmonic: every step is the simplest possible interval.

## 144 = 8 × 18

Iverson (p.82):
> "Using each of the notes as a 'Keynote' and taking the eighteen low fractional values of each note produced 144 different notes in eight keys."

The arithmetic: 8 Satellite keynotes × 18 notes per key = 144.

QA significance: 144 = 12² = (3·4)² — the square of a harmonic dozen. The Cattle Problem and I-Ching both encode the Satellite orbit structure with exactly this count.

## Connection to Prior Certs

- **[322] Harmonic Aliquot**: the 7-smooth requirement is why scale denominators stop at 7 — they must be aliquot-compatible with the Cosmos d-values ≤ 17
- **[323] LCM Chemistry**: the 18-note scale is the "myriad" in which the C(7,2)=21 harmonic pairs (from cert [323] C3) live
- **[318] Synchronous Harmonics Ceiling**: the 5040 threshold bounds the largest meaningful harmonic cycle; the 18-note scale operates far below this ceiling (max denominator = 7 << 5040)
