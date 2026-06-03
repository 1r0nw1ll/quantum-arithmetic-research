# [311] QA Archaeogeometry Orbit Classification

**Family**: `qa_archaeogeometry_orbit_cert_v1`  
**Depends on**: [310] QA Rational Surveying (squaring map, primitivity criterion), [178] QA Megalithic (Thom data, MY quantum)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | All 5 documented Thom construction Pythagorean triples are primitive and have Cosmos-family generators in both mod-9 and mod-24 (structural from [310] C3: primitive ↔ gcd(b,e)=1 AND b odd → Cosmos) | PASS |
| C2 | The two most common construction triangles (3-4-5 and 5-12-13) both have generators in the Fibonacci sub-orbit (|f|=1, ground state); they are the unique primitives with G ≤ 13 and Fibonacci sub-orbit generators | PASS |
| C3 | The 5 triangles span all 3 mod-9 Cosmos sub-orbits: 2/5 Fibonacci (ranks 1,2), 1/5 Lucas (rank 4), 2/5 Third (ranks 3,5) — complete excitation ladder; minimum-energy sub-orbit contains the 2 most common triangles | PASS |
| C4 | Among 31 distinct integer-MY diameters in Thom 1962 (84 circles, range 4–55 MY), exactly 4 are expressible as primitive Pythagorean hypotenuses G: {5,13,17,29}; 2 smallest → Fibonacci, 2 larger → Third, 0 → Lucas | PASS |
| C5 | Theorem NT: diameter in feet is observer (float); integer-MY diameter is QA layer (int); orbit_fam is discrete QA classification; all 5 triangle spreads F²/G² are exact Fractions with s_F + s_C = 1 | PASS |

## The excitation ladder

Thom documented five Pythagorean triangles in megalithic construction. Their QA generators (b,e) from the squaring map G = d²+e², d = b+e:

| Rank | Triangle | Generator | Sub-orbit | |f| |
|------|----------|-----------|-----------|-----|
| 1 | 3-4-5 | (1,1) | **Fibonacci** | 1 |
| 2 | 5-12-13 | (1,2) | **Fibonacci** | 1 |
| 3 | 8-15-17 | (3,1) | Third | 11 |
| 4 | 7-24-25 | (1,3) | Lucas | 5 |
| 5 | 12-35-37 | (5,1) | Third | 29* |

*\* (5,1) has f-norm 29 in unbounded integers but reaches seed (1,4) under mod-9 T-stepping, placing it in the Third orbit. Mod-9 wraps multiple unbounded orbits into one 24-state cycle.*

The sub-orbit is identified by which mod-9 canonical seed appears in the 24-step T-orbit:
- Fibonacci: orbit contains (1,1)
- Lucas: orbit contains (2,1)  
- Third: orbit contains (1,4)

**The two most common construction triangles (rank 1 and 2) are in the ground state.**  
Both 3-4-5 and 5-12-13 achieve minimum Eisenstein norm |f|=1, placing them at the lowest energy level of the QA excitation ladder. They are the only two primitive Pythagorean triples with hypotenuse G ≤ 13 that achieve this.

## Wildberger spreads (Theorem NT boundary)

For each construction triangle, the bearing angle (arctan F/C) is transcendental — observer only. The Wildberger spread is exact:

| Triangle | F²/G² | C²/G² | Sum |
|----------|-------|-------|-----|
| 3-4-5 | 9/25 | 16/25 | 1 |
| 5-12-13 | 25/169 | 144/169 | 1 |
| 8-15-17 | 225/289 | 64/289 | 1 |
| 7-24-25 | 49/625 | 576/625 | 1 |
| 12-35-37 | 1225/1369 | 144/1369 | 1 |

## Diameter expressibility (C4)

Fermat's sum-of-two-squares theorem: an integer N is expressible as d²+e² with gcd(d,e)=1, d≢e(mod 2) iff N has no prime factor ≡3 (mod 4) raised to an odd power. Only 4 of the 31 distinct Thom 1962 integer-MY diameters meet this condition:

| G (MY) | Generator | Sub-orbit | Circles using this D |
|--------|-----------|-----------|---------------------|
| 5 | (1,1) | Fibonacci | 2 (Loch Creran, Spittal of Glenshee) |
| 13 | (1,2) | Fibonacci | 2 (Nine Ladies, Trecastle SW) |
| 17 | (3,1) | Third | 2 (Usk River East, Esslie Minor) |
| 29 | (3,2)→(b=3,e=2) | Third | 1 |

The Fibonacci-orbit diameters (5 and 13 MY) are the two smallest; the Third-orbit ones are larger. Zero circles have a Lucas-orbit primitive hypotenuse in this dataset.

## Honest caveats

- The 5-triangle sample is small (n=5); the excitation-ladder "coverage" claim is structural but not statistically strong at this sample size.
- Most Thom circle diameters are NOT expressible as primitive Pythagorean hypotenuses — this is expected from number theory (many integers have prime factors ≡3 mod 4). The Pythagorean triangle interpretation applies to Thom's documented construction templates, not to the full diameter distribution.
- The fathom preference (74.3% even MY from cert [178]) is a stronger statistical result; this cert adds the discrete orbit structure of the Pythagorean construction template subset.
