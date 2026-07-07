# Family [143] QA_CUBE_SUM_CERT.v1

## One-line summary

The fundamental QA triple (F,C,G)=(3,4,5) satisfies both 3²+4²=5² (Pythagorean/2D) and 3³+4³+5³=6³=216 (cube sum/3D), the unique such identity for consecutive integers.

## Mathematical content

### The identity

For the fundamental QA direction (d,e)=(2,1):

```
F = d²−e² = 3
C = 2de    = 4
G = d²+e²  = 5
```

**2D (Pythagorean):** F²+C² = 9+16 = 25 = G² ✓

**3D (Cube sum):**
```
F³ + C³ + G³ = 27 + 64 + 125 = 216 = 6³
```

The triple (3,4,5) is simultaneously the fundamental Pythagorean triple and the fundamental cube-sum triple.

### Proof via the consecutive-integer formula

For any three consecutive integers (k−1, k, k+1):

```
(k-1)³ + k³ + (k+1)³ = 3k³ + 6k = 3k(k² + 2)
```

For k=4:

```
3 × 4 × (16 + 2) = 12 × 18 = 216 = 6³
```

### Uniqueness

**k=4 is the unique positive integer** where (k−1)³+k³+(k+1)³ is a perfect cube.

Verified computationally: no k∈[1,10000] other than k=4 satisfies 3k(k²+2) = n³.

*Note:* this is equivalent to asking whether 3k(k²+2) is a perfect cube. For k=4: factors are 3,4,18=2×3²; product=216=2³×3³=6³. The specific factorization 12×18=(4×3)×(2×9)=6²×6=6³ works only for k=4.

### QA connections

**1. QN product:**

For the fundamental QN (b,e,d,a)=(1,1,2,3) (the Fibonacci-root quantum number):

```
b × e × d × a = 1 × 1 × 2 × 3 = 6
6³ = 216 = F³ + C³ + G³
```

The cube of the QN product equals the cube sum of the fundamental triple.

**2. QA orbit moduli:**

```
216 = 9 × 24 = mod-9 × mod-24
```

216 equals the product of both QA orbit moduli (the Cosmos orbit period in mod-9 is 9, in mod-24 is 24).

**3. Pythagorean tree root:**

Cert [135] establishes (d,e)=(2,1) as the root of the Pythagorean tree. The cube sum identity is a root-specific property — no other tree node (d,e) has F³+C³+G³ a perfect cube (verified for small d,e).

**4. Origin of 24:**

Cert [130]: for the fundamental triple (3,4,5), H²−G²=G²−I²=2CF=24. The minimum of 2CF is exactly 24. And 216=9×24 ties the cube sum to this minimum.

### Newton identity verification

The Newton/Waring identity for power sums:
```
a³+b³+c³ = 3abc + (a+b+c)(a²+b²+c²−ab−bc−ca)
```

For (a,b,c)=(3,4,5):
- 3abc = 3×60 = 180
- a+b+c = 12
- a²+b²+c²−ab−bc−ca = 50−47 = 3
- 180 + 12×3 = 180 + 36 = **216** ✓

## Checks

| ID | Description |
|----|-------------|
| CS_1 | schema_version == 'QA_CUBE_SUM_CERT.v1' |
| CS_2 | F=d²−e², C=2de, G=d²+e², F²+C²=G² |
| CS_IDEN | F³+C³+G³=216=6³ for fundamental (F,C,G)=(3,4,5) |
| CS_DUAL | F²+C²=G² for the same triple (Pythagorean 2D) |
| CS_MOD | 216=9×24 (product of QA orbit moduli) |
| CS_QN | 6=b×e×d×a for QN (1,1,2,3); 6³=216 |
| CS_UNIQ | k=4 unique positive integer in [1,N] with (k−1)³+k³+(k+1)³ a perfect cube |
| CS_W | ≥1 witness (fundamental must be present) |
| CS_F | Fundamental (d,e)=(2,1): (F,C,G)=(3,4,5); F³+C³+G³=216=6³ |

## Connection to other families

- **[130] QA_ORIGIN_OF_24_CERT.v1**: 2CF=24 for fundamental (3,4,5); 216=9×24=mod9×mod24
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: (2,1) is the tree root; cube sum is a root-specific property
- **[137] QA_KOENIG_TWISTED_SQUARES_CERT.v1**: (I²,2CF,G²,H²)=(1,24,25,49) for (2,1); 6=√(G²+H²−I²)/... connection is through the 216=9×24 identity
- **[138] QA_PLIMPTON322_CERT.v1**: Babylonian knowledge of Pythagorean triples; 3³+4³+5³=6³ is not in Plimpton but is implicit in the triple structure
- **[143] this cert**: The 3D partner of [130]'s 2D Pythagorean identity

## Fixture files

- `fixtures/cs_pass_fundamental.json` — main proof: identity, uniqueness (k≤10000), QN product, moduli
- `fixtures/cs_pass_extended.json` — extended: 5 witnesses (1 cube + 4 non-cube confirming uniqueness)
- `fixtures/cs_fail_bad_cube_sum.json` — falsifier: wrong cube_sum + wrong uniqueness solution set (added 2026-07-06)

## Verification Note (2026-07-06)

Independently recomputed F, C, G, and F³+C³+G³ by hand for every witness
in both PASS fixtures — all correct, including a striking near-miss at
(4,3): cube_sum=29792 vs 31³=29791 (off by exactly 1, correctly
identified as non-cube). Independently re-ran the brute-force
`uniqueness_check(10000)` — confirms k=4 is the unique solution, exactly
matching the cert's declared claim. Confirmed `CS_UNIQ` genuinely
brute-forces this (not fixture-trusted) by checking the source: `validate()`
calls `uniqueness_check(up_to)` and compares against declared solutions.
No bugs found — this validator is genuinely computed throughout.

**Found and closed one real gap**: this family had zero FAIL fixtures,
the same gap found in sibling cert [142]. Added
`fixtures/cs_fail_bad_cube_sum.json`, planting two independent,
genuinely-detectable violations (wrong `cube_sum` for the fundamental
witness, and a wrong uniqueness `solutions` list including a spurious
k=7) — confirmed both are caught by `check_direction`/`uniqueness_check`
when checked directly. Updated `_self_test()` accordingly.
