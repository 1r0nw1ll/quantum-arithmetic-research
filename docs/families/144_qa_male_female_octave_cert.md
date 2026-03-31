# Family [144] QA_MALE_FEMALE_OCTAVE_CERT.v1

## One-line summary

The Male→Female transform on QA Quantum Numbers (double e, swap b↔e) always multiplies the QN product by exactly 4, placing the female QN exactly 2 octaves above the male.

## Mathematical content

### The transform

Given a male QN (b, e, d, a) with d=b+e and a=b+2e:

```
Step 1: double e  → (b, 2e)
Step 2: swap b↔e → (2e, b)
Derived:
  d_f = b_f + e_f = 2e + b = (b+2e) = a_male
  a_f = b_f + 2e_f = 2e + 2b = 2(b+e) = 2d_male
```

**Female QN = (2e, b, a_male, 2d_male)**

Key structural observations:
- The female's direction (`d_f`) equals the male's apogee (`a`)
- The female's apogee (`a_f`) equals twice the male's direction (`2d`)
- The female's base (`b_f = 2e`) is always even

### Algebraic proof of the 4× ratio

```
male_product   = b × e × d × a
female_product = 2e × b × a × 2d
               = 4 × (e × b × a × d)
               = 4 × male_product  ✓
```

This is an algebraic identity — true for **all** QNs (b,e,d,a), not just special cases.

### Musical interpretation

In frequency terms:
- 1 octave = 2× frequency
- 2 octaves = 4× frequency

Therefore: **female QN is always exactly 2 octaves above the corresponding male QN**.

### Fundamental example

| QN | (b,e,d,a) | Product | Octave position |
|----|-----------|---------|-----------------|
| Male | (1,1,2,3) | 6 | base (0 octaves) |
| Female | (2,1,3,4) | 24 = 4×6 | +2 octaves |

Note that (2,1,3,4) — the female of the fundamental — appears as the fundamental QA direction (d,e)=(2,1) direction vector elsewhere in the cert ecosystem.

### Chaining: the octave tower

The transform composes indefinitely:

| Step | QN | Product | Octaves above male |
|------|----|---------|-------------------|
| 0 | (1,1,2,3) | 6 | 0 |
| 1 | (2,1,3,4) | 24 = 4×6 | +2 |
| 2 | (2,2,4,6) | 96 = 4×24 | +4 |
| 3 | (4,2,6,8) | 384 = 4×96 | +6 |

Each application of the transform adds exactly 2 more octaves.

### More witnesses

| Male (b,e,d,a) | Product | Female (2e,b,a,2d) | Product |
|----------------|---------|---------------------|---------|
| (1,2,3,5) | 30 | (4,1,5,6) | 120 = 4×30 |
| (1,3,4,7) | 84 | (6,1,7,8) | 336 = 4×84 |
| (2,3,5,8) | 240 | (6,2,8,10) | 960 = 4×240 |
| (3,5,8,13) | 1560 | (10,3,13,16) | 6240 = 4×1560 |

## Checks

| ID | Description |
|----|-------------|
| MF_1 | schema_version == 'QA_MALE_FEMALE_OCTAVE_CERT.v1' |
| MF_2 | d=b+e, a=b+2e for all declared QNs |
| MF_TRANS | Female transform: b_f=2e, e_f=b, d_f=a, a_f=2d |
| MF_PROD | female product = 4 × male product |
| MF_OCT | 4× ratio = 2 octaves (musical interpretation declared) |
| MF_W | ≥3 male/female QN pairs |
| MF_F | Fundamental (1,1,2,3)→(2,1,3,4): products 6→24=4×6 |

## Source grounding

- **Ben Iverson QA framework**: male/female QN taxonomy — male QNs typically have b=e (symmetric unit structure); female QNs have b_f=2e (doubled)
- **Dale Pond SVP / Keely**: male and female vibrations as fundamental duality in vibrational science; 2-octave separation is a core Keely claim
- **QA source texts** (memory/qa_source_texts.md): "Male→Female transform: double e, then swap b↔e; result b must be even. (1,1,2,3)→(2,1,3,4). Direct swap stays male."
- **QA source texts**: "Female product = 4× male product; female always exactly 2 octaves above male."

## Connection to other families

- **[143] QA_CUBE_SUM_CERT.v1**: fundamental male (1,1,2,3) product=6; 6³=216=3³+4³+5³
- **[130] QA_ORIGIN_OF_24_CERT.v1**: female product of fundamental = 24 = the QA origin-of-24 constant
- **[135] QA_PYTHAGOREAN_TREE_CERT.v1**: QN (b,e,d,a) structure — d_male = root direction (d,e)=(2,1); female d_f = a_male = 3 = next Fibonacci
- **[128] QA_SPREAD_PERIOD_CERT.v1**: Cosmos orbit period = 24 = female product of fundamental

## Fixture files

- `fixtures/mf_pass_fundamental.json` — algebraic proof + 3 pairs (fundamental + 2 Fibonacci)
- `fixtures/mf_pass_witnesses.json` — 5 pairs including chained transform demonstration
