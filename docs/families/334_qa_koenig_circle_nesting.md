# [334] QA Koenig Circle Nesting

**Family**: `qa_koenig_circle_nesting_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Ch.7 pp.55-70 — "THE KOENIG SERIES AND THE TREE OF LIFE"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | H²-G²=G²-I²=2CF for (I,G,H) from 7 prime Pythagorean triangles | PASS |
| C2 | Annular area=24L where L=CF/12; CF always divisible by 12 | PASS |
| C3 | Nesting H_n=I_{n+1}: chain (1,5,7)→(7,13,17)→(17,25,31) | PASS |
| C4 | H²-G²=G²-I²=2CF and area divisible by 24 for all 32 primitive triples with G≤200 | PASS |
| C5 | Extended nesting: I=7→5 options; I=17→4 options; I=31→3 options | PASS |

## Core Structural Result

### Setup

For any primitive Pythagorean triangle with legs C, F and hypotenuse G:
$$I = C - F, \quad G = \text{hyp}, \quad H = C + F$$

Three concentric circles with radii I, G, H. The two annular areas are:

$$H^2 - G^2 = G^2 - I^2 = 2CF$$

**Proof**: $H^2 - G^2 = (C+F)^2 - (C^2+F^2) = 2CF$  
$G^2 - I^2 = (C^2+F^2) - (C-F)^2 = 2CF$ ✓

G **bisects** the circular area between I and H.

### Area = 24L

Since one leg of every primitive Pythagorean triple is divisible by 4 and the other by 3:
$$CF \equiv 0 \pmod{12} \implies 2CF \equiv 0 \pmod{24}$$

$$L = \frac{CF}{12} = \frac{\text{annular area}}{24}$$

| Triangle | C | F | G | I | H | L | Area |
|----------|---|---|---|---|---|---|------|
| 4-3-5 | 4 | 3 | 5 | **1** | 7 | 1 | 24 |
| 12-5-13 | 12 | 5 | 13 | **7** | 17 | 5 | 120 |
| 24-7-25 | 24 | 7 | 25 | **17** | 31 | 14 | 336 |
| 8-15-17 | 15 | 8 | 17 | **7** | 23 | 10 | 240 |
| 40-9-41 | 40 | 9 | 41 | **31** | 49 | 30 | 720 |

### Nesting Property (C3, C5)

$$H_n = I_{n+1}$$

The H-value of one step becomes the I-value of the next. The standard chain:

$$1 \xrightarrow{(1,5,7)} 7 \xrightarrow{(7,13,17)} 17 \xrightarrow{(17,25,31)} 31 \xrightarrow{\cdots} \cdots$$

The full radius sequence: 1, **5**, 7, **13**, 17, **25**, 31, ... (G values in bold)

Each "step" value (1, 7, 17, 31, ...) appears as both the I of one triple and the H of the previous. The chain is not unique at each step: I=7 has 5 valid continuation triangles, I=17 has 4, I=31 has 3.

## Observer Projection Note (Theorem NT)

"Circle", "annular area", "Koenig series" are observer-layer labels. The causal structure: the algebraic identity $H^2-G^2=(C+F)^2-G^2=2CF=G^2-(C-F)^2=G^2-I^2$. The 24-divisibility follows from the number-theoretic structure of primitive Pythagorean triples (one leg divisible by 4, one by 3). No continuous geometry enters.

**Depends on**: [135] Pythagorean Tree; [289] Koenig-Pell-Ford Circle; [292] Koenig Spread Optimality
