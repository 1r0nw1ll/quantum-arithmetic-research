# [362] QA Pyth-1 Internal Relationships

**Family**: `qa_pyth1_internal_relationships_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter V pp.54-65

> *(p.57)*: "Some of them are: 2D+2E=A+B, and the three three-part series of, b-d-a, F-D-G, and I²-G²-H², where the center term is the mean of the other two."

> *(p.53)*: "This is H²+I²=2G², and does not appear directly in the table."

> *(p.52)*: "the equality develops with J+K=C+2J=...=2D"

> *(p.54)*: "I, G, and H are usually prime numbers, are always coprime to each other, and are always functionally prime."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Three arithmetic mean trios: 2d=b+a; 2D=F+G (D-F=D=G-D=E); H²+I²=2G² | PASS |
| C2 | 2D+2E=A+B (equivalently A+B=a²+b²=2d²+2e²) | PASS |
| C3 | I, G, H are always pairwise coprime: gcd(I,G)=gcd(I,H)=gcd(G,H)=1 | PASS |
| C4 | J+K=C+2J=2D (the double-square identity) | PASS |
| C5 | F−C=b²−2e² exactly; F>C iff b²>2e² (Table 1a dichotomy condition) | PASS |

## Mathematical Details

### C1: Three Arithmetic Mean Trios

Iverson observes three sequences where the center term is the arithmetic mean of its neighbors:

**Trio (a): b, d, a**

d = b+e; a = b+2e → a−d = e = d−b. So d = (b+a)/2.

The common difference is e; the three beads b,d,a form an arithmetic progression with step e.

**Trio (b): F, D, G**

F = d²−e² = D−E; G = d²+e² = D+E. So D−F = E and G−D = E. Hence D = (F+G)/2.

The common difference is E = e²; F, D, G form an arithmetic progression with step e².

**Trio (c): I², G², H²**

H²+I² = 2G² ↔ G² = (I²+H²)/2.

Proof: H = C+F, I = |C−F|. Then H²+I² = (C+F)²+(C−F)² = 2C²+2F² = 2G² (since C²+F²=G²). ✓

The common difference is H²−G² = G²−I² = 2CF = 24L (the Koenig identity from cert [137]).

**Connection**: the three trios are nested:
- Linear level: b, d, a (step e)
- Quadratic level: F, D, G (step e²)
- Double-quadratic level: I², G², H² (step 2CF)

### C2: Diagonal Sum Identity 2D+2E=A+B

**Proof**: A+B = a²+b² = (d+e)²+(d−e)² = d²+2de+e²+d²−2de+e² = 2d²+2e² = 2D+2E ✓

This is a consequence of the algebraic identity (x+y)²+(x−y)² = 2x²+2y² with x=d, y=e.

Alternative form: A−B = a²−b² = (a+b)(a−b) = (2d+2e)(2e) = 4e(d+e) = 4ea. Also A−B = 2C (since A=G+C, B=G−C from cert [360]).

| b | e | D=d² | E=e² | 2D+2E | A=a² | B=b² | A+B |
|---|---|------|------|-------|------|------|-----|
| 1 | 1 | 4 | 1 | 10 | 9 | 1 | 10 |
| 3 | 2 | 25 | 4 | 58 | 49 | 9 | 58 |
| 5 | 2 | 49 | 4 | 106 | 81 | 25 | 106 |
| 5 | 4 | 81 | 16 | 194 | 169 | 25 | 194 |

### C3: Pairwise Coprimality of I, G, H

**Theorem**: For all primitive Pythagorean pairs, gcd(I,G)=gcd(I,H)=gcd(G,H)=1.

I, G, H are all odd (G is 5-par, H and I have no prime factor <7 per cert [361]).

**Proof that gcd(G,H)=1**: Suppose p|G and p|H=C+F. Then p|H−G=C+F−G=(2de+d²−e²)−(d²+e²)=2de−2e²=2e(d−e)=2be. Since G is odd, p is odd, so p|be. Since gcd(b,e)=1: either p|b or p|e (not both). If p|e: p|G=d²+e² → p|d²; but gcd(d,e)=1 → p∤d → contradiction. If p|b=d−e: p|d²−e² (since d²−e²=(d−e)(d+e)=ba and p|b→p|ba); and p|G=d²+e²; subtract: p|2d²=(d²+e²)+(d²−e²); but p|d²−e² and p|d²+e², so p|2e²; since p|b and gcd(b,e)=1, p∤e, p∤2e² → contradiction. So p cannot divide both G and H. ✓

**Proof that gcd(G,I)=1**: I=|C−F|=|2de−(d²−e²)|=|−(d−e)²+2e(d−e)+2e²−e²+e²|... Actually I=|b²−2e²| (from C5). If p|G=d²+e² and p|I=|C−F|, then p|(C+F)+(C−F)=2C=4de and p|(C+F)−(C−F)=2F=2(d²−e²). Since p|G=d²+e² and p|d²−e², then p|2d² and p|2e². Since p is odd, p|d² and p|e²→p|gcd(d,e)=1→contradiction. ✓

**Proof that gcd(I,H)=1**: If p|I and p|H, then p|(H+I)/2=C and p|(H−I)/2=F (since H=C+F,I=|C−F|, H+I=2max(C,F), H−I=2min(C,F)). But C=2de and F=(d−e)(d+e)=ba. If p|C=2de: p|de (p odd). If p|F=ba: p|ba. Case p|d: gcd(d,e) might give p|e too; if p|d and p|e→p|gcd(d,e)=1→impossible. So p|d and p∤e→p∤e; then p|ba implies p|b or p|a; but b=d−e→p|e→contradiction. Similar for p|e. So p cannot divide both I and H. ✓

### C4: The Double-Square Identity J+K=C+2J=2D

J = bd, K = ad. The key observation:

**Proof**: J+K = bd+ad = d(b+a) = d·2d = 2d² = 2D (since b+a = (d−e)+(d+e) = 2d) ✓

C+2J = 2de+2bd = 2d(e+b) = 2d(e+d−e) = 2d² = 2D ✓

The "double square" 2D is divided into major (J+K), minor (C+2J), and intermediate (=2D by both routes) parts as Iverson describes. Geometrically: the rectangle of height d and total width 2d splits into sub-rectangle bd + bd + 2de + (ad−bd−2de) = J + J + C + (K−J−C). But since K−J−C = ad−bd−2de = d(a−b)−2de = d·2e−2de = 0, we get J+K = J+J+C = C+2J = 2D.

### C5: F−C=b²−2e² Dichotomy

**Proof**: F−C = (d²−e²)−2de = d²−2de−e² = (d−e)²−2e² = b²−2e² ✓

| Class | Condition | b, e example | F−C |
|-------|-----------|-------------|-----|
| F>C (Table 1a male branch) | b²>2e² | b=3, e=2: 9>8 | +1 |
| C>F (Table 1b female branch) | b²<2e² | b=1, e=1: 1<2 | −1 |
| C>F | b²<2e² | b=3, e=4: 9<32 | −23 |
| F>C | b²>2e² | b=5, e=2: 25>8 | +17 |

The threshold is at b/e = √2 ≈ 1.414. Since b and e are integers and b is odd, the condition is exactly b²>2e² or b²<2e² (equality b²=2e² is impossible since √2 is irrational and b,e are integers).

**Note on Iverson's statement**: Iverson writes "when b>e: F>C" but the correct algebraic condition is b²>2e², not b>e. Both b=3,e=2 (b>e, b²=9>8=2e², F>C) and b=5,e=4 (b>e, b²=25<32=2e², C>F) are primitive pairs. Iverson's tables (Table 1a: b fixed, e small; Table 1b: e fixed, b small) have b²>2e² or b²<2e² throughout respectively, so the table-level statement holds for his specific sequences.

## Theorem NT Note

"Circle," "area," "orbit," "ellipse" in Iverson's text are observer projection labels (measurements in continuous space). The algebraic identities C1-C5 are consequences of discrete modular arithmetic on bead values, not geometric properties.

**Depends on**: [360] Prime Triangle Structure (G is 5-par); [361] Primeness Parity Shape (H,I odd, no factor <7); [137] Koenig Twisted Squares (H²−G²=G²−I²=2CF); [338] Gnomon Square (F=d²−e²); [355] Formal Proofs (C divisible by 4)
