# [340] QA Pythagorean A+B=2G and a+b=2d Bead Median Identities

**Family**: `qa_pythagorean_ab_2g_median_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter VII pp.62-64

> "(d+e)² + (d-e)² = 2G. But since d+e=a and d-e=b the formula reduces to  
>  a²+b²=2G, or A+B=2G."  
> "the value of G is the median value between A and B."  
> "d is the median value of a and b."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | a+b=2d — d is the bead median of a and b; verified 15 pairs | PASS |
| C2 | A+B=2G — G is the mediant square (G=(A+B)/2); verified 15 pairs | PASS |
| C3 | Algebraic proof: A+B=(b+2e)²+b²=2b²+4be+4e²=2(d²+e²)=2G; 78 pairs | PASS |
| C4 | F+G=2D — D is the mediant of F and G; verified 15 pairs | PASS |
| C5 | All three median identities hold simultaneously for 15 coprime pairs | PASS |

## Three Bead-Number Median Identities

**Bead median**: The value x is the mediant of y and z when y+z=2x (x is their arithmetic mean).

### Identity C1 — d is median of a and b

$$a + b = (b + 2e) + b = 2b + 2e = 2(b+e) = 2d$$

### Identity C2 — G is mediant of A and B

$$A + B = a^2 + b^2 = (b+2e)^2 + b^2 = 2b^2 + 4be + 4e^2$$

$$2G = 2(d^2 + e^2) = 2(b+e)^2 + 2e^2 = 2b^2 + 4be + 2e^2 + 2e^2 = 2b^2 + 4be + 4e^2 \checkmark$$

Equivalently from Iverson's form: $(d+e)^2 + (d-e)^2 = 2d^2 + 2e^2 = 2G$, and $d+e=a$, $d-e=b$.

### Identity C4 — D is mediant of F and G

$$F + G = ab + d^2 + e^2 = b^2 + 2be + (b+e)^2 + e^2 = 2(b+e)^2 = 2d^2 = 2D$$

### Mediant Structure of the 16 Identities

All three identities place one square as the arithmetic mean of two others:

| Mediant | Identity Pair | Mean |
|---------|---------------|------|
| a+b=2d | a, b linear | d linear |
| A+B=2G | A=a², B=b² squares | G=d²+e² |
| F+G=2D | F=ab, G=d²+e² | D=d² |

## Observer Projection Note (Theorem NT)

"Median," "average," "midway" are observer-layer labels. The causal structure is the integer bead algebra: a+b=2d, A+B=2G, F+G=2D — three exact integer identities with algebraic proofs.

**Depends on**: [336] Pythagorean 16 Identities; [337] Ellipse J,K 2D Triple; [338] Gnomon Square
