# [360] QA Pyth-1 Prime Triangle Structure

**Family**: `qa_pyth1_prime_triangle_structure_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter II pp.33-38

> *(p.35)*: "the hypotenuse, G, of every prime Pythagorean triangle must be a 5-par, (4n+1), number."

> *(p.35)*: "the difference between C and G is also a square, to be designated as B and is the square of the parametric (bead) number b. The sum of C and G is also a square, to be designated as A, which is the square of the parametric number, a. B and A must be numbers in the form of p^(2n) where p represents an odd prime number."

> *(p.38)*: '"a" is a common factor of A, F, and K; "b" is a common factor of B, F, and J; "d" is a common factor of C, D, J, and K; "e" is a common factor of C and E.'

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | G=d²+e² always ≡1 (mod 4) (5-par); proof: exactly one of (d,e) even | PASS |
| C2 | G+C=A=a² and G-C=B=b²; hence A+B=2G and A-B=2C | PASS |
| C3 | a=d+e always odd; A=a² always 5-par (≡1 mod 4) | PASS |
| C4 | a\|{A,F,K}; b\|{B,F,J}; d\|{C,D,J,K}; e\|{C,E}; abde≡0 (mod 6) | PASS |
| C5 | A, B, G all 5-par; C always 4-par; F spans only 3-par and 5-par (no fixed class) | PASS |

## Mathematical Details

### C1: G is Always 5-par

For all prime Pythagorean pairs (b odd, gcd(b,e)=1), G = d²+e² ≡ 1 (mod 4).

**Proof**: From cert [355] C1, exactly one of {d, e} is even and the other is odd:
- Case d even, e odd: d²≡0 (mod 4), e²≡1 (mod 4) → G≡0+1=1 (mod 4) ✓
- Case d odd, e even: d²≡1 (mod 4), e²≡0 (mod 4) → G≡1+0=1 (mod 4) ✓

G is always 5-par — the hypotenuse lies in the 5-par class with no exceptions.

### C2: The A-B-G Relations

The three quantities A, B, G satisfy a closed system:

| Relation | Formula | Proof |
|---------|---------|-------|
| G+C = A | d²+e²+2de = (d+e)² = a² | Complete square |
| G-C = B | d²+e²-2de = (d-e)² = b² | Complete square (b=d-e) |
| A+B = 2G | a²+b² = 2(d²+e²) | Sum of above |
| A-B = 2C | a²-b² = 4de = 2C | Difference of above |

**Geometric interpretation**: G lies exactly midway between A and B on the number line (A+B=2G). C is exactly half the gap A-B. Every prime Pythagorean triple is embedded in this 3-term arithmetic mean.

Example (3-4-5 triple, b=1,e=1,d=2,a=3): C=4, F=3, G=5, A=9, B=1.
- G+C=5+4=9=3²=A ✓; G-C=5-4=1=1²=B ✓; A+B=10=2×5=2G ✓; A-B=8=2×4=2C ✓

### C3: a is Always Odd

a = d+e where exactly one of {d, e} is even:
- e odd → d=b+e=odd+odd=even → a=even+odd=**odd**
- e even → d=b+e=odd+even=odd → a=odd+even=**odd**

Since b is odd by definition and a is always odd, both B=b² and A=a² lie in the 5-par class.

**Contrast with C and F**:
- C=2de: always even (4-par for male, 2-par for female)
- F=ab: b is odd, a is odd → F is always **odd** (3-par or 5-par, never even)

### C4: Common Factor Structure

Every bead is a factor of specific upper-case identities:

| Bead | Divides | Reason |
|------|---------|--------|
| a | A=a², F=ab, K=ad | a appears as factor in each |
| b | B=b², F=ab, J=bd | b appears as factor in each |
| d | C=2de, D=d², J=bd, K=ad | d appears as factor in each |
| e | C=2de, E=e² | e appears as factor in each |

Note: bead numbers do NOT always divide L. For (b=1,e=1,d=2,a=3): L=1, a=3 → 3∤1. The correct statement is that abde is always divisible by 6 (ensuring L=abde/6 is an integer), not that individual beads divide L.

### C5: Par-Class Summary

| Identity | Par-class | Reason |
|---------|-----------|--------|
| A=a² | always **5-par** | a always odd, odd²≡1(mod 4) |
| B=b² | always **5-par** | b always odd (by definition) |
| G=d²+e² | always **5-par** | one even, one odd (C1) |
| C=2de | always **4-par** (male) | cert [355] C3 |
| D=d² | 4-par if d even, 5-par if d odd | depends on e parity |
| E=e² | 4-par if e even, 5-par if e odd | depends on e parity |
| F=ab | **3-par or 5-par** (never even) | a,b both odd |
| H=C+F | varies | C is 4-par, F is odd → H is odd |
| I=|C-F| | varies | same reasoning |

**Key result**: A, B, G are all in the fixed 5-par class. C is fixed 4-par. F, D, E, H, I vary. The three "corner" identities of the triangle (A,B,G) share the same par-class as each other — a structural triple.

## Theorem NT Note

"Hypotenuse G," "base C," "altitude F," "ellipse apogee/perigee" are observer projection labels for integer arithmetic. The 5-par property of G is a consequence of integer parity arithmetic on beads, not a geometric property of triangles.

**Depends on**: [355] Formal Proofs (parity of d,e; C always 4-par); [337] J,K Parameters (J=bd, K=ad)  
**Extends**: the 5-par class is the identity element of the {3-par, 5-par} multiplication group (from cert [359]); the fact that A, B, G are all 5-par means they are "identity-class" elements in this group structure
