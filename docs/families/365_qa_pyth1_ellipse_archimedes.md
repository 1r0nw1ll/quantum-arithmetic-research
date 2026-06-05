# [365] QA Pyth-1 Ellipse of Archimedes

**Family**: `qa_pyth1_ellipse_archimedes_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter VIII pp.87-93

> *(Fig.8 caption)*: "The major diameter is 2D which is equal to J+K, and equal to C+2J, or to F+G."

> *(p.91)*: "the value of 2D/C = d/e = c, has an additional service to perform in Quantum Arithmetic"

> *(p.93)*: "(df)² = D² − (C/2)² = D(D−E) = DF"

> *(p.93)*: "F can be a square number only when both of its parametric numbers, (F=ab) are also square numbers."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 2D=J+K=F+G=C+2J (three equal forms for the major ellipse diameter) | PASS |
| C2 | K−J=C and K+J=2D (focal-chord sum/difference identities) | PASS |
| C3 | (semiminor diameter)²=DF; semiminor is integer iff F is a perfect square | PASS |
| C4 | QA eccentricity = 2D/C = d/e (bead-number ratio, always >1) | PASS |
| C5 | J·K=DF; also K/J=a/b (apogee/perigee ratio equals bead ratio a to b) | PASS |

## Mathematical Details

### C1: Major Diameter — Three Equal Forms

**2D = J+K**:
J + K = bd + ad = d(b+a) = d·2d = 2d² = 2D ✓

(since b+a = (d−e)+(d+e) = 2d)

**2D = F+G**:
F + G = (d²−e²)+(d²+e²) = 2d² = 2D ✓

**2D = C+2J**:
C + 2J = 2de + 2bd = 2d(e+b) = 2d·d = 2d² = 2D ✓

(since b+e = (d−e)+e = d)

Verified for all 512 prime pairs (b,e)≤35.

### C2: Focal-Chord Decomposition

**K−J = C** (distance between foci = base of triangle):
K − J = ad − bd = (a−b)d = 2e·d = 2de = C ✓

(since a−b = (d+e)−(d−e) = 2e)

**K+J = 2D** (same as J+K = 2D from C1, restating as sum):

This means J and K are symmetric about D: J = D−C/2 and K = D+C/2, so the perigee and apogee differ from the semimajor diameter D by exactly C/2 (half the interfocal distance).

Verified for all 512 prime pairs (b,e)≤35.

### C3: Semiminor Diameter via Pythagorean Theorem

In the ellipse constructed from a prime Pythagorean triangle, there is an internal right triangle with:
- hypotenuse = D (semimajor diameter, from focus to end of minor axis)
- base = C/2 = de (half the interfocal distance)
- altitude = semiminor diameter

**Proof**:

(semiminor)² = D² − (C/2)²  
= D² − (de)²  
= D² − D·E    (since d²e² = d²·e² = D·E)  
= D(D − E)  
= D·F          (since D − E = d²−e² = (d−e)(d+e) = b·a = F) ✓

**Integer semiminor condition**: semiminor = √(DF) = d·√F (since D=d² is always a perfect square). The semiminor is an integer if and only if F=ab is a perfect square.

For a primitive pair, gcd(a,b)=1 (proven: gcd(a,b)=gcd(d+e,d−e) divides gcd(2d,2e)=2gcd(d,e); since gcd(b,e)=1 implies gcd(d−e,e)=1 and d odd or e odd, gcd(a,b)=1). For coprime a,b: ab=□ iff a=□ and b=□ separately.

Found 20 cases with F a perfect square up to (b,e)≤100. First three: (b=1,e=4): F=9=3², semiminor=3d=15; (b=1,e=12): F=25=5², semiminor=5d=65; (b=1,e=24): F=49=7², semiminor=7d=175.

### C4: QA Eccentricity

Iverson defines the eccentricity of the QA ellipse as:

c = 2D/C = 2d²/(2de) = d/e

This is the bead-number ratio d/e, expressed as an exact rational. It is always greater than 1 (since d=b+e>e for all b≥1), while the conventional eccentricity of an ellipse satisfies 0<ε<1. Iverson notes this is the "reciprocal of the currently accepted practice."

Example: (b=1,e=2): d=3, e=2, C=12, D=9; eccentricity = 2×9/12 = 3/2 = d/e ✓

Verified exactly (Fraction arithmetic, no floating point) for all 512 prime pairs (b,e)≤35.

### C5: J·K = DF (Product Identity)

**Proof**:
J·K = bd·ad = ab·d² = F·D = DF ✓

This has a structural consequence: from C3, DF = (semiminor diameter)². Therefore J·K = (semiminor diameter)² — the product of the perigee and apogee distances from the primary focus equals the square of the semiminor diameter.

**Also**: K/J = ad/bd = a/b. The ratio of apogee to perigee equals the ratio of the bead lengths a to b. Since a>b always, K>J always.

Verified for all 512 prime pairs (b,e)≤35.

## Theorem NT Note

"Ellipse," "apogee," "perigee," "orbit," "eccentricity," "pin and string," "orbit of Earth around Sun" are all observer projections. Chapter VIII applies the integer bead arithmetic of Ch.I-VII to label the parts of an ellipse, but the algebraic identities C1-C5 are purely about the six bead-arithmetic quantities J, K, D, F, G, C — no geometric objects enter the QA computation layer.

The continuous-coordinate ellipse picture (Fig.7, Fig.8) is an observer projection of the discrete quantum group (b,e,d,a) → (C,D,E,F,G,J,K). Theorem NT is satisfied: the bead arithmetic causes the ellipse structure, not the reverse.

**Depends on**: [360] Prime Triangle Structure; [362] Internal Relationships (J+K=2D identity); [363] External Relationships (J,K,C,D relationships)
