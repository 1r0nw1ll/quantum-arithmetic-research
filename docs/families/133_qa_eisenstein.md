# [133] QA Eisenstein Cert

**Schema**: `QA_EISENSTEIN_CERT.v1`
**Directory**: `qa_alphageometry_ptolemy/qa_eisenstein_cert_v1/`
**Validator**: `qa_eisenstein_cert_validate.py`

## What It Certifies

Two universal Eisenstein-norm identities arising from QA elements, holding for **all** QA tuples (b,e,d,a) with d=b+e, a=b+2e:

| Identity | Formula | Elements |
|----------|---------|---------|
| **Eisenstein 1** | F² − FW + W² = Z² | F=ab, W=d(e+a), Z=e²+ad |
| **Eisenstein 2** | Y² − YW + W² = Z² | Y=A−D=a²−d²=e(2b+3e) |

The expression `a² − ab + b² = c²` is the **Eisenstein integer norm**: N(a + bω) = c where ω = e^(2πi/3). Elements W and Z sit naturally in the Eisenstein/triangular lattice per QA Law 15 (elements.txt): W is the **equilateral side** and Z is the **Eisenstein companion**.

## Algebraic Proof

Let u = b²+3be. Then:
- F = ab = b(b+2e)
- W = d(e+a) = (b+e)(b+3e)
- F + W = b(b+2e) + (b+e)(b+3e) = 2u + 3e²
- FW = b(b+2e)·(b+e)(b+3e) = u(u+2e²)
- Z = e² + ad = b² + 3be + 3e² = u + 3e²

Therefore:
> (F+W)² − 3FW = (2u+3e²)² − 3u(u+2e²) = u² + 6ue² + 9e⁴ = (u+3e²)² = Z²

which is exactly the Eisenstein norm identity F²−FW+W²=Z² (since a²−ab+b²=c² ↔ (a+b)²−3ab=c²).

For Identity 2: Y = e(2b+3e) satisfies the same identity by computational verification (121 pairs b,e≤11, 0 failures). A direct algebraic proof follows the same u-substitution path.

## Fundamental Witness

(b,e,d,a) = (1,1,2,3):

| Element | Value | Formula |
|---------|-------|---------|
| F | 3 | ab = 3·1 |
| W | 8 | d(e+a) = 2·4 |
| Z | 7 | e²+ad = 1+6 |
| Y | 5 | a²−d² = 9−4 |

- F²−FW+W² = 9−24+64 = **49 = 7²** ✓
- Y²−YW+W² = 25−40+64 = **49 = 7²** ✓

The classical **Eisenstein triple (3,5,7)** appears in the isosceles case b=e=1: F²+F·Y+Y² = 9+15+25 = 49 = Z² (triangular form).

## Checks

| ID | Check |
|----|-------|
| EIS_1 | `schema_version == 'QA_EISENSTEIN_CERT.v1'` |
| EIS_2 | `F = a·b` |
| EIS_3 | `W = d·(e+a)` |
| EIS_4 | `Z = e² + a·d` |
| EIS_5 | `Y = a² − d²` |
| EIS_6 | `F·F − F·W + W·W = Z·Z` |
| EIS_7 | `Y·Y − Y·W + W·W = Z·Z` |
| EIS_W | ≥3 witnesses (witness fixture) |
| EIS_U | Fundamental witness (b,e,d,a)=(1,1,2,3) present |

## Fixtures

| Fixture | Type | Expected |
|---------|------|----------|
| `eisenstein_pass_fundamental.json` | Anchor — (1,1,2,3), F=3,W=8,Z=7,Y=5 | PASS |
| `eisenstein_pass_witnesses.json` | 6 witnesses + general theorem + algebraic proof | PASS |

## Connection to Prior Art Convergence Stack

See `docs/QA_PRIOR_ART_CONVERGENCE.md` for the full stack.

QA elements W and Z are Eisenstein integer lattice elements. The norm N(F+Wω)=F²−FW+W² is the natural norm in ℤ[ω] (the Eisenstein integers, or equivalently the Hurwitz quaternions restricted to the Eisenstein plane). Every QA tuple produces two Eisenstein triples for free — one via the semi-latus rectum F, one via the differential Y=A−D.

- Predecessor: **[130]** QA_ORIGIN_OF_24 (W,Z from equilateral dissections — QA Law 15)
- Predecessor: **[125]** QA_CHROMOGEOMETRY (C,F,G as colored quadrances; W,Z are composites)
- Predecessor: **[127]** QA_UHG_NULL (QA triples = null points in UHG over ℤ)
- Cert gap successor: **QA_CYCLIC_QUAD_CERT.v1** (Ptolemy + Eisenstein → null quadrangle)

## Sources

- Ben Iverson, elements.txt — QA Law 15: Eisenstein Lattice & Equilateral Dissections (W, Z, Y, F)
- Eisenstein integer theory: ℤ[ω], ω = e^{2πi/3}, norm N(a+bω) = a²−ab+b²
- Computational verification: all (b,e) with 1≤b,e≤11, 121 pairs, 0 failures for both identities
