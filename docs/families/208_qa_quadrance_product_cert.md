# Family [208] QA_QUADRANCE_PRODUCT_CERT.v1

## One-line summary

Every QA area element is irreducibly a two-factor product of role-distinct base elements. Quadrances (A=a*a, B=b*b) are products, not powers. S1 (always b*b product form) is structural, not just numerical. A square is a rectangle whose sides happen to be equal — the product never collapses (Will Dale, 2026-04-08).

## Mathematical content

### The irreducible product principle

Every QA element of area type is a product of two base-level elements (b, e, d, a):

| Element | Product form | Factor 1 role | Factor 2 role |
|---------|-------------|---------------|---------------|
| A | a * a | fourth coordinate | multiplier |
| B | b * b | base state | multiplier |
| D | d * d | first derived | multiplier |
| E | e * e | generator | multiplier |
| C | 2 * d * e | derived coord | generator |
| F | b * a | base state | fourth coordinate |
| J | b * d | base state | derived coord |
| X | e * d | generator | derived coord |
| L | C * F / 12 | area (green) | area (red) |

Even for quadrances where both factors have the same VALUE (e.g., B = b * b), the two factors have distinct ROLES: the first `b` is the base state coordinate of the tuple, the second `b` is a scaling multiplier. Same value, different function.

### S1 as structural principle

S1 mandates `b*b`, never the power-operator form. The stated reason is floating-point ULP drift, but the deeper structural reason is:

- `b*b` = product of two operands → preserves two-factor structure
- Power form = unary operation on one operand → collapses two factors into one, destroying role-distinction

S1 is the axiom-level encoding of the irreducible product principle.

### The square/rectangle parallel

**Standard view**: A square is a "special case" of rectangle where length = width. The rectangle loses a degree of freedom.

**QA view**: A square is a rectangle whose two side-roles happen to have equal values. The product (length * width) is IRREDUCIBLE — it never collapses to a single factor.

**Key example**: 1 * 1 = 1. Standard interpretation: "1 squared = 1; the product collapses to a scalar." QA interpretation: 1 * 1 = area of 1 — the product of two unit lengths. Numerically identical to the length 1, but dimensionally distinct.

### Parallel to [207] Circle Impossibility

| Standard "special case" | What it claims | QA reality |
|------------------------|----------------|------------|
| Circle = ellipse with C=0 | Eccentricity vanishes | C=2de >= 2 always; C is hidden along viewing axis |
| Square = rectangle, equal sides | Product collapses to power | Product NEVER collapses; two factors with distinct roles always preserved |

Both are observer projections: the observer sees value-equality and concludes structural degeneracy, but QA preserves the full structure.

### Dimensional type system

QA elements have implicit types:

- **Length type**: b, e, d, a (base coordinates)
- **Area type**: A, B, C, D, E, F, J, X (products of two lengths)
- **Area-squared type**: L = CF/12 (product of two areas)

No operation in QA collapses an area to a length. Even when B = b * b = 1 (numerically equal to the length b = 1), B remains area-typed. The 16 identities [148] preserve dimensional types throughout.

## Checks

| ID | Description |
|----|-------------|
| QP_1 | schema_version == 'QA_QUADRANCE_PRODUCT_CERT.v1' |
| QP_PRODUCT | Product table has >= 6 area elements |
| QP_ROLE | All product elements have role-distinct factors |
| QP_S1 | No formula uses power notation |
| QP_AREA_MIN | Minimum area verified (unit area 1*1 acknowledged) |
| QP_DIM | Dimensional types (length/area) present |
| QP_SQUARE | Square/rectangle parallel articulated |
| QP_SRC | Source attribution to Will Dale |
| QP_WITNESS | >= 3 witnesses with explicit product forms |
| QP_F | fail_ledger well-formed |

## Examples

**Unit area** (1,1): tuple (1,1,2,3). B = 1*1 = 1, E = 1*1 = 1, C = 2*2*1 = 4, F = 1*3 = 3. The minimum quadrances. Still products of two factors. B=1 is NOT the scalar 1.

**Equal-value quadrances** (3,3): tuple (3,3,6,9). B = 3*3 = 9, E = 3*3 = 9. B = E because b = e = 3. But B is base-squared and E is generator-squared — distinct roles. C = 2*6*3 = 36.

**State (2,1)**: tuple (2,1,3,4). B = 2*2 = 4, E = 1*1 = 1, C = 2*3*1 = 6, F = 2*4 = 8. Every element is a product. Note: as a QA state, d=3 (not d=2 as in the Pythagorean direction).

## Connection to other families

- **[207] QA_CIRCLE_IMPOSSIBILITY_CERT.v1**: Parallel impossibility/non-collapse. Circle: C=0 impossible. Square: product structure irreducible. Both are observer projections of "special cases."
- **[148] QA_SIXTEEN_IDENTITIES_CERT.v1**: The 16 identities relate area elements. Every identity preserves product structure.
- **[133] QA_EISENSTEIN_NORM_CERT.v1**: F=ba (product) enters Eisenstein norm. Crystal structure is built from products of products.
- **[137] QA_KOENIG_TWISTED_SQUARES_CERT.v1**: H=C+F, I=C-F. Sums/differences of area elements remain area-type. H*H-G*G=2CF is a product of products.
- **[183] QA_EISENSTEIN_CRYSTAL_CERT.v1**: J=bd certified. Two-factor product structure is load-bearing for crystal constant encoder.
- **[125] QA_CHROMOGEOMETRY_CERT.v1**: F=Qr, C=Qg, G=Qb. All chromogeometric quadrances are products or sums-of-products.

## Source

Will Dale, 2026-04-08. Insight articulated during SVP wiki propositions mapping session, extending the circle impossibility observation: "1 * 1 = 1 squared, not 1. Many QA elements are area — F = b * a, J = b * d."

## Status

- Validator: `qa_alphageometry_ptolemy/qa_quadrance_product_cert_v1/qa_quadrance_product_cert_validate.py`
- Fixtures: 1 PASS + 1 FAIL
- Self-test: pending
