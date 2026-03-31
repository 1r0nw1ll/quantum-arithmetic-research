# Family [152] QA_EQUILATERAL_TRIANGLE_CERT.v1

## One-line summary

The equilateral triangle identities W, Y, Z satisfy two Eisenstein norm relations (F²-FW+W²=Z², Y²-YW+W²=Z²) and the key dual identity Y=A-D=C+E bridging square and product layers.

## Mathematical content

### Definitions

| Symbol | Formula | Equivalent |
|--------|---------|-----------|
| W | d(e+a) | X+K = de+da |
| Y | a²-d² | C+E = 2de+e² |
| Z | e²+ad | E+K |

### Dual identity for Y

Y = A-D = a²-d² = (d+e)²-d² = 2de+e² = C+E

This bridges:
- The **square layer**: A=a², D=d² (individual element squares)
- The **product layer**: C=2de (chromogeometric green quadrance), E=e² (eccentricity square)

### Eisenstein norm relations

Both (F,W,Z) and (Y,W,Z) are **Eisenstein triples**: they satisfy x²-xy+y²=z².

- F²-FW+W² = Z² (proved in [133])
- Y²-YW+W² = Z² (new in this cert)

### Sum rule

F + Y = W

Proof: F = d²-e² = (d-e)(d+e) = ba. Y = 2de+e². F+Y = d²-e²+2de+e² = d²+2de = d(d+2e) = d·a+(de-de) ... actually: W = de+da = d(e+a). F+Y = (d²-e²)+(2de+e²) = d²+2de = d(d+2e). And d+2e = e+a (since a=d+e, d+2e=d+e+e=a+e). So F+Y = d(a+e) = da+de = W. QED.

## Checks

| ID | Description |
|----|-------------|
| ET_1 | schema_version == 'QA_EQUILATERAL_TRIANGLE_CERT.v1' |
| ET_DEF | W, Y, Z computed correctly from (d,e) |
| ET_DUAL | Y = A-D = C+E (both equalities hold) |
| ET_EIS | F²-FW+W² = Z² AND Y²-YW+W² = Z² |
| ET_SUM | F + Y = W |
| ET_W | ≥3 direction witnesses |
| ET_F | fundamental (2,1) present |

## Source grounding

- **Ben Iverson, QA-2 Ch 7**: "Equilateral Triangles" — W, Y, Z definitions and their geometric meaning
- **Ben Iverson, elements.txt**: Y=A-D canonical; W=X+K; Z=E+K
- **Cert [133]** QA_EISENSTEIN_CERT.v1: already certifies (F,W,Z) and (Y,W,Z) as Eisenstein triples

## Connection to other families

- **[133] Eisenstein**: both Eisenstein norms; this cert adds the dual identity Y=C+E=A-D and the sum F+Y=W
- **[148] Sixteen Identities**: A, B, C, D, E, F, X, K all defined there; W, Y, Z complete the identity set
- **[125] Chromogeometry**: C=Qg, F=Qr; Y=C+E adds the equilateral layer to the chromogeometric picture

## Fixture files

- `fixtures/et_pass_eisenstein.json` — 6 directions: both Eisenstein norms + dual identity + F+Y=W
- `fixtures/et_pass_dual_identity.json` — 4 UNIFORM_A chain directions showing Y=C+E=A-D pattern
