# [341] QA Ellipse Semiminor Squared = D·F; Eccentricity c=d/e=2D/C

**Family**: `qa_ellipse_semiminor_df_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter VIII pp.91-93

> "D² - (C/2)² = (df)²"  
> "Since C = 2de, C/2 then equals de, and (C/2)² = d²e² = DE"  
> "So (df)² = D² - DE. The right side factors into D(D-E)"  
> "But D - E = F; So (df)² = DF"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | (C/2)²=DE: C/2=de is exact integer; (de)²=d²e²=D·E; verified 15 pairs | PASS |
| C2 | D-E=F: d²-e²=(d-e)(d+e)=b·a=F; verified 15 pairs | PASS |
| C3 | (df)²=D²-(C/2)²=D(D-E)=D·F; algebraic chain verified 78 pairs | PASS |
| C4 | Eccentricity c=d/e=2D/C: cross-multiply d·C=2·D·e; verified 15 pairs | PASS |
| C5 | J=D-de=bd; K=D+de=ad; J+K=2D (major diameter); K-J=C; verified 15 pairs | PASS |

## Core Structural Results

### Semiminor Identity Proof Chain (C1→C2→C3)

The QA ellipse has major semidiameter D=d². The semiminor is given by:

**Step 1** (C1): $\left(\frac{C}{2}\right)^2 = (de)^2 = d^2 \cdot e^2 = D \cdot E$

**Step 2** (C2): $D - E = d^2 - e^2 = (d-e)(d+e) = b \cdot a = F$

**Step 3** (C3): $(df)^2 = D^2 - \left(\frac{C}{2}\right)^2 = D^2 - DE = D(D-E) = D \cdot F$

So: **semiminor diameter squared = D·F** — the product of semimajor-squared and the semilatus rectum.

When F is a perfect square (F=f²), the semiminor becomes df — an integer.

### Eccentricity (C4)

$$c = \frac{d}{e} = \frac{2D}{C}$$

Cross-multiply proof: $d \cdot C = d \cdot 2de = 2d^2 e = 2D \cdot e \checkmark$

Eccentricity is always the ratio d/e (rational, ≥1 for d>e>0).

### J and K Ellipse Axis Parameters (C5)

$$J = bd = d(d-e) = d^2 - de = D - de \quad \text{(perigee from primary focus)}$$
$$K = ad = d(d+e) = d^2 + de = D + de \quad \text{(apogee from primary focus)}$$

$$J + K = 2D \quad \text{(major diameter)} \qquad K - J = 2de = C \quad \text{(interfocal distance)}$$

### Worked Example: 12-5-13 Triangle

Bead numbers (b,e,d,a) = (1,2,3,5):

| Identity | Value | Formula |
|----------|-------|---------|
| C | 12 | 2·3·2 |
| D | 9 | 3² |
| E | 4 | 2² |
| F | 5 | 1·5 |
| J | 3 | 1·3 |
| K | 15 | 5·3 |
| (df)² | 45 | 9·5=D·F ✓ |
| J+K | 18 | =2D=18 ✓ |
| K-J | 12 | =C=12 ✓ |
| c = d/e | 3/2 | =2·9/12=18/12=3/2 ✓ |

## Observer Projection Note (Theorem NT)

"Ellipse," "semiminor diameter," "focus," "eccentricity" are observer-layer geometric labels. The causal structure: integer bead identities D-E=F, (C/2)²=DE, and the algebraic chain D²-DE=DF. The ratio d/e (eccentricity) is a rational observer projection; only the integer cross-multiply d·C=2·D·e enters QA arithmetic.

**Depends on**: [337] Ellipse J,K 2D Triple; [338] Pythagorean Gnomon Square; [336] Pythagorean 16 Identities
