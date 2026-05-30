<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary sources cited in mapping_protocol_ref.json and validator -->

# [283] QA RT Quadrance Orbit Divisibility

**Cert family**: `qa_rt_quadrance_orbit_cert_v1`
**Primary source**:
- Wildberger, N. J. (2005). *Divine Proportions: Rational Trigonometry to Universal Geometry*. Wild Egg Books. ISBN 978-0-9757492-0-8. Chapter 1: quadrance Q(A,B) = (x₂−x₁)² + (y₂−y₁)²
- Mechanism: cert [279] QA Orbit Access Theorem; `qa_orbit_rules.orbit_family on (b, e, 9)`

## Claim

For (b,e) ∈ {1,...,9}², the Wildberger quadrance G = b²+e² has 3-adic valuation determined exactly by the orbit family:

| Orbit Family | Count | G structure | v₃(G) |
|---|---|---|---|
| Cosmos | 72 | gcd(G,3) = 1 | 0 |
| Satellite | 8 | G = 9k, gcd(k,3)=1 | 2 |
| Singularity | 1 | G = 162 = 2×3⁴ | 4 |

Equivalently: **v₃(G) = 2 × v₃(gcd(b,e))** for all 81 pairs in {1,...,9}². Verified exhaustively.

## Structural Significance

- **Why the formula holds**: If gcd(b,e) = 3^k × m with 3∤m, then b = 3^k b', e = 3^k e' with gcd(b',e')=m and 3∤gcd(b',e'). Then G = 9^k(b'²+e'²). Since 3∤gcd(b',e'), we have 3∤(b'²+e'²) (as b'²+e'² ≡ 1 or 2 mod 3 when not both divisible by 3). So v₃(G) = 2k = 2×v₃(gcd(b,e)).

- **Why spread denominators are orbit-class-neutral**: The spread s = (b₁e₂−b₂e₁)²/(G₁G₂) is scale-invariant — replacing (b,e) with (3b,3e) leaves the direction unchanged, so the spread unchanged. Formally (Lagrange identity): the cross product (b₁e₂−b₂e₁) scales by 3^(k₁+k₂) when both pairs scale by their respective 3^k, exactly matching G₁G₂ which scales by 9^(k₁+k₂). The 3-adic factors cancel completely. **The orbit-class 3-adic structure lives in the quadrance G, not in the spread.**

- **Cosmos pair (3,1) is instructive**: b=3 is divisible by 3, but e=1 is not. gcd(3,1)=1, v₃(gcd)=0. G=9+1=10, v₃(10)=0. Correct: even when one component is divisible by 3, if gcd is not, G is coprime to 3. This distinguishes cosmos pairs with one component ≡ 0 mod 3 (gcd still 1) from satellite pairs (gcd divisible by 3).

- **Wildberger RT connection**: In rational trigonometry, quadrance is the primary "distance" object. This cert shows that the 3-adic orbit structure of QA direction pairs is completely captured in the quadrance — the quadrance factorization reveals the orbit class.

## Scope Boundaries

- Does **not** certify spread denominator structure (the spread is scale-invariant; denominators are orbit-class-neutral — see §Structural Significance)
- Does **not** claim G values for pairs outside {1,...,9}²
- Does **not** claim the v₃(G) result extends to mod-24 QA
- The corollary about spread scale-invariance is stated in the scope note but not formally certified here

## Gates

- **RTQ_1**: Cosmos: v₃(G) = 0 (G coprime to 3)
- **RTQ_2**: Satellite: v₃(G) = 2 (G = 9k, gcd(k,3)=1)
- **RTQ_3**: Singularity: v₃(G) = 4 (G = 162 = 2×81)
- **RTQ_4**: v₃(G) = 2×v₃(gcd(b,e)) exhaustive for all 81 pairs
- **SRC**: `mapping_protocol_ref.json` present and well-formed
- **F**: every FAIL fixture declares `expected_fail_type` and fires

6 PASS fixtures, 4 FAIL fixtures. Validator: `qa_rt_quadrance_orbit_cert_validate.py --self-test`.

## Lineage

Discovered 2026-05-30 while formulating the Wildberger RT denominator domain sweep claim. Initial framing ("spread denominators depend on orbit class") failed on the first computation — the spread between (1,2) and (3,3) is 1/10, denominator coprime to 3 despite one satellite pair. The scale-invariance argument resolved the contradiction: the orbit-class 3-adic structure is in G, not in spreads. This refinement is the scientific contribution: the quadrance claim survived; the spread denominator claim was correctly retired.
