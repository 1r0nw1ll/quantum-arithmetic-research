# Family [181] QA_SATELLITE_PRODUCT_SUM_CERT.v1

## One-line summary

The sum of products Σ(b·e·d·a) over all satellite pairs equals M⁴ for any modulus M divisible by 3, with the normalized sum always equal to 81 = 3⁴.

## Mathematical content

### Satellite product sum identity

For modulus M (where 3 | M), the satellite orbit contains exactly 8 pairs (b, e) with d = (b+e) % M, a = (b+2e) % M. Define the product-sum:

    SPS(M) = Σ_{satellite} b · e · d · a

**Theorem**: SPS(M) = M⁴ for all M divisible by 3.

**Corollary**: Normalizing by (M/3)⁴ gives SPS_norm = 81 = 3⁴ for every valid modulus. This means the satellite total QA volume equals the singularity volume (singularity state (M/3, M/3) has product M⁴/81, and 81 × that = M⁴).

### Verification

Verified across 33 moduli: M ∈ {3, 6, 9, 12, 15, ..., 99}. All satisfy SPS(M) = M⁴ exactly.

### Algebraic structure

- The 8 satellite pairs are closed under the QA step operator
- Each pair contributes a product b·e·d·a that is a quartic polynomial in M
- The sum telescopes to M⁴ due to the symmetric placement of satellite states

## Checks

| ID | Description |
|----|-------------|
| SPS_1 | schema_version == 'QA_SATELLITE_PRODUCT_SUM_CERT.v1' |
| SPS_PROOF | algebraic proof that Σ(beda) = M⁴ |
| SPS_COUNT | satellite orbit has exactly 8 pairs per modulus |
| SPS_SUM | computed sum matches M⁴ for all tested moduli |
| SPS_TUPLES | all satellite tuples satisfy d = b+e, a = b+2e (mod M) |
| SPS_CLOSURE | satellite orbit is closed under QA step |
| SPS_COROL | normalized sum = 81 = 3⁴ for every modulus |
| SPS_W | ≥3 witnesses (distinct moduli) |
| SPS_F | ≥1 falsifier (wrong sum value rejected) |

## Source grounding

- **Ben Iverson**: QA orbit theory — satellite orbit structure and state-space partition
- **QA axiom A1**: states in {1,...,N}, never zero-indexed

## Connection to other families

- **[128] Spread Period**: satellite period 8 = pi(9)/3; orbit structure underlies product sum
- **[139] Orbit Partition**: satellite as one of three orbit classes; partition completeness required for SPS

## Fixture files

- `fixtures/sps_pass_multi_modulus.json` — product sums for 33 moduli, all equal M⁴
- `fixtures/sps_fail_wrong_sum.json` — falsifier with incorrect sum value
