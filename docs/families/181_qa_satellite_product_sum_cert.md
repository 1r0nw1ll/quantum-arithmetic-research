# Family [181] QA_SATELLITE_PRODUCT_SUM_CERT.v1

## One-line summary

The sum of products Σ(b·e·d·a) over all satellite pairs equals M⁴ for any modulus M divisible by 3, with the normalized sum always equal to 81 = 3⁴.

## Mathematical content

### Satellite product sum identity

For modulus M (where 3 | M), the satellite orbit contains exactly 8 pairs (b, e) with d = (b+e) % M, a = (b+2e) % M. Define the product-sum:

    SPS(M) = Σ_{satellite} b · e · d · a

**Theorem**: SPS(M) = M⁴ for all M divisible by 3.

**Corollary**: Normalizing by (M/3)⁴ gives SPS_norm = 81 = 3⁴ for every valid modulus. Corrected 2026-07-06: this means the satellite total exactly equals the singularity's own product — the singularity fixed point is (M,M,M,M) (not (M/3,M/3) as a previous version of this doc stated), and its product b·e·d·a = M·M·M·M = M⁴ directly (not M⁴/81). Independently verified: SPS(M) = M⁴ = singularity_product(M) for every tested modulus.

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

## Verification Note (2026-07-06)

Self-contained QA-internal number theory, no external citation needed.
Independently brute-force verified SPS(M)=M⁴ for all M divisible by 3
from M=3 to M=99 (33 moduli, matching the claimed test range exactly,
not just the 3 sampled witnesses in the fixture) — 0 mismatches. Also
independently verified the normalized-sum corollary (SPS(M)/(M/3)⁴=81)
for 5 test moduli.

**Found and fixed a real error in the doc's "Corollary" paragraph**
(fixture and validator were already correct): the doc described the
singularity state as "(M/3, M/3)" with product "M⁴/81" — but the actual
singularity fixed point, consistent with every other cert in this
project, is (M,M,M,M), not (M/3,M/3). Independently verified: for
M=9,12,15,24,99 the singularity tuple (M,M,M,M) has product exactly M⁴
(e.g. M=9: 9⁴=6561), not M⁴/81. The fixture's own `corollary` section
already stated this correctly ("Singularity product = M^4 (since
b=e=d=a=M)"), so this was purely a documentation-precision bug, not a
validator or fixture bug. Corrected the doc to match.

Validator confirmed genuinely computing (not fixture-trusting): reads
`qa_satellite_product_sum_cert_validate.py`'s own docstring, which
already correctly states "SPS_COROL — singularity product = M^4", and
its checks independently recompute the satellite enumeration and sum
from the declared modulus at runtime. `--self-test` passes on both
fixtures.
