<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary source cited in mapping_protocol_ref.json and validator -->

# [279] QA Orbit Access Theorem

**Cert family**: `qa_orbit_access_theorem_cert_v1`
**Primary source**: Wall, D. D. (1960). Fibonacci series modulo m. *American Mathematical Monthly* 67(6), 525–532. DOI: [10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541)

## Claim

For mod-9 route enumeration — all pairs (b, e) with b+2e=a, b≥1, e≥1, A1-reduced — the orbit classes reachable for a given *a* are determined exclusively by gcd(a, 3):

| Divisibility class | Condition | Cosmos | Satellite | Singularity |
|---|---|---|---|---|
| `coprime_to_3` | gcd(a,3)=1 | ✓ | — | — |
| `mul_3_not_9` | 3\|a and 9∤a | ✓ | ✓ | — |
| `mul_9` | 9\|a | ✓ | ✓ | ✓ |

## Algebraic Mechanism

Satellite period = 8 = Pisano(3). Cosmos period = 24 = Pisano(9). Divisibility by 3 (resp. 9) governs which Pisano orbit classes the A1-reduced route set can reach.

## Verified Corpus

- `coprime_to_3`: a ∈ {8, 13, 28, 50}
- `mul_3_not_9`: a ∈ {12, 15, 21}
- `mul_9`: a ∈ {9, 27, 126, 144}

## Cross-Domain Implications

**Nuclear magic numbers**: {8, 20, 28, 50, 82, 184} are all coprime to 3 → pure Cosmos. Only a=126 (= 14×9) is in `mul_9` and has Satellite + Singularity access. This makes a=126 structurally distinct from all other nuclear shell closures.

**Fibonacci sequence**: F_n is divisible by 3 iff 4∣n. The first Fibonacci number divisible by 9 is F₁₂=144. Singularity access opens exactly at F₁₂.

**FST/Briddell**: All powers of 3 and the proton a=1836 are in `mul_9`. The STF hierarchy exclusively inhabits the full Cosmos+Satellite+Singularity regime.

## Scope Boundaries

- Does NOT claim route proportions per orbit class (those converge asymptotically, not fixed per a)
- Does NOT address unique (b,e) pairs generating each orbit
- Does NOT claim Satellite fraction is exactly 2/9 (it is only a large-a limit)
- Does NOT address Pythagorean F values (orthogonal domain — F is a product, not a sum b+2e)

## Gates

- **OAT_1**: For `coprime_to_3` fixtures, satellite=0 AND singularity=0
- **OAT_2**: For `mul_3_not_9` fixtures, satellite>0 AND singularity=0
- **OAT_3**: For `mul_9` fixtures, satellite>0 AND singularity>0
- **SRC**: `mapping_protocol_ref.json` present and well-formed
- **F**: Every FAIL fixture declares `expected_fail_type` and the declared mode fires

6 PASS fixtures, 4 FAIL fixtures. Validator: `qa_orbit_access_theorem_cert_validate.py --self-test`.

## Lineage

Discovered 2026-05-30 during FST/Briddell domain sweep. The theorem emerged from an orbit fingerprint sweep across nuclear magic numbers, Fibonacci, Keely harmonics, and Wildberger RT denominators. Zero exceptions found in ~60 test values. Sister certs: [277] Pisano 5-Factor Boundary, [278] No-3-Divisor Overclaim (all three ground in Wall 1960).
