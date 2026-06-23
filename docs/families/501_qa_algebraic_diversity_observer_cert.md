# [501] QA Algebraic Diversity Observer Cert

## What this is

Machine-checkable integer formulation of Thornton's Algebraic Diversity framework (arXiv:2604.03634, arXiv:2604.19983) applied to QA orbit observer channels.

## Claim (narrow)

The G-function G(b,e) = (b+e)² + e² over QA mod-9 orbits defines algebraically diverse observer channels:

| Check | Claim |
|-------|-------|
| **AD_1** | G injective on each Cosmos orbit: 24 pairwise-distinct G-values |
| **AD_2** | G injective on the Satellite orbit: 8 pairwise-distinct G-values |
| **AD_3** | No proper divisor of 24 (resp. 8) is a G-period in any Cosmos (resp. Satellite) orbit |
| **AD_4** | Satellite G-set {45,90,117,153,180,225,261,306} is disjoint from all Cosmos G-sets |

where G(b,e) = (b+e)² + e² (raw d=b+e, A2-compliant).

## Why this matters

### Integer form of the Replacement Theorem

Thornton (arXiv:2604.03634) proves that a group-averaged estimator over a *matched* cyclic group recovers the full orbit spectral structure from a single observation. The matched group for QA's Cosmos orbit is Z/24Z; for the Satellite, Z/8Z.

The integer formulation of "Z/NZ is the minimal matched group for the G-sequence" is:
1. **G-injectivity**: the N G-values on the orbit are pairwise distinct (AD_1, AD_2)
2. **No sub-period**: no proper divisor of N is a period of the G-sequence (AD_3)

If either condition failed, a smaller cyclic group Z/kZ (k | N, k < N) would suffice as the matched group, and Z/NZ would be non-minimal.

### Integer form of Blind Group Matching

Thornton (arXiv:2604.19983) proves that the matched group is uniquely identifiable from the signal's covariance structure. AD_3 is the integer certificate of this uniqueness: Z/24Z cannot be "compressed" to any proper subgroup while preserving the G-sequence structure.

### Algebraically diverse channels (AD_4)

AD_4 establishes that the Satellite and Cosmos observer channels produce completely non-overlapping G-values. No G-value can serve double duty as a Satellite reading and a Cosmos reading — the two channels are algebraically orthogonal. This formalizes Theorem NT (Observer Projection Firewall): the discrete Satellite and Cosmos layers project into distinct, non-overlapping ranges of the observer output.

## Explicit Satellite G-set

{45, 90, 117, 153, 180, 225, 261, 306} — these are the 8 G-values of the Satellite orbit under QA mod-9. All 8 are distinct and none appears in any Cosmos orbit G-set.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_algebraic_diversity_observer_cert_v1/qa_algebraic_diversity_observer_cert_validate.py` |
| Mapping ref | `qa_algebraic_diversity_observer_cert_v1/mapping_protocol_ref.json` |
| PASS fixture | `fixtures/pass_ad_observer.json` |
| FAIL: G not injective (Cosmos O1) | `fixtures/fail_g_not_injective.json` |
| FAIL: wrong satellite G-set | `fixtures/fail_wrong_satellite_g_set.json` |
| FAIL: Satellite not injective | `fixtures/fail_satellite_not_injective.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_algebraic_diversity_observer_cert_v1
python3 qa_algebraic_diversity_observer_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`, 1 PASS + 3 FAIL fixtures.

## Checks

| Check | Meaning |
|-------|---------|
| `AD_1` | G injective on each Cosmos orbit (24 distinct G-values per orbit) |
| `AD_2` | G injective on Satellite orbit (8 distinct G-values) |
| `AD_3` | No proper divisor of 24 (or 8) is a period of the corresponding G-sequence |
| `AD_4` | Satellite G-set disjoint from all Cosmos G-sets |
| `SRC` | schema_version = `QA_ALGEBRAIC_DIVERSITY_CERT.v1` |
| `F` | fail_ledger well-formed if present |

## QA Axiom Compliance

- **A1**: all (b,e) ∈ {1,...,9}; σ maps {1,...,9}² to itself
- **A2**: G = (b+e)² + e² uses raw d = b+e (not reduced)
- **T1**: orbit periods are integer path counts
- **T2**: all arithmetic is pure integer; no floats
- **S1**: no `**` operator; G uses `d*d`, `e*e`
- **S2**: b, e are int throughout

## Primary Sources

- Thornton, M.A. (2026). Algebraic Diversity: Group-Theoretic Spectral Estimation from Single Observations. arXiv:2604.03634.
- Thornton, M.A. (2026). Algebraic Diversity: Principles of a Group-Theoretic Approach to Signal Processing. arXiv:2604.19983.
- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8.

## Relation to other certs

- **[496] `qa_e8_satellite_chamber_cert_v1`** — characterizes the Satellite orbit via E8 Weyl chamber; cert [501] adds that the Satellite G-set is disjoint from all Cosmos G-sets
- **[499] `qa_pisano_all_initializations_cert_v1`** — proves orbit periods exhaustively; cert [501] proves G-injectivity within those orbits
- **[500] `qa_cosmos_chamber_cert_v1`** — proves G-sum arithmetic progression; cert [501] proves G-injectivity (a per-element claim stronger than G-sum)

## Scope boundary

**The cert does NOT:**
- Compute DFT coefficients (float observer, Theorem NT violation)
- Run the blind group matching algorithm (Lie algebra eigenvalue problem)
- Invoke the continuous Algebraic Diversity framework (arXiv:2605.00848)
- Claim the quantum extension (arXiv:2604.03725)

**The cert DOES:**
- Provide integer certificates for the two key discrete properties that make Z/24Z and Z/8Z minimal matched groups
- Establish complete non-overlap between Satellite and Cosmos observer channels
- Ground QA's orbit structure in Thornton's independently published AD framework
