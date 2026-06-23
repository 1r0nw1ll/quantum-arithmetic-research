# [502] QA Collatz-Fibonacci Spectral Cert

## What this is

Machine-checkable integer certificate establishing that QA mod-9 exhibits the same
three structural features that enable the Collatz-Fibonacci spectral result of
Reyes Jiménez (arXiv:2606.02621).

## Claim (narrow)

| Check | Claim |
|-------|-------|
| **CF_1** | QA Singularity (9,9) is the unique period-1 fixed point under sigma |
| **CF_2** | All 80 non-Singularity pairs have period in {8, 24} — the Pisano hierarchy for m=9 |
| **CF_3** | Pisano integer witness: F(24)≡0 (mod 9), F(25)≡1 (mod 9); the only k∈{1,...,23} with F(k)≡0 is k=12 with F(13)≡8≠1, proving π(9)=24 is minimal |

## Collatz-QA structural analogy

Reyes Jiménez (arXiv:2606.02621 §2) proves: for each m≥1, exactly F(m+1) odd integers in {1,...,2^m} have Collatz orbits avoiding residue 4 (mod 6). The Collatz transition graph mod 6 has absorbing SCC G'={1,2,4,5}; removing vertex 4 gives a subgraph with spectral radius φ instead of ρ=2.

The QA analog:

| Collatz mod-6 | QA mod-9 |
|---------------|----------|
| Vertex 4 — absorbing fixed point, removed to reveal Fibonacci dynamics | Singularity (9,9) — period-1 fixed point outside Fibonacci-attractor orbits |
| G'={1,2,4,5} absorbing SCC | Cosmos (72 pairs, period 24) + Satellite (8 pairs, period 8) |
| Spectral radius φ after removing vertex 4 | π(9) = 24 — the Fibonacci period for mod-9 |
| F(m+1) count (integer, exact) | Period-24 orbit has 72 = 3×24 pairs (integer, exact) |

## Theorem NT compliance (CF_3)

The golden ratio φ is the spectral radius of the Fibonacci subgraph — a **continuous observer measurement** of discrete structure. In QA: φ = lim F(n+1)/F(n) is a continuous observer projection. The primary discrete fact is π(9) = 24, certified by pure integer Fibonacci arithmetic (CF_3). φ never enters QA dynamics as a causal input. This is the T2 (Observer Projection Firewall) compliance check.

## Pisano minimality certificate

The only zero of F(k) mod 9 for k ∈ {1,...,23} is k=12, with F(12)≡0 but F(13)≡8≠1.
Therefore π(9) is NOT 12. F(24)≡0 AND F(25)≡1, and 24 is the smallest k with both
properties → π(9) = 24.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_collatz_fibonacci_spectral_cert_v1/qa_collatz_fibonacci_spectral_cert_validate.py` |
| Mapping ref | `qa_collatz_fibonacci_spectral_cert_v1/mapping_protocol_ref.json` |
| PASS fixture | `fixtures/pass_collatz_fibonacci_spectral.json` |
| FAIL: wrong singularity period | `fixtures/fail_wrong_singularity_period.json` |
| FAIL: wrong Pisano period | `fixtures/fail_wrong_pisano_period.json` |
| FAIL: wrong period set | `fixtures/fail_wrong_period_set.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_collatz_fibonacci_spectral_cert_v1
python3 qa_collatz_fibonacci_spectral_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`, 1 PASS + 3 FAIL fixtures.

## Checks

| Check | Meaning |
|-------|---------|
| `CF_1` | Singularity (9,9) has period 1 (unique fixed point) |
| `CF_2` | All 80 non-Singularity pairs have period ∈ {8, 24} |
| `CF_3` | Pisano integer witness: F(24)≡0, F(25)≡1 (mod 9); decoy F(12)≡0 but F(13)≡8≠1 |
| `SRC` | schema_version = `QA_COLLATZ_FIBONACCI_CERT.v1` |
| `F` | fail_ledger well-formed if present |

## QA Axiom Compliance

- **A1**: all (b,e) ∈ {1,...,9}; σ maps {1,...,9}² to itself
- **T1**: orbit periods are integer path counts
- **T2**: φ is an observer projection, NOT a QA state; all arithmetic is pure integer
- **S1**: no `**` operator; period computation uses iteration, not exponentiation
- **S2**: b, e are int throughout

## Primary Sources

- Reyes Jiménez, A.E. (2025). Collatz conjecture, Fibonacci numbers and the spectral radius of matrices. arXiv:2606.02621.
- Wall, D.D. (1960). Fibonacci Series Modulo m. *American Mathematical Monthly* 67(6):525–532. DOI:10.2307/2309169.
- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8.

## Relation to other certs

- **[128]** — Lean 4 proof that π(9)=24; cert [502] provides the integer Fibonacci witness (CF_3) that grounds the same claim without Lean
- **[499] `qa_pisano_all_initializations_cert_v1`** — proves the full {period-1, period-8, period-24} partition of all 81 pairs; cert [502] adds the Collatz structural analogy and Pisano minimality witness
- **[500] `qa_cosmos_chamber_cert_v1`** — proves G-arithmetic of the 3 Cosmos sub-orbits; cert [502] identifies the Singularity as the structural "vertex 4" analog

## Scope boundary

**The cert does NOT:**
- Verify the Collatz conjecture
- Compute φ as a QA state (Theorem NT violation)
- Claim that F(m+1) counts in Collatz equal QA orbit counts numerically
- Invoke continuous spectral theory

**The cert DOES:**
- Certify the structural role equivalence between the QA Singularity and Collatz vertex 4
- Provide a pure-integer Pisano witness proving π(9)=24 is minimal
- Establish Theorem NT compliance for φ in this context
