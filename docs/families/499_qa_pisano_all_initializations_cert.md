# [499] QA Pisano All-Initializations Cert

## What this is

Machine-checkable equivalence between QA's three-orbit partition of {1,...,9}² and Pudelko's (arXiv:2510.24882) complete Pisano period classification of all m²=81 initial pairs in (ℤ/9ℤ)².

This cert closes the gap between QA's internal orbit structure (derived from the no-zero dynamics) and an independently published math.NT result that classifies every possible initial condition for the Fibonacci recurrence mod 9.

## Claim (narrow)

The QA step function σ(b,e) = (e, ((b+e−1) % 9) + 1) partitions {1,...,9}² into exactly three orbit families:

| Family | Count | Period | Content-ideal class |
|--------|-------|--------|---------------------|
| Cosmos | 72 | 24 = π(9) | min(v₃(b), v₃(e)) = 0 |
| Satellite | 8 | 8 | min(v₃(b), v₃(e)) = 1 |
| Singularity | 1 | 1 | min(v₃(b), v₃(e)) = 2 |

Total: 72 + 8 + 1 = 81 = 9².

Pudelko independently proves that these are the **only** achievable Pisano periods from all m²=81 initial pairs, with exactly these counts. The period set {1, 8, 24} is complete — no other period exists in (ℤ/9ℤ)².

Additional structural results verified exhaustively:

- **Swap symmetry** (QAP_3): orbit_family on (b,e,9) = orbit_family on (e,b,9) for all 81 pairs
- **Negation parity** (QAP_5): orbit_family on (b,e,9) = orbit_family on (neg(b), neg(e), 9) for all 81 pairs, where neg(b) = 9 − (b mod 9) if b mod 9 ≠ 0, else 9 — mirrors Pudelko's parity observation that additive inversion preserves the period distribution

## Why this matters

- **External validation**: Pudelko's paper is an independent math.NT result with no QA connection. The fact that the same three orbit families with the same counts emerge from a completely different framing (all random initial conditions vs. QA's discrete dynamics) is the strongest independent confirmation QA's orbit partition has received.
- **Completeness**: Previous certs validated π(9)=24 for specific initial conditions; this cert extends the claim to ALL 81 initial pairs.
- **Content-ideal link**: The three period values exactly correspond to the three content-ideal classes from cert [261] at the prime p=3, modulus 9=3².

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_pisano_all_initializations_cert_v1/qa_pisano_all_initializations_cert_validate.py` |
| Mapping ref | `qa_pisano_all_initializations_cert_v1/mapping_protocol_ref.json` |
| PASS fixture | `fixtures/pass_full_partition_m9.json` |
| FAIL: wrong count | `fixtures/fail_wrong_cosmos_count.json` |
| FAIL: wrong period | `fixtures/fail_wrong_period_witness.json` |
| FAIL: wrong period set | `fixtures/fail_wrong_period_set.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_pisano_all_initializations_cert_v1
python3 qa_pisano_all_initializations_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`, 1 PASS + 3 FAIL fixtures.

## Checks

| Check | Meaning |
|-------|---------|
| `QAP_1` | Exhaustive partition: cosmos_count + satellite_count + 1 = m²=81; counts match declared values |
| `QAP_2` | Period witnesses: declared (b,e) pairs have the stated period under σ |
| `QAP_3` | Swap symmetry: orbit_family(b,e) = orbit_family(e,b) for all 81 pairs |
| `QAP_4` | Period completeness: exactly 3 distinct periods, period 1 present, declared period_set matches computed |
| `QAP_5` | Negation parity: orbit_family(b,e) = orbit_family(neg(b),neg(e)) for all 81 pairs |
| `SRC` | schema_version = `QA_PISANO_ALL_INIT_CERT.v1` |
| `F` | fail_ledger is a well-formed list if present |

## QA Axiom Compliance

- **A1**: all (b,e) ∈ {1,...,9}; the step σ maps {1,...,9}² to itself; no zero element
- **A2**: the step σ computes d = (b+e−1) mod 9 + 1 as the derived coordinate; a is not explicitly used here but the orbit is driven by d = b+e (raw, before modular reduction)
- **T1**: orbit length k is an integer path count — period = k steps under σ
- **T2**: all arithmetic is pure integer; no floats; negation and period computation are mod arithmetic over ℤ
- **S1**: no `**` operator; period computation uses `%` and `+` only
- **S2**: b, e are int throughout; no NumPy or float state

## Primary Sources

- Pudelko, M.T. (2025). Modular Periodicity of Random Initialized Recurrences. arXiv:2510.24882 v5 (2026-04-09). Results 1.1–1.5: complete period classification of all m² initial pairs; parity symmetry theorem.
- Wall, D.D. (1960). Fibonacci series modulo m. *Amer. Math. Monthly* 67(6):525–532. doi:10.2307/2309169. Pisano periods, prime-power structure.
- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. QA step operator, no-zero convention.

## Relation to other certs

- **[128] `qa_fibonacci_matrix_orbit_periods_cert_v1`** — proves π(9)=24 exactly via Lean 4 (`pisano_period_9_exact`); cert [499] extends this to the full 81-pair classification
- **[261] `qa_orbit_stratification_cert_v1`** — proves the content-ideal classification min(v₃(b),v₃(e))∈{0,1,2} is the algebraic ground for the three orbit families; cert [499] provides the period interpretation of those three classes
- **[496] `qa_e8_satellite_chamber_cert_v1`** — uses the Satellite orbit (8 pairs, period 8) as its anchor; cert [499] proves the Satellite partition is the complete second class

## Scope boundary

**The cert does NOT:**
- Claim the full Pudelko distribution P(n) = P(1−n) symmetry (that requires the parity-Fibonacci recurrence comparison across all periods, not just period preservation under negation)
- Extend beyond m=9 (the validator takes m as a fixture parameter but is tested only at m=9)
- Prove the fractal prime→prime-power structure (p=3 to p²=9 lifting); that is cert-candidate material for a future Pudelko-lifting cert

**The cert DOES:**
- Exhaustively verify all 81 pairs with pure integer arithmetic
- Confirm period_set = {1, 8, 24} is complete (no other periods exist)
- Prove both swap and negation symmetries exhaustively
- Anchor QA's orbit count claims to an independently published classification
