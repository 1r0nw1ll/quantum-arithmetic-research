<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary sources cited in mapping_protocol_ref.json and validator -->

# [281] QA Pisano-Orbit Correspondence

**Cert family**: `qa_pisano_orbit_correspondence_cert_v1`
**Primary source**:
- Wall, D. D. (1960). Fibonacci series modulo m. *American Mathematical Monthly*, 67(6), 525-532. DOI: [10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541)
- Mechanism for cert [279]: QA Orbit Access Theorem

## Claim

For m=9, under `qa_step(b,e) = (e, ((b+e-1)%9)+1)`, the orbit period of every pair (b,e) in {1,...,9}² is exactly:

| Orbit Family | Count | Period | Pisano Period |
|---|---|---|---|
| Singularity | 1 | 1 | — |
| Satellite | 8 | **8** | π(3) = 8 |
| Cosmos | 72 | **24** | π(9) = 24 |

Pisano period π(k) is defined as the smallest i>0 such that F_i ≡ 0 and F_{i+1} ≡ 1 mod k (Wall 1960, §3).

Verified exhaustively across all 81 pairs in {1,...,9}². π(3)=8 and π(9)=24 confirmed independently via Fibonacci recurrence mod k.

## Structural Significance

- **`qa_step` IS the Fibonacci recurrence**: `qa_step(b,e) = (e, ((b+e-1)%9)+1)` is the A1-compliant Fibonacci shift. The orbit structure of the QA state space is determined by the Fibonacci sequence modulo 9.
- **Why Satellite ↔ π(3)**: The 8 Satellite pairs all satisfy 3|b and 3|e (after A1-reduction). The Fibonacci sequence restricted to multiples of 3 mod 9 has period 8 — the Pisano period of 3.
- **Why Cosmos ↔ π(9)**: Generic (b,e) pairs not divisible by 3 generate the full Fibonacci cycle mod 9, which has period 24 — the Pisano period of 9.
- **Grounding for cert [279]**: The Orbit Access Theorem (cert [279]) says orbit class reachability is governed by gcd(a,3). Cert [281] explains WHY: it is because the cycle length of each orbit class equals the Pisano period of the corresponding divisibility threshold. The Satellite orbit IS the Fibonacci sequence mod 3 embedded in mod 9.

## Scope Boundaries

- Does **not** claim the Pisano-period correspondence extends to arbitrary m
- Does **not** certify the period structure of mod-24 QA (separate future cert candidate)
- Does **not** prove the identity algebraically (empirical verification of all 81 pairs)
- The Pisano period computation uses Wall's (1960) definition — smallest i>0 with F_i≡0, F_{i+1}≡1 mod k

## Gates

- **POC_1**: singularity period = 1
- **POC_2**: all satellite orbits have period = 8 = π(3)
- **POC_3**: all cosmos orbits have period = 24 = π(9)
- **POC_4**: π(3) = 8 verified via Fibonacci recurrence
- **POC_5**: π(9) = 24 verified via Fibonacci recurrence
- **SRC**: `mapping_protocol_ref.json` present and well-formed
- **F**: every FAIL fixture declares `expected_fail_type` and fires

6 PASS fixtures, 4 FAIL fixtures. Validator: `qa_pisano_orbit_correspondence_cert_validate.py --self-test`.

## Lineage

Discovered 2026-05-30 during domain sweep across cert [279] (Orbit Access Theorem). The Orbit Access Theorem showed that orbit class reachability depends on gcd(a,3); cert [281] grounds this algebraically by identifying the orbit periods as exactly the Pisano periods π(3) and π(9). Companion certs: [279] (mechanism — which classes are reachable), [280] (nuclear magic number application). The three certs form a complete chain: period identity ([281]) → access theorem ([279]) → domain application ([280]).
