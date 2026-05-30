<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary sources cited in mapping_protocol_ref.json and validator -->

# [282] QA Fibonacci-Orbit Index Correspondence

**Cert family**: `qa_fibonacci_orbit_index_cert_v1`
**Primary source**:
- Wall, D. D. (1960). Fibonacci series modulo m. *American Mathematical Monthly*, 67(6), 525-532. DOI: [10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541)
- Mechanism: cert [281] (Pisano-Orbit Correspondence) + cert [279] (Orbit Access Theorem)

## Claim

For n ≥ 1, the mod-9 orbit class of the Fibonacci number F_n (used as an a-value in mod-9 route enumeration b+2e=a) is determined solely by **n mod 12**:

| n mod 12 | Orbit Class | Condition on F_n | Example |
|---|---|---|---|
| 0 | `mul_9` | 9 \| F_n | F_12=144=9×16 |
| 4 or 8 | `mul_3_not_9` | 3\|F_n, 9∤F_n | F_4=3, F_8=21 |
| otherwise | `coprime_to_3` | 3∤F_n | F_1=1, F_6=8 |

Wall (1960) proves: rank of apparition α(3)=4 (so 3\|F_n iff 4\|n) and α(9)=12 (so 9\|F_n iff 12\|n). Verified exhaustively for n=1..48 (two full Pisano periods of 9; π(9)=24 per cert [281]).

## Structural Significance

- **F_12=144 is the first mul_9 Fibonacci number**: this is the first Fibonacci number with full orbit access (Cosmos+Satellite+Singularity) under the Orbit Access Theorem [279]. 144=12²=F_12.
- **The pattern is exact**: every 12th Fibonacci number lands in mul_9. Between them, every 4th that isn't 12th lands in mul_3_not_9. No exceptions in either direction within the verified range.
- **Connection to cert [281]**: cert [281] proved that orbit periods under qa_step equal Pisano periods (π(3)=8=Satellite period, π(9)=24=Cosmos period). Cert [282] is the complementary claim: the Fibonacci NUMBER F_n's orbit class as an a-value is determined by Wall's rank-of-apparition theory, which is the divisibility side of the same Pisano structure.
- **The two claims are dual**: [281] characterizes the operator (qa_step); [282] characterizes the inputs (Fibonacci numbers). Together they close the loop between Wall (1960) and the QA state space.

## Scope Boundaries

- Does **not** claim the correspondence extends to Fibonacci numbers mod m for arbitrary m
- Does **not** assert that the orbit class of F_n determines the orbit class of F_{n+1}
- Does **not** certify the orbit access structure for arbitrary Fibonacci-indexed sequences
- The rank of apparition argument applies only for p=3 and p^2=9; higher powers require separate verification

## Gates

- **FOI_1**: coprime_to_3 class: 3 does not divide F_n
- **FOI_2**: mul_3_not_9 class: 3\|F_n but 9∤F_n
- **FOI_3**: mul_9 class: 9\|F_n
- **FOI_4**: declared class matches both the n mod 12 rule AND actual F_n divisibility (Wall consistency check)
- **SRC**: `mapping_protocol_ref.json` present and well-formed
- **F**: every FAIL fixture declares `expected_fail_type` and fires

6 PASS fixtures, 4 FAIL fixtures. Validator: `qa_fibonacci_orbit_index_cert_validate.py --self-test`.

## Lineage

Discovered 2026-05-30 as a direct corollary of cert [281] and Wall (1960) rank-of-apparition theory. The cert family sweep across nuclear magic numbers ([280]), Pisano periods ([281]), and Fibonacci structure ([282]) forms a unified Wall-1960-grounded cluster. All three certs draw on the same primary source, with each cert addressing a distinct structural level: period identity ([281]) → orbit access by divisibility class ([279]) → Fibonacci number classification ([282]).
