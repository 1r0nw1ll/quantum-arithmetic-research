# Family [131] QA_PRIME_BOUNDED_CERTIFICATE_SCALING_CERT.v1

## One-line summary

Empirical scaling cert for bounded factor-certificate witness caps on tested intervals [2,N]; validates that certified witness bounds match recomputed values.

## Mathematical content

For each interval [2,N], the cert records the maximum certificate witness size needed to certify all primes up to N. The validator recomputes witness caps from the experiment artifact and checks consistency with declared values.

## Checks

| ID | Description |
|----|-------------|
| Gate 1 | Schema validity |
| Gate 2 | Canonical hash match |
| Gate 3 | Artifact integrity (results file exists and matches) |
| Row check | Per-row recomputation of witness bounds |
| Honesty | PASS/FAIL result honestly reflects row outcomes |

## Source grounding

- Empirical experiment: `results/qa_prime_bounded_certificate_scaling_experiment.json`
- QA prime factorization theory (Iverson)

## Fixture files

- `fixtures/pass_scaling_100_1000.json` — PASS: exact-match cert for [100,250,500,1000]
- `fixtures/fail_scaling_100_500.json` — FAIL: mock 500 mismatch
