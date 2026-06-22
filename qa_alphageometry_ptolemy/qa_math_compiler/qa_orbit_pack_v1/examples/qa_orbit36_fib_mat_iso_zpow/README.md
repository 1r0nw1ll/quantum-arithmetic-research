# qa_orbit36_fib_mat_iso_zpow

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_iso_zpow (k : ℤ) : fib_mat_iso_ZMod24 (Multiplicative.ofAdd k) = fib_gen ^ k`

**Proof tactic**: `exact`

**Cert refs**: [128] SP2

**NL**: The isomorphism sends Multiplicative.ofAdd k to the k-th power F^k for any integer k.
