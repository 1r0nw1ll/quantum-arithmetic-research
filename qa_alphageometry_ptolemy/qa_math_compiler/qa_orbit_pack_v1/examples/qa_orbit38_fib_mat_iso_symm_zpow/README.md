# qa_orbit38_fib_mat_iso_symm_zpow

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_iso_symm_zpow (k : ℤ) : fib_mat_iso_ZMod24.symm (fib_gen ^ k) = Multiplicative.ofAdd (k : ZMod 24)`

**Proof tactic**: `exact`

**Cert refs**: [128] SP2

**NL**: The inverse isomorphism sends F^k to Multiplicative.ofAdd (k mod 24) for any integer k.
