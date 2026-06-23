# qa_orbit43_fib_mat_pow_eq_one_iff

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_pow_eq_one_iff (m : ℕ) : fib_mat ^ m = 1 ↔ 24 ∣ m`

**Proof tactic**: `rcases`

**Cert refs**: [128] Pisano period π(9)=24 — minimality

**NL**: fib_mat^m = 1 iff 24 | m: the Fibonacci matrix has exact multiplicative order 24 in M₂(ZMod 9).
