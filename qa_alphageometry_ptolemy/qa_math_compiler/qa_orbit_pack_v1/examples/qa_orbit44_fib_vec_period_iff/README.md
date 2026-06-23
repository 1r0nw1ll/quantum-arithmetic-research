# qa_orbit44_fib_vec_period_iff

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_vec_period_iff (m : ℕ) : fib_vec m = fib_vec 0 ↔ 24 ∣ m`

**Proof tactic**: `rcases`

**Cert refs**: [128] Pisano period π(9)=24 — minimality

**NL**: fib_vec m = fib_vec 0 iff 24 | m: the Fibonacci column vector returns to its initial state exactly when 24 divides m.
