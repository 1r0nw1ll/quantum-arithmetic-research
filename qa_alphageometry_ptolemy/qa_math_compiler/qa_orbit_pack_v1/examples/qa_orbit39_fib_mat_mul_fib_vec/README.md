# qa_orbit39_fib_mat_mul_fib_vec

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_mul_fib_vec (n : ℕ) : fib_mat *ᵥ fib_vec n = fib_vec (n + 1)`

**Proof tactic**: `simp`

**Cert refs**: [128] Pisano period π(9)=24

**NL**: Multiplying fib_mat by the Fibonacci column vector fib_vec n advances it by one step: fib_mat *ᵥ fib_vec n = fib_vec (n+1).
