# qa_orbit40_fib_mat_pow_fib_vec

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_pow_fib_vec (n m : ℕ) : (fib_mat ^ n) *ᵥ fib_vec m = fib_vec (n + m)`

**Proof tactic**: `simp`

**Cert refs**: [128] Pisano period π(9)=24

**NL**: Applying fib_mat^n to fib_vec m yields fib_vec (n+m): iterated matrix action = shift by n in Fibonacci sequence.
