# qa_orbit41_fib_vec_periodic

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_vec_periodic (n : ℕ) : fib_vec (n + 24) = fib_vec n`

**Proof tactic**: `rw`

**Cert refs**: [128] Pisano period π(9)=24

**NL**: The Fibonacci column vector fib_vec is periodic with period 24 mod 9: fib_vec (n+24) = fib_vec n.
