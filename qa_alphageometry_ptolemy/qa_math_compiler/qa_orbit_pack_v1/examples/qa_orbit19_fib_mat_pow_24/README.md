# qa_orbit19_fib_mat_pow_24

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_pow_24 : fib_mat ^ 24 = 1`

**Proof tactic**: `native_decide`

**Cert refs**: [128] SP2

**NL**: The Fibonacci matrix F = [[1,1],[1,0]] over ZMod 9 satisfies F^24 = I: the 24th power of F is the identity matrix in M₂(ZMod 9).
