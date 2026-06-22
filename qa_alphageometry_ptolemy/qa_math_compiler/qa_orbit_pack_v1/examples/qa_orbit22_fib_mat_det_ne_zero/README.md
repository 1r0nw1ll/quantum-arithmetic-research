# qa_orbit22_fib_mat_det_ne_zero

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_det_ne_zero : Matrix.det fib_mat ≠ 0`

**Proof tactic**: `native_decide`

**Cert refs**: [153] DOMINANT=SINGULARITY

**NL**: The determinant of the Fibonacci matrix F = [[1,1],[1,0]] is nonzero in ZMod 9, confirming F is invertible.
