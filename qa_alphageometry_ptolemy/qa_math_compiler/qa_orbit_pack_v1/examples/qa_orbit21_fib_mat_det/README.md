# qa_orbit21_fib_mat_det

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_det : Matrix.det fib_mat = 8`

**Proof tactic**: `native_decide`

**Cert refs**: [153] DOMINANT=SINGULARITY

**NL**: The determinant of the Fibonacci matrix F = [[1,1],[1,0]] over ZMod 9 equals 8 (= -1 mod 9). Since 8 ≠ 0, F is invertible: F ∈ GL₂(ZMod 9).
