# qa_orbit05_t_period_divides_24

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem qa_t_period_divides_24 : ∀ s : ZMod 9 × ZMod 9, (qa_t_step9^[24]) s = s`

**Proof tactic**: `native_decide`

**Cert refs**: [128] SP2, [128] SP3

**NL**: Every state in ZMod 9 × ZMod 9 has orbit period dividing 24 under the QA T-step; equivalently, the Fibonacci matrix F satisfies F^24 = I in GL₂(ZMod 9).
