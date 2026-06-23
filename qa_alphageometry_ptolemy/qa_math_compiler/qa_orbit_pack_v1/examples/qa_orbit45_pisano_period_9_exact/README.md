# qa_orbit45_pisano_period_9_exact

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem pisano_period_9_exact (m : ℕ) :`

**Proof tactic**: `calc`

**Cert refs**: [128] Pisano period π(9)=24 — exact characterization

**NL**: Exact Pisano period π(9)=24: (Nat.fib (n+m) : ZMod 9) = Nat.fib n for all n iff 24 | m. This fully characterises π(9)=24.
