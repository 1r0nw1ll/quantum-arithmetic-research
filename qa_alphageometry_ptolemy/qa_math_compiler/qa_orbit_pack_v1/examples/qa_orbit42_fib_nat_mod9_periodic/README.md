# qa_orbit42_fib_nat_mod9_periodic

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_nat_mod9_periodic (n : ℕ) : (Nat.fib (n + 24) : ZMod 9) = Nat.fib n`

**Proof tactic**: `simp`

**Cert refs**: [128] Pisano period π(9)=24

**NL**: The Fibonacci sequence is periodic mod 9 with Pisano period 24: (Nat.fib (n+24) : ZMod 9) = Nat.fib n for all n.
