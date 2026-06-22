# qa_orbit24_fib_mat_iter

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `theorem fib_mat_iter :`

**Proof tactic**: `native_decide`

**Cert refs**: [126] orbit-structure, [128] SP2

**NL**: Iterating the QA T-step k times equals applying the k-th matrix power F^k: for all k ∈ {0,...,23} and all (b,e) ∈ (ZMod 9)², the k-fold iterate of T on (b,e) equals F^k · [b,e]ᵀ.
