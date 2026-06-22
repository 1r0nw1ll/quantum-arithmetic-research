# qa_orbit34_fib_mat_iso_ZMod24

**QA Orbit Pack v1** — machine-checked Lean 4 proof.

**Theorem**: `noncomputable def fib_mat_iso_ZMod24 : Multiplicative (ZMod 24) ≃* ↕(Subgroup.zpowers fib_mat_unit)`

**Proof tactic**: `exact`

**Cert refs**: [128] SP2, [126] orbit-structure

**NL**: There exists an explicit group isomorphism Multiplicative(ZMod 24) ≃* ⟨F⟩ built by zmodMulEquivOfGenerator, sending Multiplicative.ofAdd k to F^k. This is the full Pisano period statement in group-isomorphism form.
