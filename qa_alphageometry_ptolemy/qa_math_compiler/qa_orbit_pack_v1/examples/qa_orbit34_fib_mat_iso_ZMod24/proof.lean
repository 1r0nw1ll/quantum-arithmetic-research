import Mathlib.GroupTheory.SpecificGroups.Cyclic
import Mathlib.Tactic

-- prerequisite defs (see QAFibMatrixGroupIso.lean for full context)
noncomputable def fib_mat_iso_ZMod24 :
    Multiplicative (ZMod 24) ≃* ↕(Subgroup.zpowers fib_mat_unit) :=
  zmodMulEquivOfGenerator fib_gen_generates fib_mat_zpowers_nat_card
