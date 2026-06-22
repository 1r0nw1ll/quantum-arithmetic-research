import Mathlib.GroupTheory.SpecificGroups.Cyclic
import Mathlib.Tactic

-- (prerequisite defs in QAFibMatrixGroupIso.lean)
theorem fib_mat_iso_zpow (k : ℤ) :
    fib_mat_iso_ZMod24 (Multiplicative.ofAdd k) = fib_gen ^ k :=
  zmodMulEquivOfGenerator_apply_ofAdd_intCast fib_gen_generates fib_mat_zpowers_nat_card k
