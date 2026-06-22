import Mathlib.GroupTheory.SpecificGroups.Cyclic
import Mathlib.Tactic

-- (prerequisite defs in QAFibMatrixGroupIso.lean)
theorem fib_mat_iso_symm_zpow (k : ℤ) :
    fib_mat_iso_ZMod24.symm (fib_gen ^ k) = Multiplicative.ofAdd (k : ZMod 24) :=
  zmodMulEquivOfGenerator_symm_apply_zpow fib_gen_generates fib_mat_zpowers_nat_card k
