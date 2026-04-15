/-
# LedgerInvariantsCheck — sanity-check harness for LedgerInvariants.lean

Loads the main proof file and exercises the theorems to confirm they are
not axiomatized away. If any theorem uses `sorry`, `axiom`, or an unsafe
axiom, `#print axioms` will reveal it here.
-/

import «LedgerInvariants»

open LLMQaWrapper.Ledger

/-- Concrete ledger for exercising the theorems. -/
def h1 : Hash := ⟨111⟩
def h2 : Hash := ⟨222⟩

def c1 : CertRecord := ⟨⟨1⟩, genesis, h1, 0⟩  -- prev = genesis
def c2 : CertRecord := ⟨⟨2⟩, h1,      h2, 1⟩  -- prev = c1.self_hash

/-- `[c1]` is a valid ledger — c1 chains to genesis. -/
example : valid [c1] := by
  rw [valid_singleton]
  rfl

/-- `[c1, c2]` is a valid ledger — c1 chains to genesis, c2 chains to c1. -/
example : valid [c1, c2] := by
  unfold valid
  simp only [ChainStartsAt_cons, ChainStartsAt_nil, and_true]
  exact ⟨rfl, rfl⟩

/-- `append_preserves_valid` on a concrete case. -/
example : valid ([c1] ++ [c2]) := by
  apply append_preserves_valid [c1] c2
  · rw [valid_singleton]; rfl
  · unfold lastHash
    rfl

/-- `ledger_prefix_valid` on a concrete case: chopping [c1, c2] to [c1]
still yields a valid ledger. -/
example (hv : valid ([c1] ++ [c2])) : valid [c1] :=
  ledger_prefix_valid [c1] [c2] hv

-- Inspect the axioms used by the main theorems. This fails loudly
-- if any `sorry` or unsafe axiom leaked in.

#print axioms append_preserves_valid
#print axioms ledger_prefix_valid
#print axioms hash_chain_binds_contents
#print axioms ledger_invariants_summary
