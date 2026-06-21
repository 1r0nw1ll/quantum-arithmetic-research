import Mathlib.Data.Nat.BitIndices

theorem qa_mathlib05_bit_indices_nodup (n : Nat) : n.bitIndices.Nodup := by
  exact Nat.bitIndices_nodup
