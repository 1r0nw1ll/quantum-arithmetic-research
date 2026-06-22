import Mathlib.Algebra.Ring.Parity

/-- Mined from mathlib_pack_v1: Even is closed under addition.
    Compresses the 3-step obtain+omega witness construction to 1 step. -/
theorem qaEvenAdd {a b : ℕ} (ha : Even a) (hb : Even b) : Even (a + b) := by
  obtain ⟨k, rfl⟩ := ha
  obtain ⟨l, rfl⟩ := hb
  exact ⟨k + l, by omega⟩
