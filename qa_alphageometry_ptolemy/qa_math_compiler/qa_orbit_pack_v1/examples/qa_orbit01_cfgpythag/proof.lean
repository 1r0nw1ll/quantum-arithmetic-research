import Mathlib.Tactic

theorem qa_cfgpythag (b e : ℤ) :
    let d := b + e
    (2 * d * e) * (2 * d * e) +
    (d * d - e * e) * (d * d - e * e) =
    (d * d + e * e) * (d * d + e * e) := by
  ring
