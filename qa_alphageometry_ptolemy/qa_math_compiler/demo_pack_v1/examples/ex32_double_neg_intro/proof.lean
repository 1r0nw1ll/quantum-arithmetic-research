theorem ex32_double_neg_intro (p : Prop) : p → ¬¬p :=
  by
    intro hp hnp
    exact hnp hp
