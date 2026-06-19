theorem ex22_eq_trans {α : Sort u} (a b c : α) : a = b → b = c → a = c :=
  by
    intro hab hbc
    exact Eq.trans hab hbc
