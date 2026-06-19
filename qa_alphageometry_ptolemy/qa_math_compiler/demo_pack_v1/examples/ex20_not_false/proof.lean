theorem ex20_not_false : ¬ False :=
  by
    intro h
    exact False.elim h
