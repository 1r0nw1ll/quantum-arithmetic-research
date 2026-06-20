theorem ex40_list_map_id {α : Type u} (xs : List α) : List.map (fun x => x) xs = xs :=
  by
    induction xs with
    | nil => rfl
    | cons x xs ih =>
        simp only [List.map, ih]
