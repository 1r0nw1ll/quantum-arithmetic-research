# Review Case: Vacuous TypeOK Invariant

Review whether this Counter spec is acceptable as a teaching example. The only
invariant is `TypeOK == counter \in Nat`, and the state definition (`counter`
starts in Nat and increment preserves Nat) makes that invariant tautologically
satisfied. Is this a legitimate minimal example or is the reviewer expected to
ask for a substantive safety property?
