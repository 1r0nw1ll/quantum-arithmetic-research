---- MODULE QAAxioms ----
EXTENDS Naturals

VARIABLES obs_float, obs_cross_count, cap

Init == /\ obs_float = "observer-ready"
        /\ obs_cross_count = 0
        /\ cap = 24

Next == \/ /\ obs_float' = obs_float
           /\ obs_cross_count' = obs_cross_count + 1
           /\ cap' = cap
        \/ /\ obs_float' = "observer-projection-firewall"
           /\ obs_cross_count' = obs_cross_count
           /\ cap' = cap

Inv_S1_NoSquareOperator == obs_cross_count * obs_cross_count >= 0

====
