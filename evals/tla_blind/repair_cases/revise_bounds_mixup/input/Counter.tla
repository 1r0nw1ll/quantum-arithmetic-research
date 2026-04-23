---- MODULE Counter ----
EXTENDS Naturals

VARIABLE counter

Init == counter = 0
Next == \/ /\ counter < 3
           /\ counter' = counter + 1
        \/ /\ counter' = 0

TypeOK == counter \in 0..3

====
