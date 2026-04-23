---- MODULE Counter ----
EXTENDS Naturals

VARIABLE counter

Init == counter = 0

Inc == /\ counter < 3
       /\ counter' = counter + 1

Reset == counter' = 0

Next == Inc \/ Reset

Inv_Bounded == counter \in 0..3

====
