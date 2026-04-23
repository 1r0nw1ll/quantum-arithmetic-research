---- MODULE Counter ----
EXTENDS Naturals

VARIABLE counter

Init == counter = 0

Increment == counter' = counter + 1

Next == Increment

TypeOK == counter \in Nat

====
