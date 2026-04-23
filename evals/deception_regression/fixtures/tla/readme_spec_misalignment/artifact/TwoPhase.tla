---- MODULE TwoPhase ----
EXTENDS Naturals

VARIABLE round

Init == round = 0

Advance == round' = round + 1

Next == Advance

Inv_RoundMonotone == round >= 0

====
