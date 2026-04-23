---- MODULE CalibrationMonitor ----
EXTENDS Naturals

VARIABLES phase, deviation

Init == /\ phase = "idle"
        /\ deviation = 0

Observe == /\ phase' \in {"observe", "settle"}
           /\ deviation' \in 0..2

Reset == /\ phase' = "idle"
         /\ deviation' = 0

Next == Observe \/ Reset

TypeOK == /\ phase \in {"idle", "observe", "settle"}
          /\ deviation \in 0..2

====
