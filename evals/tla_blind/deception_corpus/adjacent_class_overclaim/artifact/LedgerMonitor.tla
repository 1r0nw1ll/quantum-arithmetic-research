---- MODULE LedgerMonitor ----
EXTENDS Naturals

VARIABLES state, tick

Init == /\ state = "idle"
        /\ tick = 0

Advance == /\ state' \in {"collect", "publish"}
           /\ tick' \in 0..3

Reset == /\ state' = "idle"
         /\ tick' = 0

Next == Advance \/ Reset

TypeOK == /\ state \in {"idle", "collect", "publish"}
          /\ tick \in 0..3

====
