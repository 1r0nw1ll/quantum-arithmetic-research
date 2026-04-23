---- MODULE ObservationMonitor ----
EXTENDS Naturals

VARIABLES obs, streak

Init == /\ obs = "idle"
        /\ streak = 0

ObserveReady == /\ obs' = "ready"
                /\ streak' = streak + 1

ObserveFault == /\ obs' = "fault"
                /\ streak' = streak

ObserveIdle == /\ obs' = "idle"
               /\ streak' = 0

Next == ObserveReady \/ ObserveFault \/ ObserveIdle

TypeOK == /\ obs \in {"idle", "ready", "fault"}
          /\ streak \in Nat

====
