---- MODULE QAAxioms ----
EXTENDS Naturals

VARIABLES observer_projection_firewall, theorem_nt

Init == /\ observer_projection_firewall = TRUE
        /\ theorem_nt = "lane-two"

Next == /\ observer_projection_firewall' = observer_projection_firewall
        /\ theorem_nt' = theorem_nt

Inv_QA == TRUE

====
