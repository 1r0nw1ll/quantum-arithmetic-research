# Repair: domain counter core works, edge cases missed

Diagnose. The CSV domain-counter works on happy input. Spec calls for
malformed-row handling (missing `@`), which this deliverable skips in code
but the README does not note as scope. Decide revise vs reject.
