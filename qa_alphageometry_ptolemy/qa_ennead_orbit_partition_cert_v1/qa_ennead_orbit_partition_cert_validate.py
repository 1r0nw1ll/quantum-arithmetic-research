#!/usr/bin/env python3
"""
QA Egyptian Ennead Orbit Partition Cert [314] — validator

Primary sources:
  Iverson, B. (1975-1996) Quantum Arithmetic vols 1-2, QA-1/QA-2.
  Budge, E.A.W. (1904) The Gods of the Egyptians, Methuen, London, Vol. 1,
    pp.86-105 (Ennead of Heliopolis: 9 deities, 8-active + 1-source partition).
  Wilkinson, R.H. (2003) The Complete Gods and Goddesses of Ancient Egypt,
    Thames & Hudson, ISBN 978-0-500-05120-7, pp.74-81.

QA mapping: The Egyptian Ennead of Heliopolis partitions 9 primary deities as
1 source-god (Atum, self-created, unmoved mover) + 8 active gods. This mirrors
the QA mod-9 orbit partition of the two lowest orbit families:
  Singularity = 1 T-fixed state (9,9) -- the discrete analog of Atum.
  Satellite   = 8 states in a single 8-cycle -- the discrete analog of the
                8 active Ennead gods.
State space: {1,...,9}^2 = 81 states total.
T-step (A1 compliant): qa_step from qa_orbit_rules (canonical).

Orbit family classification: orbit_family on (b, e, 9) from qa_orbit_rules
  (canonical, period-based: period 1=Singularity, period 8=Satellite, else Cosmos).

Five claims:
  C1  Ennead count: |Satellite| + |Singularity| = 8 + 1 = 9, matching the
      Ennead cardinality; total partition covers all 81 = 1+8+72 states.
  C2  Singularity is the unique T-fixed point in {1,...,9}^2:
      qa_step(9,9,9) returns (9,9); no other state satisfies qa_step(b,e,9)=(b,e).
  C3  Satellite 8-cycle: orbit from (3,3) has period exactly 8 and visits
      all 8 Satellite states:
      (3,3)->(3,6)->(6,9)->(9,6)->(6,6)->(6,3)->(3,9)->(9,3)->(3,3).
  C4  3-divisibility characterization exact: orbit_family_divisor_shortcut on
      (b,e,9) agrees with canonical orbit_family on (b,e,9) for all 81 pairs;
      Satellite = {(b,e): 3|b and 3|e and (b,e)!=(9,9)}; counts {1,8,72}=81.
  C5  Theorem NT: mythological attributes of each Ennead deity (domain,
      iconography, narrative role) are observer projections onto the discrete
      QA orbit structure; orbit_family and T-period are the falsifiable
      integer claims; no float state enters the QA layer.
"""

QA_COMPLIANCE = (
    "cert_validator -- A1 no-zero arithmetic via qa_step from qa_orbit_rules; "
    "canonical orbit_family on (b,e,9) from qa_orbit_rules; "
    "3-divisibility shortcut vs canonical agreement; "
    "Theorem NT Ennead observer"
)

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qa_orbit_rules import (  # noqa: E402
    orbit_family,
    orbit_family_divisor_shortcut,
    orbit_period,
    qa_step,
)

checks = {}
passed = 0
failed = 0

M = 9

singularity_states = [(b, e) for b in range(1, M + 1) for e in range(1, M + 1)
                      if orbit_family(b, e, M) == "singularity"]
satellite_states   = [(b, e) for b in range(1, M + 1) for e in range(1, M + 1)
                      if orbit_family(b, e, M) == "satellite"]
cosmos_states      = [(b, e) for b in range(1, M + 1) for e in range(1, M + 1)
                      if orbit_family(b, e, M) == "cosmos"]

# C1 -- Ennead count: |Satellite|+|Singularity| = 9; full partition = 81
ok_c1 = (len(singularity_states) == 1
          and len(satellite_states) == 8
          and len(singularity_states) + len(satellite_states) == 9
          and len(cosmos_states) == 72
          and len(singularity_states) + len(satellite_states) + len(cosmos_states) == 81)
checks["C1_ennead_count_satellite8_singularity1_total9"] = "PASS" if ok_c1 else "FAIL"
if ok_c1: passed += 1
else:     failed += 1

# C2 -- Singularity is the unique T-fixed point
step_9_9 = qa_step(9, 9, M)
fixed_points = [(b, e) for b in range(1, M + 1) for e in range(1, M + 1)
                if qa_step(b, e, M) == (b, e)]
ok_c2 = (step_9_9 == (9, 9)
          and singularity_states == [(9, 9)]
          and fixed_points == [(9, 9)])
checks["C2_singularity_unique_t_fixed_point"] = "PASS" if ok_c2 else "FAIL"
if ok_c2: passed += 1
else:     failed += 1

# C3 -- Satellite 8-cycle: (3,3) orbit visits all 8 states with period 8
EXPECTED_SAT_ORBIT = [(3,3),(3,6),(6,9),(9,6),(6,6),(6,3),(3,9),(9,3)]
b, e = 3, 3
traversed = []
for _ in range(8):
    traversed.append((b, e))
    b, e = qa_step(b, e, M)
period_33 = orbit_period(3, 3, M)
ok_c3 = (traversed == EXPECTED_SAT_ORBIT
          and (b, e) == (3, 3)
          and period_33 == 8
          and frozenset(traversed) == frozenset(satellite_states))
checks["C3_satellite_8cycle_period8_visits_all8_states"] = "PASS" if ok_c3 else "FAIL"
if ok_c3: passed += 1
else:     failed += 1

# C4 -- 3-divisibility shortcut agrees with canonical orbit_family for all 81 pairs;
#        partition counts {1,8,72}=81
shortcut_agree = all(
    orbit_family_divisor_shortcut(b, e, M) == orbit_family(b, e, M)
    for b in range(1, M + 1) for e in range(1, M + 1)
)
sat_by_div = frozenset(
    (b, e) for b in range(1, M + 1) for e in range(1, M + 1)
    if b % 3 == 0 and e % 3 == 0 and not (b % M == 0 and e % M == 0)
)
ok_c4 = (shortcut_agree
          and sat_by_div == frozenset(satellite_states)
          and len(singularity_states) + len(satellite_states) + len(cosmos_states) == 81)
checks["C4_3divisibility_shortcut_agrees_canonical_counts_1_8_72"] = "PASS" if ok_c4 else "FAIL"
if ok_c4: passed += 1
else:     failed += 1

# C5 -- Theorem NT: all orbit data are integers; no float state;
#        Satellite period=8, Singularity period=1, Cosmos seed (1,1) period=24
period_sing   = orbit_period(9, 9, M)
period_cosmos = orbit_period(1, 1, M)
all_int_states = all(isinstance(b, int) and isinstance(e, int)
                     for b, e in singularity_states + satellite_states + cosmos_states)
ok_c5 = (period_33    == 8
          and period_sing   == 1
          and period_cosmos == 24
          and all_int_states)
checks["C5_theorem_nt_periods_int_satellite8_sing1_cosmos24"] = "PASS" if ok_c5 else "FAIL"
if ok_c5: passed += 1
else:     failed += 1

print(json.dumps(checks, indent=2))
print(f"\nTotal: {passed} PASS, {failed} FAIL")
if failed:
    sys.exit(1)
