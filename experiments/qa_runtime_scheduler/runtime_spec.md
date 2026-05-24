# QA Substrate Ladder — Level 3: Runtime/Scheduler Benchmark

## What Is Being Tested

Tasks are not opaque jobs. Each task carries a QA state packet `(b,e)` with
integer-only derived fields (`d=b+e`, `a=e+d`, `C=2ed`, `F=ab`, `G=D+E`,
`J=bd`, `X=ed`, `K=da`, `D=d²`, `I=|C-F|`). Legal transitions are generator
moves (sigma, mu, lambda2, nu). Tasks have target orbit classes and failure
orbits. The QA scheduler reasons about reachability, avoids fail orbits by
move selection, and uses BFS to find guaranteed escape paths when recovery is
needed.

## Schedulers

| Scheduler       | Move selection | Recovery |
|-----------------|---------------|----------|
| fifo            | Random generator | Random walk ≤ k steps |
| priority        | Random generator | Random walk ≤ k steps |
| qa_scheduler    | Greedy: minimize return_distance to target_orbit_9; avoid fail_orbit_9 | BFS escape path (guarantees recovery when path exists within k) |

## Hypotheses

**H1 (qa_lawful mode)**: QA scheduler reduces failure rate (tasks entering
fail_orbit at least once) vs FIFO and priority. Mechanism: move selection
avoids fail_orbit_9 proactively.

*What would falsify H1*: QA fail_rate ≥ FIFO fail_rate on qa_lawful workloads.
This occurs when: (a) fail orbits occupy all legal neighbor orbits (forced
entry), or (b) the recovery window k is so small that even BFS cannot escape.

**H2 (qa_lawful mode)**: QA scheduler reduces wasted recovery steps vs FIFO
and priority. Mechanism: BFS finds the shortest escape path; random walk may
use up to k steps even when 1 suffices.

*What would falsify H2*: QA mean_wasted_steps ≥ FIFO mean_wasted_steps. This
occurs when QA's greedy move selection causes it to enter fail states more
often than expected (negative side of avoiding fail_orbit while target is
close to fail orbit).

**H3 (random_opaque mode)**: QA scheduler does NOT beat FIFO/priority on
random opaque workloads. Mechanism: LCG transitions have no QA structure. BFS
return_distance computation is overhead with no benefit.

*H3 is a structural falsifier of QA overclaiming*: if QA beats FIFO on random
tasks, that is evidence of an implementation bug (the comparison is unfair).

**H4 (adversarial_trap mode)**: QA's return_distance-first task selection is
exploitable by a bait-and-trap workload. Bait tasks (dist=0, prio=1, generous
deadline) have low QA score → QA drains them first. Trap tasks (dist=3, prio=9,
tight deadline) have high QA score → QA defers them until past their deadline.
Priority's urgency metric (`priority×100 - remaining`) handles trap tasks first,
meeting all deadlines. All schedulers use greedy (informed) execution on
adversarial_trap to isolate task-selection order, not execution quality.

*What would falsify H4*: QA deadline_miss_rate < priority deadline_miss_rate +
0.05 on adversarial_trap. This would indicate QA's distance-based selection
adapts to deadline pressure quickly enough to avoid deferring urgent tasks.

## Benchmark Results (N=100, 500 tasks, seed=42)

| Mode | Scheduler | fail | unrec | wasted | steps | dl_miss |
|------|-----------|------|-------|--------|-------|---------|
| qa_lawful | fifo | 79 | 0 | 0.40 | 8.9 | — |
| qa_lawful | priority | 76 | 0 | 0.40 | 8.6 | — |
| qa_lawful | qa_scheduler | **0** | 0 | **0.00** | **2.8** | — |
| random_opaque | fifo | 1 | 1 | 0.00 | 17.7 | — |
| random_opaque | priority | 1 | 1 | 0.00 | 17.7 | — |
| random_opaque | qa_scheduler | 1 | 1 | 0.00 | 17.7 | — |
| deadline_only | all | 0 | 3 | 0.00 | 17.9 | — |
| adversarial_trap | fifo | 0 | 0 | 0.00 | 1.2 | **0.100** |
| adversarial_trap | priority | 0 | 0 | 0.00 | 1.2 | **0.000** |
| adversarial_trap | qa_scheduler | 0 | 0 | 0.00 | 1.2 | 0.080 |

## Where QA Wins

- **qa_lawful**: 0 failures vs 79 (FIFO); 0 wasted steps vs 0.40; 2.8 mean steps vs 8.9.
  Orbit-based move avoidance eliminates all failure events. BFS recovery gives
  zero wasted steps because no recovery is ever needed.
- **mixed_runtime**: QA half (qa_lawful tasks) pulls total wasted steps to 0.00.

## Where QA Fails / Does Not Win

- **random_opaque**: Identical outcomes across all schedulers (17.7 steps, 1 unrec).
  LCG transitions have no QA structure; BFS overhead is pure cost.
- **deadline_only**: All schedulers equally constrained by tight deadlines. 3 tasks
  timeout regardless. QA adds no deadline-awareness advantage.
- **adversarial_trap**: Priority dominates (dl_miss=0.000 vs QA=0.080 vs FIFO=0.100).
  QA's return_distance-first selection drains dist=0 bait tasks before dist=3 trap
  tasks, causing trap tasks to miss tight deadlines. Priority's urgency metric
  (`priority×100 - remaining`) correctly identifies and processes urgent trap tasks
  first. QA's score formula lacks pure urgency weighting — the `remaining` component
  is additive, not multiplicative with priority. QA is slightly better than FIFO
  (0.08 vs 0.10) because `remaining` decreases as trap deadline approaches, but
  this is too weak to match priority's explicit urgency ordering.

## What This Does Not Claim

- QA scheduler is not a drop-in OS scheduler replacement. It requires tasks whose
  state transitions are generator-lawful — real OS tasks are not.
- Steps-to-completion advantage (2.8 vs 8.9) is partly a product of the greedy
  BFS toward target_orbit — this is not free in a real runtime (BFS costs compute).
- No continuous QA phase surrogate is used. All timing is perf_counter_ns on
  in-memory integer operations.

## Level 4 (Next Rung)

Natural extension: **QA network/routing benchmark**. Packets are QA state packets;
routing = generator-legal path finding; congestion = orbit saturation. Compare
QA-native router (exploit orbit structure for load balancing) vs shortest-path
and random routing.
