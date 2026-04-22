# qa_tla_blind — QA-vs-TLA+ Blind Reproduction Benchmark

> **Honesty sentence.** A blind match to validated TLA+ results is necessary but
> not sufficient; QA only passes strongly when it contributes non-ornamental
> structure beyond plain state-machine recovery.

**Purpose.** Take validated specs (both the `tlaplus/Examples` corpus and
QA's own machine-validated control theorems), hide the solution from the
reproducing session, have a fresh session re-express each spec in QA-native
terms (discrete orbits, Theorem NT firewall, integer path-time, no-zero
states), then diff the QA result against the verified ground truth. Failure
modes become first-class data.

Motivation: stop settling for hand-wavy QA-as-spec-language claims. Run the eval.

## Two reference poles

The benchmark uses both directions to pin the scoring scale:

- **Neutral/negative controls** (`tlaplus/Examples` — DieHard, Bakery, Paxos,
  etc.): non-QA specs where QA may be compatible but non-contributory.
- **Positive control** (QA's Jan-10-2026 control theorems — `QA_CONTROL_THEOREMS.md`
  + `QARM_v02_Stats.tla` + `ALL_INVARIANTS_VALIDATED.md`): QA's own
  machine-validated result. If QA framing cannot blind-reproduce these, the
  whole benchmark is mis-calibrated and results on non-QA specs are
  uninterpretable.

Without the positive control, the benchmark is asymmetric — it can show
ornamentality but cannot calibrate decisive contribution.

## Layout

```
qa_tla_blind/
├── ground_truth/        # tlaplus/Examples clone (GITIGNORED — do not read from main session)
├── prompts/             # <spec>.md — English statement + invariant names, no bodies
├── attempts/            # <spec>.md — QA reproduction written from prompt only
├── diffs/               # <spec>.md — scored comparison vs ground truth
└── scripts/             # extraction + diff harnesses
```

## Protocol

### Blindness

1. **Main session never reads `ground_truth/**/*.tla`.** Ever. This is load-bearing.
2. A subagent (with its own context) extracts the English problem statement and
   the *names* of `INVARIANT`/`THEOREM`/`PROPERTY` declarations into
   `prompts/<spec>.md`. No predicate bodies, no `Next`-state actions, no `Init`.
3. Main session writes `attempts/<spec>.md` using only the prompt.
4. A second subagent (or the first re-invoked) scores the attempt against the
   ground truth, writing `diffs/<spec>.md`.

### Reproduction task (per spec)

Given only `prompts/<spec>.md`, produce:

1. **QA state encoding.** Define (b,e) — what concrete integers represent each
   system variable, under mod-9 or mod-24. Derived coords d=b+e, a=b+2e.
2. **Observer projection.** What continuous/external data enters the read
   layer? What integer bucketing crosses the firewall into QA? (Theorem NT.)
3. **QA dynamics.** Discrete step rule `((b+e-1) % m) + 1`. Orbit classification
   of reachable states (Cosmos / Satellite / Singularity).
4. **Invariant restatement.** For each named invariant in the prompt, write its
   QA-native form. Prove or demonstrate by orbit enumeration.
5. **Liveness (if applicable).** QA path-time (integer k) argument.
6. **Counterexamples.** Any state the QA encoding forbids that TLA+ allowed,
   or vice versa.

### Scoring (diffs/<spec>.md)

Per invariant: classify as one of

- `reproduced` — QA invariant is equivalent to TLA+ invariant
- `strengthened` — QA invariant implies TLA+ invariant strictly
- `weakened` — TLA+ invariant implies QA invariant but not vice versa
- `missed` — QA attempt did not cover this invariant at all
- `wrong` — QA attempt contradicts ground truth (QA claims a property that
  model-checker refuted, or refutes one it verified)

Plus overall tags from the failure taxonomy below.

## Failure Taxonomy (pre-registered)

Every reproduction result gets one or more tags. Tags are not mutually exclusive
— a single attempt can be both `ornamental-overlay` and have a `proof-gap`.

| Tag | Meaning |
|---|---|
| `no-mapping-exists` | No defensible (b,e) encoding for the system state |
| `wrong-observer-projection` | Picked the wrong quantity to bucket across the firewall |
| `orbit-mismatch` | States claimed in one orbit actually belong in another |
| `invariant-inexpressible` | The property has no QA-native form under this mapping |
| `proof-gap` | Mapping is sound but the QA proof/enumeration is incomplete |
| `qa-stronger-than-tla` | QA forbids states TLA+ allowed (may be a real QA refinement OR an encoding error — flag for review) |
| `qa-weaker-than-tla` | QA allows states TLA+ refuted (encoding lost information) |
| `ornamental-overlay` | Benchmark was solved, but the QA layer was pasted on top — axioms satisfied syntactically, no discriminating power, orbit/derived/firewall machinery contributes nothing |

`ornamental-overlay` is the failure mode that distinguishes "QA works" from
"QA is compatible with working." An attempt can reproduce all invariants
(Recovery 2/2) and still be entirely ornamental (Contribution 0). Flagging
this explicitly blocks the drift from "reproduced the invariants" to
"validated QA."

## Two-axis scoring

Every scored attempt gets **both** axes. They are independent.

### Recovery score (per named invariant)

Each invariant gets one of:

- `reproduced` — QA invariant is equivalent to TLA+ invariant
- `strengthened` — QA invariant implies TLA+ invariant strictly (rules out states TLA+ allowed)
- `weakened` — TLA+ invariant implies QA invariant but not vice versa
- `missed` — QA attempt did not cover this invariant at all
- `wrong` — QA attempt contradicts ground truth

Aggregate as `X/Y reproduced (+ sharpenings/weakenings counted separately)`.

### Contribution score (per artifact, 0-4)

Judged relative to the **QA control theorem shape** (closed-form counts over
a discrete generator algebra, SCC structure, failure-class algebra,
monotonicity under generator expansion). Not judged by generic "insight."

- **0 — Decorative.** QA vocabulary added no real compression or insight.
  Axioms satisfied syntactically only. A plain state-machine restatement
  would be equivalent or better. → `ornamental-overlay`.
- **1 — Compatible.** QA framing fits cleanly but adds little. E.g. T1
  path-time count coincides with TLA+ step count; no new structure exposed.
- **2 — Useful.** QA framing simplifies an invariant, reachability argument,
  or classification. E.g. a modular encoding makes a proof shorter.
- **3 — Strong.** QA framing exposes structure the original presentation
  leaves implicit. E.g. reveals an SCC/orbit structure; reveals a symmetry
  the TLA+ form obscures; produces a closed-form count.
- **4 — Decisive.** QA framing yields a genuinely better proof,
  classification, or predictive handle. Produces a result of the shape of
  `QA_CONTROL_THEOREMS.md` (closed forms over generators; SCC/edge/failure
  counts; monotonicity lemmas). This is the positive-control endpoint.

**DieHard retroactively:** Recovery 2/2, Contribution 0 (`ornamental-overlay`).

**QA control theorems (positive control):** Expected Recovery = full,
Contribution = 4. If not, the harness itself is broken.

### Specific markers of Contribution ≥ 3

The scorer should check whether the attempt produces any of these (the QA
control-theorem signature):

- Generator-relative structure (named σ/μ/λ₂/ν-style operators)
- SCC / orbit organization of the reachable state graph
- Closed-form counts (not enumeration, not "by BFS")
- Failure-class algebra (OUT_OF_BOUNDS, PARITY, etc. counted by closed form)
- Monotonicity under generator expansion (adding Σ can only merge, not split)

If none of these are present, Contribution ≤ 1, even if Recovery is full.

## Scope ladder

First run in order:

1. **DieHard** — jug puzzle, trivial, shakedown (shipped; Recovery 2/2, Contribution 0)
2. **QA_CONTROL_THEOREMS** — positive control, QA's own Jan-10-2026 result (SCC + edge + failure counts over `{σ, μ, λ₂, ν}` on `Caps(N,N)`). Calibrates the Contribution=4 endpoint.
3. **Bakery-Boulangerie** — mutex via tickets; first non-toy test of QA's modular machinery.
4. **MissionariesAndCannibals** — integer state, small; sanity check.
5. **Prisoners** / **DiningPhilosophers** — small concurrency.
6. **TwoPhase** / **Paxos** — distributed, hard; where QA's SCC/orbit framing either contributes on quorum/ballot structure or doesn't.

Stop early if (a) positive control fails (harness is broken) or (b) two
consecutive specs show `ornamental-overlay` on axes that QA should dominate.

## Hygiene

- `ground_truth/` is gitignored (see repo `.gitignore`). Commit only prompts,
  attempts, diffs, scripts, README.
- If the blindness pact is broken (main session reads a ground-truth file),
  that spec is tainted — mark the attempt file `TAINTED` and skip scoring.
- Never edit a `prompts/<spec>.md` after writing `attempts/<spec>.md` —
  backfilling the prompt to match the attempt defeats the test.
