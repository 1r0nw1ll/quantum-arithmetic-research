# QA Lab Inventory — 2026-04-05

**Session**: `lab-consolidation`
**Scope**: read-only audit of `qa_lab/` against actual files (memory was 9+ days stale).
**Purpose**: establish ground truth for the Learning Levels bridge and the SelfImprovementAgent design doc.

> **Key correction vs. memory / kickoff doc**: `qa_lab/agents/self_improvement_agent.py` **already exists** (377 LoC, deterministic rule engine). `TaskType.IMPROVE` is a first-class task type, and the kernel auto-enriches IMPROVE tasks with introspect + orbit + spawn-queue context. The next session is not a greenfield implementation — it is a v2 refactor against the [191]/[192] framework.

---

## 1. Top-level layout (qa_lab/)

Relevant subtrees for this session:

```
qa_lab/
├── qa_core/           # canonical math substrate (algebra + orbits)
├── kernel/            # Von Neumann loop (SENSE→REASON→ACT→VERIFY→LEARN→REPLICATE)
│   ├── loop.py        (1032 LoC)
│   ├── results_log.jsonl           (874 rows)
│   ├── orbit_transition_log.jsonl  (823 rows)
│   ├── task_state_trace.jsonl     (2463 rows)
│   └── spawn_queue/   (7 pending specs)
├── agents/            # QAAgent subclasses — one per TaskType
│   ├── base.py
│   ├── cert_agent.py
│   ├── experiment_agent.py
│   ├── query_agent.py
│   ├── synthesis_agent.py
│   ├── theorem_agent.py
│   ├── stem_agent.py
│   └── self_improvement_agent.py   ← already exists
└── protocols/organ.py # (referenced from loop.py, OrganRegistry)
```

(The rest of `qa_lab/` — ARC, E8, JEPA, SVP, Rust, swarm, Cargo, etc. — is out of scope for the self-improvement track and is not touched here.)

---

## 2. `qa_core/` — canonical substrate

### `algebra.py` (431 LoC)
Integer-only QA algebra: `qa_step`, `qa_norm_mod`, inert primes, `v_p`, `is_prime*`, canonical JSON, SHA-256 helpers. **Convention note**: pre-A1 `{0,…,m-1}` with `qa_step=(b+e)%m`, not the A1-compliant `{1,…,m}` used by Track D scripts. This is documented in-file and intentional — the kernel operates on pre-A1 state but orbit topology (cosmos=24, satellite=8, singularity=1) is identical.

### `orbits.py` (497 LoC)
- `orbit`, `orbit_length` — cycle computation under `T = qa_step`.
- `classify_state` — returns `"cosmos" | "satellite" | "singularity"` by orbit length (cached max-length per modulus).
- `precompute_all_families(m)` — one-shot map of every (b,e) → family (LRU cached).
- `same_orbit`, `is_reachable` — orbit-membership reachability.
- `structural_obstruction(r,m)` — true iff target norm `r` has `v_p(r)=1` for any inert prime (geometric failure algebra).
- `state_successors`, `build_state_graph`, `shortest_witness`, `reachable_states`, `reachable_subgraph` — BFS over `Q` and/or `T` generators; returns integer-only JSON witnesses.
- `nearest_prime_norm_targets`, `nearest_semiprime_norm_targets` — used by QueryAgent.
- **Convergence metrics** (Track C):
  - `kappa(b,e,m,lr) = 1 − (1 − lr·H_QA)²`, where `H_QA = |f(b,e)|/m`.
  - `orbit_contraction_factor` — `ρ(O) = ∏(1−κ_t)²` over the full orbit (Finite-Orbit Descent ρ).
  - `orbit_family_score` — `{cosmos:1.0, satellite:0.5, singularity:0.0}`.

**Observation relevant to the bridge**: `ρ(O)` IS a per-orbit Lyapunov witness for the scalar-quadratic loss inside the Unified Curvature paper. It is **already a concrete numeric quantity on every kernel cycle** (computed in `_sense` and attached to every kernel result as `kernel_rho`). The self-improvement design should not invent a new Lyapunov — it should use `kappa`/`rho` or an aggregate over `results_log.jsonl`.

---

## 3. `kernel/loop.py` — the Von Neumann kernel

### Structure
`Task`, `KernelResult`, `TaskType` (10 values including `IMPROVE`, `SPAWN`, `DEDIFFERENTIATE`, `ORGAN`).

`QALabKernel` holds:
- Own orbit state `(self._b, self._e)` starting at `(1,1)` cosmos — **advanced one qa_step per cycle**.
- `_families = precompute_all_families(modulus)` — the orbit map.
- `_results: List[KernelResult]` — in-process history.
- `_agents: Dict[TaskType, QAAgent]` — one handler per task type.
- `_spawn_queue: List[Dict]` — specs awaiting operator approval.
- `_agent_perf: Dict[str, {ok, fail, verdicts, orbit_drift}]` — per-type counters.
- `_satellite_streak` + `metamorphosis_threshold` (default 5) — Levin metamorphosis trigger.
- Operator safety knobs: `dry_run`, `require_spawn_approval` (default ON), `max_cycles`, `max_agents`.

### Six phases (actual behavior)

| Phase | What it does | Side effects |
|---|---|---|
| **SENSE** | Reads kernel orbit family, `kernel_rho`, `kernel_kappa`, agent availability. | None. Pure read. |
| **REASON** | Warns if family is satellite/singularity. Selects bare agent vs. organ. Sets `spawn_needed` if no handler. | None. |
| **ACT** | Calls `agent.safe_handle(task)` (or `organ.handle`). Auto-enriches `SYNTHESIZE` tasks with injected OB context + introspect report. Auto-enriches `IMPROVE` tasks with `introspect()`, `orbit_context`, `spawn_queue`, `ob_context`, and a **live `kernel_ref`**. | Calls out to agents. |
| **VERIFY** | For `CERTIFY` tasks, runs `<validator> --self-test` via subprocess. Otherwise checks `output["ok"]`. | subprocess (30s timeout). |
| **LEARN** | Updates `_agent_perf` counters. Appends a compact record to `results_log.jsonl`. | Append to JSONL. |
| **REPLICATE** | If `spawn_needed`, produces `QA_AGENT_SPAWN_SPEC.v1` (gated by `require_spawn_approval`), appends to `_spawn_queue`, writes `spawn_queue/spawn_<task_id>.json`. | File write. |

After the six phases, `_advance_orbit()` applies one `qa_step` to kernel state, then `_log_transition_traces` appends rows to `orbit_transition_log.jsonl` and (if any task-semantic `(b,e)` fields are found) `task_state_trace.jsonl`. Finally `_check_metamorphosis` increments/resets `_satellite_streak` and triggers `_trigger_metamorphosis` if threshold reached.

### Self-improvement surface (already present)

- `agent_performance()` → per-type `{total, success_rate, verdicts, orbit_drift_count, needs_improvement}`.
- `introspect()` → `{kernel_orbit, kernel_family, cycles, agent_performance, weak_agents, improvement_suggestions, health}`.
- `_format_introspect_for_context()` → human-readable string auto-appended to SYNTHESIZE context after cycle 2.
- **Bootstrap closure**: `approve_spawn(id)` → `create_agent_stub(spec)` → `register_from_stub(path, task_type)`. The kernel can literally write a Python stub file and load it at runtime.
- **Levin metamorphosis**: after 5 consecutive satellite cycles, dedifferentiates any satellite-phase or progenitor agents, adds a `QA_METAMORPHOSIS_SPEC.v1` to the spawn queue, and resets kernel state to `(1,1)` cosmos.

### Ledger schemas (observed)

`results_log.jsonl` row (sample):
```json
{"description":"...","kernel_orbit":[1,1],"ok":false,"task_id":"3aecc708",
 "task_type":"certify","timestamp":"2026-03-26T00:41:13.821493","verified":false}
```

`orbit_transition_log.jsonl` row:
```json
{"cycle_idx":1,"description":"...","move":"qa_step","ok":true,
 "orbit_family_after":"cosmos","orbit_family_before":"cosmos",
 "run_id":"kernel-20260329T140315Z-c2588ef7",
 "state_after":[1,2],"state_before":[1,1],
 "task_id":"5ab77e81","task_type":"query","timestamp":"...","verified":true}
```

`task_state_trace.jsonl` row: same base plus `{source, path, state, ref_idx}` for every well-typed `(b,e)` in task inputs or agent output.

**Ledger cardinality as of 2026-04-05**: 874 results, 823 kernel transitions, 2463 task-semantic states. Non-trivial corpus — the self-improvement agent can be trained against real history, not synthetic.

### Spawn queue state

7 specs under `qa_lab/kernel/spawn_queue/`, e.g. `spawn_014e893b.json`:
```json
{"cert_family_name":"qa_theorem_agent","required_capability":"theorem",
 "description":"Prove orbit closure theorem for mod-24",
 "gate_compliance":[0,1,2,3,4,5],"schema":"QA_AGENT_SPAWN_SPEC.v1",...}
```
These are pending operator approval from prior sessions.

---

## 4. `agents/` — the QAAgent ecosystem

### `base.py` (212 LoC) — `QAAgent` ABC
Every agent has:
- Orbit state `(b, e)` mod m — **advanced one `qa_step` per call** via `safe_handle`.
- `capabilities: List[str]`, `cert_family: Optional[str]`.
- `orbit_family`, `orbit_score`, `convergence_rate` (`kappa`) properties.
- `can_handle(task)` by `task.task_type.value in capabilities`.
- `certify(body, verdict, evidence, …)` → emits a `QA_EMPIRICAL_OBSERVATION_CERT.v1` artifact with agent orbit state embedded and a SHA-256 hash. Verdicts: `CONSISTENT | CONTRADICTS | PARTIAL | INCONCLUSIVE`; `CONTRADICTS` requires a non-empty `fail_ledger`.
- `safe_handle(task)` = `handle(task)` wrapped in try/except, always advancing orbit and incrementing `_handled_count`.

**Critical property**: every agent is orbit-tracked identically to the kernel. The kernel, each agent, and each task-semantic `(b,e)` field live in the same mod-9 state space.

### Agent roster (LoC, task type, one-liner)

| File | LoC | TaskType | Behavior |
|---|---|---|---|
| `cert_agent.py` | 205 | `CERTIFY` | Wraps meta-validator; runs `--self-test` on cert families. |
| `experiment_agent.py` | 314 | `EXPERIMENT` | Runs experiment scripts. |
| `query_agent.py` | 333 | `QUERY` | Orbit/reachability queries via `orbits.py`. |
| `theorem_agent.py` | 344 | `THEOREM` | Proof-like claim checks against orbit computations. |
| `synthesis_agent.py` | 320 | `SYNTHESIZE` | Reads context + results, proposes next tasks. Receives injected OB + introspect context. |
| `stem_agent.py` | 387 | `STEM`, `DEDIFFERENTIATE` | Totipotent Levin cell: singularity → progenitor → committed (cosmos). Can accept any task in TRIAL mode, commits on 3 successes, dedifferentiates on demand. |
| `self_improvement_agent.py` | 377 | `IMPROVE` | Deterministic 7-rule engine (see §5). |

### `self_improvement_agent.py` — the pre-existing v1 design

**Role** (per docstring): "immune system + growth regulator". Deterministic, O(n_agents), always terminates, starts at cosmos `(1,1)`.

**Inputs** (populated by the kernel's ACT auto-enrichment):
- `introspect` — from `kernel.introspect()`.
- `recent_results` — recent `KernelResult` records.
- `orbit_context` — `{satellite_streak, metamorphosis_threshold, kernel_family}`.
- `spawn_queue` — live list of pending specs.
- `ob_context` — string injected via `kernel.inject_context()`.
- `kernel_ref` — the live kernel (for direct `ADJUST_THRESHOLD` writes).

**Rule engine** (priority-sorted):

| Rule | Trigger | Action | Governed? |
|---|---|---|---|
| R1 SINGULARITY_STUCK | `kernel_family=="singularity"` and `cycles≥2` | `ADJUST_THRESHOLD metamorphosis_threshold=1` | Yes (applied directly) |
| R2 SATELLITE_LOOP | `satellite_streak/meta_threshold > 0.5` | `ADJUST_THRESHOLD meta_threshold-=1` | Yes |
| R3 WEAK_AGENT | `success_rate<0.4 and total>3` | `SPAWN_IMPROVED <agent>` | **Deferred to operator** |
| R4 ORBIT_DRIFT | `orbit_drift_count>2` | `DEDIFFERENTIATE <agent>` | **Deferred** |
| R5 UNHANDLED_CAPABILITY | unapproved specs in spawn_queue | `PROMOTE_STEM <capability>` (max 2) | **Deferred** |
| R6 OB_INSIGHT | keyword match in `ob_context` | `LOG_ONLY <topic>` or `SPAWN_IMPROVED` | Yes (log only) |
| R7 HEALTHY | no other rules fire | `LOG_ONLY "health=good"` | Yes |

**Governed-vs-deferred split**: `ADJUST_THRESHOLD` and `LOG_ONLY` are applied in-cycle. `SPAWN_IMPROVED`, `DEDIFFERENTIATE`, `PROMOTE_STEM` are staged for operator review. This is already the "Level-III operators emit proposals that Will reviews" pattern called for in the kickoff doc — just not yet framed in those words.

**Output**: a certified `QA_EMPIRICAL_OBSERVATION_CERT.v1` with verdict `CONSISTENT | PARTIAL | INCONCLUSIVE`, plus a history record appended to `self._improvement_history`.

### `stem_agent.py` — Levin totipotency

Relevant because the metamorphosis path depends on it. Lifecycle: `UNDIFFERENTIATED (singularity, (0,0)) → PROGENITOR (satellite) → COMMITTED (cosmos)` after 3 successful commits. Exposes `dedifferentiate()` for reset. This is the mechanism through which `DEDIFFERENTIATE` proposals from the SelfImprovementAgent actually take effect.

---

## 5. What is missing (honest gaps)

Findings per kickoff Step 1:
1. **No Lyapunov function is currently computed across cycles.** `kappa`/`rho` exist per state, but nothing aggregates them into a monotonically decreasing quantity over the kernel's trajectory. The `results_log` is a raw stream; no smoothed / trend-aware statistic is derived.
2. **No kernel-quality metric beyond per-agent success rate.** The `health` field in `introspect()` is a 2-value classifier (`good | needs_attention`); there is no scalar that can be compared across sessions.
3. **No proof gating on improvements.** `ADJUST_THRESHOLD` writes to the kernel unconditionally when `kernel_ref` is present. There is no Gödel-machine-style "only apply if provably non-worsening" check.
4. **The kernel never applies Level-III operators to itself.** The modulus is fixed at construction time; the agent set is fixed to whatever was registered. There is no path by which `kernel.modulus 9 → 24` or a structural rewiring of `_agents` could happen autonomously — and per the [191]/[192] framework that is exactly correct for now (Level-III is operator-gated, by design).
5. **No Pisano-style fixed-point analog is defined for the kernel.** This is the interesting open question for Step 5 of the design doc.
6. **OB keyword table is hand-rolled.** `_OB_KEYWORDS` in `self_improvement_agent.py` contains 10 hard-coded topic strings. Not grounded in any cert or formal taxonomy.
7. **Cert family is `None` for `SelfImprovementAgent`**. The class docstring says "will become [123] or similar". A proper cert family for self-improvement doesn't exist yet; this is a gap on the cert side.

---

## 6. What the next session inherits

- A working Von Neumann kernel with real 874/823/2463-row ledgers from prior runs.
- A deterministic `SelfImprovementAgent` v1 that already implements the governed/deferred split correctly for L_1/L_2 actions (even though it doesn't use that vocabulary).
- A Levin totipotency pipeline (`StemAgent`, `dedifferentiate`, metamorphosis trigger) that is the mechanism for L_2b restructuring.
- An orbit-tracked agent base class that gives every handler a Lyapunov-friendly state coordinate for free.
- Two new load-bearing certs ([191] strict filtration, [192] Pisano fixed point at m=24) that the kernel has not yet been taught about.

The consolidation's job is to make those last two certs actually govern the lab.
