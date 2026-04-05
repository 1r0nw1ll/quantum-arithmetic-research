# Learning Levels Bridge — qa_lab kernel ↔ [191]/[192]

**Session**: `lab-consolidation` (2026-04-05)
**Status**: permanent reference. Every claim about "Level-X" behavior in the qa_lab kernel or its agents should cite this doc and the underlying cert.
**Prerequisites**: read `LAB_INVENTORY_2026-04-05.md` first. Ground truth on what the kernel actually does lives there.
**Authority**:
- [191] `QA_BATESON_LEARNING_LEVELS_CERT.v1` — strict filtration $\mathcal{L}_0 \subsetneq \mathcal{L}_1 \subsetneq \mathcal{L}_{2a} \subsetneq \mathcal{L}_{2b} \subsetneq \mathcal{L}_3$
- [192] `QA_DUAL_EXTREMALITY_24_CERT.v1` — Pisano period $\pi$ is the canonical Level-III operator on moduli; $\pi(9)=24$; $\pi(24)=24$ (minimum non-trivial fixed point).
- `docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md` — full sketch with proofs.

---

## 1. The filtration, stated in kernel-native language

Every operator the kernel or an agent can apply acts on some QA state. The state in question can be one of three:

- **task-semantic state** `s_task = (b,e)` inside a task's inputs or outputs (e.g. QueryAgent's `(3,6)`),
- **agent internal state** `s_agent = (b_agent, e_agent)` advanced one `qa_step` per `safe_handle`,
- **kernel state** `s_kernel = (self._b, self._e)` advanced one `qa_step` per cycle.

All three live in the same ambient $S_m$ for the configured modulus $m$ (default $m=9$). The invariants of [191] — orbit, family, modulus, ambient category — apply to each.

| Level | Invariant preserved | Invariant broken | Cert-[191] identifier |
|---|---|---|---|
| $\mathcal{L}_0$ | Identity on a point | — | Fixed point (singularity) |
| $\mathcal{L}_1$ | Orbit membership | — | "Doing the same thing, different place in the cycle" |
| $\mathcal{L}_{2a}$ | Orbit **family** (cosmos/satellite/singularity) | Orbit membership | Reframe within realm |
| $\mathcal{L}_{2b}$ | Modulus $m$ | Orbit family | Change of realm |
| $\mathcal{L}_3$ | Ambient category | Modulus | Change of modulus / algebra |

Strictness witnesses live in [191] ((Z/9Z)* scalar action for $\mathcal{L}_{2a}$; `×3 mod 9` and constant maps for $\mathcal{L}_{2b}$; Pisano $\pi$ or explicit reductions for $\mathcal{L}_3$).

---

## 2. Kernel phases → operator classes

Each phase of `QALabKernel.run_cycle` is classified below. "Acts on" names which state the operation moves.

| Phase (`loop.py`) | Acts on | Operation | Level | Invariant preserved | Cert hook |
|---|---|---|---|---|---|
| `_sense` | — | pure read (no state change) | — | everything | — |
| `_reason` | — | selection only; no state change | — | everything | — |
| `_act` (agent handler) | $s_{\text{agent}}$ via `safe_handle` → `_advance_orbit` | one `qa_step` | $\mathcal{L}_1$ | orbit of agent | base.py:110 |
| `_act` (task-semantic work) | $s_{\text{task}}$ (e.g. QueryAgent BFS, TheoremAgent claim check) | depends on agent, but all current handlers use `qa_step`/`t_step`/BFS inside one $S_m$ | $\mathcal{L}_1$ | orbit of task state | orbits.py BFS stays inside $S_m$ |
| `_verify` | — | subprocess validator check | — | — | — |
| `_learn` | ledger | append-only log | — | — | — |
| `_replicate` → adds stub to `_spawn_queue` | `_agents` dict | **extends** the set of handlers without changing modulus | $\mathcal{L}_{2a}$ on the agent-population state | modulus + ambient algebra; kernel $m$ unchanged | loop.py:571, `QA_AGENT_SPAWN_SPEC.v1` |
| `_advance_orbit` | $s_{\text{kernel}}$ | one `qa_step` on kernel state | $\mathcal{L}_1$ | orbit of kernel | loop.py:630 |
| `_check_metamorphosis` ≥ threshold → `_trigger_metamorphosis` | agent population + $s_{\text{kernel}}$ | dedifferentiate satellite agents **and** reset kernel to $(1,1)$ cosmos | $\mathcal{L}_{2b}$ | modulus | loop.py:806 |
| `approve_spawn` → `create_agent_stub` → `register_from_stub` | `_agents` dict | operator-gated addition of a new handler | $\mathcal{L}_{2a}$ | modulus + family of existing agents | loop.py:907–1032 |

### The only Level-III operation in the current kernel

Autonomous $\mathcal{L}_3$ operators — operators that change the modulus, the ring, or the ambient category — **do not exist** in the current loop. The kernel's `self.modulus` is set at construction and never written after that. This is by design and is consistent with the [191] recommendation that higher-tier operators require operator approval (double bind avoidance).

A human operator instantiating a new `QALabKernel(modulus=24)` is itself a Level-III action, but it is fully external to the running kernel.

### Metamorphosis reclassification

The inventory doc calls metamorphosis a "Level-III protocol" loosely. Against the [191] invariants it is actually $\mathcal{L}_{2b}$:
- modulus $m$ is unchanged (still 9),
- ambient algebra is unchanged ($\mathbb{Z}[\varphi]/9$),
- orbit family of the kernel state *does* change — the reset $(self._b, self._e) \leftarrow (1,1)$ replaces whatever family was current (satellite, by trigger condition) with cosmos.

Family change, modulus preserved ⇒ $\mathcal{L}_{2b}$.

---

## 3. Agents → operator classes

The claim of this section: **every existing agent is a Level-I operator on its own agent state, even when the task it handles lives at a higher level.** The level of an agent is the level of its induced transition on `(self._b, self._e)` — which for every current handler is one `qa_step` per call.

| Agent (`qa_lab/agents/`) | Handles | Agent-state transition | Level |
|---|---|---|---|
| `QueryAgent` | `QUERY` | one `qa_step` on agent state; task work is BFS inside $S_m$ | $\mathcal{L}_1$ |
| `TheoremAgent` | `THEOREM` | one `qa_step`; proof checks over $S_m$ | $\mathcal{L}_1$ |
| `ExperimentAgent` | `EXPERIMENT` | one `qa_step`; runs external script | $\mathcal{L}_1$ |
| `CertAgent` | `CERTIFY` | one `qa_step`; calls `--self-test` | $\mathcal{L}_1$ |
| `SynthesisAgent` | `SYNTHESIZE` | one `qa_step`; proposes next tasks | $\mathcal{L}_1$ |
| `StemAgent` (undifferentiated) | `STEM`, `DEDIFFERENTIATE` | $(0,0) \to (0,0)$ in trial mode | $\mathcal{L}_0$ (fixed at singularity) |
| `StemAgent` (progenitor) | any trial task type | one `qa_step`; orbit family = satellite | $\mathcal{L}_1$ |
| `StemAgent` (committed) | committed task type | one `qa_step`; orbit family = cosmos | $\mathcal{L}_1$ |
| `SelfImprovementAgent` v1 | `IMPROVE` | one `qa_step` on its own state *plus* governed writes to `kernel._agents`/kernel attributes | see §4 |

### SelfImprovementAgent v1 — per-rule level classification

This is the critical table for the design doc. Every rule in the v1 rule engine is pinned to a level and an invariant:

| Rule | Action | Target | Level | Invariant broken on target |
|---|---|---|---|---|
| R1 SINGULARITY_STUCK | `ADJUST_THRESHOLD metamorphosis_threshold=1` | kernel scalar | $\mathcal{L}_1$ on agent; parameter tweak on kernel (not a state-space operator at all) | none — parameter lives outside $S_m$ |
| R2 SATELLITE_LOOP | `ADJUST_THRESHOLD meta_threshold-=1` | kernel scalar | same | none |
| R3 WEAK_AGENT | `SPAWN_IMPROVED <agent>` | `_agents` dict (proposed) | $\mathcal{L}_{2a}$ on agent-population state | orbit assignment within population |
| R4 ORBIT_DRIFT | `DEDIFFERENTIATE <agent>` | agent's `(b,e)` | $\mathcal{L}_{2b}$ on that agent | family changes (cosmos/satellite → singularity) |
| R5 UNHANDLED_CAPABILITY | `PROMOTE_STEM <capability>` | StemAgent internal lifecycle | $\mathcal{L}_{2a}$ on StemAgent | orbit (progenitor → committed) |
| R6 OB_INSIGHT | `LOG_ONLY` | ledger only | — | none |
| R7 HEALTHY | `LOG_ONLY` | ledger only | — | none |

**Key observation**: the v1 governed/deferred split already matches the [191] safety stratification, by accident. `ADJUST_THRESHOLD` (not a state-space operator) and `LOG_ONLY` are applied in-cycle. Every rule that is actually $\mathcal{L}_{2a}$ or $\mathcal{L}_{2b}$ on any state (R3, R4, R5) is deferred to operator approval. This is what the [191] double-bind analysis recommends — $\mathcal{L}_2$ and above require higher-tier reasoning to avoid runaway.

**What v1 is missing**: the split is implicit, not cert-backed. The rule engine does not know that R3/R4/R5 are higher-level operators; it just happens to defer them for unrelated reasons ("operator review"). This is fine as v1, but v2 should name the levels and use them as the gate.

---

## 4. Gap analysis (kickoff Step 3, answered against actual files)

### 4.1 Is there a Lyapunov function on kernel state?
Not aggregated, but the raw materials exist. Candidates, in descending order of readiness:

1. **Per-cycle `ρ(O)` (already computed)**. `orbits.orbit_contraction_factor(b,e,m,lr) = ∏(1−κ_t)²` over the full orbit of the current state. The Unified Curvature paper proves $L_{t+L} = \rho(O) L_t$ exactly for scalar quadratic loss, so for any scalar-quadratic objective **$\rho$ is a certified contraction factor, not a proxy**. The kernel attaches it to every `KernelResult` as `kernel_rho` (loop.py:657). **What is missing**: aggregation. There is no "running $\rho$ over $N$ cycles" statistic.
2. **Per-agent `success_rate` (already computed)**. Monotonically bounded in $[0,1]$. Not a Lyapunov (can decrease as more tasks come in), but can be aged with EWMA.
3. **Satellite fraction over a sliding window**. `_satellite_streak` is point-in-time; it should be generalized to "fraction of the last $N$ cycles in satellite family" as a leading indicator.
4. **Ledger compressibility / trace entropy**. The research thread mentioned in OB ("compression kernel trace") suggests: a kernel that has improved itself should produce a trace that is more compressible by its own codec. This is a Kolmogorov-style Lyapunov with a strong self-referential flavor and is a natural Level-III quantity. **Do not commit to this in v1.** Design doc will treat it as a candidate to test.

**Recommendation for v2**: use an EWMA-smoothed `ρ(O)` as the primary Lyapunov, with satellite-fraction-in-window and weighted-success-rate as secondary indicators. All three are integer-only / rational (or rational after EWMA with integer-fraction weights).

### 4.2 Is there a metric on kernel quality?
Currently only the 2-value `health ∈ {good, needs_attention}` string in `introspect()`. No cross-session scalar exists. v2 should emit a `kernel_quality_vector = (ρ_smoothed, p_cosmos, p_verified, n_weak_agents)` that is logged to its own ledger per IMPROVE cycle and can be diffed across sessions.

### 4.3 Are improvements currently gated by any test?
Partially. `ADJUST_THRESHOLD` is applied without any pre/post check. Every other rule defers to operator approval — this is gating, but it is human-in-the-loop gating, not machine-checkable Gödel-style proof gating. No Level-III operator is applied autonomously, which is the important safety property per [191].

**Recommendation for v2**: introduce a `safe_apply(proposal)` wrapper that:
1. Snapshots the relevant Lyapunov before.
2. Applies the proposal.
3. Runs a short post-check (e.g. 3 forward cycles in `dry_run=True`).
4. Accepts iff post Lyapunov $\leq$ pre Lyapunov + tolerance; otherwise rolls back.

This is not a full Gödel machine (no formal proof), but it is an **empirical non-worsening gate** on $\mathcal{L}_{2a}$ actions. $\mathcal{L}_{2b}$ actions remain operator-gated.

### 4.4 Does the kernel ever apply Level-III operators to itself?
No. And per [191]/[192] it should not — Level-III on the kernel is reserved for human operators running `lab-self-improvement` sessions, where the decision "should the modulus change? should a new ambient algebra be introduced?" lives.

**But**: the kernel CAN emit $\mathcal{L}_3$ **proposals** without applying them. See §5.

### 4.5 What would a Pisano-style fixed point look like for the kernel?
Answered in §5.

---

## 5. Pisano-analog fixed point for the kernel

From [192]: the Pisano period operator $\pi$ is the canonical Level-III operator on moduli. It is classical (Wall 1960), it takes a modulus to the length of its Fibonacci orbit, and $m=24$ is its minimum non-trivial fixed point. $\pi(9)=24$ is the canonical pre-fixed-point, linking QA theoretical $(m=9)$ to QA applied $(m=24)$ in one Level-III step.

The question for this session: **is there an analog operator $\Pi_{\text{kernel}}$ whose fixed point characterizes "the kernel has improved itself as far as it can within its current architecture"?**

Three candidates, documented here, none committed:

### Candidate A — Trace-compression fixed point

Define $\Pi_A$: run the kernel for $N$ cycles under a fixed task distribution, then compress the resulting `orbit_transition_log.jsonl` + `task_state_trace.jsonl` with the kernel's own current codec (agent choice policy + routing rules). The compressed length is a scalar $C(\text{kernel})$.

A fixed point is a kernel whose **next improvement cycle does not reduce $C$**. In Kolmogorov-ish terms: the trace of a stable kernel is maximally compressed by that kernel's own model of itself.

- Pros: self-referential in the right way; connects to the research thread already in Open Brain.
- Cons: no closed form; $N$ has to be chosen; compression ratio depends on the codec choice.
- **Test condition (for next session)**: instrument the current ledgers with a trivial codec (orbit-family 3-letter run-length encoding) and check whether the v1 SelfImprovementAgent monotonically decreases $C$.

### Candidate B — Orbit-barycenter routing equilibrium

For each agent-type bucket of tasks, compute the barycenter of the $(b,e)$ states that that agent was asked to handle. Define the kernel routing policy's distance from equilibrium as
$$D_B = \sum_{\text{agent}} \| \text{mean}(s_{\text{task}}) - s_{\text{agent}} \|_{S_m}$$
where the norm is shortest `qa_step` distance within $S_m$.

$\Pi_B$: one "reassign tasks to agents" pass. A fixed point is a routing policy where no reassignment reduces $D_B$.

- Pros: Nash-equilibrium flavor; trivially integer; uses existing `orbits.shortest_witness`.
- Cons: `task_state_trace.jsonl` is the only source, and 2463 rows may be too sparse to estimate a per-agent barycenter robustly. Ignores task type semantics.

### Candidate C — Pisano reduction of the trace itself

This is the most speculative and the most interesting. The kernel's trace in `orbit_transition_log.jsonl` is a sequence of $(b,e)$ states over time. Compute the longest contiguous segment where the state sequence matches the Fibonacci orbit of the current modulus (i.e. periodic with period $\pi(m) = 24$ for $m=9$).

$\Pi_C$: a reduction that projects the trace onto its Pisano-periodic skeleton. A fixed point is a trace that is already a single Pisano period — i.e. the kernel's history has collapsed to "one full Fibonacci cycle" per self-improvement epoch.

- Pros: directly invokes [192]; gives a geometric meaning to "the kernel improved itself maximally" as "the kernel's trace became as periodic as the underlying modulus allows".
- Cons: requires the kernel's actual trace to be driven by Fibonacci-like dynamics, which is true for `_advance_orbit` on the kernel's own state **but not** for the task-semantic states in `task_state_trace.jsonl`. So the candidate restricted to `orbit_transition_log.jsonl` is immediately testable; extended to the full trace, it is only aspirational.

### Recommendation

Test candidate A (trace compression) and a restricted form of candidate C (kernel-only Pisano periodicity) in the `lab-self-improvement` session. Candidate B is simpler but less cert-connected — keep as fallback. **Do not commit to one as "the" kernel fixed point until empirical data says which one monotonically decreases under v1's rule engine.** If the v1 agent does not decrease any of them, that is itself the result: v2 needs a new rule.

---

## 6. Invariants summary — what the kernel already guarantees

For the record, against the [191] filtration:

1. **$\mathcal{L}_0$ containment**: any agent stuck at `(0,0)` is a fixed point and no orbit-advancing handler exists that can move it (StemAgent in undifferentiated mode is the only one, by design).
2. **$\mathcal{L}_1$ closure**: every existing agent's `safe_handle` preserves the orbit of its own state (one `qa_step` is intra-orbit). The kernel's `_advance_orbit` is intra-orbit on kernel state. Level-I closure is **structural**, not just empirical.
3. **$\mathcal{L}_{2a}$ closure of the agent population**: `_replicate` and `approve_spawn` can only extend `_agents`; they cannot change the modulus of any existing agent. Population grows within a fixed ambient algebra.
4. **$\mathcal{L}_{2b}$ closure of metamorphosis**: `_trigger_metamorphosis` changes kernel and agent orbit families but never touches `self.modulus`. Verified by inspection of loop.py:806–846.
5. **No $\mathcal{L}_3$ autonomous operators**. `self.modulus` is written exactly once, in `__init__`. This is the safety guarantee that [191]/[192] recommend — Level-III is human-approved.

These invariants are currently guaranteed by code convention, not by an assertion or a validator. **Adding an explicit runtime check would be a cheap Level-II cert upgrade** (e.g. `assert self.modulus == self._initial_modulus` at start of every cycle). Noted as a low-cost improvement for v2.

---

## 7. Load-bearing claims in one line each

- Every existing agent is $\mathcal{L}_1$ on its own state.
- `_replicate` / `approve_spawn` / `create_agent_stub` are $\mathcal{L}_{2a}$ on the agent population, operator-gated.
- `_trigger_metamorphosis` is $\mathcal{L}_{2b}$ on the kernel+agent orbit family, auto-triggered on satellite streak ≥ 5, and is the highest-tier autonomous operator in the current loop.
- `self.modulus` is never written autonomously: no autonomous $\mathcal{L}_3$.
- `SelfImprovementAgent` v1 already implements a governed/deferred split that aligns with the [191] safety stratification by accident; v2 should make that split explicit.
- `ρ(O)` from the Unified Curvature paper is a ready-made per-cycle Lyapunov; it is computed but not aggregated.
- A Pisano-analog fixed point for the kernel is an open empirical question with three testable candidates.
