# Session Kickoff — `lab-consolidation`

**Created**: 2026-04-05 (end of prior session `claude-main-0300`)
**Purpose**: Bridge existing qa_lab kernel work to the new formal foundation from certs [191] + [192], producing a permanent reference document and a properly-specced SelfImprovementAgent design.
**Next session after this**: `lab-self-improvement` (implementation of the design produced here).

---

## Why this session exists

Will wants to focus hard on QA Lab self-improvement. The last time qa_lab work happened (2026-03-25 through 2026-03-29), the team didn't yet have formal language for what "self-improvement" means in QA terms. Between then and now (2026-04-04), two certs shipped that change everything:

- **[191] QA_BATESON_LEARNING_LEVELS_CERT.v1** — strict invariant filtration on QA state spaces. Four operator classes $\mathcal{L}_1 \subsetneq \mathcal{L}_{2a} \subsetneq \mathcal{L}_{2b} \subsetneq \mathcal{L}_3$ defined by which invariant (orbit / family / modulus / ambient category) is preserved. Tiered Reachability Theorem: only 26% of $S_9$ pairs are Level-I reachable.

- **[192] QA_DUAL_EXTREMALITY_24_CERT.v1** — Pisano period operator $\pi$ is the canonical Level-III operator on moduli. $m=24$ is the minimum non-trivial fixed point (OEIS A235702), $\pi(9)=24$, joint extremality with Carmichael $\lambda$.

Together, these give the project its first formal answer to "what is self-improvement, and when is it stable?" The consolidation session's job is to MAP the existing kernel work onto this framework without inventing anything new.

---

## Do not skip: session startup protocol

Per `CLAUDE.md` (Multi-Session Parallelism Protocol):

1. Pick session name: **`lab-consolidation`**
2. Run `mcp__open-brain__recent_thoughts(since_days=3)` before starting work
3. Read existing file locks: `collab_get_state(key="file_locks")` — check if anyone else is touching qa_lab
4. Register self: `collab_set_state(key="session:lab-consolidation", value={"scope": "qa_lab/ read-only + docs/qa_lab/ write", "ts": "<ISO8601>"})`
5. Check disk: `df -h /` — if < 2GB free, flag to Will

---

## Required reading (in order)

Do NOT skip these. They are the foundation of this session's work.

### Tier 1 — must read before doing anything
1. **`docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md`** — full formal sketch of the learning level hierarchy. This is the theoretical foundation.
2. **`docs/families/191_qa_bateson_learning_levels_cert.md`** — the cert summary with witnesses and theorem statements.
3. **`docs/families/192_qa_dual_extremality_24_cert.md`** — Pisano fixed point cert, the Level-III answer.
4. **`memory/project_qalab_self_improvement.md`** — Will's stated priority (note: memory is 9+ days old, verify against current state).
5. **`memory/project_levin_morphogenetic_architecture.md`** — agents-as-cells framing (related).

### Tier 2 — inventory the actual qa_lab state
**IMPORTANT**: The memory is stale. Read the ACTUAL files, not what memory says.

- `ls qa_lab/` and `ls qa_lab/*/` to see current structure
- `qa_lab/qa_core/algebra.py` — canonical math substrate
- `qa_lab/qa_core/orbits.py` — orbit classification, reachability, structural obstruction
- `qa_lab/kernel/loop.py` — the Von Neumann kernel (SENSE → REASON → ACT → VERIFY → LEARN → REPLICATE)
- `qa_lab/agents/base.py` — QAAgent abstract base class
- `qa_lab/agents/cert_agent.py` — CertAgent (wraps meta-validator)
- Any `qa_lab/kernel/results_log.jsonl` or `qa_lab/kernel/spawn_queue/` if present
- Any `orbit_transition_log.jsonl` or `task_state_trace.jsonl` (ledgers from instrumentation)
- Any other agent files that may have been added since 2026-03-29

### Tier 3 — research context (only if relevant questions arise)
- Open Brain search: `QA lab self-improvement kernel` — returns the 2026-03-25 through 2026-03-29 thread
- Open Brain search: `compression kernel trace` — the real-trace compression experiments

---

## The work of this session (6 steps)

### Step 1: Inventory (30 min)
Read every file in qa_lab. Build a mental (and written) map of what exists:
- What does the kernel's SENSE step actually do?
- What does REASON do? How does it decide?
- What does ACT do? How is it orbit-tracked?
- What does VERIFY check?
- What does LEARN update?
- What does REPLICATE produce?
- What agents exist? What do they handle? What's their orbit state?
- What instrumentation exists? What ledgers are being written?

Write findings to `docs/qa_lab/LAB_INVENTORY_2026-04-05.md` (new file).

### Step 2: Formal mapping (45 min)
For each kernel component and each agent, map to Learning Level operators:

| Component | Action | Level | Invariant preserved | Invariant broken |
|---|---|---|---|---|
| Kernel qa_step cycle | advance (b,e) by T | $\mathcal{L}_1$ | orbit | — |
| Agent handling a task | agent state → agent state | $\mathcal{L}_1$ | orbit of agent | — |
| Spawn new agent | kernel has new agent set | $\mathcal{L}_{2?}$ | ? | ? |
| Kernel detects satellite drift → correction | orbit family change trigger | $\mathcal{L}_{2?}$ | ? | ? |
| ... | ... | ... | ... | ... |

The point is to find the existing Level-III operations (if any) — or to discover there are none yet, which is itself the gap.

Write to `docs/qa_lab/LEARNING_LEVELS_BRIDGE.md` (new file, becomes permanent reference).

### Step 3: Gap analysis (30 min)
Questions to answer in the bridge doc:

1. **Is there a Lyapunov function on kernel state?** (decreasing quantity under normal operation)
   - Candidates: orbit-cycle-count, HI (harmonic index), task-completion-rate, agent-orbit-health
2. **Is there a metric on "kernel quality"?** (so improvements can be measured)
3. **Are improvements currently gated by any test?** (Gödel-machine style proof gating)
4. **Does the kernel ever apply Level-III operators to itself?** (modulus change, agent-set structural change)
5. **What would a Pisano-style fixed point look like for the kernel?** (does $\pi(\text{kernel}) = \text{kernel}$ have an analog?)

### Step 4: Design the SelfImprovementAgent (45 min)
Now with the formal framework AND the existing kernel context, spec the agent properly:

- **Type**: `TaskType.INTROSPECT` (stated in memory; verify this is still the intended type)
- **Input**: kernel trace ledgers (`orbit_transition_log.jsonl`, `task_state_trace.jsonl`, `results_log.jsonl`)
- **Measurement**: Lyapunov function from Step 3, computed over a window of cycles
- **Operator class**: primarily $\mathcal{L}_{2a}$ (reframe agent routing within existing realm), escalating to $\mathcal{L}_{2b}$ (family change — spawn new agent types) only when Lyapunov stalls
- **Stability gate**: Level-III operators (kernel structural changes) are NOT applied autonomously. Design them to emit proposals that Will reviews.
- **Output**: certified action (accept/reject, with reason), logged to a new ledger
- **Failure modes (from Thread 2 research)**:
  - DEQ-style saddle fixed points (looks stable, actually meta-unstable)
  - Löbian procrastination (never accepts any improvement)
  - Chaotic runaway (accepts every proposal)
  - Lyapunov stall at local minimum

Write to `docs/qa_lab/SELF_IMPROVEMENT_AGENT_DESIGN.md` (new file, becomes the spec for the next session's implementation).

### Step 5: Define the Pisano-analog fixed point for the kernel (20 min)
This is the interesting theoretical question. For the kernel, what is the operator whose fixed point = "the kernel has improved itself as much as it can within its current architecture"? Candidates:

- **Trace-compression fixed point**: the kernel is stable when its own trace is maximally compressed by its own codec (Kolmogorov-style)
- **Agent-orbit barycenter**: the kernel is stable when all agents sit at the orbit barycenter of their task distribution (Reptile analog)
- **Routing equilibrium**: task → agent routing is stable iff no rerouting improves the Lyapunov function (Nash-equilibrium analog)
- **Pisano reduction**: does the kernel trace have a Pisano-like period structure that stabilizes?

Document 2-3 candidates in the bridge doc. Don't commit to one yet — the next session will test them.

### Step 6: Session handoff (15 min)
Write the kickoff doc for `lab-self-improvement`:
- Summary of what the consolidation produced
- The chosen Lyapunov function (or "to be tested")
- The SelfImprovementAgent spec to implement
- Required reading list (the bridge doc + the design doc)

---

## Deliverables from this session

One git commit containing:

1. `docs/qa_lab/LAB_INVENTORY_2026-04-05.md` — what actually exists in qa_lab today
2. `docs/qa_lab/LEARNING_LEVELS_BRIDGE.md` — permanent reference mapping existing kernel to formal operator hierarchy (this is the load-bearing document)
3. `docs/qa_lab/SELF_IMPROVEMENT_AGENT_DESIGN.md` — spec for the agent, ready to implement
4. `docs/qa_lab/SESSION_KICKOFF_lab_self_improvement.md` — handoff to the next session
5. Open Brain captures for key findings

**No code changes** in this session. Design only. The next session (`lab-self-improvement`) does the implementation.

---

## Rules for this session

1. **Read actual files, not memory.** Memory is 9+ days old. The `SessionStart` warning on `project_qalab_self_improvement.md` specifically flags this.
2. **Do not modify `qa_lab/` files.** This session is read-only on qa_lab. All writes go to `docs/qa_lab/`.
3. **Do not invent new kernel concepts.** If the kernel doesn't have a Lyapunov function, say so — don't propose one in the inventory. The design doc (Step 4) is where new proposals live.
4. **Do not skip Step 1 (inventory).** Any mapping work (Step 2) that isn't grounded in actual file contents is worthless. Memory claims must be verified.
5. **Do not build the agent.** This is a design session. The implementation happens next session with a fresh window and a clean handoff.
6. **Ground every claim in [191]/[192].** When you assert "this is Level-II-a", point at the specific invariant in [191]. When you invoke Pisano, point at the specific theorem in [192].
7. **Honest gaps are more valuable than pretty diagrams.** If something doesn't map cleanly, say "this doesn't map, and here's why" — that's a research result.

---

## What success looks like

At the end of the consolidation session:

- Anyone reading `LEARNING_LEVELS_BRIDGE.md` can see exactly how the existing kernel fits into the formal framework.
- Anyone reading `SELF_IMPROVEMENT_AGENT_DESIGN.md` can implement the agent without needing to re-derive the theory.
- The next session starts from a clean foundation with zero context re-derivation.
- The question "what does it mean for the QA Lab to improve itself?" has a specific, formal, verifiable answer.

---

## Context from the previous session (`claude-main-0300`)

Shipped [191] and [192] in a single session. The work traced from Will's observation that NLP's "modeling excellence" is QA-native → Bateson Learning Levels as the deepest thread → formal sketch → computational verification → two certs. Full context in Open Brain (capture IDs: d6cec94b, 7bc9cd8b, 3b7d7be4, 96c08f93, 1b575f71, eaf22865).

The key conceptual shift: **self-improvement is not a vague aspiration, it's a specific type of operator on a specific state space, and its stability has a classical number-theoretic answer (Pisano fixed point at m=24).** The consolidation session's job is to make this concrete for the actual qa_lab kernel.

Good hunting.
