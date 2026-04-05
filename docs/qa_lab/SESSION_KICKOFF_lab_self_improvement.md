# Session Kickoff — `lab-self-improvement`

**Created**: 2026-04-05 (by `lab-consolidation`)
**Purpose**: implement `SelfImprovementAgentV2` against the `LEARNING_LEVELS_BRIDGE.md` + `SELF_IMPROVEMENT_AGENT_DESIGN.md` produced by the consolidation session. Produce a baseline fixed-point readout Will can look at.
**Predecessor**: `lab-consolidation` — design-only session, no code changes in `qa_lab/`.

---

## What the consolidation session produced

Three load-bearing docs in `docs/qa_lab/`:

1. **`LAB_INVENTORY_2026-04-05.md`** — ground-truth audit of `qa_lab/qa_core/`, `qa_lab/kernel/`, `qa_lab/agents/`. Memory was stale by 9+ days; the inventory is what the files actually say.
2. **`LEARNING_LEVELS_BRIDGE.md`** — permanent reference mapping every kernel phase and every agent to the [191] filtration $\mathcal{L}_0 \subsetneq \mathcal{L}_1 \subsetneq \mathcal{L}_{2a} \subsetneq \mathcal{L}_{2b} \subsetneq \mathcal{L}_3$. Includes gap analysis and three Pisano-analog fixed-point candidates.
3. **`SELF_IMPROVEMENT_AGENT_DESIGN.md`** — the implementation contract for v2. Read this as the spec; implement to it.

### Key correction carried over from consolidation

`qa_lab/agents/self_improvement_agent.py` **already exists** as v1 (377 LoC, deterministic 7-rule engine). `TaskType.IMPROVE` is a first-class task type. The kernel auto-enriches IMPROVE tasks with introspect + orbit_context + spawn_queue + ob_context + a live `kernel_ref`. v1's governed/deferred action split already matches the [191] safety stratification **by accident**. The v2 refactor makes that split explicit and adds a Lyapunov gate.

**v2 subclasses v1**. Do not rewrite v1. Do not touch `loop.py` or `base.py`.

---

## Session startup protocol

1. Session name: **`lab-self-improvement`**
2. `mcp__open-brain__recent_thoughts(since_days=3)` before starting. The consolidation session produced OB captures you will want to read first.
3. `collab_get_state(key="file_locks")` — check for conflicts.
4. Register: `collab_set_state(key="session:lab-self-improvement", value={"scope": "qa_lab/agents/ + docs/qa_lab/ + qa_alphageometry_ptolemy/qa_self_improvement_cert_v1/ write", "ts": "<ISO8601>"})`
5. Claim locks:
   - `qa_lab/agents/self_improvement_agent_v2.py` (new file — technically doesn't need a lock, but claim for visibility)
   - Do NOT claim `self_improvement_agent.py` (v1) — you don't edit it.
6. `df -h /` — consolidation session observed 28GB free; fine.

---

## Required reading (in order, no shortcuts)

### Tier 1 — must read before writing any code
1. `docs/qa_lab/SELF_IMPROVEMENT_AGENT_DESIGN.md` — this is your contract.
2. `docs/qa_lab/LEARNING_LEVELS_BRIDGE.md` — justifies every design decision.
3. `docs/qa_lab/LAB_INVENTORY_2026-04-05.md` — ground truth on what's in qa_lab.
4. `qa_lab/agents/self_improvement_agent.py` — v1, which you are subclassing.
5. `qa_lab/kernel/loop.py` lines 570–690 (REPLICATE, run_cycle, orbit advance) and 700–790 (status, introspect, agent_performance).
6. `qa_lab/agents/base.py` — understand `QAAgent.certify` (especially the verdict invariants and the `fail_ledger` requirement for `CONTRADICTS`).

### Tier 2 — must read before implementing the fixed-point readout
7. `docs/families/192_qa_dual_extremality_24_cert.md` — Pisano period operator. You do not need the proofs, but you need the fact that $\pi(9)=24$ and the $(\mathbb{Z}/9\mathbb{Z})^\times$ doubling cycle.
8. `qa_lab/qa_core/orbits.py` lines 440–497 — `kappa`, `orbit_contraction_factor`, `orbit_family_score`.

### Tier 3 — reference only
9. `docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md` §3–§5 for the operator definitions and the Tiered Reachability Theorem.
10. `papers/in-progress/...` Unified Curvature §8.1 for the $L_{t+L} = \rho(O) L_t$ identity — if you want to sanity-check the `rho_ewma` Lyapunov choice.

---

## Deliverables

All must exist at session end:

1. `qa_lab/agents/self_improvement_agent_v2.py` — the class per `SELF_IMPROVEMENT_AGENT_DESIGN.md` §2–§10.
2. `qa_lab/agents/tests/test_self_improvement_v2.py` (or wherever the existing test dir is) — synthetic 5-cycle harness with all four assertions from design doc §12.
3. `qa_alphageometry_ptolemy/qa_self_improvement_cert_v1/` — cert family **stub** per §9. Validator can be minimal; passing fixture is required, failing fixture is required.
4. One real IMPROVE cycle executed against a live `QALabKernel` registered with v2, producing a baseline fixed-point readout logged to a new file `qa_lab/kernel/self_improvement_v2_baseline.json`.
5. Open Brain capture summarizing the v1→v2 delta in one observation.
6. No changes to `qa_lab/kernel/loop.py`, `qa_lab/agents/base.py`, `qa_lab/agents/self_improvement_agent.py`.
7. No new certs registered in the meta-validator during this session — the cert stub is not promoted until Will reviews it. (Meta-validator untouched.)

---

## The chosen Lyapunov (or: "to be tested")

**Default**: `rho_ewma` (EWMA of per-cycle `kernel_rho` over last 64 cycles, α=0.2). This is the only candidate with a published theorem behind it (Unified Curvature §8).

**Alternatives to try** in the first cycle's baseline readout, but not to commit to:
- `sat_fraction` — fraction of last 64 cycles in satellite family.
- `weighted_sr` — `1 - weighted_success_rate` across all agent types.

The baseline readout produced at the end of the session should log all three. Will decides which to use going forward.

---

## The three Pisano-analog fixed-point candidates (report, do not enforce)

From `LEARNING_LEVELS_BRIDGE.md` §5:
- **A** — Trace compression ratio over `orbit_transition_log.jsonl`.
- **B** — Routing barycenter distance over `task_state_trace.jsonl`.
- **C** — Pisano periodicity of kernel-state sequence (restricted form; should be ~1.0 trivially — sanity check).

Implement all three per design doc §8. The session ends successfully if all three are populated in the baseline readout with numeric values, even if they are all zero or all meaningless. The **next** session interprets them.

---

## Guardrails for this session

1. **No kernel edits.** v2 is a pure agent. If you find a kernel bug, file it in Open Brain and in a new `docs/qa_lab/KERNEL_TODOS.md`, do not patch.
2. **No v1 edits.** If v1 needs a fix, you're not doing v2 — stop and ask.
3. **No autonomous $\mathcal{L}_3$ operators.** None. Proposals only.
4. **No meta-validator registration.** The cert family stub is scaffolding, not a live cert.
5. **Snapshot-and-restore is non-negotiable** for `_apply_with_probe`. The single biggest risk in this session is corrupting the real kernel trajectory with probe cycles. Design doc §10 item 6 is the correctness-critical property.
6. **Commit discipline**: the consolidation session did not commit. Coordinate with Will before committing (per CLAUDE.md multi-session protocol). One clean commit at session end is preferred.

---

## Infrastructure notes from consolidation

- Bus session tag: consolidation registered as `claude-main-2159` → `session:lab-consolidation`. Clean up the state key when you start: `collab_set_state(key="session:lab-consolidation", value={"closed": "<ts>", "successor": "lab-self-improvement"})`.
- There was an MCP bus outage mid-session (stdio subprocess staleness). If the bus fails again in your session, note it, proceed with read-only checks via `git` and the filesystem, and carry on. The fix for the outage landed in `.mcp.json` and `qa_lab/qa_mcp_servers/qa-collab/server.py` before the consolidation docs were finalized.
- 95 files uncommitted at session start (per the `SessionStart` hook). Consolidation only added 4 new files under `docs/qa_lab/`; everything else is pre-existing state.
- Stale `file_locks` from prior sessions (`claude-main-0742`, `claude-main-0300`) were observed and left alone — they are >2 days old and represent dead sessions, but none conflict with the next session's write scope either.

---

## Success looks like

- v2 runs one real IMPROVE cycle against a kernel that has real ledgers.
- The cycle emits a certified `QA_EMPIRICAL_OBSERVATION_CERT.v1` with every proposal level-tagged and a numeric `lyap_pre`/`lyap_post`.
- No $\mathcal{L}_{2b}$ or $\mathcal{L}_3$ proposal ended up in `applied`.
- The baseline fixed-point readout exists as a JSON file Will can open.
- An OB entry exists summarizing the delta.
- Zero edits to the kernel or to v1.

If all of that is true, you're done. Hand off to whatever comes next.

Good hunting.
