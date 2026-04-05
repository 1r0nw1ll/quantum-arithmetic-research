# SelfImprovementAgent v2 — Design Spec

**Session**: `lab-consolidation` (2026-04-05)
**Audience**: whoever runs `lab-self-improvement` next. This doc is the contract. Implement against it.
**Prerequisites**: `LAB_INVENTORY_2026-04-05.md`, `LEARNING_LEVELS_BRIDGE.md`, [191], [192].

> **v1 already exists** at `qa_lab/agents/self_improvement_agent.py` (377 LoC). This spec is a **refactor** against the [191]/[192] framework, not a greenfield build. v1 is treated as prior art and baseline.

---

## 1. What changes from v1

v1 is correct in its **actions** but not in its **vocabulary**. The refactor adds three things and removes none:

1. **Level tagging**. Every proposal carries `level ∈ {L_1, L_2a, L_2b, L_3}` and `invariant_broken ∈ {orbit, family, modulus, category, none}`, cited against [191]. The gating predicate becomes a function of level, not of ad-hoc action names.
2. **A Lyapunov layer**. A single numeric quantity the agent is minimizing across IMPROVE cycles, with a monotone-non-worsening gate on every $\mathcal{L}_{2a}$ auto-applied action.
3. **A Pisano-analog fixed-point readout**. Reported on every cycle for the three candidates in the bridge doc. **Reported, not enforced** — the next session's empirical job is to pick one.

v1's rule engine stays verbatim as the seed ruleset; the refactor wraps it, does not rewrite it.

---

## 2. Class signature

```python
# qa_lab/agents/self_improvement_agent_v2.py (NEW FILE — leave v1 in place)
from agents.self_improvement_agent import SelfImprovementAgent as _V1

class SelfImprovementAgentV2(_V1):
    capabilities = ["improve"]
    cert_family  = "qa_self_improvement_cert"   # [TBD] — new cert family

    def __init__(self, modulus: int = 9, lyapunov: str = "rho_ewma",
                 lyapunov_alpha: float = 0.2, dry_run_probe_cycles: int = 3):
        super().__init__(modulus=modulus)
        self.lyapunov_name = lyapunov           # "rho_ewma" | "sat_fraction" | "weighted_sr"
        self.alpha = lyapunov_alpha
        self.dry_run_probe_cycles = dry_run_probe_cycles
        self._lyap_history: list[tuple[int, float]] = []   # (cycle_idx, value)
        self._fp_history:   list[dict] = []                # Pisano-candidate readouts
```

Subclassing v1 preserves the existing rule engine (`_generate_proposals`) and the OB keyword hook without duplication. v2 overrides only `handle`, `_apply_governed`, and adds three new helpers.

---

## 3. Inputs contract (unchanged from v1)

The kernel's `_act` auto-enriches `IMPROVE` tasks with:
- `introspect`, `recent_results`, `orbit_context`, `spawn_queue`, `ob_context`, `kernel_ref`.

v2 adds one optional input:
- `task.inputs["lyap_override"]` (`str`) — lets a caller pin the Lyapunov for this cycle.

---

## 4. Cycle structure

```
handle(task):
  1. proposals  = self._generate_proposals(...)   # v1, unchanged
  2. tagged     = self._tag_levels(proposals)     # NEW — attach {level, invariant_broken, cert_ref}
  3. lyap_pre   = self._compute_lyapunov(kernel_ref)
  4. fp_readout = self._compute_fixed_point_candidates(kernel_ref)
  5. applied, deferred = self._apply_level_gated(tagged, kernel_ref, lyap_pre)
  6. lyap_post  = self._compute_lyapunov(kernel_ref)
  7. cert       = self.certify(
         observation_body=summary,
         verdict=self._verdict(tagged, applied, lyap_pre, lyap_post),
         evidence=[..., {"lyap_pre": lyap_pre, "lyap_post": lyap_post,
                         "delta": lyap_post - lyap_pre,
                         "fixed_point_candidates": fp_readout,
                         "applied": applied, "deferred": deferred,
                         "proposals": [strip(p) for p in tagged]}],
         fail_ledger=(fail_ledger if verdict=="CONTRADICTS" else None),
       )
  8. self._lyap_history.append((cycle_idx, lyap_post))
     self._fp_history.append(fp_readout)
  9. return {"ok": True, "verdict": ..., "proposals": tagged, "applied": ...,
             "deferred": ..., "lyapunov": {"pre": lyap_pre, "post": lyap_post},
             "fixed_point_candidates": fp_readout, "cert": cert}
```

### Verdict rule

| Condition | Verdict |
|---|---|
| Any $\mathcal{L}_{2a}$ auto-applied action causes `lyap_post > lyap_pre + tol` (rollback fires) | `CONTRADICTS` + `fail_ledger` |
| Structural issue detected (R1 or R4 fires) | `PARTIAL` |
| At least one proposal generated, all applied or deferred cleanly | `CONSISTENT` |
| Only R7 (HEALTHY) fires | `INCONCLUSIVE` |

`tol = 1e-6 * |lyap_pre|` is the default; override via `task.inputs["lyap_tol"]`.

---

## 5. Level tagging (Step 2 of cycle)

`_tag_levels(proposals)` attaches to each proposal:

```python
{
  "level": "L_1" | "L_2a" | "L_2b" | "L_3",
  "invariant_broken": "orbit" | "family" | "modulus" | "category" | "none",
  "cert_ref": "QA_BATESON_LEARNING_LEVELS_CERT.v1",
  "tier_reachability_claim": "<rule cites [191] §5.3 by tier>",
}
```

Mapping table (from `LEARNING_LEVELS_BRIDGE.md` §3):

| Rule | action | level | invariant_broken |
|---|---|---|---|
| R1 SINGULARITY_STUCK | ADJUST_THRESHOLD | — (parameter, not state) | `none` |
| R2 SATELLITE_LOOP | ADJUST_THRESHOLD | — | `none` |
| R3 WEAK_AGENT | SPAWN_IMPROVED | `L_2a` | `orbit` (agent population) |
| R4 ORBIT_DRIFT | DEDIFFERENTIATE | `L_2b` | `family` |
| R5 UNHANDLED_CAPABILITY | PROMOTE_STEM | `L_2a` | `orbit` |
| R6 OB_INSIGHT | LOG_ONLY / SPAWN_IMPROVED | depends | — |
| R7 HEALTHY | LOG_ONLY | — | `none` |

---

## 6. Level-gated application (Step 5)

```python
def _apply_level_gated(self, tagged, kernel_ref, lyap_pre):
    applied, deferred = [], []
    for prop in tagged:
        lvl = prop.get("level")
        action = prop.get("action")

        # Non-state parameter tweaks and logs: apply directly.
        if action in ("LOG_ONLY",) or (action == "ADJUST_THRESHOLD" and lvl is None):
            applied.append(self._apply_single(prop, kernel_ref))
            continue

        # L_1 and "no-level" state ops: apply directly.
        if lvl in (None, "L_1"):
            applied.append(self._apply_single(prop, kernel_ref))
            continue

        # L_2a: apply with empirical non-worsening gate.
        if lvl == "L_2a":
            result = self._apply_with_probe(prop, kernel_ref, lyap_pre)
            (applied if result["committed"] else deferred).append(result)
            continue

        # L_2b and above: NEVER auto-apply. Always defer.
        deferred.append({**prop, "applied": False,
                         "note": f"level={lvl} requires operator approval per [191]"})
    return applied, deferred
```

`_apply_with_probe`:

1. Snapshot kernel state: `(_b, _e, _agents, _agent_perf, _spawn_queue, metamorphosis_threshold, _satellite_streak)`.
2. Apply the proposal to the live `kernel_ref`.
3. Run `dry_run_probe_cycles` forward cycles in `kernel_ref.dry_run=True` mode against a synthetic probe task list (supplied as class attribute `_probe_tasks`).
4. Compute `lyap_post_probe`.
5. If `lyap_post_probe <= lyap_pre + tol`: keep the change, mark `committed=True`.
6. Else: restore from snapshot, mark `committed=False`, reason=`"probe_worsened_lyapunov"`.

**Probe tasks** are 3 stock `TaskType.QUERY` tasks covering `(1,1)`, `(3,3)`, `(1,3)` — one from each orbit family. This costs ~milliseconds and gives a minimal signal on whether the change broke basic routing.

---

## 7. Lyapunov layer (Step 3, 6)

Primary candidate: **$\rho$-EWMA**.

```python
def _compute_lyapunov(self, kernel_ref) -> float:
    name = self.lyapunov_name
    if name == "rho_ewma":
        # EWMA of kernel_rho across the tail of results_log
        vals = [r.get("kernel_rho") for r in kernel_ref._results[-64:]
                if r.get("kernel_rho") is not None]
        if not vals: return 1.0
        ewma = vals[0]
        for v in vals[1:]:
            ewma = self.alpha * v + (1 - self.alpha) * ewma
        return ewma    # lower is better (faster contraction)
    if name == "sat_fraction":
        recent = kernel_ref._results[-64:]
        if not recent: return 0.0
        return sum(1 for r in recent if r.orbit_family == "satellite") / len(recent)
    if name == "weighted_sr":
        perf = kernel_ref.agent_performance()
        if not perf: return 1.0
        # we want to minimize failure, so return 1 - weighted success
        total = sum(p["total"] for p in perf.values()) or 1
        ws = sum(p["success_rate"] * p["total"] for p in perf.values()) / total
        return 1.0 - ws
    raise ValueError(f"unknown lyapunov: {name}")
```

All three are bounded in $[0,1]$, monotone-interpretable (lower=better), and use only integer / rational arithmetic modulo the EWMA step. Defaults to `rho_ewma` because it is the only one backed by a published theorem (`ρ(O) < 1 ⇒ convergence`, Unified Curvature §8).

**Note**: `kernel_rho` must be attached to `KernelResult` for this to work. Inspection confirms it is already attached via `KernelResult(rho=rho, ...)` at loop.py:676. Good.

---

## 8. Fixed-point candidates readout (Step 4)

Cheap, read-only. Computed every IMPROVE cycle, logged, not acted on.

```python
def _compute_fixed_point_candidates(self, kernel_ref) -> dict:
    return {
        "A_trace_compression":  self._fp_A(kernel_ref),
        "B_routing_barycenter": self._fp_B(kernel_ref),
        "C_pisano_periodicity": self._fp_C(kernel_ref),
    }
```

### 8.A Trace compression
Read the tail of `orbit_transition_log.jsonl` (last 256 rows). Encode each row as a single char from `{C, S, G}` (Cosmos, Satellite, sinGularity) using `orbit_family_after`. Compute `zlib.compress(s.encode()).__len__() / len(s)`. Report as a ratio in $[0,1]$.
- Fixed point signal: the ratio **plateaus**.

### 8.B Routing barycenter
For each task type present in `task_state_trace.jsonl` (last 256 rows), compute the mean `(b,e)` of rows where `source == "task.inputs"`. The "routing distance" is $\sum_{\text{type}} \|\text{mean}(s) - s_{\text{agent}}\|$ where the agent state is pulled from the kernel's registered agent for that type, and the norm is min shortest-path length on $S_m$ via `shortest_witness` (BFS-bounded to 24 steps).
- Fixed point signal: distance stops decreasing.

### 8.C Pisano periodicity (restricted to kernel state)
Read the last $N = 48$ rows of `orbit_transition_log.jsonl`. Extract the `state_after` sequence. Compute the longest contiguous slice that equals itself shifted by $\pi(m)$ — i.e., periodic with period $\pi(9)=24$. Report `longest_periodic / len(sequence)` in $[0,1]$.
- Fixed point signal: ratio → 1.0 means the kernel's own state sequence has become a perfect Pisano period. For `_advance_orbit`, which applies one `qa_step` per cycle to `(1,1)` cosmos, **this should already be trivially 1.0** — which is a sanity check: if it isn't, something in the kernel advancement is broken.

The first cycle of v2 should log all three and produce the **baseline readout** that the next session tunes against.

---

## 9. Cert integration

Create a new cert family stub (real implementation is a next-session deliverable, outside the scope of this design doc):

```
qa_alphageometry_ptolemy/qa_self_improvement_cert_v1/
├── cert.json                 # metadata, parents: [122], [191], [192]
├── validator.py              # checks that:
│    V1: every improve cycle emits level-tagged proposals
│    V2: every L_2b or L_3 proposal is in deferred, not applied
│    V3: lyap_post <= lyap_pre + tol for every CONSISTENT-verdict cycle
│    V4: fixed_point_candidates has all three keys A/B/C with numeric values
│    V5: --self-test exits 0 on a synthetic 5-cycle fixture
├── fixtures/
│    ├── pass_trace.jsonl     # 5 improve cycles, lyap monotone non-increasing
│    └── fail_trace.jsonl     # 1 improve cycle where L_2b was auto-applied (must fail)
└── mapping_protocol_ref.json
```

This is enough for a cert number assignment under Gate 0 when the next session lands the implementation.

---

## 10. Failure modes to guard against

From the kickoff doc, plus two discovered during this session:

1. **DEQ-style saddle fixed points** — Lyapunov flat but underlying state unstable.
   *Mitigation*: require $\rho$-EWMA to **decrease** by at least `tol` over any 10-cycle window before the next $\mathcal{L}_{2a}$ auto-apply is allowed. Otherwise escalate to deferred.
2. **Löbian procrastination** — never accept any improvement.
   *Mitigation*: track `accept_rate = applied / (applied+deferred)` and warn if it drops below 0.1 for 5 cycles (operator surface, not auto-reset).
3. **Chaotic runaway** — accept every proposal.
   *Mitigation*: $\mathcal{L}_{2b}$ is hard-gated to deferred; $\mathcal{L}_{2a}$ is probe-gated. R1/R2 (parameter tweaks) are the only unbounded applied actions; rate-limit to one per cycle (v1 already sorts by priority, but does not rate-limit — add a `max_applied_per_cycle = 3` cap).
4. **Lyapunov stall at local minimum** — the kernel is "good enough" and no proposal fires.
   *Mitigation*: when R7 HEALTHY fires for 10 consecutive IMPROVE cycles, inject a stochastic (operator-supplied!) exploration task. This is explicitly an operator action, not agent autonomy.
5. **Stale `kernel_ref`** — the kernel handle is a reference; if the kernel is recreated, writes go to a ghost.
   *Mitigation*: check `kernel_ref._kernel_run_id` matches at start and end of cycle; abort with `INCONCLUSIVE` if not.
6. **Probe contamination** — running probe cycles advances the kernel orbit, corrupting the real trajectory.
   *Mitigation*: the `_apply_with_probe` snapshot must include `(_b, _e, _cycle_count, _satellite_streak, _results)` and restore them regardless of probe outcome. This is the most critical correctness property of the design.

---

## 11. Out of scope (explicit non-goals for the implementation session)

- **No new cert beyond the stub described in §9.** Full validator + fixtures is its own multi-hour task.
- **No $\mathcal{L}_3$ operators**, automatic or otherwise. The agent proposes, Will disposes.
- **No changes to v1's rule engine.** R1–R7 stay verbatim. If a new rule is needed, it goes in a new method on the v2 class.
- **No kernel modifications.** Everything lives in `agents/self_improvement_agent_v2.py` + the cert family stub. The kernel already exposes everything v2 needs.
- **No Open Brain writes from inside the agent.** The kernel's LEARN step handles capture; v2 only returns certified output.

---

## 12. Definition of done for `lab-self-improvement`

The implementation session ends when:

1. `qa_lab/agents/self_improvement_agent_v2.py` exists, imports v1 as base, implements every method in this spec.
2. A synthetic 5-cycle test harness (`tests/test_self_improvement_v2.py`) runs against a mock kernel and asserts:
   - Every proposal has a `level` key.
   - No `L_2b` / `L_3` proposal ends up in `applied`.
   - `lyap_post <= lyap_pre + tol` for every `CONSISTENT` cycle.
   - All three fixed-point candidate fields are populated.
3. The agent is registered against a live `QALabKernel` via `register_agent(TaskType.IMPROVE, SelfImprovementAgentV2())` and one real cycle runs end-to-end against the actual ledgers.
4. The run produces a baseline fixed-point readout that Will can look at.
5. No changes to `loop.py`, `base.py`, or v1.
6. Open Brain capture summarizes the delta from v1 → v2 in one observation.

Any of these failing means the session is not done — not that the spec is wrong.
