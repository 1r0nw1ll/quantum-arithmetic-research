# QA Lab Kernel — TODO / Known Bugs

Filed during `lab-self-improvement` session (2026-04-04/05).  The kickoff
doc forbids kernel edits from that session; bugs are recorded here for a
future session to patch.

---

## K-002 — RESOLVED 2026-04-04 — `run_cycle` lookups use pre-A1 `% self.modulus` reduction

**STATUS**: Fixed. All 6 sites in `qa_lab/kernel/loop.py` that did
`(self._b % self.modulus, self._e % self.modulus)` have been replaced with
`(self._b, self._e)`. The internal kernel state is maintained in the A1
space {1..m} by `qa_step` (2026-04-05 rehab), so the mod-m reduction was a
pre-A1 vestige that mapped state 9 → 0, which is not a valid key in the
A1-compliant `_families` dict.

**Severity**: latent — only fires after the kernel orbit reaches a state
where `b` or `e` equals the modulus. Under the default `(1,1)` cosmos
start, this first happens at cycle 6: state sequence
`(1,1) → (1,2) → (2,3) → (3,5) → (5,8) → (8,4) → ... → (1,8) → (8,9)`.
Before 2026-04-05 rehab the qa_step was pre-A1 and never produced 9, so
the reduction was a no-op. After the rehab, qa_step produces the full
A1 range {1..9}, and the stale reduction becomes a hard bug.

**Reproduction**: any runner that executes ≥11 consecutive cycles from
`(1,1)` cosmos. Caught by `tools/cert_batch_route_through_kernel.py`
after cycle 10.

**Fix**:
```diff
- family = self._families[(self._b % self.modulus, self._e % self.modulus)]
+ family = self._families[(self._b, self._e)]
```
Applied at lines 213, 221, 388, 412, 657, 706.

**Regression protection**: `run_batch_v2_hardening.py` ran 4 cycles
pre-fix without triggering; cert_batch_route_through_kernel.py runs 14
cycles and triggered the bug on cycle 11. Going forward, any kernel
runner that executes >10 cycles serves as a regression check.

---

## K-001 — RESOLVED 2026-04-04 — `introspect()` crashes when any agent has `needs_improvement=True`

**STATUS**: Fixed. `loop.py:756` now reads `p["success_rate"] < 0.5` from the formatted
`agent_performance()` view (equivalent to `fail > ok` since `success_rate = ok/(ok+fail)`).
Verified by re-running `run_batch_v2_hardening.py` phase-3 with a seeded weak-agent perf
record — kernel's introspect() runs cleanly, no workaround needed in the runner.

**File**: `qa_lab/kernel/loop.py`
**Lines**: 755–758 (approx.)
**Severity**: was latent — only fires when the kernel actually has a weak agent,
which is exactly the case the self-improvement track needs to exercise.

### The bug

```python
# loop.py:741  introspect()
perf = self.agent_performance()          # formatted dict
weak_agents = [tt for tt, p in perf.items() if p["needs_improvement"]]
...
for tt in weak_agents:
    p = perf[tt]
    suggestions.append({
        "agent": tt,
        "issue": "high failure rate" if p["fail"] > p["ok"] else "satellite drift",
                                         ^^^^^^^^^     ^^^^^^^
        ...
    })
```

`agent_performance()` (loop.py:727) returns a dict with keys
`{total, success_rate, verdicts, orbit_drift_count, needs_improvement}`.
**It does NOT include `"ok"` or `"fail"` keys.** Those are only in the raw
`_agent_perf` counters.  The `introspect()` method assumes the formatted
output still has the raw keys, so it raises `KeyError: 'fail'` the first
time the loop body runs.

In a healthy lab this is never triggered (empty `weak_agents`), which is
why the bug has gone unnoticed.  It fires immediately when
`lab-self-improvement` seeds a weak-agent perf record to exercise v2's
L_2a probe path.

### Reproduction

```python
k = QALabKernel(...)
k._agent_perf["query"] = {"ok": 1, "fail": 6, "verdicts": {}, "orbit_drift": 0}
k.introspect()  # KeyError: 'fail'
```

### Fix (one-line)

```python
# loop.py:756 — read from the raw counters instead of the formatted dict
raw = self._agent_perf[tt]
"issue": "high failure rate" if raw["fail"] > raw["ok"] else "satellite drift",
```

Or equivalently: expose `fail`/`ok` in `agent_performance()` output so
callers don't need to reach into raw state.

### Workaround (used by `lab-self-improvement` session)

`run_batch_v2_hardening.py` hand-crafts the IMPROVE task inputs for phase
3, pre-populating `task.inputs["introspect"]` so the kernel's auto-enrich
in `_act` short-circuits before calling `self.introspect()`.  See the
phase-3 block of that runner.

### Related

Does not affect v2 correctness — v2 only reads `introspect`'s output if
the kernel provides it.  The crash is fully inside the kernel's ACT
enrichment path before v2.handle sees anything.

---

*Filed by: `lab-self-improvement` 2026-04-04/05.*
