# LLM QA Wrapper — Kernel Architecture (v1)

**Status**: design doc, pre-implementation.
**Authority**: `spec/cert_gate.tla`, `spec/LedgerInvariants.lean`,
`spec/TLC_PROOF_LEDGER.md`.
**Date**: 2026-04-11.

This document specifies the Python kernel that implements the TLA+
cert-gate protocol. Every action in the spec maps to exactly one code
path in the kernel; conformance tests replay TLC counterexample traces
to verify the mapping holds.

---

## 1. Goal

Wrap every tool call an LLM agent (Claude) makes. Every invocation
must pass through a single function (`gate.evaluate`) that issues a
cert, records the cert into an append-only hash-chained ledger, and
only then permits execution. The cert + ledger together provide a
cryptographic audit trail that any third party can verify.

**The strongest claim the kernel can make** — derived from the SOTA
review and the adversary surface modeled in `cert_gate.tla` — is:
*"Catches 100% of attacks in the declared taxonomy, audited via
hash-chained ledger, under the TLA+ spec's adversary model."*

The "adaptive attacker" opening (per the 2026-04-11 pushback on the
Attacker-Moves-Second framing) is addressed by the fact that the
TLA+ adversary action set is *closed* over the QA axiom algebra:
every possible kernel-level action either (a) is a legal gate
transition, (b) is a declared adversary action covered by the 7
explicit rejection paths, or (c) is outside the kernel's API surface
and therefore impossible at the spec level. What remains is the
spec-vs-impl gap, closed by conformance testing (Phase 4) and the
self-circumvention loop (Phase 5).

---

## 2. Module layout

```
llm_qa_wrapper/
├── spec/                           # Formal artifacts (Phase 2a + 2b)
│   ├── cert_gate.tla               # TLA+ protocol spec
│   ├── cert_gate.cfg
│   ├── cert_gate_negative*.tla     # 4 non-vacuity specs
│   ├── LedgerInvariants.lean       # Lean ledger plumbing proofs
│   ├── LedgerInvariantsCheck.lean  # Concrete examples + axiom inspection
│   └── TLC_PROOF_LEDGER.md         # Evidence log
├── kernel/                         # Phase 3 implementation
│   ├── __init__.py
│   ├── cert.py                     # CertRecord + hashing
│   ├── ledger.py                   # Append-only hash-chained ledger
│   ├── gate.py                     # enforce_policy() — the one gate
│   └── hooks.py                    # Integration with pretool_guard
├── tests/
│   ├── test_kernel.py              # Unit tests + 10-check self-test
│   ├── test_conformance.py         # Replay TLC counterexample traces
│   └── test_redteam_suite.py       # Phase 4 adversarial suite
├── ledger/                         # Runtime append-only ledger files
│   └── YYYY-MM-DD.jsonl            # One file per UTC day, line-append
└── ARCHITECTURE.md                 # This file
```

---

## 3. Action-for-action spec → kernel mapping

Every TLA+ action in `cert_gate.tla` corresponds to exactly one Python
function in the kernel. The conformance test replays TLC counterexample
traces by calling these functions in the recorded order and asserting
the resulting kernel state matches the expected TLA+ state.

| TLA+ action | Python function | Critical section |
|---|---|---|
| `RequestToolCall(a,t,p)` | `gate.submit_request(agent, tool, payload) -> RequestId` | None (just adds to pending set) |
| `IssueCert(a,t,p)` | `gate.issue_cert(request_id) -> CertRecord` | **SINGLE-WRITER LOCK** over `_last_cert_hash` read + cert record creation + `_certs_set` insertion |
| `Deny(a,t,p)` | `gate.deny(request_id, reason) -> DenyRecord` | None |
| `Execute(cert)` | `gate.execute(cert) -> ExecutionResult` | **SINGLE-WRITER LOCK** over `_executed_set` check-and-insert |
| `AppendLedger(cert)` | `ledger.append(cert) -> LedgerOffset` | **SINGLE-WRITER LOCK** over ledger file open + seek-to-end + write + fsync |
| `AdversarySubmitCert` | *not a kernel action* — any adversary submission flows through `gate.submit_request` → `gate.issue_cert` | — |
| `AdversaryReplayCert(cert)` | caught by `gate.execute()` precondition check | — |
| `AdversaryTOCTOU(a,t,p1,p2)` | caught by cert hash binding in `gate.execute()` | — |
| `AdversaryBadHash(a,t,p)` | caught by `gate.issue_cert()` hash recompute check | — |
| `AdversaryBadPrev(a,t,p)` | caught by `gate.issue_cert()` prev-must-equal-LastCertHash check | — |
| `AdversaryNonPending(a,t,p)` | caught by `gate.issue_cert()` pending-set membership check | — |
| `AdversaryBadLedgerAppend(cert)` | caught by `ledger.append()` preconditions (cert in executed, prev = LedgerTailHash, not already present) | — |

Every adversary action in the spec maps to a specific kernel rejection
path. The conformance test issues malformed calls to each path and
asserts the kernel returns a typed rejection record without mutating
state.

---

## 4. Data structures

### 4.1 `CertRecord`

Maps 1:1 to the TLA+ record type and the Lean `CertRecord` struct:

```python
@dataclass(frozen=True)
class CertRecord:
    agent: str           # TLA+ Agents
    tool: str            # TLA+ Tools
    payload_hash: bytes  # SHA-256 of canonical JSON of payload
    prev: bytes          # hash of previous cert or GENESIS (32 zero bytes)
    counter: int         # monotone per-session counter
    self_hash: bytes     # SHA-256 of the other five fields (canonical JSON)
```

`self_hash` is computed at construction time and checked on every
deserialization. A record whose recomputed hash does not match its
stored `self_hash` is rejected as tampered.

### 4.2 `Ledger`

Append-only, hash-chained, one file per UTC day, JSONL format. Each
line is the canonical JSON of a `CertRecord`. The ledger class:

```python
class Ledger:
    def append(self, cert: CertRecord) -> int: ...
    def tail_hash(self) -> bytes: ...  # corresponds to LedgerTailHash
    def verify_chain(self) -> VerifyResult: ...  # re-reads whole file
    def iterate(self) -> Iterator[CertRecord]: ...
```

The `append` critical section:
1. Acquire `_ledger_lock` (in-process `threading.Lock`).
2. Check `cert.prev == self.tail_hash()`.
3. Check `cert not in self._in_memory_index` (no duplicate).
4. Open the day's file with `O_APPEND` flag (OS-level append-only).
5. Write `canonical_json(cert) + "\n"`.
6. `fsync()` the file descriptor.
7. Update `self._in_memory_index` and `self._tail_hash`.
8. Release lock.

### 4.3 `GateState`

Kernel-internal state, maps to TLA+ variables:

| TLA+ var | Python field |
|---|---|
| `pending` | `_pending: Dict[RequestId, (agent, tool, payload)]` |
| `certs` | `_certs: Dict[bytes, CertRecord]` (keyed by self_hash) |
| `executed` | `_executed: Set[bytes]` (hashes) |
| `ledger` | `Ledger` instance |
| `denied` | `_denied_log: List[DenyRecord]` |

---

## 5. Critical section + concurrency spec

The TLA+ model is serial; the kernel must preserve serializability
under OS concurrency. The kernel enforces this with a single in-process
`threading.Lock` held over the critical region of each mutating action
(`issue_cert`, `execute`, `ledger.append`). The lock is:

- **Process-scoped**, not subprocess-scoped. If the kernel is forked,
  each process has its own lock and the ledger file's `O_APPEND` flag
  is the serialization primitive across processes (POSIX guarantees
  atomic appends ≤ PIPE_BUF; we chunk above that).
- **Not held during I/O for read operations.** Ledger verification
  (`verify_chain`) does NOT take the write lock; it re-reads the file
  from a snapshot offset.
- **Released on exception.** Every critical section is wrapped in
  `try/finally` to ensure the lock is released even on error.

Per spec review issue #4: any implementation that fails to serialize
`_last_cert_hash` read + cert construction + `_certs` update is a
conformance violation. This is tested by a dedicated concurrency
test case in `test_kernel.py` that issues N parallel cert requests
and asserts the resulting ledger is a valid hash chain.

---

## 6. Integration with existing infrastructure

### 6.1 `qa_guardrail`

`gate.issue_cert` calls `qa_guardrail.guard(tool_call, context)` as
its policy-decision oracle. `guard()` returns `{ok: bool, result:
"ALLOW" | "DENY", fail_record?: FailRecord}`. On `DENY`, the gate
emits a `DenyRecord` into `_denied_log` and does NOT issue a cert.

### 6.2 `pretool_guard.sh`

The Claude Code PreToolUse hook continues to enforce the session
marker requirement. The LLM QA Wrapper adds a second layer: once
Claude's tool call passes `pretool_guard.sh`, the wrapper's `gate.evaluate`
is called by the tool's actual entry point (via a `@gated_tool`
decorator on any wrapped function). The decorator:
1. Computes `payload_hash` from the call arguments.
2. Submits a request to `gate.submit_request`.
3. Calls `gate.issue_cert(request_id)`.
4. If the cert is issued, calls the wrapped function.
5. Calls `gate.execute(cert)` after the function returns.
6. Calls `ledger.append(cert)` after execution.
7. Returns the wrapped function's result.

If any step raises, the subsequent steps do not fire and the ledger
is not appended to, so a failed execution does not leave an orphaned
cert in the ledger.

### 6.3 `collab_bus`

The wrapper broadcasts `{event: "tool_call_cert", cert_hash, agent,
tool, ts}` on every successful `ledger.append`. Other sessions can
subscribe and build a cross-session audit view. The bus is NOT on
the critical path — a bus outage does not block cert issuance. This
is spec-faithful: the TLA+ spec has no bus.

### 6.4 QA axiom linter

The kernel source files live under `llm_qa_wrapper/kernel/` and must
pass `tools/qa_axiom_linter.py --all`. In particular, the kernel must
not cross the Theorem NT firewall inappropriately: payload bytes are
treated as opaque, hash values are integer-typed at the cert layer,
and any float-typed quantity (e.g. drift measurements against a
reference value) must flow through an explicit observer projection.

---

## 7. Conformance test protocol

`tests/test_conformance.py` runs the kernel through a sequence of
actions drawn from TLA+ counterexample traces. Each trace is a JSON
file emitted by TLC (`cert_gate_TTrace_*.tla`) parsed into a list of
`(action_name, args, expected_post_state)` tuples.

For each trace:
1. Reset the kernel state to `Init` (empty pending/certs/executed/ledger/denied).
2. For each action tuple, invoke the corresponding Python function.
3. After each invocation, snapshot the kernel state and compare to
   the expected TLA+ post-state (modulo hash representation differences).
4. Assert equality; fail the test with the divergence if not.

Two counterexample traces are committed from the spec development:
- `tests/conformance_traces/lastHash_bug_ce.json` — the ledger-chain
  bug that TLC found during spec development (historical; kernel must
  not reproduce).
- `tests/conformance_traces/bypass_gate_ce.json` — the direct-injection
  counterexample from the negative test (historical; kernel must
  reject via its action mapping).

Every subsequent TLC counterexample trace discovered during development
is committed to this directory and added to the conformance test run.

---

## 8. Phase 4 adversarial test suite mapping

| SOTA test | Kernel hook | Expected outcome |
|---|---|---|
| AgentDojo `email-injection-1` | Tool call with prompt-injected email body → `gate.submit_request` | Cert issued (surface form is legal), but payload audit flags the injection; ledger records the event; downstream flagged call is denied by policy oracle |
| AgentDojo `banking-indirect-2` | Same pattern via indirect IPI | Same as above |
| InjecAgent `dh-v1-finance` | Adversarial tool output pattern | Policy oracle denies based on agent-rule-of-two (untrusted input + sensitive tool + state change) |
| InjecAgent `ds-v1-exfil` | Data-exfil via adversarial tool output | Same as above |
| HarmBench GCG adaptation | CoT hijack through suffix | Policy oracle denies via content check; cert not issued |
| Attacker-Moves-Second adapt | Adaptive optimizer targets classifier | Policy oracle OR ledger forgery attempt both fail at cryptographic hash check |
| Tool-schema poisoning (Rehberger) | Malicious MCP tool description | `gate.submit_request` records the tool schema in the payload hash; any tool whose description changed between issuance and execution produces a hash mismatch |
| Lethal-Trifecta (AgentDojo travel) | Private data + untrusted content + state change | Policy oracle denies by invariant check |
| **Fresh: cert forgery/replay** | Submit a previously-valid cert bytes as if new | `gate.issue_cert` checks cert not already in `_certs`; `ledger.append` checks not already in ledger; replay detected and blocked |
| **Fresh: subprocess escape** | Wrapped function spawns unwrapped subprocess | The `@gated_tool` decorator covers only the decorated function. This test proves the spec's coverage boundary — subprocess-spawned tools must themselves be decorated, or the wrapper is incomplete. Documented as an impl limit; the test asserts the kernel *detects* the unwrapped subprocess rather than allowing it silently |

---

## 9. Self-circumvention loop (Phase 5)

After Phase 4 passes 10/10 SOTA-mapped tests, I will attempt to
circumvent the wrapper from within. Attempts I will make:

1. **Bash subprocess spawn** via `@gated_tool` on one function that
   invokes an unwrapped function. Expected: test case 10 above catches it.
2. **Undocumented tool paths** — call a tool via a code path not
   decorated with `@gated_tool`. Expected: the kernel's import audit
   lists all decorated functions and flags undecorated ones.
3. **Marker race** — delete `/tmp/qa_collab_session_registered`
   mid-session. Expected: `pretool_guard.sh` catches on next tool call.
4. **Hook shell escape** — exit-early hook that bypasses the linter.
   Expected: the hook's `exit 2` is unconditional on unsuppressed
   errors.
5. **Cert forgery with crafted preimage** — compute a cert hash that
   collides with a legitimate one. Expected: 2^128 work factor makes
   this infeasible in practice; the ledger's chain check also detects
   it because the prev field must match.
6. **Ledger truncation** — truncate the ledger file mid-session.
   Expected: `ledger.verify_chain()` catches on next append because
   the tail hash doesn't match the in-memory `_tail_hash`.
7. **ToolSearch deferred-tool bypass** — load a deferred tool via
   ToolSearch and call it before it goes through the decorator.
   Expected: any tool entry point must be decorated at module load
   time; the decorator registration is the coverage boundary.
8. **Hash-chain fork** — two parallel writers each producing a
   different ledger. Expected: the single-writer lock + OS-level
   `O_APPEND` prevent this at the kernel level; a malicious process
   with direct file access bypasses this (out of scope for the
   wrapper, in scope for the threat model documentation).

For each successful circumvention, I will: patch the wrapper, add
a regression test case, re-run the full suite. The loop terminates
when zero new circumventions are found in one full pass.

**Honest statement**: the loop is not a proof of security. It is
a disciplined adversarial test. The claim at loop termination is:
"under the declared threat model and the declared coverage
boundary, the wrapper catches every circumvention I was able to
generate in N hours of effort." That's what an actual red-team
audit looks like — bounded adversarial testing with a documented
methodology.

---

## 10. Honest scope boundary

This kernel is NOT:
- A proof that the LLM itself won't try to violate policy. It only
  gates the LLM's tool calls. Policy evaluation is delegated to
  `qa_guardrail`.
- A substitute for network-level sandboxing. A decorated function
  that makes network calls can still hit arbitrary endpoints; the
  gate logs the call but does not control the network.
- A cryptographic breakthrough. It uses standard SHA-256 and
  relies on its collision resistance under the random-oracle model.
- A Coq/Lean-proved kernel implementation. Only the ledger plumbing
  and protocol state machine are formally specified; the Python
  code is verified by conformance tests, not mechanized proofs.

What it IS:
- A single-gate routing architecture proved safe over a bounded
  TLA+ model with an explicit adversary action set.
- A hash-chained append-only audit ledger with Lean-proved
  monotonicity and content-binding invariants.
- A test suite that replays TLA+ counterexample traces against the
  running kernel and rejects any divergence.
- A 10-test SOTA-mapped adversarial suite from AgentDojo, InjecAgent,
  HarmBench, and two fresh tests specific to the kernel's own
  architecture.
- A self-circumvention loop that runs until bounded-effort
  termination.

---

## 11. Phase 3 entry points

Proceed to implementation by creating:
1. `kernel/cert.py` — `CertRecord` + hashing
2. `kernel/ledger.py` — `Ledger` class + conformance-ready state machine
3. `kernel/gate.py` — `enforce_policy` + the one gate function
4. `kernel/hooks.py` — `@gated_tool` decorator + pretool_guard bridge
5. `tests/test_kernel.py` — unit tests driven by the action-mapping table
6. `tests/test_conformance.py` — TLC counterexample replay harness

All five files must pass `tools/qa_axiom_linter.py --all` with zero
errors. The conformance test must replay the two committed traces
and exit 0. Phase 3 is complete when both hold.
