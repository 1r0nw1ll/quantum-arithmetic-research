# QA ↔ TLA+ State Audit

**Date:** 2026-04-20
**Session:** `audit-tla-plus` / claude-main-1556
**Trigger:** Will — "we need to up our TLA+ game ... map into QA among other useful things"
**Scope:** inventory every `.tla` artifact, machine-check status, strategic gap, recommend three extension lanes with sequenced first steps.

TLA+ primary references: (Lamport, 1994) for the Temporal Logic of Actions, (Lamport, 2002) for the TLA+ specification language and TLC model checker, (Cousineau et al., 2012) for TLAPS. Cert-gate protocol primary references already in `llm_qa_wrapper/spec/cert_gate.tla` header: arXiv:2603.18829 (Agent Control Protocol) and arXiv:2603.23801 (AgentRFC composition safety).

---

## 1. Inventory (16 `.tla` files, two regions)

### Region A — `llm_qa_wrapper/spec/` (MATURE, actively worked Apr 19)

| File | Size | Role | Machine-check |
|---|---|---|---|
| `cert_gate.tla` | — | Main protocol spec: cert-gate state machine + 5 adversary actions + 4 rejection-path actions | **TLC 81,629 states, 0 err, depth 22, 4s** |
| `cert_gate.cfg` | — | Bounded model: 2 agents × 1 tool × 1 payload × MaxLedgerDepth=3, AgentSymmetry | live |
| `cert_gate_negative.tla` | — | Non-vacuity: `BrokenDirectInject` → `Inv_NoExecWithoutCert` fires | TLC: 2 states, error as expected |
| `cert_gate_negative_chain.tla` | — | Non-vacuity: broken ledger prev-chain → `Inv_LedgerChainValid` fires | TLC: 2 states, error as expected |
| `cert_gate_negative_bind.tla` | — | Non-vacuity: `ch ≠ Hash(fields)` → `Inv_CertBindsPayload` fires | TLC: 2 states, error as expected |
| `cert_gate_negative_composition.tla` | — | Non-vacuity: ledger entry missing from `executed` → `Inv_Composition` fires | TLC: 2 states, error as expected |
| `cert_gate_scaled.cfg` | — | Alt config for larger parameter set | present |
| `cert_gate*_TTrace_17759*.tla` (×7) | 5-9 KB | TLC counterexample trace exports for trace-explorer | **all dated 2026-04-19 19:48 — yesterday** |

Plus paired Lean 4 layer: `LedgerInvariants.lean` (10 theorems, axioms = `propext` + `Quot.sound` only, no `sorry`); `LedgerInvariantsCheck.lean`; `LedgerInvariants.olean` (compiled). Theorem-proving authority: (de Moura & Ullrich, 2021) for Lean 4.

Proof record: `TLC_PROOF_LEDGER.md` — disciplined Phase 2a/2b/3/4/5 ledger following the pair-proof pattern from (Lamport, 2002). Python kernel 18/18 tests, SOTA red-team 10/10, self-circumvention 6 caught / 2 boundary / 0 fail. One real bug caught by the loop (`hash-chain-fork` → `gate._wrapper_lock` fix).

**Status: production-grade formal spec. Protocol + data-layer proved. Load-bearing.**

### Region B — `qa_alphageometry_ptolemy/` (STALE since Jan)

| File | Size | Date | Role | Machine-check |
|---|---|---|---|---|
| `QACertificateSpine.tla` | 13 KB | Jan 21 | Cert architecture for 7 domains (Policy / MCTS / Exploration / Inference / Filter / RL / Imitation). Bundle coherence rules + `FailureFirstClass` theorem | **no `.cfg` on disk; never TLC-checked in recent era** |
| `QARM_v02_Failures.tla` | 6 KB | Dec 30 | QARM generator algebra: σ / μ / λ_k moves; fail ∈ {OK, OUT_OF_BOUNDS, FIXED_Q_VIOLATION, ILLEGAL}; 5 invariants declared; qtag = 24·φ9 + φ24 | **no `.cfg`; never TLC-checked in recent era** |
| `QARM_v02_NoMu.tla` | 4 KB | Dec 30 | Variant: μ removed from Next, to test failure-count invariance under generator change | same |
| `QARM_v02_Stats.tla` | 4 KB | Dec 30 | Variant: `PrintT` instrumentation for per-action failure tallies | same |
| `tla2tools.jar` | 4 MB | Dec 30 | Lamport's TLC + SANY + Toolbox JAR (Lamport, 2002 distribution) | used by Region A |

**Status: dormant. QA semantic algebra encoded but unverified.** No `.cfg`, no proof ledger entry, no TTrace, not referenced from any Apr-era cert or doc.

---

## 2. Prior work never migrated to repo

ChatGPT export A-RAG surfaced a 5-KB structured document titled **"Formal Specification and Verification of Quantum Arithmetic Using TLA+"** (internal ID `qa_tla_summary`). Contains Metadata-for-Archiving headers, implying it was drafted as a repo-spec candidate. **It never landed in `docs/specs/`.** Three chunks recoverable:
- `chatgpt/user` b=6 e=1 sector=(7,8) — problem statement + framing (~5200 chars)
- `chatgpt/assistant` b=9 e=2 sector=(2,4) — structured TLA+ summary doc (~5400 chars)
- `chatgpt/assistant` b=2 e=2 sector=(4,6) — "TLA⁺ is extremely well-suited to QA, but not in the way most people initially think" (~4800 chars)

Zero Open Brain entries on "TLA+" as of this audit — knowledge loss since whenever the ChatGPT thread was authored.

---

## 3. Strategic positioning (from `docs/specs/VISION.md` §1)

QA is positioned as a **peer** of TLA+ (Lamport, 2002), not a consumer:

| System | Specifies |
|---|---|
| Hoare logic (Hoare, 1969) | Pre/post-conditions on program states |
| **TLA+ / Lamport (Lamport, 2002)** | **Behaviors of distributed systems over time** |
| Lean / Coq (de Moura & Ullrich, 2021) | Types as propositions; proofs as programs |
| **QA** | **Reachable vs unreachable structure in modular dynamical systems + failure algebra** |

QA's published differentiator: **first-class failure algebra** (geometry of impossibility, machine-checked). This is *already partially encoded* in `QARM_v02_Failures.tla` — `fail` is a structural variable with ILLEGAL / OUT_OF_BOUNDS / FIXED_Q_VIOLATION states that persist as absorbing stuttering. That's the VISION §1 thesis in live TLA+ form, but it's never been run.

---

## 4. What's stale vs live

- **Live:** cert-gate protocol layer (TLA+ + Lean + Python kernel + red-team + self-circumvention; all green Apr 19).
- **Stale:** QA semantic layer (QARM + Certificate Spine; written Dec-Jan; no machine-check since).
- **Missing:** (a) QA axioms A1/A2/T2/S1/S2/T1 as TLA+ temporal invariants — not encoded anywhere; enforced only by the Python linter; (b) ChatGPT `qa_tla_summary` content migration; (c) `qa_core_spec/` is JSON-schema + Python, *not* cross-linked to any `.tla`.

---

## 5. Three extension lanes (sequenced, concrete first steps)

### Lane 1 — **Resurrect QARM/Spine to Region-A parity** (highest leverage)

**Rationale:** the QA semantic layer IS the differentiator per VISION §1. Wrapper-layer proof discipline exists as a template (`TLC_PROOF_LEDGER.md` Phase 2a/2b pattern per Lamport, 2002 §14); apply it to Region B. First proof in the repo of QA's own failure algebra.

**First step (single PR):**
1. Author `QARM_v02_Failures.cfg` — CONSTANTS `CAP=20, KSet={2,3}`, INVARIANTS `Inv_TupleClosed / Inv_InBounds / Inv_QDef / Inv_FailDomain / Inv_MoveDomain`.
2. Author `QARM_v02_Failures_negative.tla` — `BrokenTupleClosure` action that writes `d' ≠ b' + e'` directly. Expected: `Inv_TupleClosed` fires, 2-state counterexample.
3. Run TLC on both; record run in `qa_alphageometry_ptolemy/QARM_PROOF_LEDGER.md` mirroring the wrapper doc.
4. Check `QARM_v02_NoMu` vs `QARM_v02_Failures` **failure-count differential** (this is what `_Stats.tla` was designed for but never executed) — does removing μ from generator set change reachable-failure tally? This is a novel empirical QA question that TLC can answer in seconds.

**Payoff:** a machine-checked proof that QARM's failure algebra is non-vacuous. Citable as "the first formally-verified QA generator spec." Output is peer of Region A's 81,629-state proof record.

### Lane 2 — **QA Axioms as TLA+ temporal invariants** (most research-legible)

**Rationale:** the 6 axioms (A1/A2/T2/S1/S2/T1) are enforced by `tools/qa_axiom_linter.py` at commit-time against source text. They are NOT encoded as *runtime temporal invariants* anywhere. Encoding them in TLA+ lifts them from lint-level to model-checker-level, which is the research contribution that maps cleanly to the `tlaplus/examples` corpus style (Paxos/Raft invariants — see Lamport, 2001 for Paxos; Ongaro & Ousterhout, 2014 for Raft).

**First step:**
1. Author `qa_alphageometry_ptolemy/QAAxioms.tla` that EXTENDS `QARM_v02_Failures`.
2. Add temporal invariants:
   - `Inv_A1_NoZero == b \in 1..CAP /\ e \in 1..CAP` (shift from 0..CAP — directly catches the A1 bug).
   - `Inv_A2_DerivedCoords == d = b + e /\ a = b + 2*e` (already present structurally; lift to named invariant).
   - `Inv_T1_IntegerPathTime == lastMove \in {"NONE","σ","μ","λ"} /\ "t" \notin DOMAIN vars` (no continuous time var).
   - Theorem NT as `Inv_NT_NoObserverFeedback` — requires lifting observer-projection boundary into the spec's variable set (design work, not trivial).
3. TLC-check; produce paired negative tests per invariant (pattern from Region A).

**Payoff:** the QA axiom system as a first-class TLA+ module, submittable to `tlaplus/examples`.

### Lane 3 — **Migrate ChatGPT `qa_tla_summary` + cross-link** (fastest payoff, lowest novelty)

**First step:**
1. Pull 3 chunks via `tools.qa_retrieval.query chunk`, reassemble as `docs/specs/QA_TLA_PLUS.md` with provenance header `Source: ChatGPT export, rehydrated 2026-04-20`.
2. Update this audit's §2 inventory + add a pointer from `docs/specs/VISION.md` §1 TLA+ row.
3. Capture as Open Brain reference so future sessions find it via search.

**Payoff:** closes 6-month knowledge-loss gap in ~30 minutes. Zero research novelty but prevents re-deriving.

---

## 6. Main tradeoff

Initial framing treated Will's ask as "start mapping TLA+ into QA." The repo disproves that framing: TLA+ is already a *dominant* formal-methods tool here at the infrastructure layer. The real leverage is **reactivating dormant QA-semantic specs and synchronizing them with Region A's mature proof discipline** — not building new infrastructure.

- Lane 1 is highest research leverage (first formal proof of QA's own differentiator) but most spec work: ~1-2 sessions.
- Lane 2 is the most externally legible contribution (submittable upstream to `tlaplus/examples`) but has a hard conceptual bit (Theorem NT as temporal formula): ~2-3 sessions.
- Lane 3 is unblock-everything hygiene: ~1 hour.

**Recommended order:** Lane 3 → Lane 1 → Lane 2. Lane 3 surfaces the prior thinking that Lanes 1 and 2 can build on; Lane 1 produces the empirical-proof substrate Lane 2 needs to encode axioms against.

---

## 7. Machine-checkable handoff

- `llm_qa_wrapper/spec/TLC_PROOF_LEDGER.md` — replicable template for Region B proof ledger.
- `qa_alphageometry_ptolemy/tla2tools.jar` — shared TLC binary; no new infra needed.
- Invocation pattern (from proof ledger):
  ```
  java -XX:+UseParallelGC -jar $REPO/qa_alphageometry_ptolemy/tla2tools.jar \
      -workers 4 -config <spec>.cfg <spec>.tla
  ```

Session-end intent: release `session:audit-tla-plus` lock, broadcast `session_done`, capture audit-complete OB entry pointing at this doc.

---

## References

- Lamport, L. (1994). *The Temporal Logic of Actions.* ACM TOPLAS 16(3), 872–923. DOI: 10.1145/177492.177726.
- Lamport, L. (2001). *Paxos Made Simple.* ACM SIGACT News 32(4), 51–58.
- Lamport, L. (2002). *Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers.* Addison-Wesley. ISBN 978-0-321-14306-8.
- Cousineau, D., Doligez, D., Lamport, L., Merz, S., Ricketts, D., & Vanzetto, H. (2012). *TLA+ Proofs.* In *FM 2012: Formal Methods* (LNCS 7436), 147–154.
- de Moura, L., & Ullrich, S. (2021). *The Lean 4 Theorem Prover and Programming Language.* In *CADE 28* (LNCS 12699), 625–635.
- Hoare, C. A. R. (1969). *An Axiomatic Basis for Computer Programming.* Comm. ACM 12(10), 576–580.
- Ongaro, D., & Ousterhout, J. (2014). *In Search of an Understandable Consensus Algorithm.* USENIX ATC '14.
- Liu et al. *Agent Control Protocol.* arXiv:2603.18829.
- *AgentRFC: Composition Safety for Agent Systems.* arXiv:2603.23801.
- TLA+ examples corpus: `github.com/tlaplus/examples` (Paxos, Raft, distributed snapshots).
