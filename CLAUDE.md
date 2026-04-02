# CLAUDE.md

Mathematical research project: **Quantum Arithmetic (QA) System** — a modular arithmetic framework with applications in signal processing, finance, neural networks, and automated theorem generation. Computational and experimental, not a traditional software project.

## QA Axiom Compliance (HARD GATE — enforced on every commit)

**Theorem NT (Observer Projection Firewall)**: Continuous functions are observer projections ONLY. They NEVER enter the QA discrete layer as causal inputs. QA dynamics are discrete; the boundary is crossed exactly twice (input → observer layer, QA layer → output).

**The six non-negotiable axioms:**
- **A1 (No-Zero)**: States in {1,...,N}. Never {0,...,N-1}. `qa_step`: `((b+e-1) % m) + 1`, not `(b+e) % m`.
- **A2 (Derived Coords)**: `d = b+e`, `a = b+2e` — always derived, never assigned independently.
- **T2 (Firewall)**: Float × modulus → int cast is a **QA violation** (T2-b). Observer outputs never feed back as QA inputs.
- **S1 (No `**2`)**: Write `b*b`, never `b**2` (libm ULP drift).
- **S2 (No float state)**: `b`, `e` must be `int` or `Fraction`. No `np.zeros`, `np.random.rand` as QA state.
- **T1 (Path Time)**: QA time = integer path length k. No continuous time variables in QA logic.

**Linter** (pre-commit hook): `python tools/qa_axiom_linter.py --staged`
Manual scan: `python tools/qa_axiom_linter.py --all`
Authority: `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md` | `QA_AXIOMS_BLOCK.md`

## Smoke Test (run to verify nothing is broken)

```bash
python tools/qa_axiom_linter.py --all && cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

## Security Audit (run daily or after infrastructure changes)

```bash
python tools/qa_security_audit.py
```

Checks: guardrail E2E (12 tests), agent security kernel (14 tests), collab bus agent registry, event log credential scan, GUARDRAIL_DENY report, bridge process status. All bridges MUST run with guardrail enabled (no `--no-guardrail` flag).

## Do Not Touch

These directories/files are off-limits — never modify, delete, or reorganize:
- `archive/` — historical session logs, handoffs, zips
- `Documents/` — working drafts, ODT exports, chat migrations
- `QAnotes/` — Obsidian vault (100+ research notes)
- `~/Desktop/qa_finance/` — private finance scripts (frozen hashes)
- `*.png` images in repo root — generated experiment outputs
- Any file under `qa_alphageometry_ptolemy/` family dirs without understanding Gate 0 (see below)

## QA Mapping Protocol (Gate 0)

Every certificate family root in `qa_alphageometry_ptolemy/` must contain **exactly one** of `mapping_protocol.json` (inline) or `mapping_protocol_ref.json` (reference). Schemas + validators at repo root: `qa_mapping_protocol/`, `qa_mapping_protocol_ref/`.

## Core QA Architecture

- **Modular arithmetic**: mod 9 (theoretical) or mod 24 (applied)
- **State**: Pairs (b, e) → tuples (b, e, d, a) where `d = (b+e) % m`, `a = (b+2e) % m`
- **Three orbits**: 24-cycle Cosmos (72 pairs, 1D), 8-cycle Satellite (8 pairs, 3D), 1-cycle Singularity (fixed at (9,9))
- **E8 Alignment**: 4D QA tuples → 8D projection → cosine similarity to E8 root system (240 vectors)
- **Harmonic Index**: `HI = E8_alignment × exp(-0.1 × loss)`

## Key Implementation Patterns

- **Resonance**: `np.einsum('ik,jk->ij', tuples, tuples)` for coupling
- **Noise annealing**: `noise * (NOISE_ANNEALING ** t)`
- **Markovian coupling**: Weight matrices update from tuple resonance (self-organizing)
- **Signal injection**: External signals enter via `b` state, influence coupling before propagation

## Working with This Codebase

- Scripts are **standalone** — run directly, no imports between experiment files
- Each script is **self-contained** with parameters at the top
- `QA_Engine`/`QASystem` is re-implemented per file with domain-specific variations — no shared module
- **Always use `_final` or latest version** — earlier versions contain bugs
- Random seeds set for reproducibility (usually `np.random.seed(42)`)
- Output PNGs saved to current directory

For full experiment list and how to run them: `docs/experiments/RUNNING_EXPERIMENTS.md`

## Architectural Invariants

1. **No zero element**: QA uses {1,...,9} or {1,...,24}, not {0,...,N-1}
2. **Geometry is emergent**: E8 alignment and orbits arise from dynamics, not explicit encoding
3. **Coupling is bidirectional**: Signal affects coupling, coupling affects evolution
4. **Statistical rigor**: Multi-trial validation, t-tests, KS tests — check for honest failure reporting

## Multi-Session Parallelism Protocol

Multiple Claude sessions may run in parallel across worktrees or within the same worktree. File-level resource locking via the `qa-collab` MCP bus prevents conflicts.

### Session Startup (REQUIRED for every session)

1. **Pick a session name**: `paper-<topic>`, `exp-<topic>`, `cert-<topic>`, `lab-<topic>`
2. **Read existing locks**: `collab_get_state(key="file_locks")` — inspect what's claimed
3. **Register yourself**: `collab_set_state(key="session:<name>", value={"scope": "<dir>", "ts": "<ISO8601>"})`
4. **Check disk**: if `df -h /` shows < 2GB free, warn Will before heavy compute

### File Locking (REQUIRED before editing any file another session might touch)

**Acquire lock before editing:**
```
1. collab_get_state(key="file_locks")          → read current locks dict
2. Check: is your target file already locked?
   - YES and lock.ts is < 5 min old → WAIT or ask Will
   - YES and lock.ts is > 5 min old → stale lock, safe to claim (dead session)
   - NO → proceed
3. collab_set_state(key="file_locks", value={...existing, "<path>": {"session": "<name>", "ts": "<ISO8601>"}})
```

**Release lock when done editing:**
```
1. collab_get_state(key="file_locks")          → read current
2. Remove your file's entry from the dict
3. collab_set_state(key="file_locks", value={...without your entry})
```

**Broadcast major changes** so other sessions react:
```
collab_broadcast(event_type="file_updated", data={"session": "<name>", "file": "<path>", "summary": "..."})
```

### Files That ALWAYS Require Locking

- `CLAUDE.md`, `MEMORY.md`, `AGENTS.md` — shared config
- `qa_alphageometry_ptolemy/qa_meta_validator.py` — cert registry
- `docs/families/README.md` — family index
- Any file you know another active session uses

### Files That NEVER Need Locking

- New files you are creating (no other session knows about them yet)
- Files inside your exclusive scope directory when no other session claims that scope
- Read-only access (reading never conflicts)

### Git Commit Rules for Parallel Sessions

- **Preferred**: Will commits from a dedicated terminal. Sessions stage nothing.
- **If a session must commit**: `collab_broadcast(event_type="commit_intent", data={"session": "<name>", "files": [...]})` then wait 5s for objections via `collab_wait_for_event(topic="commit_veto", timeout_s=5)` before proceeding.
- **Never force-push from a parallel session.**

### Session Shutdown

1. Release all your file locks (remove entries from `file_locks`)
2. `collab_broadcast(event_type="session_done", data={"session": "<name>", "summary": "..."})`
3. Remove `session:<name>` state key
4. Capture key findings in Open Brain

### Worktree Layout

| Directory | Branch | Workstream |
|---|---|---|
| `/home/player2/signal_experiments/` | `main` | qa_lab, domain experiments, heavy compute |
| `/home/player2/wt-papers/` | `wt/papers` | Paper writing/editing |
| `/home/player2/wt-certs/` | `wt/certs` | Cert families + validator |

Multiple sessions MAY work in the same worktree if they lock files properly and have disjoint scopes.

## Research Documentation

- **GEMINI.md**: High-level project overview
- **CONSTITUTION.md**: Strategic governance
- **docs/specs/VISION.md**: Full audit and roadmap
- **docs/specs/PROJECT_SPEC.md**: Technical spec
- **qa_formal_report.tex**: Formal mathematical foundations
