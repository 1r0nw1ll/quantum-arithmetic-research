# AI Platform Integration Guide: Working with QA on Your AI Platform

This guide is for Tier 4 patrons working with Claude, ChatGPT, Gemini, Grok, or any other AI platform. It covers how to use QA as a formal framework within AI-assisted work.

---

## The Problem QA Solves for AI Work

AI platforms are powerful but have a **drift problem**: over a long session, or across sessions, the AI's understanding of your domain accumulates inconsistencies. It starts simplifying your formulas. It uses "approximately" when you need exact. It conflates orbit families. It invents generators you didn't define.

QA solves this with a **kernel spec that the AI must respect**:
- Axioms are machine-checkable (not just stated)
- Failures are named (the AI can't silently drop them)
- Results are certifiable (a certificate either passes or fails)

The session header in `SESSION_HEADER.md` is the practical tool. Use it every session.

---

## Three Modes of QA-AI Work

### Mode 1: QA as a Verification Layer
You use AI to explore or compute, and QA to verify the results. The AI generates candidate answers; you check them against the axiom system.

**Setup**:
1. Paste the session header (see `SESSION_HEADER.md`)
2. Give the AI a computation task
3. Ask it to produce a certificate structure alongside its answer
4. Verify the certificate with the appropriate validator

**Example prompt**:
```
Working under QA Canonical Reference v1.0.
State: (b=3, e=5). Compute the full invariant packet.
Also produce a JSON certificate with: state, all 21 invariant values, orbit_family, and a list of which generators are legal from this state with their successor states.
Verify: does a = b + 2e? Does d = b + e? Is I = |C - F| > 0?
```

### Mode 2: QA as a Planning Tool
You have a target state or behavior. Use QA to plan the generator sequence that reaches it.

**Setup**:
1. Paste the session header
2. Describe your current state and target state in QA terms
3. Ask the AI to perform BFS and produce a planner cert structure
4. Check: is the target in the obstruction spine? (If v_p(r) = 1 for inert p, stop here)

**Example prompt**:
```
Working under QA Canonical Reference v1.0.
Domain: [your domain]. State space: Caps(9,9), mod-9.
Current state: (b=1, e=1) → orbit family: cosmos (Fibonacci family).
Target: reach orbit family cosmos with norm f(b,e) ∈ {4,5} (Lucas family).
Perform BFS. What is the minimal generator sequence? What is path_length_k?
Produce a planner cert with: algorithm, depth_bound, path, minimality_witness.
```

### Mode 3: QA as a Domain Mapper
You have a physical or conceptual domain. Use AI to help map it onto QA orbit families.

**Setup**:
1. Describe your domain: what are its states, its operations, its failure modes?
2. Ask the AI to propose the QA mapping using the Cross-Domain Principle
3. Check the mapping against the axioms
4. Build the Tier 1 (recognition) cert for your domain

**Example prompt**:
```
Working under QA Canonical Reference v1.0.
I want to map [tuning fork / crystal bowl / water cymatic / plant growth / etc.] to QA.
My system has these states: [describe].
My control operations are: [describe].
My observable failure modes are: [describe].
Using the QA generator algebra (σ, μ, λ, ν), propose a mapping. Identify orbit families for each state, and name the QA failure type for each observed failure mode.
Produce a draft Tier 1 recognition cert schema.
```

---

## The QA-Axiom AI Mapping

The QA system maps formally onto AI/theorem-proving systems:

| AI concept | QA concept |
|-----------|-----------|
| Proof state | QA state (b, e, d, a) |
| Proof tactic | QA generator (σ, μ, λ, ν) |
| Proof kernel | QA invariant oracle |
| Valid proof step | Legal generator application |
| Invalid proof step | Generator failure (OUT_OF_BOUNDS, PARITY, etc.) |
| Completed proof | Target orbit family reached |
| Proof certificate | QA cert with planner + control + compiler |

This mapping (documented in `QA_MAP__AXIOM_AI.yaml`) means: **QA can be used as a formal framework for any AI system that progresses through states toward goals** — theorem provers, planning agents, language models with tool use.

---

## The Generator Injection Principle

From `QA_MAP__GENERATOR_INJECTION.yaml`:

> Capability = reachability under a generator set.

An AI agent's capabilities are exactly the states it can reach with its available tools/actions. QA formalizes this:
- Agent actions = generators
- Agent state = (b, e) pair in some mapped space
- Agent goal = target orbit family
- Agent failure = classified generator failure
- Agent certification = planner cert + control cert + compiler cert

This means: **every AI agent has an implicit QA structure**. Making it explicit gives you:
1. A formal bound on what the agent can achieve (reachability)
2. A formal description of what will fail and why (failure algebra)
3. A verifiable record of what the agent did (certificate chain)

---

## Connecting QA to Your Open Brain / Memory System

If you use an AI memory system (Open Brain or similar), QA captures have a specific structure:

**When to capture a QA result**:
1. A theorem was proved — capture: statement, proof sketch, cert ID, verification status
2. A new domain mapping was established — capture: domain, orbit family mapping, generator mapping
3. A result was corrected/retracted — capture: the error AND the fix (e.g. the GF(9) remark correction)
4. A paper status changed — capture when arXiv-ready, frozen, submitted

**QA-specific capture template** (for your memory system):
```
QA result: [statement]
Cert family: [ID if applicable]
Status: [proved / conjectured / retracted]
Verification: [passed X/Y checks / not yet verified]
Cross-domain: [which domains this applies to]
```

---

## Multi-AI Orchestration

If you use multiple AI platforms (Claude + ChatGPT + Gemini):

1. **One platform = orchestrator** (reads certs, specs next task, verifies outputs)
2. **Other platforms = executors** (write code, run experiments, produce cert structures)
3. **The cert system = the shared language** between platforms

The orchestrator never needs to trust that an executor "understood" — it verifies the cert. A cert either passes the validator or it doesn't. Platform-specific interpretation doesn't matter.

This is documented in the project as the ORCHESTRATION RULE: Claude = orchestrator/manager, others = code writing and doc writing.

---

## Common Mistakes to Avoid

| Mistake | Why it matters | Correct approach |
|---------|---------------|-----------------|
| Letting AI use `d` and `a` as free variables | Breaks the invariant; derived coords are never independent | Enforce A1: always `d = b+e`, `a = b+2e` |
| Approximating L = (C*F)/12 | Introduces floating point error in a rational quantity | L is exact rational; always compute exactly |
| Silently ignoring generator failures | Hides information about why a path doesn't work | Every failure must be classified and logged |
| Truncating gate sequence to [0,1,2,3] | Removes tamper-evidence; cert cannot be externally audited | Always enforce full [0,1,2,3,4,5] gate sequence |
| Confusing obstruction spine with control spine | Wrong tool for the job | Obstruction = what's impossible; Control = how to reach the possible |
| Using mod-9 for applied work and forgetting to switch | Orbit structures differ between mod-9 and mod-24 | Applied work uses mod-24 unless explicitly doing Pythagorean theory |

---

## Source References

- Session header: `SESSION_HEADER.md`
- Axiom AI mapping: `qa_alphageometry_ptolemy/QA_MAP__AXIOM_AI.yaml`
- Generator injection: `qa_alphageometry_ptolemy/QA_MAP__GENERATOR_INJECTION.yaml`
- Capture templates: `CAPTURE_TEMPLATES.md`
- Open Brain companion: `Documents/BUILD_A_BRAIN_PROMPT_KIT.md`
- Multi-AI guide: `MULTI_AI_COLLABORATION_GUIDE.md`
