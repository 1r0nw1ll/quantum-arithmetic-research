<!-- PRIMARY-SOURCE-EXEMPT: reason=Phase 5 determinism fixture; frozen OB snapshot -->
- 2026-04-16T09:00:00Z (observation) [qa-mem,phase-5] Phase 5 scope anchor
  Cert [228] validates canonical graph hashing. References axioms A1 and T2.
  Complements cert [225] consistency and cert [254] authority ranker.

- 2026-04-16T09:05:00Z (insight) [qa-mem,ranker] Authority-tiered ranker stable
  Cert [254] closed-form ranker; factor set is frozen by ranker_spec.json.

- 2026-04-16T09:10:00Z (claim) [firewall,provenance] Theorem NT boundary holds
  The firewall in cert [227] keeps agent-authority causal edges gated by promote.

- 2026-04-16T09:15:00Z (observation) [claude-session-qa-mem-phase-5] Session trace
  Session claude-session-qa-mem-phase-5 is writing canonicalize.py.

- 2026-04-16T09:20:00Z (reference) [cert-family,supersedes] Cert lifecycle
  Cert [225] v4 replaces v3; v3 frozen per KG13 supersedes DAG.

- 2026-04-16T09:25:00Z (claim) [codex-session,firewall] Promotion discipline
  Agent authority nodes cannot emit causal edges without a valid promote
  via cert [227] ledger freshness.
