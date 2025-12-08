Handoff: Continuous Vault Mining + Knowledge Graph Refresh

Purpose
- Keep the new Codex agent running hands‑off: mine the Obsidian vault, refresh QA fast‑path signals, and maintain the knowledge graph with clear artifacts and daily summaries.

Operating Mode
- Goal: Continuously mine the vault (.md), refine QA‑native fast‑path signals and the KG, and produce daily summaries and trend snapshots.
- Loop Driver: Use QA Lab’s agent loop + Make targets so it runs automatically with minimal noise.

One‑Time Setup (new instance)
1) Create/activate venvs and install packages per HANDOFF_TO_NEW_INSTANCE.md
2) Build Rust extension (optional, recommended):
   - make -C qa_lab rust-py-build
3) Ensure E8 roots exist:
   - python qa_lab/scripts/generate_e8_roots.py

Environment Defaults (safe)
- Fast‑path gates/weights:
  - QA_FP_ENABLE_WHEEL=1, QA_FP_ENABLE_FAMILY=1, QA_FP_FAMILY_TOL=0.05, QA_FP_POS_MIN=0.0
  - Optional QE weights: QA_QE_CURV_WEIGHT=0.25; keep others at 0 unless testing
- E8 strategy:
  - QA_E8_PREFER_NUMPY=0, QA_E8_DISABLE_RUST=0
  - Optional chunking: QA_E8_VEC_CHUNK=200000; leave QA_E8_ROOT_CHUNK unset unless needed

Continuous Loop (every cycle)
- Run the agent loop (includes fast‑eval + daily summary):
  - make -C qa_lab agent_loop
- Produces automatically:
  - Fast‑path: artifacts/evals/fastpath_eval.txt/.json
  - Daily summary: artifacts/evals/daily_summary_latest.txt
  - Trends: artifacts/evals/fastpath_trends.json/.png

Vault Mining + KG Refresh
- Refresh the Knowledge Graph each loop (already wired into agent_loop):
  - make -C qa_lab phase1-kg
- This runs (with repo‑root outputs):
  - entities-extract → ../artifacts/knowledge/qa_entities.json
  - entities-encode → ../artifacts/knowledge/qa_entity_encodings.json
  - graph-build → ../artifacts/knowledge/qa_knowledge_graph.graphml
  - graph-viz → ../artifacts/plots/qa_knowledge_graph.png

Daily Summary Hook (KG)
- The agent loop now appends a KG section to the daily summary automatically:
  - qa_lab/scripts/qa_kg_summary.py appends entity/edge counts and top‑k by HI (fallback E8)
  - Output: qa_lab/artifacts/evals/daily_summary_latest.txt (appended below fast‑path block)

What to refine (autonomous, low‑noise)
- QA‑native filters/features: φ‑based family detectors (a/d, e/b), wheel routing (mod‑24 sectors), triangle/positivity checks. Keep defaults safe.
- E8 path: NumPy for M≫N; Rust batch with chunking otherwise. Adjust chunks if trend timings regress.
- Streaming scale: for very large N, use fast‑prune‑stream and record top‑k stats under artifacts/evals/.

Alerting (minimal)
- Append a single line at the end of daily_summary_latest.txt only if:
  - Speedup < 1.5× for default config
  - post_gates or post_qe collapse to ~0 unexpectedly
  - E8 roots missing and generator failed
  Otherwise stay quiet; rely on daily summaries and plots.

File/Command Pointers
- Artifacts (repo root):
  - artifacts/evals/*.txt/.json/.png
  - artifacts/knowledge/*.json, *.graphml
  - artifacts/plots/*.png
- Key scripts:
  - Fast‑eval: qa_lab/qa_fast_eval.py
  - Daily summary/trends: qa_lab/scripts/daily_summary.py, qa_lab/scripts/append_fastpath_trends.py, qa_lab/scripts/plot_fastpath_trends.py
  - E8 roots: qa_lab/scripts/generate_e8_roots.py
  - KG: qa_entity_extractor.py, qa_entity_encoder.py, qa_knowledge_graph.py, qa_graph_viz.py
  - KG daily appender: qa_lab/scripts/qa_kg_summary.py
  - Streaming: qa_lab/scripts/fast_prune_stream.py

Make Integration (one‑button)
- Agent loop (continuous): make -C qa_lab agent_loop
- Metrics (fast‑eval + daily summary + trends + plot): make -C qa_lab metrics
- Knowledge Graph (Phase 1, repo‑root artifacts): make -C qa_lab phase1-kg

Notes
- Phase 1 encodings are deterministic hash‑based with manual overrides (qa_entity_overrides.yaml). E8/HI reuse qa_fastpath functions when available, and fall back to simplified metrics if not.
- The loop is low‑friction and quiet by design. Daily outputs and trend plots are the primary surfaces.

