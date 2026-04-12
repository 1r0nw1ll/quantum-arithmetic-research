# Family [210] QA_CONVERSATION_ARAG_CERT.v1

## One-line summary

The conversation graph store at `_forensics/qa_retrieval.sqlite` is a QA-native A-RAG datastore instance per [QA_ARAG_INTERFACE_CERT.v1] (three views: keyword_search / semantic_search / chunk_read), with cross-source messages (ChatGPT / Claude.ai / Google AI Studio) mapped to canonical integer tuples via **Candidate F**: `b = dr(sum(ord(c) for c in raw_text))`, `e = role_rank[canonical_role]`, grounded in the certified Aiq Bekar digital root [family 202] and Theorem NT observer firewall.

## Background

**Problem**: We had three years of conversational research output scattered across three provider-specific export formats (ChatGPT tree-structured `conversations.json`, Claude.ai flat `chat_messages` lists, Google AI Studio per-conversation JSON files with `chunkedPrompt.chunks`). None of the existing QA-native retrieval primitives applied directly; the 2025-11-14 GraphRAG design predated the Observer Projection Firewall (Theorem NT) formalization and violated A2 by assigning `(b, e, d, a)` as four independent learned coordinates.

**Approach**: Extend the existing post-axiom `QA_ARAG_INTERFACE_CERT.v1` cert ([18]+[20] composition — datastore + view + A-RAG) with a conversation-specific instance. Derive `(b, e)` deterministically from axiom-grounded integer measurements (gematria digital root + canonical role rank). Use `(d, a) = (b+e, b+2e)` raw per the 2026-04-09 rule. Use the T-operator (Aiq Bekar `dr`) for sector labels in `{1..9}`. No embeddings. No learned codebooks. No float state anywhere.

**Cross-source normalization**: each source's native role field maps to the canonical 5-role taxonomy `{user:1, assistant:2, tool:3, system:4, thought:5}`. A content-type-sensitive promotion rule moves thinking/reasoning messages (ChatGPT `reasoning_recap`, Claude `content[].type == 'thinking'`, Gemini `chunk.isThought`) onto the `thought` diagonal regardless of their native role field.

**Architecture**: the three A-RAG views get three substrates:

| View | Substrate | SOTA analog |
|---|---|---|
| KEYWORD_VIEW | SQLite FTS5 (unicode61), filtered by orbit sector | SPLADE-v3 inverted index |
| SEMANTIC_VIEW | Personalized PageRank over `parent / cite / succ / ref` edges, α=0.5 | HippoRAG |
| CHUNK_STORE | Direct `messages.raw_text` lookup by `msg_id` | Any K-V store |

The query pipeline follows PLAID's four-stage cheap-first-stage pattern: (1) compute query's `(b,e)` via gematria → sector lookup, (2) FTS5 narrow within sector, (3) PPR walk over edge graph, (4) return full text + metadata. Stages 2 and 3 use float arithmetic only at the observer layer — scores are measurements, never inputs to QA state.

## Mathematical content

### TUPLE: Candidate F derivation

```
dr(n) = 1 + ((n - 1) mod 9)            # Aiq Bekar, certified by family [202]
b     = dr(sum(ord(c) for c in raw_text))   # integer in {1..9}
e     = role_rank[canonical_role]           # integer in {1..5}
d     = b + e                          # raw, per 2026-04-09 rule
a     = b + 2*e                        # raw
d_label = dr(d)                        # T-operator output in {1..9}
a_label = dr(a)                        # T-operator output in {1..9}
sector  = (d_label, a_label)           # 81 possible, axiom-compliant labels
```

### DIAG: Role-diagonal property

Because `a - d = e`, and `e` is discrete-integer role rank, each canonical role occupies exactly one mod-9 diagonal:

| Role | role_rank | `(a_label − d_label) mod 9` |
|---|---|---|
| user | 1 | 1 |
| assistant | 2 | 2 |
| tool | 3 | 3 |
| system | 4 | 4 |
| thought | 5 | 5 |

Each role lives on 9 sectors (one per value of `d_label`), giving 5 × 9 = 45 maximum structural sectors for the full 5-role taxonomy. In practice, sources without a system role use fewer diagonals — Claude shows 2 (user + assistant), ChatGPT shows 3-4 depending on data, Gemini shows 3 (user + model + thought).

### CROSS: Cross-source invariance

The same formula applies to all three sources without modification. Per-source logic is limited to:
1. Role string normalization (e.g., Claude `"human"` → `"user"`)
2. Content-type promotion rule firing for `thought` (per-source detection)
3. Text extraction from the source's native content shape

The empirical verification on Claude + Gemini (5361 messages total) shows **27 sector/role combinations, 0 axiom violations**.

### A1/A2/S2/T2 compliance

- **A1**: All `b ∈ {1..9}` (dr output), `e ∈ {1..5}` (role_rank), `d_label, a_label ∈ {1..9}` (dr T-operator). CHECK constraints in schema enforce these at the DB level.
- **A2**: Only `b` and `e` are stored as columns. `d` and `a` are derived on read via SQL expressions. The index expression uses `1 + ((b + e - 1) % 9)` form (the A1-compliant T-operator).
- **S2**: All QA tuple columns are SQLite INTEGER. No embeddings, no float vectors, no learned parameters.
- **T2**: Raw text enters ingest as an observer projection. FTS5 BM25 scores and PPR probabilities are observer-layer measurements used only for ranking display. They never feed back into QA state. The generator/observer boundary is crossed exactly twice: (1) text → integer measurements at ingest, (2) integer sector structure → ranked observer output at query.

### THOUGHT: Promotion rules per source

| Source | Detection | Native → canonical |
|---|---|---|
| ChatGPT | `content.content_type ∈ {thoughts, reasoning_recap}` | assistant → **thought** |
| Claude.ai | `content[].type == 'thinking'` on any content item | assistant → **thought** |
| Google AI Studio (Gemini) | `chunk.isThought == true` | model → **thought** |

Observed thought-promotion rates (empirical on full ingest):
- Claude.ai: **~46%** of assistant content is extended thinking (91/200)
- Gemini: **~29%** of model output is marked thought (874/3013)
- ChatGPT: pending full ingest measurement

## Checks

| ID | Description |
|----|-------------|
| CAV_1 | `schema_version == 'QA_CONVERSATION_ARAG_CERT.v1'` |
| CAV_SCHEMA | `arag_tool_set == {keyword_search, semantic_search, chunk_read}` |
| CAV_TUPLE | `tuple_derivation.b_formula` references `dr()` and `ord()`; `e_formula` references `role_rank`; `dr_cert_ref == 'family_202'` |
| CAV_DIAG | For each witness: `(a_label - d_label) mod 9 == role_rank[role] mod 9` |
| CAV_CROSS | Witnesses cover all three sources: `{chatgpt, claude, gemini}` |
| CAV_PROMO | `thought_promotion_rules` defined for each source as nonempty strings |
| CAV_A1 | All witness `b ∈ {1..9}`, `e ∈ {1..5}`, `d_label ∈ {1..9}`, `a_label ∈ {1..9}` |
| CAV_A2 | `a2_compliance.d_derived_on_read == True`, `a2_compliance.stored_columns == ['b', 'e']` |
| CAV_T2 | `t2_firewall.raw_text_role == 'observer_projection'`, `fts_score_role == 'observer_measurement'`, `qa_state_feedback_allowed == False` |
| CAV_VIEWS | `tool_to_view_kind` maps each tool to a valid `VIEW_KIND` |
| CAV_W | At least 9 witnesses in total |
| CAV_F | Falsifier has `expect_fail == True`, well-formed `fail_kind` and `fail_reason`, demonstrates an actual axiom violation (e.g., sector label == 0 for A1_SECTOR_ZERO kind) |

## Source grounding

- **Aiq Bekar digital root**: certified by family [202] `QA_HEBREW_MOD9_IDENTITY_CERT.v1` (PASS). `dr(n) = 1 + ((n-1) mod 9)` is A1-compliant by construction. The Kabbalistic Nine Chambers are structurally identical to QA mod-9 state space.
- **A-RAG interface cert**: composes with `QA_ARAG_INTERFACE_CERT.v1` (existing, post-axiom). Shares the `tool_set` / `tool_to_view_kind` / `fail_types` schema.
- **TIGER**: Rajput et al. NeurIPS 2023, *"Recommender Systems with Generative Retrieval"*. Structural twin: items encoded as integer tuples, retrieval as prefix lookup. We borrow the tuple-as-shard-label pattern and discard the learned RQ-VAE codebook + seq2seq decoder in favor of axiom-grounded deterministic derivation.
- **HippoRAG**: Gutiérrez et al. NeurIPS 2024. We borrow the Personalized PageRank ranking (α=0.5, node-specificity weighting) and discard the LLM-extracted entity graph + embedding synonym linking — our graph is already discrete and axiom-compliant.
- **PLAID**: Santhanam et al. CIKM 2022. We borrow the four-stage pipeline structure (cheap candidate gen → filter → rerank) and the posting-list substrate insight, discarding the MaxSim float scoring in favor of FTS5 BM25 + PPR.
- **SPLADE-v3**: Lassance et al. 2024. Referenced as the sparse-lexical baseline to beat on BEIR subsets if this work ever moves to publication.

## Connection to other families

- **[18]** `QA_DATASTORE_SEMANTICS_CERT.v1` — foundational datastore cert; conversation datastore is an instance
- **[20]** `QA_DATASTORE_VIEW_CERT.v1` — view semantics; KEYWORD_VIEW / SEMANTIC_VIEW / CHUNK_STORE all implement
- **[existing]** `QA_ARAG_INTERFACE_CERT.v1` — the A-RAG tool/view contract this cert instantiates
- **[202]** `QA_HEBREW_MOD9_IDENTITY_CERT.v1` — provides the certified `dr` function
- **[122]** `QA_EMPIRICAL_OBSERVATION_CERT.v1` — bridge from Open Brain observation to cert ecosystem; conversation retrieval operationalizes OB search
- **[191]** `QA_BATESON_LEARNING_LEVELS_CERT.v1` — orbit classification as invariant filtration; conversation sectors sit in this framework

## Artifacts

- Schema spec: `chat_data/QA_CONVERSATION_RETRIEVAL_SCHEMA.md`
- Shared schema module: `tools/qa_retrieval/schema.py`
- Ingest adapters: `tools/qa_retrieval/ingest_{chatgpt,claude,gemini}.py`
- Storage: `_forensics/qa_retrieval.sqlite` (gitignored, regenerable)
- Validator: `qa_alphageometry_ptolemy/qa_conversation_arag_cert_v1/qa_conversation_arag_cert_validate.py`
- Fixtures:
  - `fixtures/cav_pass_core.json` (9 witnesses across all three sources)
  - `fixtures/cav_fail_a1_violation.json` (demonstrates A1 sector=0 violation)

## Meta-validator registration (COMPLETE — 2026-04-11)

Registered in `qa_alphageometry_ptolemy/qa_meta_validator.py` FAMILY_SWEEPS via `_validate_conversation_arag_cert_family` subprocess wrapper. The registered validator runs the cert's own `--self-test` and returns None (= PASS) when both fixtures behave correctly. Self-test payload: `{"ok": true, "schema_version": "QA_CONVERSATION_ARAG_CERT.v1", "family": 210, "n_pass": 2, "n_fail": 0}`.

## Programmatic wrapper

A multi-relational KG wrapper is available at `qa_lab/qa_graph/knowledge_graph.py`:

```python
from qa_graph.knowledge_graph import QAKnowledgeGraph

# Load from the real [210] store
kg = QAKnowledgeGraph.from_qa_retrieval_db('_forensics/qa_retrieval.sqlite')

# Or build an in-memory demo
kg = QAKnowledgeGraph.build_demo()

kg.typed_edges_of_type('parent')          # list of (src, dst) pairs
kg.sector_of_node(msg_id)                 # (d_label, a_label)
kg.nodes_on_diagonal(e=2)                 # all assistant-role nodes
kg.multi_hop_neighbors(msg_id, ['parent', 'succ'], max_hops=3)
kg.verify_role_diagonal_theorem()         # exhaustive proof on {1..9}^2
```

### Role-diagonal theorem (rigorously stated)

For any QA state (b, e) with b ∈ {1..9}, e ∈ {1..9}, let

```
d_label = 1 + ((b + e - 1) mod 9)
a_label = 1 + ((b + 2e - 1) mod 9)
```

Then `(a_label − d_label) mod 9 ≡ e (mod 9)` identically over the 81-state space.

**Proof**: `a_label − d_label = (b + 2e) − (b + e) = e (mod 9)` by direct substitution, before the A1 offset. The A1 offsets cancel. ∎

**Corollary**: the 81 sector labels partition into **9 disjoint role-diagonals**, each containing exactly 9 sectors. Each canonical role (user=1, assistant=2, tool=3, system=4, thought=5, note=6) occupies exactly one diagonal. Role-based queries have O(9) structural cost regardless of corpus size. Exhaustively verified: 81/81 match, partition total 9 × 9 = 81.

## Limitations and future work

- **`ref` edges** (lexicon-term co-occurrence) are enabled in v1 but not yet measured for density. If edge count scales poorly we add a per-term top-k cap in v1.1.
- **Gematria is a weak content signal** — the sub-bucketing within a role diagonal is style-level at best. Semantic meaning comes from FTS5 + graph walks, not from sector neighborhood.
- **ChatGPT content-type rule** currently promotes `reasoning_recap` and `thoughts` to `thought`. Other reasoning content types may emerge in future exports; the map is a single constant table that can be extended.
- **Temporal queries** are not yet a first-class view. `create_time_utc` is stored; a `temporal_view` could be added as a fourth A-RAG view in v2.
- **CRT-join across moduli** (mod-9 ↔ mod-24) is a v2 feature. v1 is mod-9 only.
