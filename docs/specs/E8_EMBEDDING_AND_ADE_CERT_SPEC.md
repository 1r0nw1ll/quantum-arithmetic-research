# Certs [248] + [249] — Codex Implementation Spec

**Orchestrator:** Claude (session `wild-e8-embed`, 2026-04-15).
**Executor:** Codex.
**Theory doc:** `docs/theory/QA_E8_EMBEDDING_AND_ADE.md`.
**Builds on:** [244] `qa_mutation_game_root_lattice_cert_v1/` — reuse BFS + mutation rule + E_8 adjacency directly.

Ship **both certs in one commit** to keep registry churn minimal.

## 1. Deliverables

```
qa_alphageometry_ptolemy/qa_e8_embedding_orbit_classifier_cert_v1/
    mapping_protocol_ref.json
    qa_e8_embedding_orbit_classifier_cert_validate.py
    fixtures/
        e8e_pass_winner_embedding.json
        e8e_fail_spoofed_invariance.json

qa_alphageometry_ptolemy/qa_ade_mutation_game_cert_v1/
    mapping_protocol_ref.json
    qa_ade_mutation_game_cert_validate.py
    fixtures/
        ade_pass_a5_d5_e6_e7_e8.json
        ade_fail_wrong_orbit_size.json
```

Plus `FAMILY_SWEEPS` entries in `qa_meta_validator.py` for [248] + [249]; family docs `docs/families/248_*.md` + `docs/families/249_*.md`; two new rows in `docs/families/README.md`.

## 2. Cert [248] — E_8 embedding orbit classifier

### 2.1 Inputs

- Canonical E_8 Cartan from [244] (import the constants directly; do not re-derive).
- m=9 T-orbit partition: compute fresh by iterating T: `(b,e) → (e, ((b+e−1) mod 9)+1)` from each of the 81 `(b,e)` in `[1..9]²`. Reference against cert [233] fixture `fixtures/uodp_pass_orbit_profile.json` (or equivalent; locate in the [233] cert dir).

### 2.2 Candidate embeddings

```python
def E_diag(b, e, m=9):
    d = ((b + e - 1) % m) + 1
    a = ((b + 2*e - 1) % m) + 1
    return (b, e, d, a, 0, 0, 0, 0)

def E_tri(b, e, m=9):
    d = ((b + e - 1) % m) + 1
    a = ((b + 2*e - 1) % m) + 1
    return (b, e, b, e, d, d, a, a)
```

### 2.3 Empirical tests

For each embedding `E ∈ {E_diag, E_tri}`:

1. Compute `Q(v) = v^T · G · v` as integer for every `(b,e) ∈ [1..9]²`.
2. Group by T-orbit (from §2.1).
3. Report per-orbit `{len, min_Q, max_Q, n_distinct_Q}`.
4. Count T-orbits where `n_distinct_Q == 1` (T-orbit-invariant under Q).

### 2.4 Claim selection

- If `E_diag` yields T-orbit-invariance on **all 9 T-orbits**: ship that as `E8E_DIAG_INVARIANT_ALL`.
- Else if `E_tri` yields T-orbit-invariance on **all 9 T-orbits**: ship that.
- Else: ship the **partial result** with explicit per-orbit data and a `E8E_BEST_EMBEDDING` label identifying which embedding covers more orbits. Both the count and the failing orbits are part of the witness.

### 2.5 Check list

| ID | Verifies |
|---|---|
| `E8E_1` | Gate 0 |
| `CARTAN_LOAD` | `G_{E_8}` imported from [244] matches canonical (hash or elementwise) |
| `T_ORBIT_CENSUS` | 9 orbits total: 1 Singularity + 2 Satellites + 6 Cosmos (counts by size `{1, 4, 4, 12, 12, 12, 12, 12, 12}` summing to 81). If cert [233] uses different counts, emit both and flag — but do NOT patch; re-read [233] fixture |
| `DIAG_PROFILE` | per-orbit `{min_Q, max_Q, n_distinct_Q}` under `E_diag` recorded in fixture |
| `TRI_PROFILE` | same under `E_tri` |
| `BEST_EMBEDDING` | winner label + its invariant orbit count |
| `SRC` | Wildberger 2020 + theory doc |
| `WITNESS` | `e8e_pass_winner_embedding.json` reproduces all per-orbit stats |
| `F` | spoofed fixture (claim invariance where data says not) is rejected |

### 2.6 Fixture schema

`e8e_pass_winner_embedding.json`:

```json
{
  "family": "qa_e8_embedding_orbit_classifier_cert",
  "m": 9,
  "t_orbits": [
    {"label": "Singularity", "size": 1, "points": [[9,9]]},
    {"label": "Sat#1", "size": 4, "points": [...]},
    ...
  ],
  "embeddings": {
    "E_diag": {"per_orbit": [{"label":"Singularity","Q":[162],"invariant":true}, ...], "total_invariant": N1},
    "E_tri":  {"per_orbit": [...], "total_invariant": N2}
  },
  "best_embedding": "E_diag" | "E_tri" | "tie",
  "best_invariant_count": max(N1, N2)
}
```

`e8e_fail_spoofed_invariance.json`: same schema, with `per_orbit[k].invariant = true` for an orbit where `len(set(Q)) > 1`. Validator must reject on inconsistency.

## 3. Cert [249] — Full ADE mutation game

### 3.1 Inputs

Five Dynkin types: `A_5, D_5, E_6, E_7, E_8`. Edge lists per theory doc §B.2. Expected orbit sizes: `30, 40, 72, 126, 240` respectively.

### 3.2 Implementation

Reuse `mutation` + BFS from [244] exactly. Parameterize over `(n, edges)`. For each type:

1. Build `A` (symmetric adjacency), `G = 2·I − A`.
2. Check `det(G) == 1`.
3. BFS from `δ_0 = (1, 0, ..., 0)` under `{s_i}` for `i ∈ {0..n-1}`.
4. Assert orbit size matches expected.
5. Split positive / negative, assert equal halves and `R− = −R+`.
6. For every `v` in orbit, assert `v^T · G · v == 2`.

### 3.3 Check list

| ID | Verifies |
|---|---|
| `ADE_1` | Gate 0 |
| `CARTAN_A5`..`CARTAN_E8` | det = 1 for each type |
| `BFS_A5`..`BFS_E8` | orbit sizes `{30, 40, 72, 126, 240}` exact |
| `ROOT_NORM_*` | `v·G·v = 2` exhaustive per type |
| `SIGN_SPLIT_*` | equal halves + negation |
| `SRC` | Humphreys GTM 9 Table 1 + Wildberger 2020 |
| `WITNESS` | `ade_pass_*.json` contains orbit sizes for all 5 types |
| `F` | corrupted edge list (drop one edge of E_6 → wrong orbit size) rejected |

### 3.4 Fixture

```json
{
  "family": "qa_ade_mutation_game_cert",
  "types": {
    "A5": {"n": 5, "edges": [[0,1],[1,2],[2,3],[3,4]], "orbit_size": 30},
    "D5": {"n": 5, "edges": [[0,1],[1,2],[2,3],[2,4]], "orbit_size": 40},
    "E6": {"n": 6, "edges": [[0,2],[2,3],[3,4],[4,5],[1,3]], "orbit_size": 72},
    "E7": {"n": 7, "edges": [[0,2],[2,3],[3,4],[4,5],[5,6],[1,3]], "orbit_size": 126},
    "E8": {"n": 8, "edges": [[0,2],[2,3],[3,4],[4,5],[5,6],[6,7],[1,3]], "orbit_size": 240}
  }
}
```

(Codex: double-check the `D_5` edge list — Bourbaki convention has a branch at the second-to-last vertex. If `orbit_size != 40` under the listed edges, fix the edges, not the expected size.)

## 4. QA compliance

Both validators carry the same `QA_COMPLIANCE` module-docstring block as [244]:

```python
"""
QA_COMPLIANCE: {
    'signal_injection': 'none (static algebraic construction)',
    'dynamics': 'integer Weyl mutations on ℤ^n',
    'float_state': false,
    'observer_projection': 'none — classification is QA-discrete-layer',
    'time': 'integer path-length in Weyl group (T1 clean)'
}
"""
```

## 5. Smoke test

```
python tools/qa_axiom_linter.py --all
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

Both certs must be green in the meta-validator output.

## 6. Commit handoff

One commit, full loop per CLAUDE.md §"Git Commit Rules for Parallel Sessions" (stage + commit + push):

```
feat(cert): [248]+[249] E_8 embedding classifier + full ADE mutation game
```

Include: the two cert dirs (8 files), meta-validator entry, two family docs, README.md edit, theory doc, spec doc.

## 7. Execution

```
nohup codex exec --skip-git-repo-check --full-auto < codex_248_249_prompt.txt > /tmp/codex_248_249.log 2>&1 &
```
