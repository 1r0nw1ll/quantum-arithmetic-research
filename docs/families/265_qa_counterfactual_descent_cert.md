# [265] QA Counterfactual Descent Cert

## What this is

The **third sharp-claim cert** derived from the Kochenderfer 2026 *Algorithms for Validation* bridge (see [bridge spec §5](../specs/QA_KOCHENDERFER_BRIDGE.md), counterfactual-explanation row). Anchored at Kochenderfer Validation §11.5 *Counterfactual Explanations* (open-candidate framing in the v1 ingestion).

**Primary source**:
- Kochenderfer, M. J., Wheeler, T. A., Katz, S., Corso, A., & Moss, R. J. (Kochenderfer, 2026). *Algorithms for Validation*. MIT Press. CC-BY-NC-ND. Chapter 11 §11.5 'Counterfactual Explanations' — "smallest input change that flips the output". §11.5 was tagged not-mapped-here in the v1 Validation excerpts ingestion (continuous-domain) but is the canonical anchor for this cert.

## Claim (narrow)

For QA finite orbit-class PASS/FAIL specifications on `S_9`, counterfactual explanations can be computed **exactly as shortest-legal-generator-paths on the orbit graph**. The counterfactual distance from state `s` to spec-flip is the minimum number of legal QA generator moves needed to land in a state where the Boolean specification predicate flips.

**Claim scope**:
- Claim does **not** generalize to continuous feature-space counterfactual explanations.
- Claim does **not** say BFS scales to arbitrary state spaces.
- Exactness holds only for QA-discrete finite-state symbolic settings on the QA-discrete side of the Theorem NT firewall.

## Construction

**Spec predicate**:
```
in_PASS(s) := orbit_family_s9(s) ∈ declared_pass_classes
```

**Legal generators (v1 set, three only):**
- `qa_step(b, e) = (e, qa_mod(b + e, 9))` — Fibonacci shift T (= L_1, orbit-preserving per cert [191] tier hierarchy)
- `scalar_mult_2(b, e) = (qa_mod(2b, 9), qa_mod(2e, 9))` — L_2a (coprime-to-9 scalar; orbit-changing, family-preserving)
- `scalar_mult_3(b, e) = (qa_mod(3b, 9), qa_mod(3e, 9))` — L_2b (multiple-of-3 scalar; family-changing — the orbit-class bridge)

**Counterfactual algorithm:**
- BFS from declared start state on the directed graph defined by the declared generator subset.
- Terminate at first state where `in_PASS(state) ≠ in_PASS(start)` (the flip).
- Path length = BFS depth = exact counterfactual distance.
- Reconstruct path via parent pointers.

## Empirical results (declared in fixtures, recomputed by validator)

For all 6 declared cosmos-start test cases under `declared_pass_classes = {cosmos}` and full v1 generator set:

| Start | Class | scalar_mult_3 → Terminal | Terminal Class | Path Length |
|-------|-------|--------------------------|---------------|-------------|
| (1, 1) | cosmos | (3, 3) | satellite | 1 |
| (1, 2) | cosmos | (3, 6) | satellite | 1 |
| (2, 5) | cosmos | (6, 6) | satellite | 1 |
| (4, 7) | cosmos | (3, 3) | satellite | 1 |
| (7, 8) | cosmos | (3, 6) | satellite | 1 |
| (8, 9) | cosmos | (6, 9) | satellite | 1 |

Under inverted spec `declared_pass_classes = {satellite, singularity}` (4 test cases), same generator set: same 1-step `scalar_mult_3` paths flip in the opposite direction (cosmos out-of-PASS → satellite in-PASS).

## Structural observation (asymmetry)

Under the v1 declared generator set, scalar multiplication on Z/9 is a **one-way contraction toward multiples of 3**. This means:

- **Cosmos states (72/81)** can move into satellite via `scalar_mult_3` — counterfactuals exist, all of length 1.
- **Satellite states (8/81)** are trapped: `qa_step` keeps you in satellite (8-cycle), `scalar_mult_2` keeps you in satellite (since `2 · multiple_of_3 ≡ multiple_of_3 mod 9` for this lattice), `scalar_mult_3` maps to satellite or singularity. Cosmos is unreachable.
- **Singularity (9, 9)** is a fixed point of all 3 v1 generators.

This asymmetry is documented in the satellite + singularity witnesses as `"counterfactual_exists": false`. v2 may add an additive generator (e.g., `add_one_b(b, e) = (qa_mod(b+1, 9), e)`) to give bidirectional symmetry; v1 scope is sufficient to establish the BFS-reduction claim.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_counterfactual_descent_cert_v1/qa_counterfactual_descent_cert_validate.py` |
| Utility (shared) | `tools/qa_kg/orbit_failure_enumeration.py` (cert [263] anchor) |
| Primary PASS fixture (PASS = {cosmos}) | `qa_counterfactual_descent_cert_v1/fixtures/pass_s9_shortest_counterfactual_paths.json` |
| Inverted-spec PASS fixture (PASS = {satellite, singularity}) | `qa_counterfactual_descent_cert_v1/fixtures/pass_generator_cost_weighted_paths.json` |
| FAIL fixture (illegal generator) | `qa_counterfactual_descent_cert_v1/fixtures/fail_illegal_generator_path.json` |
| FAIL fixture (non-minimal path) | `qa_counterfactual_descent_cert_v1/fixtures/fail_nonminimal_counterfactual_path.json` |
| Mapping ref | `qa_counterfactual_descent_cert_v1/mapping_protocol_ref.json` |
| Bridge spec | `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §5 counterfactual-explanation row |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_counterfactual_descent_cert_v1
python qa_counterfactual_descent_cert_validate.py --self-test
```

## Gates

- **CFD_1** — `schema_version == "QA_COUNTERFACTUAL_DESCENT_CERT.v1"`.
- **CFD_DECL** — `declared_pass_classes` is a non-empty list of orbit-class labels; `declared_generators` is a non-empty subset of `{qa_step, scalar_mult_2, scalar_mult_3}` (v1 closed legal-generator set).
- **CFD_LEGAL** — every step in each declared `counterfactual_test_cases[i].path` uses a generator name in `declared_generators` AND applying that generator to the prior state produces the declared successor (independent recomputation).
- **CFD_FLIP** — terminal state of each path flips the spec predicate (`in_PASS(start) ≠ in_PASS(terminal)`).
- **CFD_MINIMAL** — declared path length matches the BFS-recomputed shortest path length under the declared generator set + declared spec, bit-exact.
- **CFD_SRC** — `source_attribution` cites Kochenderfer 2026 + cert [263].
- **CFD_WIT** — at least 3 witnesses, one per orbit class (`witnesses[i].orbit_class` covers `{singularity, satellite, cosmos}`); satellite + singularity witnesses may declare `counterfactual_exists: false` to document the v1 generator-set asymmetry.
- **CFD_F** — `fail_ledger` is well-formed; FAIL fixtures (with `result == "FAIL"`) early-return after `CFD_1` + `CFD_F`.

## Theorem NT compliance

Integer-only state path throughout. `qa_step`, `scalar_mult_2`, `scalar_mult_3` are integer functions of integer inputs. BFS produces integer path lengths and discrete state sequences. No float operations. No observer projection. The cert lives entirely on the QA-discrete side of the firewall.

## Why mod-9 only in v1

Same constraint as certs [263] and [264]: the canonical mod-9 orbit-family classifier `orbit_family_s9` is the only canonical orbit-class predicate currently published; mod-24 extension would need a new classifier landing first.

## Cross-references

- [263] `qa_failure_density_enumeration_cert_v1` — utility provider; `orbit_family_s9` reused from `tools/qa_kg/orbit_failure_enumeration.py`.
- [264] `qa_runtime_odd_monitor_cert_v1` — complementary cert: [264] verifies orbit-class membership exactness; [265] verifies counterfactual-distance exactness on the same orbit graph.
- [191] `qa_bateson_learning_levels_cert_v1` — tier hierarchy aligns with declared generator labels: `qa_step = L_1`, `scalar_mult_2 = L_2a`, `scalar_mult_3 = L_2b`.
- [194] `qa_cognition_space_morphospace_cert_v1` — canonical mod-9 orbit-family classifier.
- `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §5 counterfactual-explanation row — flips from `open` to `established` once this cert lands.

## Future work

- v2: add an additive generator (e.g., `add_one_b`) to give bidirectional symmetry — counterfactuals from satellite/singularity states would then exist.
- v2: weighted Dijkstra with declared per-generator costs (e.g., `cost(qa_step) = 1, cost(scalar_mult_2) = 1, cost(scalar_mult_3) = 2`) to model "expensive" tier-changing moves.
- mod-24 extension once a canonical mod-24 orbit-family classifier lands.
- Connect to cert [259] HeartMath: counterfactual paths could explain "what HRV change would flip the orbit-class label?" as an applied use case.
