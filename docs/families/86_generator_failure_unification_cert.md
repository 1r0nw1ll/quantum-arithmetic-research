# Family [86]: QA Generator–Failure Algebra Unification Cert v1

## What it is

This family certifies that a generator set G, acting on a named domain, **structurally binds** to an independently certified failure algebra (Family [76]) and an independently certified energy cert (Family [80]).

It is not a standalone theorem. It is a **cross-family spine** that compresses:

```
[76] Failure Algebra (carrier, poset, join, compose)
        ↑ carrier membership gate (Gate 1)
        │ τ: G → F  (tagging function — every generator maps to an F element)
        │
[86] Generator–Failure Unification Cert
        │
        │ BFS energy + SCC structure (Gate 4, Gate 5)
        ↓
[80] Energy Cert (reference_state, generator_set, energy_map)
```

## Four theorems

| ID | Claim | Gate |
|---|---|---|
| T1 | F(G) = {τ(g) \| g ∈ G} is finite (⊆ carrier of [76]) | 3 |
| T2 | SCC count and max SCC size are determined by G | 4 |
| T3 | For any path g₁...gₖ, the failure image = τ(g₁) ⊔ ... ⊔ τ(gₖ) | 4 |
| T4 | States reachable via OK-only generators have strictly lower energy than states requiring non-OK generators | 5 |

## Six gates

| Gate | What it checks |
|---|---|
| 0 | Cross-family ref guard: referenced [76]/[80] files exist, are tracked (when `.git` is present), and match expected sha256 |
| 1 | Schema validation + every `failure_tag` in `generator_tagging` ∈ referenced [76] carrier |
| 2 | `canonical_sha256` and `schema_sha256` digest recompute |
| 3 | T1: recompute F(G), verify `image_sha256` and F(G) ⊆ carrier |
| 4 | T2: recompute SCC count + max size + `scc_hash`; T3: spot-check sampled paths via [76] join table |
| 5 | T4: BFS from [80] `reference_state` — compute ok-only vs all-gen reachability, verify monotone energy separation |

## Current fixtures

| File | Result | Notes |
|---|---|---|
| `valid_caps_tr_fear_love.json` | PASS | CAPS_TR N=2, 6 generators (fear/love families), 1 SCC (9 states), F(G)={OK,PARITY_BLOCK,INVARIANT_VIOLATION} |
| `invalid_tag_not_in_carrier.json` | FAIL Gate 1 | `fear_up` tagged PANIC which is not in [76] carrier |

## Cross-family bindings (Option B)

The valid fixture binds to:

- **[76] reference**: `qa-failure-algebra-pass-tiny-001` (`canonical_sha256=2cb33c72...`)
  - Carrier: `{OK, PARITY_BLOCK, INVARIANT_VIOLATION, OUT_OF_DOMAIN}`
  - Poset: `OK ≤ PARITY_BLOCK, INVARIANT_VIOLATION ≤ OUT_OF_DOMAIN`
  - Join: least upper bound table
  - Unit: `OK`

- **[80] reference**: `energy_caps_tr_N2_mixed` (`canonical_sha256=5463f9b4...`)
  - Domain: CAPS_TR, N=2, reference_state={T:0,R:2}
  - Generator set: fear_up/down/lock, love_soothe/support/reframe

## Tagging function τ for CAPS_TR

| Generator | Family | τ(g) | Semantic |
|---|---|---|---|
| fear_up | fear | PARITY_BLOCK | Threat escalation → parity imbalance |
| fear_down | fear | OK | De-escalation is a valid/recovering move |
| fear_lock | fear | INVARIANT_VIOLATION | Regulation depletion breaks recovery invariant |
| love_soothe | love | OK | Coupled T↓R↑ is the canonical recovery move |
| love_support | love | OK | Pure regulation increase |
| love_reframe | love | OK | Threat reduction |

## T2 result: one SCC (all 9 states)

With N=2 and all 6 generators, the reachability graph is **strongly connected** — every state can reach every other state. This reflects that `love_reframe + fear_up` can cycle through the full grid. SCC count = 1, max_scc_size = 9.

This demonstrates the **SCC collapse** predicted by the [19] SCC monotonicity law: adding generators can only merge SCCs, never split them.

## T4 result: OK-reachable = {reference only}

From reference state {T:0,R:2}, using *only* OK-tagged generators (fear_down, love_soothe, love_support, love_reframe):
- Only 1 reachable state: the reference itself (energy=0)
- All 8 other states require at least one PARITY_BLOCK or INVARIANT_VIOLATION generator
- min_energy_non_ok = 1 > min_energy_ok = 0 ✓

This confirms: non-OK generators are the **only mechanism for state space escape** from the reference.

## How to run

```bash
python qa_generator_failure_unification_cert_v1/validator.py --self-test
python qa_generator_failure_unification_cert_v1/validator.py fixtures/valid_caps_tr_fear_love.json
```

## If CI fails

Gate 1: a generator's `failure_tag` is not in the [76] carrier — update tagging or the [76] reference.
Gate 3: F(G) changed (generator added/removed) — recompute `image_elements` and `image_sha256`.
Gate 4: SCC structure changed (N changed or generators changed) — recompute `scc_count`, `max_scc_size`, `scc_hash`.
Gate 5: Energy cert ref changed — update `energy_cert_ref.canonical_sha256` to match new [80] cert.
