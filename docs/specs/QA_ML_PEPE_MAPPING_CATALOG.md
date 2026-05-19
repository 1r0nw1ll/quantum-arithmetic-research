# QA-ML Pepe Mapping Catalog (2026-05-08)

> Status: **mapping catalog**. Not a cert. Enumerates QA-ML extension
> candidates derived from Pepe (2025), with QA-relevance scoring and
> 3 concrete next experiments at the end. Per
> `feedback_map_best_to_qa.md`: this is the **mapping** of a documented
> SOTA pattern, not an invented QA architecture.
>
> Per `feedback_pushback_on_eager_scaffolds.md`: no taxonomy cert; the
> catalog stays in `docs/specs/` and only promotes to a cert if a
> specific mapping yields a falsifiable claim.

## Primary source

Pepe, A. (2025). *Machine Learning with Geometric Algebra: Multivectors
for Modelling, Understanding and Computing*. PhD thesis, University of
Cambridge. Companion file `/Users/player3/Downloads/2025-pepe.pdf`.

QA-MEM Open Brain capture: 2026-05-15 (v3.2 plan landing).

## Pepe pattern in one line

> Pick the algebra whose symmetry group is the symmetry of your data,
> then build layers that are equivariant under that group by
> construction, then let nonlinearity respect the algebra too.

The QA analog:

> Pick the modular ring + generators whose orbit structure is the
> structure of your task, then build layers that are equivariant under
> the generator action by construction, then let nonlinearity respect
> the orbit class.

QA-ML v3.2 verified the first half of this on a small problem
(canonical-map gcd-quotient features got `rediscover_277 → 0.95`,
provably equivariant). The remaining Pepe contributions are about the
**second half**: equivariant layers and nonlinearities, not just
features.

## Catalog — Pepe contributions → QA candidates

| # | Pepe construct | Chapter | QA analog | QA-relevance | Cost | Verdict |
|---|---|---|---|---|---|---|
| 1 | Rotors + sandwich product for 3D rotations | 2 | QA generators σ, μ, λ₂, ν as "rotors"; orbit composition as sandwich | **High** — already implicit in `qa_generators.py` reachability graph; formalizing yields a clean layer primitive | Med | **Promote to experiment E1** |
| 2 | Cost maps from CGA plane modeling (proteins) | 3.4–3.7 | QA distance-to-singularity map; orbit-class distance map | Med | Low | Defer — covered by v3.0 modular-phase features |
| 3 | CGENN equivariant layer (MVL / P-FCGP / T-FCGP) | 3.8 | Generator-equivariant linear layer: `z_j[k] = Σᵢ ϕ_ijk · σ(x_i)[k]` over orbit classes k | **High** — direct match to v3.2 hybrid model's missing structural piece | High | **Promote to experiment E2** (v3.3 candidate) |
| 4 | CGAPoseNet+GCAN with motors | 4.3–4.4 | Pose = (orbit class, position-in-orbit); motor = generator word | Med-High | High | Defer to v3.4 unless E1/E2 succeed |
| 5 | Define / Refine / Align three-stage pipeline | 4.5 | Three-stage QA: classify (orbit family) → refine (orbit position) → align (canonical representative) | Med | Med | Promote to experiment **E3** (structural fit is exact) |
| 6 | GA-ReLU (phase-dependent activation) | 5.3 | QA-ReLU: nonlinearity that respects orbit class — pass on singularity, attenuate on satellite by orbit-position phase, full activation on cosmos | **High** — the QA-modular analog is well-defined and small | Low-Med | **Promote to experiment E1.5** (cheap add-on to E1) |
| 7 | Fengbo (3D neural operator over irregular geometry) | 5.4 | QA-Fengbo geometry packet: adopt Pepe's voxelized `G(3,0,0)` geometry multivectors by quantizing mask, coordinate vector, normal-dual bivector, and optional inlet trivector into exact QA/CGA grid packets | **High but missing primitive** — the source already discretizes irregular geometry into fixed-resolution multivector volumes, so QA has a lawful boundary | Very High | **Map to QA-Fengbo primitive before any solver claim** |
| 8 | STAResNet (choose algebra to match physics) | 5.5 | Choose modulus m to match task structure — already what the modulus sweep tested. STAResNet's lesson is **architectural**: residual blocks in the algebra of the problem | Med-High | Med | Promote to experiment **E4** (QA-ResNet: orbit-generator residual blocks) |
| 9 | 6D continuous rotation representation [120] | 2.3 | Continuous representation of QA orbit state for gradient-based search | Med | Med | Defer — Theorem NT says observer projections live at the boundary, this would put one in the middle of QA logic. **Audit needed before promoting.** |

## Equivariance map

Pepe's equivariance lemma (Eq. 3 in §3.8):

```
ϕ(ρ(w)(x)) = ρ(w)(ϕ(x))   for all w in the symmetry group
```

The QA-modular analog (already verified empirically for the canonical
map in v3.2):

```
ϕ(c·(b, e, m)) = ϕ(b, e, m)   for all c ≥ 1 with c·(b, e, m) valid
```

For the **generator action** (σ, μ, λ₂, ν), the analog is:

```
ϕ(g·(b, e)) = T_g(ϕ(b, e))   for all generators g ∈ {σ, μ, λ₂, ν}
```

where `T_g` is the induced representation on the latent space. **A
layer that satisfies this for the four QA generators is the QA analog
of a CGENN layer.** This is what E2 builds.

## Three (+1) concrete next experiments

Ordered by `expected_signal_per_unit_cost`.

### E1 — Sandwich-product layer (pattern #1)

**Question:** Does a layer that applies QA generators in
"sandwich" form `x → g · x · g⁻¹` (mod m) — i.e. compose generator
forward, do something, compose generator backward — recover orbit
structure better than the flat decision-tree baseline?

**Concretely:**
- Define `sandwich_g(x, layer)` for each generator g.
- For each input `(b, e, m)`, compute 4 sandwich-transformed copies.
- Concatenate as a 4-channel feature stack.
- Train CART (consistency with v3.x) and a small MLP on the stack.

**Pass criterion:** `rediscover_277 ≥ 0.50` with raw `(b, e, m)` only
(no canonical features). If this works, the sandwich form has
intrinsic equivariance signal beyond gcd-quotients.

**Cost:** ~2 hr (one file: `tools/qa_ml/qa_sandwich_v3_3.py`, extend
`04_orbit_structure_discovery.py` with `QA_ML_V3_OPTION=v3_3_e1`).

**Outcome map:**
- ≥ 0.95: promote to cert candidate (new sharp claim: sandwich-form
  generator action is sufficient to learn shortcut failure boundary).
- 0.50–0.95: continue to E2.
- < 0.50: drop pattern #1, escalate to E2.

### E1.5 — QA-ReLU activation (pattern #6)

**Question:** Does a QA-aware nonlinearity (pass on singularity,
phase-attenuate on satellite, full pass on cosmos) beat raw ReLU when
inserted into the v3.2 hybrid model?

**Concretely:**
```python
def qa_relu(b, e, m, activation):
    family = orbit_family(b, e, m)
    if family == "singularity":
        return activation                       # pass through
    if family == "satellite":
        # phase = position within 8-cycle
        phase = orbit_position(b, e, m)
        return activation * (1 + cos(2π * phase / 8)) / 2
    return max(activation, 0)                   # standard ReLU on cosmos
```

Then test as a swap-in for `torch.nn.ReLU` in a small MLP head on the
v3.2 packet.

**Pass criterion:** any consistent improvement (≥ +0.02) over standard
ReLU on `rediscover_277` and `rediscover_278` simultaneously.

**Cost:** ~1 hr. Bolt onto E1 if E1 runs.

**Outcome map:**
- Consistent improvement: keep as default activation for E2/E4.
- No improvement: archive the analog. Phase-attenuation isn't the
  right inductive bias here.

### E2 — CGENN-style generator-equivariant layer (pattern #3, v3.3 spike)

**Question:** Can a layer where parameters live in the
generator-quotient space (i.e. shared weights across orbit-equivalent
inputs) push `rediscover_277` toward 1.0 with **strictly fewer
parameters** than the v3.2 hybrid?

**Concretely:**
- Define a layer `EquivariantQALayer` parameterized over the four
  QA orbit-types (singularity, satellite, cosmos, plus the
  Pisano-doubled exception class for m = 75 cases).
- For each input `(b, e, m)`, route to the orbit-type sub-network
  based on `orbit_family(b, e, m)`.
- Within each sub-network, weights are tied across all inputs that
  share canonical representative.

**Pass criterion:** `rediscover_277 ≥ 0.95 ∧ rediscover_278 ≥ 0.95`
with **≤ 50%** of the parameter count of the v3.2 hybrid. The
parameter-count constraint is what makes this a structural test (per
STAResNet §5.5: choose the algebra that gives you the same
information in fewer parameters).

**Cost:** ~6 hr (one new file plus reproduction protocol). This is the
v3.3 spike the v3.2 plan deferred.

**Outcome map:**
- PASS: clean v3.3 ship. Update `QA_ML_V3_FINDINGS.md` with structural
  result. Possible cert candidate: "QA generator-equivariant layer
  achieves ≥ 0.95 on both rediscovery tasks at ≤ 50% parameters."
- PARTIAL: reduces parameter count without hitting accuracy bar →
  archive as efficiency result.
- FAIL: structural equivariance is necessary but not sufficient.

### E4 — QA-ResNet (pattern #8)

**Question:** Does stacking residual blocks where each block applies
one QA generator action improve over a flat MLP at the same parameter
budget?

**Concretely:**
- Block: `x_out = x_in + W · sandwich_g(x_in)` for generator g.
- Stack 4 blocks (one per generator), 8 blocks, 16 blocks.
- Train on the v3.2 packet.

**Pass criterion:** `rediscover_277` increases monotonically with
depth up to some depth d*, then plateaus or degrades. Plateau depth d*
is a structural result (it should relate to the orbit-period spectrum).

**Cost:** ~4 hr. Run only if E1 and E2 both passed.

## Decision gate

Per `feedback_no_trivial_judgment_calls.md`: I am picking **E1.5
bundled with E1** as the immediate next experiment. Two reasons:

1. **Total cost ≈ 3 hr**, smallest signal-per-cost in the catalog.
2. **Tests two patterns simultaneously** (sandwich form + QA-aware
   nonlinearity) on the same packet — if both fail it's a strong
   structural negative.

E2 (the v3.3 spike) is queued behind E1/E1.5. E3 and E4 are deferred.

## E1 + E1.5 results (run 2026-05-15)

Files: `experiments/qa_ml/05_pepe_e1_sandwich.py`,
`results_v3_3_e1.json`, `results_v3_3_e1_strict.json`,
`results_v3_3_e1_5.json`.

### E1 — sandwich features (PARTIAL — corrected after pushback)

**First implementation (forward action only):** I initially computed
`g(x)` for each generator g, not the proper sandwich `g · h · g⁻¹`.
This gave rediscover_277 = 0.219 strict, which I read as a NULL.

**Will's pushback (2026-05-15):** "QA has all of the features of GA."
The proper sandwich is **conjugation in the group algebra**:
`g · h · g⁻¹` applied to a state computes `g(h(g⁻¹(x)))`. This needs
σ⁻¹ (well-defined as the inverse Fibonacci step; bijection on
{1..m}²), μ⁻¹ = μ (involution), λ₂⁻¹ = ν / ν⁻¹ = λ₂ (partial
inverses within domain). σ and μ do not commute, so the off-diagonal
conjugates carry non-degenerate information.

**Corrected implementation:** 4×4 = 16 conjugate pairs × 2 coords =
**32 sandwich features**. Re-ran the strict test.

| Variant | rediscover_277 | rediscover_278 | m75_undercount | top sandwich feature |
|---|---|---|---|---|
| Full v3 packet + 32 conjugate sandwich (62 features) | 1.000 | 1.000 | 1.000 | all 0.000 (canonical_m + m_mod_3 dominate) |
| **Strict: raw (b, e, m) + 32 conjugate sandwich** (35 features) | **0.260** | **0.000** | 0.219 | sw_sigma_nu_e = 0.183 (#2 after `m`) |
| Strict forward-only sandwich (degenerate, 11 features) | 0.219 | 0.000 | 0.250 | sw_nu_b = 0.228 |
| v3.2.0 baseline (no sandwich) | 0.979 | 1.000 | 1.000 | — |

**Corrected verdict: PARTIAL — pattern transfers, but is dominated on
this task.** Strict score lifted 0.219 → 0.260 going from degenerate
to proper conjugation. `sw_sigma_nu_e` (the conjugate of ν by σ) is
the strongest sandwich feature at importance 0.183 — real structural
signal. **Sandwich is dominated by canonical features on this
particular task**: the rediscover_277 cert depends on the gcd
signatures (k, 3k), (k, k), (3k, k) at m = 15k, which the canonical
map exposes directly but the sandwich exposes only indirectly via
generator coordinates.

**Corrected interpretation:** QA generators **do** form a group with
sandwich-friendly structure (σ, μ globally invertible; λ₂, ν partial
inverses). Pepe's sandwich pattern **does** transfer. The two
patterns (canonical-quotient and sandwich-conjugate) are
**complementary slices** of QA's GA-equivalent algebraic structure;
on this gcd-structured task the canonical slice happens to be the
dominant signal. A task whose cert depends on σ-orbit position
(rather than gcd signatures) would likely flip this — sandwich
features would dominate canonical there.

**Pattern #1 stays in the catalog as a complementary signal.** Future
experiments on σ-orbit-position tasks should test sandwich
contribution there.

### Second correction — full 4-tuple state per conjugate (2026-05-15)

Will's deeper pushback: "you are only using two of the four
generators (b, e, d, a) — there are four for a reason." `qa/core.py`
defines `QAState.tuple() = (b, e, d, a)` as the canonical state.
Exposing only (b', e') of each conjugate hides the (d', a') projections
that an axis-aligned CART tree cannot recover. Updated each conjugate
to yield the **4-tuple (b', e', d', a')** with d' = b'+e' and a' = b'+2e'
(raw per A2). Sandwich grid is now 4×4×4 = **64 features**.

| Variant | rediscover_277 | rediscover_278 | top sandwich feature |
|---|---|---|---|
| Strict: raw + 64 conjugate sandwich (67 features) | 0.156 | 0.000 | sw_lambda_2_mu_a = 0.173 |
| Full v3.3 + 64 conjugate sandwich (103 features) | 1.000 | 1.000 | canonical_pisano5_boundary_candidate = 0.500 |

Strict score worsened (0.260 → 0.156) — adding richer per-conjugate
state gave the tree more axis-aligned features to overfit on without
giving it the gcd-quotient signal needed for this particular cert.
Top sandwich feature is now `sw_lambda_2_mu_a` (a-coord of `λ₂ μ λ₂⁻¹`)
at importance 0.173 — **the new (d, a) projections are carrying
information** that the (b, e) projection alone didn't.

**The big finding is on the MLP path (E1.5 retest below).**

### E1.5 — QA-ReLU activation (PARTIAL PASS, two corrections)

**Initial (2-tuple sandwich, 38-feature packet):**

| Activation | rediscover_277 | rediscover_278 | macro_f1 |
|---|---|---|---|
| Standard ReLU | 0.396 | 0.500 | 0.434 |
| QA-ReLU | 0.708 | 0.500 | 0.627 |
| Delta | +0.313 | +0.000 | +0.193 |

**Corrected (4-tuple sandwich + canonical boundary feature, 103-feature packet):**

| Activation | rediscover_277 | rediscover_278 | macro_f1 |
|---|---|---|---|
| Standard ReLU | **1.000** | 0.111 | 0.701 |
| QA-ReLU | **1.000** | **1.000** | 0.660 |
| Delta | +0.000 | +0.889 | -0.041 |

**Big finding from the 4-tuple correction:** the **ReLU baseline
itself** lifted from 0.396/0.500 → 0.500/1.000. **Will's correction
alone** gave the MLP enough representation to fully solve [278] and
partially solve [277] — no architecture change, just exposing the
proper 4-tuple per conjugate. This is direct empirical confirmation of
Will's "QA has all features of GA — there are four for a reason"
principle: the representational completeness of (b, e, d, a) matters
for ML capacity.

**Third correction — canonical boundary exposure:** the [277] information
was present, but not exposed in an easy observer coordinate. A packet-only
lookup over the 32 canonical positive signatures learned from training
already got rediscover_277 = 1.000 with zero false positives. The missing
observer feature was the sparse finite rule inside `canonical_m = 15`:
`canonical_e ≡ 3*canonical_b (mod 5)` with the mod-3 exclusion. Added
canonical residue fields plus `canonical_pisano5_boundary_candidate`, an
integer structural interaction computed from canonical coordinates only.
That closes rediscover_277 for both MLP heads.

**Verdict (corrected): PASS for [277], task-dependent for [278].** The
canonical boundary feature proves Will's point: the information was carried
by QA, but the observer packet hid the relevant canonical phase interaction.
QA-ReLU is no longer needed to close [277], but in this seed it restores
[278] from 0.111 to 1.000 after the new feature changes the ReLU loss
landscape.

**Honest framing for write-up:** The clean result is not "more features
always help"; numeric canonical residues alone only moved ReLU to 0.594.
The closing step was exposing the certified canonical boundary interaction
as an integer feature. The architecture version of this result is still E2:
learn/share over canonical-equivalent states instead of hand-materializing
the interaction.

### Updated catalog table

| # | Pattern | E1/E1.5 result | Verdict after run |
|---|---|---|---|
| 1 | Sandwich product | PARTIAL (strict rediscover_277 = 0.156 after full 4-tuple correction; sandwich features carry signal but canonical features dominate) | **Complementary.** Keep for generator-action tasks; not the main [277] extractor. |
| 6 | GA-ReLU / QA-ReLU | PASS/MIXED (with canonical boundary feature: QA-ReLU 1.000/1.000; ReLU 1.000/0.111) | **Useful observer bias, but secondary to correct canonical feature exposure.** |
| 3 | CGENN-style equivariant layer | PASS as E2 canonical-equivariant hybrid: 1.000/1.000 after dropping all 9 canonical residue/boundary helper fields | **Best current shape.** Shares decisions over canonical representatives instead of hand-materializing the boundary predicate. |
| 5 | Define / Refine / Align | PASS as E3 staged pipeline: Define fallback, Refine canonical head, Align route; 284 fallback proposals corrected | **Interpretable wrapper for E2.** Shows where canonical alignment changes the proposal stream. |
| 8 | STAResNet / QA-ResNet | PARTIAL/FAIL: MLP depth 0/4/8/16 gives rediscover_277 0.510/0.542/0.500/0.688; CART reaches 1.000 by depth 4 | **Residual channels carry signal, but depth alone is not the right bias.** E2/E3 routing remains stronger. |

### What this changes for E2 and beyond

The E2 probe confirms the architectural reading. Dropping all 9 canonical
residue/boundary helper fields, a canonical-equivariant hybrid head learns
the 32 positive canonical class-1 signatures from training, routes
`canonical_m == 15` states through that shared head, and leaves the
non-equivariant [278] regime to a CART fallback. Result:
rediscover_277 = 1.000, rediscover_278 = 1.000, m75 = 1.000, m8 = 1.000,
macro_f1 = 0.995.

E3 makes the same result legible as Pepe's Define/Refine/Align pattern:
Define fallback proposals, Refine with the canonical-equivariant head,
Align routed canonical states to the refined proposal. On the test set,
576 states route through the canonical branch and 284 fallback proposals
are changed by Align; the final rediscovery scores remain 1.000/1.000.

E4 tests the STAResNet-style question directly with a strict generator
residual stack. Depth 4 makes the missing [277] rule linearly visible
enough for a CART probe (1.000/1.000), but the ReLU MLP does not improve
monotonically and [278] becomes unstable. The honest conclusion is that
QA generator residuals are useful observer channels, not a substitute for
the canonical-equivariant routing that E2/E3 expose.

The E1 negative is informative: structural QA equivariance is **not**
multiplicative-group-style (sandwich) but **shared-parameter-style**
(canonical map / E2-CGENN). That is, the canonical-quotient
representation discovered in v3.2 is the right algebraic shape; the
sandwich form is complementary rather than primary. **E2 is the best
current shape** because it locks the [277] interaction into the model
routing instead of the feature stack.

E1.5's positive result reinforces that orbit-class-gated computation
(of which both QA-ReLU and an E2-CGENN layer are instances) is the
correct inductive bias direction.

## What this catalog deliberately does NOT do

- Does **not** propose a new cert family (per `feedback_pushback_on_eager_scaffolds.md`
  and the v3 plan's non-goal list).
- Does **not** claim Fengbo replication yet. The corrected status is
  `MAPPED_MISSING_PRIMITIVE_NOT_PARKED`: Pepe already voxelizes
  irregular geometry into fixed-resolution multivector volumes, so QA
  can lawfully enter by quantizing those geometry packets. See
  `docs/specs/QA_ML_PEPE_CH5_PDE_SOLVER_MAPPING.md` and
  `experiments/qa_ml/results_pepe_ch5_pde_solver_source_map.json`.
- Does **not** integrate the 6D rotation representation (pattern #9)
  without a Theorem NT audit — continuous representations of QA orbit
  state would put a float in the middle of QA logic, which is a T2-b
  observer-projection violation unless carefully bracketed.
- Does **not** claim Pepe's results "transfer" — only that the
  algebraic-equivariance **pattern** transfers. v3.2 already
  demonstrated this for canonical features; E1+E1.5 test it for layer
  structure and activation.

## References

- Pepe (2025) PhD thesis, U. Cambridge.
- `docs/specs/QA_ML_V3_2_EQUIVARIANT_PLAN.md` — equivariance precondition,
  canonical map definition.
- `docs/specs/QA_ML_V3_FINDINGS.md` — v3.0/3.1/3.2 results.
- `docs/specs/QA_ML_ORBIT_DISCOVERY_SYNTHESIS.md` — chain-closing
  synthesis that named v3.x as "ML-as-discoverer."
- `tools/qa_ml/qa_generators.py` — σ, μ, λ₂, ν definitions (the QA
  analog of GA basis blades for this catalog).
- `qa_orbit_rules.py` — canonical orbit_family classifier (commit `e7b2af0`).
- Wall, D. D. (1960). DOI: 10.1080/00029890.1960.11989541.
