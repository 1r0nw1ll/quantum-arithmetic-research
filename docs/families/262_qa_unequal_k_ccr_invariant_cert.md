# [261] QA Unequal-k CCR Invariant Cert

## What this is

Delivers MC-1 and MC-2 of `docs/theory/QA_QFT_ETCR_CROSSMAP.md` §4.1 — an explicit QA-native unequal-`k` propagator on `S_9`. Companion to [260] `qa_orbit_dirac_bracket_cert_v1` which delivers MC-3 / MC-4 (Blaschke-Gieres side). Together the two certs close out the ETCR / Dirac cross-map thread shipped in commit `aa2053c`.

**Primary source**:
- Mannheim, P.D. *Equivalence of light-front quantization and instant-time quantization.* Phys. Rev. D 102 025020 (2020). arXiv:1909.03548. Equations (2.2) + (8.9) / (8.16) — the unequal-time commutator as a c-number invariant and the Lehmann spectral representation.

**Construction**:

```
i_Δ_QA(Δk; (b, e), (b', e')) := 1 if T^|Δk|(b, e) = (b', e') else 0
```

The integer-valued T-orbit trajectory indicator. Depends on `Δk` only — slice-independent. Integer-valued throughout — Theorem NT compliant. The QA-layer analog of Mannheim's c-number `i∆(x − y)` eq (2.2).

**Equal-k limit** (the MC-1 equivalence):

```
i_Δ_QA(0; (b, e), (b', e')) = δ_{(b, e), (b', e')}
```

which recovers the stipulated observer-side CCR `[â_{b, e}, â†_{b', e'}] = δ_{b, b'} δ_{e, e'}` exactly. Note: this CCR is a **proposed observer-side canonical encoding**, not a theorem derived from QA orbit dynamics. The cert certifies the unequal-`k` invariant and its internal consistency — nothing more.

**Lehmann-type trace formula**:

```
Tr i_Δ_QA(Δk) = Σ_O |O| · 1[period(O) divides Δk]
              = 72 · 1[24 | Δk] + 8 · 1[8 | Δk] + 1             for m = 9, T_F
```

Each orbit class contributes its density `|O|` weighted by the indicator that `Δk` is a multiple of its period. The Singularity contributes 1 at every `Δk` (period 1 divides everything); the Satellite adds 8 when 8 divides `Δk`; the Cosmos adds 72 when 24 divides `Δk`. This is QA's discrete Lehmann representation — the analog of Mannheim's `ρ(σ²)` spectral density, with orbit period playing the role of mass squared.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_unequal_k_ccr_invariant_cert_v1/qa_unequal_k_ccr_invariant_cert_validate.py` |
| Pass fixture | `qa_unequal_k_ccr_invariant_cert_v1/fixtures/ukc_pass_m9_three_orbits.json` |
| Fail fixture | `qa_unequal_k_ccr_invariant_cert_v1/fixtures/ukc_fail_wrong_lehmann.json` |
| Mapping ref | `qa_unequal_k_ccr_invariant_cert_v1/mapping_protocol_ref.json` |
| Cross-map | `docs/theory/QA_QFT_ETCR_CROSSMAP.md` §4.1 |
| Prior-art audit | `docs/theory/QA_QFT_COMMUTATORS_PRIOR_ART.md` §5a |
| Paper section | `papers/in-progress/qft-etcr-orbit-quotient/section.md` §3, §6 |
| Companion cert | `qa_orbit_dirac_bracket_cert_v1/` [260] |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_unequal_k_ccr_invariant_cert_v1
python qa_unequal_k_ccr_invariant_cert_validate.py --self-test
```

## Semantics

- **UKC_1**: `schema_version` matches `QA_UNEQUAL_K_CCR_INVARIANT_CERT.v1`; `modulus` is `9` in v1.
- **UKC_A1**: every `witnesses[].start_point` is in `{1..9}²`; `orbit_class` is in `{Cosmos, Satellite, Singularity}`.
- **UKC_WITNESS**: the witness list covers all three orbit classes at least once.
- **UKC_EQ_LIMIT**: exhaustively on `{1..9}² × {1..9}²` (6561 pairs), `propagator(0; w, w') = δ_{w, w'}`. The equal-k limit recovers the stipulated delta-function CCR exactly.
- **UKC_ORBIT_DECOMP**: for every pair of orbit representatives in distinct T-orbits, `propagator(Δk; w, w') = 0` at every `Δk ∈ [0, max_period]`. Combined with periodicity this gives 0 at all `Δk`.
- **UKC_PERIODICITY**: for each witness `w` with orbit period `P(w)`, `propagator(0; w, w') = propagator(P(w); w, w')` for every target `w'`. Additional spot-check at `Δk + P(w)` for `Δk ∈ {1, 2, P(w)/2}`.
- **UKC_LEHMANN**: at `Δk ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16, 24}`, the observed trace `Tr propagator(Δk) = #{w : T^Δk(w) = w}` equals the Lehmann prediction `Σ_O |O| · 1[period(O) | Δk]`. The sample mixes coprime values (1, 2, 3, 4), divisors of 8 (2, 4, 8), divisors of 24 (2, 3, 4, 6, 8, 12, 24), and non-divisors (16 is divisible by 8 but not 24 — exercises the mid-period separation).
- **UKC_SRC**: `source_attribution` references arXiv:1909.03548, Mannheim, the cross-map, and the companion cert [260].
- **UKC_F**: `fail_ledger` is a list.

## Relation to other certs

- **[260] `qa_orbit_dirac_bracket_cert_v1`** — companion cert. Cert-D delivers the Blaschke-Gieres side (Dirac bracket, orbit quotient); cert-C delivers the Mannheim side (unequal-time propagator, Lehmann representation). The two certs together complete the ETCR / Dirac cross-map.
- **[191] `qa_tiered_reachability_theorem_cert`** — substrate. The orbit classifier is shared across cert-C and cert-D; the Tiered Reachability Theorem structurally constrains which orbits exist on S_9.
- **[214] `qa_norm_flip_signed_temporal_cert_v1`** — orbit-family classification. The witness-per-orbit-class pattern in cert-C (Fibonacci/Tribonacci/Ninbonacci representatives) lines up with the five-family classification in [214].
- **[256] `qa_orbit_resonance_attention_cert_v1`** — shares the orbit-invariance discipline. Cert-C's propagator is orbit-invariant by construction (cross-orbit values vanish); cert-256's attention rules are orbit-invariant under T evolution. Same structural principle applied to different operators.

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `UKC_1` | `schema_version` or `modulus` mismatch. | Set `schema_version` to `QA_UNEQUAL_K_CCR_INVARIANT_CERT.v1` and `modulus` to `9` in v1. |
| `UKC_A1` | A witness `.start_point` is not in `{1..9}²`, or `.orbit_class` is not in the allowed set. | Use integer pairs and one of `Cosmos`, `Satellite`, `Singularity`. |
| `UKC_WITNESS` | Fewer than 3 witnesses, or the 3 orbit classes are not all represented. | Provide one witness per class; see the PASS fixture for canonical choices `(1, 1)`, `(3, 3)`, `(9, 9)`. |
| `UKC_EQ_LIMIT` | The recomputed equal-k propagator value at some `(w, w')` disagrees with the δ-function. | This is a structural invariant — a failure indicates a validator bug or corrupted T_F implementation. |
| `UKC_ORBIT_DECOMP` | A nonzero cross-orbit propagator value. | Should never happen for a correct T_F implementation; investigate the QA step function. |
| `UKC_PERIODICITY` | `propagator(Δk)` differs from `propagator(Δk + period)` for some witness and target. | Same — structural invariant; a failure indicates a bug. |
| `UKC_LEHMANN` | Declared trace disagrees with the recomputed trace, OR the Lehmann prediction disagrees with the observed trace. | Either correct the declared values, or (if the Lehmann formula itself is wrong) re-derive; for m=9 the formula is `72*1[24|Δk] + 8*1[8|Δk] + 1`. |
| `UKC_SRC` | `source_attribution` is missing a required primary-source or cross-reference string. | Add the missing citation. |

## Scope boundary

**The cert does NOT:**
- Derive the equal-k CCR from QA primitives. The `[â, â†] = δ` encoding is stipulated observer-side and only the unequal-k invariant is certified. This is the cross-map §4.1 precision guardrail, explicit.
- Cover m=24 — the Lehmann trace formula generalizes but has more terms; on S_24 there are 30 orbits across periods `{1, 3, 6, 8, 12, 24}`, and the trace decomposition is `Σ_O |O| · 1[period(O) | Δk]` with six contributing period classes.
- Cover alternative step operators (Lucas, Tribonacci, etc.). Each defines its own orbit structure and propagator. Slice-independence across T-operator choice is a deeper invariant not addressed in v1.
- Handle the interacting case. Mannheim's Lehmann representation (eqs 8.9 / 8.16) has `ρ(σ²)` as a continuous integral over masses; the QA free-case analog is the discrete orbit-period decomposition. An interacting QA analog would introduce a non-trivial "spectrum" beyond the orbit-period delta functions — out of scope for v1.

**The cert DOES:**
- Provide a concrete, integer-valued, slice-independent unequal-k commutator object whose equal-k limit is exactly the stipulated CCR.
- Verify the orbit-quotient decomposition exhaustively: propagation never crosses orbit boundaries.
- Establish the discrete Lehmann representation in QA vocabulary: each orbit class contributes its density to the trace whenever `Δk` is a multiple of its period.
- Close MC-1 / MC-2 of the cross-map, complementing cert-D's MC-3 / MC-4.
