# [260] QA Orbit-Dirac Bracket Cert

## What this is

Delivers the MC-4 construction of `docs/theory/QA_QFT_ETCR_CROSSMAP.md` §4.2 — an explicit QA-native Dirac-bracket construction on `S_9`. Backfills the ETCR / Dirac cross-map thread shipped in commit `aa2053c` (2026-04-20).

The cert certifies the **structural machinery**, not the mapping claim: constraint-function family, base bracket, X-matrix computation, invertibility on the physical subspace, instantiation of the orbit-Dirac bracket for a concrete observable pair, and the strong-zero property that parallels Blaschke-Gieres eq (5.38).

**Primary sources**:
- Blaschke, D.N., Gieres, F. *On the canonical formulation of gauge field theories and Poincaré transformations.* Nucl. Phys. B 965 (2021). arXiv:2004.14406. Equations (5.32)–(5.39) define the Dirac-bracket construction from second-class constraints.
- Mannheim, P.D. *Equivalence of light-front quantization and instant-time quantization.* Phys. Rev. D 102 025020 (2020). arXiv:1909.03548. Motivates the slice/path (observer/invariant) framing that the QA-side firewall reads as Theorem NT.

**Construction (m = 9)**. Constraint functions on `ℋ_raw = {1..9}²`:

```
phi_1(b, e) = b² - b·e - e² - 1       (Cassini +1 branch of I = 1)
phi_2(b, e) = b - 1                   (gauge-fix picking b = 1)
```

Base bracket: the symplectic lift of the tuple wedge (prior-art C3 recast),

```
[F, G] := (∂_b F)(∂_e G) − (∂_e F)(∂_b G),     {b, e} = 1.
```

X-matrix entries computed symbolically via `Z[b, e]` partial derivatives:

```
X_11 = 0            X_12 = b + 2·e
X_21 = −(b + 2·e)   X_22 = 0
det X = (b + 2·e)²
```

Orbit-Dirac bracket:

```
[F, G]_orbit := [F, G] − [F, phi_a] (X⁻¹)^{ab} [phi_b, G].
```

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_orbit_dirac_bracket_cert_v1/qa_orbit_dirac_bracket_cert_validate.py` |
| Pass fixture | `qa_orbit_dirac_bracket_cert_v1/fixtures/odb_pass_m9_b1_i1.json` |
| Fail fixture | `qa_orbit_dirac_bracket_cert_v1/fixtures/odb_fail_wrong_physical_subspace.json` |
| Mapping ref | `qa_orbit_dirac_bracket_cert_v1/mapping_protocol_ref.json` |
| Cross-map | `docs/theory/QA_QFT_ETCR_CROSSMAP.md` §4.2 |
| Prior-art audit | `docs/theory/QA_QFT_COMMUTATORS_PRIOR_ART.md` |
| Preliminary invariant | `docs/theory/empirical/etcr_t_invariant_check.py` |
| Paper section | `papers/in-progress/qft-etcr-orbit-quotient/section.md` §4.4 |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_orbit_dirac_bracket_cert_v1
python qa_orbit_dirac_bracket_cert_validate.py --self-test
```

## Semantics

- **ODB_1**: `schema_version` matches `QA_ORBIT_DIRAC_BRACKET_CERT.v1`; `modulus` is `9` in v1.
- **ODB_PHI**: `constraint_polynomials` declares two polynomials in `Z[b, e]`; validator parses declared coefficient terms and requires exact equality to the canonical `phi_1`, `phi_2` above.
- **ODB_WIT_A1**: every witness `.point` is in `{1..9}²` (A1 compliance — no zero residue).
- **ODB_WIT_PHYSICAL**: every witness satisfies `phi_1 ≡ 0 (mod 9)` and `phi_2 ≡ 0 (mod 9)`. This is the physical subspace definition.
- **ODB_X_MATRIX**: validator independently recomputes `X_{ab} = [phi_a, phi_b]` symbolically and evaluates mod 9 at each witness; declared `X_matrix_mod_m` must match.
- **ODB_INV**: at every witness, `det X mod 9` must be a unit — i.e. `gcd(det X, 9) = 1`, so `X⁻¹` exists via `pow(det_X, -1, 9)`.
- **ODB_DB_BE_ZERO**: `[b, e]_orbit ≡ 0 (mod 9)` at every witness. Parallels the Blaschke-Gieres observation that quotiented-bracket-of-two-gauge-redundant-variables vanishes.
- **ODB_STRONG_ZERO**: `[phi_a, F]_orbit ≡ 0 (mod 9)` for every `a ∈ {1, 2}` and every `F ∈ {b, e}` at every witness. Parallels Blaschke-Gieres eq (5.38) — constraint functions become strong invariants under the Dirac bracket.
- **ODB_SRC**: `source_attribution` references arXiv:2004.14406, arXiv:1909.03548, Blaschke, Mannheim, the cross-map doc, and cert [191].
- **ODB_WITNESS**: at least 2 distinct physical-subspace points witnessed.
- **ODB_F**: `fail_ledger` is a list.

## Relation to other certs

- **[191] `qa_tiered_reachability_theorem_cert`** — substrate for the orbit quotient; the physical subspace of cert-D lives on one Cosmos orbit at I=1 which is structurally bounded by [191]'s Level-I ceiling.
- **[234] `qa_chromogeometry_pythagorean_identity_cert_v1`** — complementary SYMMETRIC bilinear on `(b, e)` cross-pairs. Cert-D's tuple wedge is the ANTISYMMETRIC complement; together they span the full bilinear structure on pair space.
- **[256] `qa_orbit_resonance_attention_cert_v1`** — shares the orbit-invariance framework. Cert-D's Dirac-bracket construction is structurally consistent with the orbit-resonance attention approach: both respect T-orbit class as the physical quotient.
- **[214] `qa_norm_flip_signed_temporal_cert_v1`** — characterizes Eisenstein-norm orbit families on S_9. Cert-D's I = Cassini-squared invariant is a different orbit invariant that separates the Cosmos orbits; the two invariants crosscut.

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `ODB_PHI` | Declared constraint polynomial does not parse or does not match canonical `b² - b·e - e² - 1` / `b - 1`. | v1 scopes to these canonical `phi_1`, `phi_2`. A future v2 can extend with period-n constraints for m=24 or alternate Cassini branches. |
| `ODB_WIT_A1` | A witness `.point` falls outside `{1..m}²` — A1 violation (zero residue). | Choose witnesses in `{1..9}²` only. |
| `ODB_WIT_PHYSICAL` | A witness does not satisfy both constraints mod `m`. | Either move the witness onto the physical subspace or declare a different constraint family. |
| `ODB_X_MATRIX` | Declared `X_matrix_mod_m` contradicts symbolic recomputation at the witness. | Recompute `X_{ab} = [phi_a, phi_b]` and evaluate at the witness; the validator's evaluation is authoritative. |
| `ODB_INV` | `det X mod m` is not a unit at a witness — X is not invertible on the physical subspace at that point. | For m=9 this happens when `(b + 2e) mod 3 = 0`; in v1 that excludes the witness. A future v2 may use `phi_1 + phi_3 = period-n` constraints inside an I-level set to retain invertibility. |
| `ODB_DB_BE_ZERO` | `[b, e]_orbit ≠ 0` mod m at a witness. | Indicates an inconsistency in the Dirac-bracket construction; recheck the X-matrix and constraint derivatives. The value should be 0 identically on the physical subspace. |
| `ODB_STRONG_ZERO` | `[phi_a, F]_orbit ≠ 0` for some `a`, `F`. | Parallels a Blaschke-Gieres eq-(5.38) violation; the Dirac construction is inconsistent. |
| `ODB_WITNESS` | Fewer than two distinct witnesses on the physical subspace. | Provide at least two points satisfying both constraints mod m. |

## Scope boundary

**The cert does NOT:**
- Certify the mapping between QA and QFT Dirac-bracket machinery as identity — the cross-map is structural analogue, not derivation.
- Claim completeness on the full orbit family at m=9 — v1 covers only the +1 Cassini branch of the I=1 Cosmos orbit (two physical-subspace points in the witness set). The −1 branch and the other two Cosmos orbits (I=4, I=7) have analogous constructions but are not in v1.
- Cover m=24 — at m=24 the invariant I has 4 distinct values across 30 orbits, coarser than orbit separation; cert-D v2 will require a mixed family `I`-level plus period-n constraints inside each level set.
- Cover dynamics beyond T_F(b, e) = (a1(b + e), b) — the T-invariance of `I` depends on the Fibonacci-like step. Other QA step operators may need a different constraint family.
- Certify cert-C (the unequal-k CCR invariant of MC-1/MC-2); cert-C is a follow-up that depends on cert-D stability.

**The cert DOES:**
- Demonstrate that the Blaschke-Gieres Dirac-bracket machinery admits a clean discrete realization on a QA pair space with integer arithmetic throughout.
- Produce concrete integer witnesses at which `X`-matrix invertibility, `[b, e]_orbit = 0`, and strong-zero all hold exactly (not approximately) mod 9.
- Anchor the QA side of the ETCR / Dirac cross-map with an auditable artifact independent of the paper section draft.

This is the orbit-Dirac-bracket counterpart to [256] attention's orbit-invariance discipline, applied to bracket quotient structure rather than pairwise resonance.
