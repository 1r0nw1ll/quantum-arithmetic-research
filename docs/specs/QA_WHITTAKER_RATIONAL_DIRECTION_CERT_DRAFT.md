# QA Whittaker Rational Direction `S¹` Cert — DRAFT (Layer 1)

**Primary source**: E. T. Whittaker, "On the partial differential equations of mathematical physics," *Math. Annalen* **57**:333–355, 1903. DOI 10.1007/BF01444290.

**Status**: DRAFT used for build prep; Layer 1 registered as cert family **[266]** on 2026-05-01 after standalone tests and hostile review. Slug: `qa_whittaker_rational_direction_s1_cert_v1`.

**Layer position**: This is **Layer 1** of a six-layer Whittaker → QA development ladder; see §8.

**Authored**: Will Dale + Claude. Refined 2026-04-29; build-prep adjustments 2026-04-30 per Will's second review (W3 full-Q1 closure with virtual endpoints, W2 all-pairs validation).

**What this cert is.** A QA-side discretization claim. The set `D_m` of QA-rational propagation directions on `S¹` is a finite, exactly-enumerable rational sample-net, and a Lipschitz angular profile sampled at this net has bounded reconstruction error in terms of the net's max angular gap.

**What this cert is *not*.** It does **not** prove Whittaker's wave-equation theorem, Maxwell's equations, electromagnetism, scalar-potential reductions, or any physics. Whittaker 1903 is the **motivation** for studying rational angular nets; the cert certifies only the QA-side discretization layer.

---

## 1. Anchor

The directional plane-wave kernel of Whittaker 1903, §3, decomposes general wave solutions over angular directions on the sphere. Restricted to the (x,z)-plane, the kernel uses a single polar angle `u ∈ [0, 2π)` with `(sin u, cos u) ∈ S¹`.

QA replaces the **continuous angular sweep** with a **discrete rational angular net** generated from QA seeds. The cert is about that net.

**On disk.** `Documents/whittaker_corpus/whittaker_1903_partial_differential_equations_mathematical_physics.pdf` (1.30 MB, Zenodo mirror).

**1904 paper.** E. T. Whittaker, "On an Expression of the Electromagnetic Field due to Electrons by Means of Two Scalar Potential Functions," PLMS s2-1:367–372, DOI 10.1112/plms/s2-1.1.367 — **bridge-spec background only**, not in cert v1. The 1904 reduction is a gauge move (eliminating the vector potential), not a direction-indexing move; mapping it requires a separate companion cert.

---

## 2. The QA-rational direction net `D_m`

### Construction (canonical QA definitions, raw)

For seed `(b, e) ∈ {1, ..., m}²` with `gcd(b, e) = 1`:

```
d := b + e
a := b + 2e
C := 2 · d · e
F := a · b              ( = d² − e² )
G := d² + e²
```

QA-rational direction:

```
(C/G, F/G) ∈ ℚ² ∩ S¹ ∩ ℝ²_{>0}
```

The full direction set:

```
D_m := { (C/G, F/G) : (b,e) ∈ {1,..,m}², gcd(b,e) = 1 }
```

`D_m` lies in the open first quadrant of `S¹`. The other three quadrants are obtained by trivial sign reflections `(±C/G, ±F/G)` and are not part of v1.

### Identities (verified by validator at construction)

- `F = a · b = d² − e²` ✓
- `C² + F² = G²` ✓ (Pythagorean closure)
- `(b, e, d, a)` integer; `(C, F, G)` integer; HARD rule preserved (no `% m` in element computation).

### Arithmetic discipline

- All construction in **exact integer arithmetic**.
- Angular order in `D_m` determined by **exact rational comparison** of `tan θ = F/C` via `fractions.Fraction(F, C)` — never by float.
- Floats appear **only** in observer-side reporting (printed gap values, plot coordinates, Lipschitz error display).

---

## 3. Cert claims

### Gate W1 — Cardinality is the count of coprime seeds (exact)

The map `(b, e) → (C/G, F/G)` is injective on coprime seeds in the `m × m` grid:

```
|D_m| = |{(b, e) ∈ {1..m}² : gcd(b, e) = 1}|
```

Bit-exact predictions (verified):

| m  | `|D_m|` |
|----|---------|
| 9  | 55      |
| 24 | 359     |
| 72 | 3,175   |

(Note: `(b, e)` and `(e, b)` give *different* directions and are both counted. E.g., `(1,2) → (12/13, 5/13)` and `(2,1) → (3/5, 4/5)` — distinct points.)

**Validator check.** Independent re-enumeration; assert `|D_m| == declared_count` exactly.

---

### Gate W2 — Angular separation lower bound (rigorous theorem, all-pairs validation)

**Theorem (all distinct pairs).** For any two distinct directions `(C_i/G_i, F_i/G_i), (C_j/G_j, F_j/G_j) ∈ D_m` (`i ≠ j`), the angular separation `Δθ_{ij} ∈ (0, π/2]` satisfies

```
sin(Δθ_{ij}) = |C_i·F_j − C_j·F_i| / (G_i·G_j) ≥ 1 / (G_i·G_j) ≥ 1 / G_max(m)²
```

and consequently `Δθ_{ij} ≥ 1 / G_max(m)²` (using `arcsin(x) ≥ x` for `x ∈ [0,1]`).

**Proof.** The cross product `|C_i·F_j − C_j·F_i|` is a non-negative integer. It vanishes iff the two unit vectors are parallel, i.e., equal in the open first quadrant — but the directions are distinct. Hence the cross product is a positive integer, so `|C_i·F_j − C_j·F_i| ≥ 1`. The standard 2D identity `|v_i × v_j| = |v_i| · |v_j| · sin(angle)` then gives `sin(Δθ_{ij}) ≥ 1/(G_i·G_j)`. The inequality `arcsin(x) ≥ x` for `x ∈ [0,1]` is elementary. ∎

**Validator check.** **All-pairs** integer test: for every `i < j` in `D_m`, assert `|C_i·F_j − C_j·F_i| ≥ 1`. Pair count is `n(n−1)/2`: 1485 at `m=9`, 64,261 at `m=24`, 5,038,725 at `m=72` — pure integer arithmetic, fits in Python at all three sizes. The consecutive-pairs subset is implicitly covered. The bound `1/G_max(m)²` is empirically loose by ~4 orders of magnitude, so the theorem bound is robust to numerical edge cases.

---

### Gate W3 — Lipschitz nearest-neighbor sampling error (main theorem gate, full-Q1 closure)

**Setup.** Define **virtual boundary anchors**:

```
E_0 := (1, 0)       at θ = 0
E_∞ := (0, 1)       at θ = π/2
```

These are **observer-side boundary anchors only** — they are NOT QA seeds, NOT counted in `|D_m|`, NOT included in W1 or W2, and NOT QA tuples. They exist solely to close the angular test interval for W3.

Define the **closed direction set**:

```
D_m⁺ := D_m ∪ {E_0, E_∞}
```

`D_m⁺` sorted by angle gives `0 = θ̃_0 < θ̃_1 < ... < θ̃_{n+1} = π/2`. Define the **closed max angular gap**:

```
Δ_max⁺(m) := max_i (θ̃_{i+1} − θ̃_i)
```

computed via exact rational `sin(Δθ)` for interior gaps, and `θ̃_1 − 0` and `π/2 − θ̃_n` for the two boundary gaps (the only float operation in `Δ_max⁺` computation), converted to float for the W3 bound check.

**Claim.** For any `L`-Lipschitz function `g : [0, π/2] → ℝ` (Lipschitz in the angular metric), nearest-neighbor sampling at points of `D_m⁺` yields the bound

```
sup_{θ ∈ [0, π/2]}  |g(θ) − g(θ_NN(θ))|  ≤  L · Δ_max⁺(m)
```

where `θ_NN(θ) = argmin_{θ̃ ∈ D_m⁺} |θ − θ̃|`. Coverage is now the **full closed quadrant**, not just the seed hull.

**Proof sketch.** For any `θ ∈ [0, π/2]`, the nearest sample in `D_m⁺` is at most `Δ_max⁺(m)/2` away (consecutive samples in sorted `D_m⁺` are at most `Δ_max⁺(m)` apart, and `θ` falls within one of those intervals). Lipschitz: `|g(θ) − g(θ_NN(θ))| ≤ L · Δ_max⁺(m)/2 ≤ L · Δ_max⁺(m)`. The loose bound `L · Δ_max⁺` (not `L · Δ_max⁺/2`) is used in v1; the half-gap improvement is reserved as a candidate for a later sharp-constant cert.

**Validator check.** For each `m ∈ {9, 24, 72}` and a declared Lipschitz function `g` at fixed `L`:
1. Construct `D_m⁺` (sample angles + two boundary anchors).
2. Compute `Δ_max⁺(m)` exactly (rational where possible; boundary gaps via float arithmetic but with `π/2` as `math.pi / 2` declared as observer-side).
3. On a dense observer-side test grid `T ⊂ [0, π/2]` (size `≥ 10·(n+2)`), compute nearest-neighbor reconstruction `ĝ(θ) = g(θ_NN(θ))`.
4. Assert `sup_{θ ∈ T} |g(θ) − ĝ(θ)| ≤ L · Δ_max⁺(m)`.

The continuous test grid `T`, the test function `g`, and `π/2` are **observer projection only**; no QA-discrete object depends on any of them.

---

### Gate W_DECL — Declarative invariants

- `(b, e, d, a)` are `int`; `(C, F, G)` are `int`.
- `gcd(b, e) = 1` enforced for every seed; non-coprime seeds rejected.
- `d = b + e` raw; `a = b + 2e` raw. **No** `% m` in element computation.
- Angular ordering of `D_m` uses `Fraction(F, C)` comparison only — never float.
- No `**2` (use `x*x`).
- `b, e` never `np.zeros` / `np.random.rand` / float / `Fraction`-of-float.

---

## 4. Validator design (sketch)

```python
from math import gcd
from fractions import Fraction

def enumerate_D_m(m: int) -> list[tuple[int, int, int, int, int]]:
    """Return [(b, e, C, F, G), ...] for all coprime seeds, ordered by angle."""
    pts = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if gcd(b, e) != 1:
                continue
            d, a = b + e, b + 2 * e
            C, F, G = 2 * d * e, a * b, d * d + e * e
            assert F == d * d - e * e
            assert C * C + F * F == G * G
            pts.append((b, e, C, F, G))
    # angle = atan2(F, C); since C,F > 0, angle in (0, pi/2). Sort by F/C exactly.
    pts.sort(key=lambda p: Fraction(p[3], p[2]))
    return pts

def gate_w1(pts, declared_count): ...
def gate_w2(pts):  # checks exact integer cross product >= 1 for all consecutive pairs
    ...
def gate_w3(pts, g, L, test_grid_size, declared_max_gap_bound):
    # exact max_gap via rational cross-product magnitudes converted to angles
    # nearest-neighbor reconstruction error sup
    # observer-side floats explicitly tagged
    ...
```

The validator is in pure Python with `int` and `fractions.Fraction` for the QA-discrete layer. Float operations are confined to:
- Reporting `Δ_max(m)` in radians (`asin(...)`).
- The Lipschitz test function `g` (declared as observer projection at input).
- Sup-error reporting in radians.

Theorem NT firewall is crossed exactly twice: input boundary at `g(θ)`, output boundary at the sup-error report.

---

## 5. Fixtures

| File | Type | Contents |
|---|---|---|
| `pass_d9_canonical_net.json` | PASS | `m=9`; declared `\|D_9\|=55`; declared `Gmax=370`; declared `Δ_max⁺ ≤ 0.20`; declared coprime-only; W2 all-pairs claim |
| `pass_d24_canonical_net.json` | PASS | `m=24`; declared `\|D_24\|=359`; declared `Gmax=2785`; declared `Δ_max⁺ ≤ 0.080` |
| `pass_d72_canonical_net.json` | PASS | `m=72`; declared `\|D_72\|=3175`; declared `Gmax=25633`; declared `Δ_max⁺ ≤ 0.028` |
| `pass_lipschitz_witness_d9_sin.json` | PASS | `g(θ)=sin θ`, `L=1`, `m=9`; declared sup error ≤ `Δ_max⁺(9)`; bit-exact validator agrees |
| `fail_noncoprime_seed.json` | FAIL | declared seed `(b,e)=(2,4)` (`gcd=2`); validator must reject seed and refuse to count it as part of `D_m` |
| `fail_nonqa_seed_form.json` | FAIL | declared seed sets `d` independently (e.g., `d=10` while `b=1, e=2`); HARD rule violation `d ≠ b+e`; validator must reject |
| `fail_w3_bound_too_strong.json` | FAIL | declared sup error bound `< L · Δ_max⁺(m) / 4` for `g(θ) = sin θ`; validator computes the actual sup and rejects the claim |
| `fail_w2_underclaimed_separation.json` | FAIL | declared `min(Δθ) > 1` (overclaim); validator finds a pair with smaller `sin(Δθ)` and rejects |

---

## 6. Theorem NT compliance

- Construction: integer / `Fraction` only.
- No `**2`; uses `x * x`.
- No `np.zeros` / `np.random.rand` as state.
- No mod reduction in element computation (raw `d`, `a`).
- No continuous time variable in QA-discrete layer.
- Firewall: input `g`, output sup error report. Two crossings, no more.

---

## 7. Out of scope (explicit non-claims)

This cert does **not** assert any of the following:

- Whittaker's 1903 theorem holds (it does, but that's classical analysis, outside QA's domain).
- The QA-rational net `D_m` is "the" right discretization of Whittaker's angular kernel for any physical purpose.
- The Whittaker phase packet `(Cx + Fz)/G + t/k` reproduces solutions to Maxwell's equations.
- Any 1904 two-scalar-potential / EM / electron / gauge-theoretic claim.
- Density of `D_m` in `S¹` as `m → ∞` (deferred to v2).
- 3D rational direction coverage on `S²` (deferred to v2).

The cert asserts only:
- **(W1)** `|D_m|` equals an exact, declared count.
- **(W2)** Distinct directions are angularly separated by at least `1/G_max²`, rigorously.
- **(W3)** Lipschitz nearest-neighbor reconstruction over `D_m⁺` (with observer-side boundary anchors `E_0`, `E_∞`) has sup error bounded by `L · Δ_max⁺(m)` on the closed quadrant `[0, π/2]`.

---

## 8. Whittaker → QA development ladder (Layers 1–6)

**v1 is the controlled seed crystal, not the endpoint.** The full research program is staged so the cert ecosystem does not overclaim before each layer is ready.

**Layer ID convention.** `[266]` is definite now that it is registered in `qa_meta_validator.py`. Layer 2-5 cert-family IDs are **unassigned** until each layer is built and reviewed. Do not reserve `[267]`-`[270]`: the current meta-validator already uses `[267]`-`[272]` for external-validation and doc/linter gate labels, so future Whittaker certs must claim the next free family ID at build time.

### Layer 1 — `[266]` `qa_whittaker_rational_direction_s1_cert_v1` *(this cert)*
- 2D `S¹` rational direction net; finite, exactly enumerable.
- Whittaker 1903-inspired angular decomposition; **no EM or wave claim** in this layer.
- Status: build prep authorized 2026-04-30.

### Layer 2 — `qa_whittaker_rational_direction_s2_cert_v1` *(ID unassigned)*
- 3D `S²` rational direction set: extend from one angular parameter to full direction cosines.
- Design draft: `docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_S2_CERT_DRAFT.md`.
- Construction options (decide at v2 design):
  - **Paired QA seeds** `((b₁,e₁), (b₂,e₂))` parameterizing two angular degrees of freedom.
  - **Wildberger 3D rational** — quadrance/spread on `S²` (see *Divine Proportions* §17–18).
  - Hybrid: paired seeds with a Wildberger-style spread invariant.
- Sharp-claim form: `D_m^{(2)} ⊂ S²` finite; angular separation lower bound from spherical cross-product integrality; spherical-cap nearest-neighbor Lipschitz error bound.

### Layer 3 — `qa_whittaker_wave_kernel_bridge_cert_v1` *(ID unassigned)*
- Bridge from Layer 2's direction net to Whittaker 1903's full angular wave-kernel decomposition.
- Sharp-claim form: a discrete superposition of plane-wave packets indexed by `D_m^{(2)}` approximates a smooth wave-equation solution `ψ` with sup error scaling explicitly in `m`.
- Claim type: **discretization / approximation / observer-projection**, NOT "QA proves Whittaker."

### Layer 4 — `[507]` `qa_whittaker_two_scalar_potential_bridge_cert_v1` *(built 2026-07-02)*
- Whittaker 1904 two-scalar-potential reduction, `Documents/whittaker_corpus/whittaker_1904_two_scalar_potential_functions_electromagnetic_field.pdf`, DOI 10.1112/plms/s2-1.1.367.
- **Naming guardrail applied**: Whittaker's scalars renamed `Phi, Psi` (never `F, G`, which are QA-reserved: `F = a·b = d²−e²`, `G = d²+e²`).
- **Built claim form** (narrower than originally sketched — no "spans the same packet space as the 4-scalar form" claim yet, since that requires comparing against an actual 4-potential `(φ,ax,ay,az)` construction not built here): for a QA-rational plane-wave packet over a registered [273] direction `ω` with declared QA-rational `k,v,c`, Whittaker's twelve raw operator coefficients `(Phi,Psi) → (dx,dy,dz,hx,hy,hz)` — transcribed verbatim from the primary source, p.370 §3 — are exact `fractions.Fraction` values; `div(h)` (both channels) and `div(d)` `Psi`-channel vanish unconditionally; `div(d)` `Phi`-channel vanishes under `v²=c²` or under the degenerate direction case `Kz=0`; for nonzero-`Kz` packets, `v²=c²` is necessary and sufficient.
- **Primary-source catch**: a pasted AI-generated summary of the 1904 paper (supplied by Will, sourced from a different assistant) matched the primary source on five of six field equations verbatim but guessed `hz` by false symmetry with `dz`. That guess breaks `div(h)=0`; the real primary-source `hz = ∂²Ψ/∂x²+∂²Ψ/∂y²` is what's implemented and is exactly what makes `div(h)=0` unconditional. Concrete instance of [[feedback_primary_sources_vs_consensus]].
- Cert dir: `qa_alphageometry_ptolemy/qa_whittaker_two_scalar_potential_bridge_cert_v1/`. 2 PASS + 7 FAIL fixtures, self-test ok.
- **Not yet built**: the original "spans the same direction-indexed packet space as the four-scalar form" comparison claim. Would require constructing a QA-rational 4-potential `(φ,ax,ay,az)` representation and a rank/spanning argument against it — left for a Layer 4.1 extension if pursued.

### Layer 5 — `qa_maxwell_scalar_pair_reconstruction_cert_v1` *(ID unassigned — mapped, not built)*
- Reconstruct numeric Maxwell field values from Layer 4's exact-rational carrier pairs by finally evaluating the formal `trig(θ)` symbol at a specific numeric `(x,y,z,t)`.
- Build-boundary continuation: `docs/specs/QA_WHITTAKER_LAYER5_LAYER6_CLAIM_BOUNDARY.md`.
- **Theorem NT accounting**: this is the *second and final* firewall crossing for this ladder (QA-exact coefficient algebra → observer-side numeric field value). The first crossing already happened going from raw physical directions into QA-rational directions in Layer 1/2. A single cert may legitimately cross the QA↔observer boundary exactly twice per Theorem NT; Layer 5 is where the second crossing belongs, not before.
- **The real new difficulty vs. Layer 3**: Layer 3's kernel-sampling error bound was 0th-derivative (sampling a scalar profile). Layer 5 differentiates twice before sampling, and Whittaker's own general solution (primary source §5, p.372) is a **continuous double integral over the full sphere of directions** `F = ∫₀^π∫₀^2π f(x sinu cosv + y sinu sinv + z cosu + ct, u, v) du dv` (and similarly for `G`), not a single plane wave. Approximating that continuous double integral with a finite sum over [273]'s `D_m^(2)` therefore needs a sup-error bound that is **explicitly `k`-dependent** (higher spatial frequency directions amplify angular sampling error by a factor scaling with the differentiation order — two derivatives means the naive bound picks up a factor of `k²` relative to Layer 3's undifferentiated case). This bound does not fall out of Layer 3's machinery for free; it is new analysis.
- **Sharp-claim form** (to build): for a target smooth vacuum EM disturbance `(dx,...,hz)` expressible in Whittaker's double-integral form with angular data `f,g` that is `L`-Lipschitz *and* band-limited to `k ≤ K_max`, a finite QA-rational-packet superposition over `D_m^(2)` (using Layer 4's exact coefficient algebra per packet) reconstructs each of the six components with sup-error `≤ C·L·K_max²·Δ_max^{(2)}(m)` for an explicit constant `C` and the [273] angular-gap bound `Δ_max^{(2)}(m)`.
- **Guarded framing**: ✅ "QA-compatible finite-packet approximation of scalar-potential-generated EM fields, with explicit error." ❌ "QA derives electromagnetism" / "QA computes Maxwell's equations."

### Layer 6 — research track (no cert ID assigned) — mapped, not built
- Build-boundary continuation: `docs/specs/QA_WHITTAKER_LAYER5_LAYER6_CLAIM_BOUNDARY.md`.
- **What's tractable in QA-exact arithmetic**: Mie scattering requires Bromwich's *spherical* generalization of Whittaker's Cartesian construction — potentials `U, V` (in `[168]`/`[169]`'s existing spherical-slicing vocabulary this would be a third, EM-flavored slicing) satisfying the scalar **Helmholtz** equation `∇²ψ + k²ψ = 0` (not the wave equation), separated as `ψ(r,θ,φ) = R(r)Θ(θ)Φ(φ)` with `R` = spherical Bessel functions `j_n(kr), y_n(kr)`, `Θ` = associated Legendre polynomials `P_n^m(cosθ)`, `Φ` = `e^{±imφ}`. The multipole indices `(n,m)` — `n ≥ 0` integer, `m ∈ {−n,...,n}`, `2n+1` values per `n` — are a genuinely discrete, finite-per-order lattice. **This is the one legitimately QA-shaped piece**: a narrow, honest Layer 6 cert would certify a QA-indexed enumeration of the `(n,m)` mode lattice up to a cutoff `N` (pure combinatorics, in the spirit of [128]'s `π(9)=24` ↔ 24 longitude-band mapping), *not* the field values themselves.
- **What is NOT tractable in QA-exact arithmetic**: `j_n(kr)`, `y_n(kr)`, and `P_n^m(cosθ)` are transcendental special functions — they are not exactly representable as `fractions.Fraction` the way a plane-wave phase argument is. Any cert touching actual Mie-scattered field *values* necessarily makes a **third** QA↔observer crossing (evaluating the special functions numerically), which Theorem NT does not license within a single cert's "exactly twice" budget without careful scoping (e.g., explicitly declaring the mode-lattice enumeration as one self-contained QA-layer cert, and any numeric field evaluation as a wholly separate, clearly-labeled observer-projection tool that consumes the cert's output — not a single cert claiming both).
- **Where the overclaim risk lives**: this is exactly the "Bearden/Pond/SVP" territory flagged in the original Layer 6 note — claims that the scalar potentials `Φ,Ψ` (or Mie-mode coefficients) carry "structured energy" prior to field manifestation, "scalar wave" free-energy framing, etc. **None of that is licensed by anything proven in Layers 1–5.** The pasted AI-generated content that prompted this ladder extension used exactly this kind of unsourced language ("energy can exist in a structured scalar state before manifesting as detectable transverse forces") — Whittaker's actual result is an orthodox linear PDE decomposition (later standardized by Bromwich into TE/TM modes — ordinary waveguide theory, nothing fringe). Any Layer 6 cert must reject this framing explicitly, the way [507]'s `claim_policy.claims_scalar_wave_energy_physics` does.
- Primary-source gating mandatory; hostile review mandatory. Cert IDs assigned only after Layers 1–5 are stable — Layer 5 is not yet built, so Layer 6 is not ready to claim an ID regardless of the mapping above.

### Cross-cutting (any layer)
- **QA-MEM ingestion** — `SourceWork × 2` (1903, 1904) + claim rows + bridge spec `docs/specs/QA_WHITTAKER_BRIDGE.md`. Do as a separate task, not part of any cert.
- **Layer-by-layer hostile review** — each cert passes its own review before the next layer's build prep starts.

---

## 9. Build plan (completed for [266] registration)

After hostile review, the completed build sequence was:

1. **`qa_alphageometry_ptolemy/qa_whittaker_rational_direction_s1_cert_v1/`** directory:
   - `mapping_protocol_ref.json` (Gate-0 ref protocol; DOI in scope_note; no `_exempt`).
   - `qa_whittaker_rational_direction_s1_cert_validate.py` with `enumerate_D_m` and gate functions.
   - `fixtures/`: 8 JSON files per §5.
2. **`docs/families/266_qa_whittaker_rational_direction_s1_cert.md`** — family doc.
3. **`docs/families/README.md`** — link added.
4. **`qa_alphageometry_ptolemy/qa_meta_validator.py`** — `FAMILY_SWEEPS` entry + validator function. Registered only after all standalone gates passed.
5. **`qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/expected_hash.json`** — regen after registry edit.

**Hostile review checklist before build:**

- [ ] W1: cardinality formula bit-exact verified for `m ∈ {9, 24, 72}`. ✅ (verified during refinement: 55, 359, 3175.)
- [ ] W2: cross-product proof tight; no hand-waving. ✅ (4-line proof in §3 W2.)
- [ ] W3: Lipschitz bound matches a standard nearest-neighbor result; `Δ_max⁺` computed exactly on closed Q1 with virtual endpoints `E_0`, `E_∞`. ✅ (`L · Δ_max⁺` is the loose form; tighter `L · Δ_max⁺ / 2` available; validator uses loose form.)
- [ ] All construction is exact integer / `Fraction`; floats observer-side only. ✅
- [ ] Negative fixtures cover non-coprime, non-QA-form, overclaimed-bound cases. ✅
- [ ] Cert does not claim any physics; only the QA discretization layer. ✅
- [x] [266] confirmed free in registry at build time; collab `family_id_claim` broadcast attempted (or noted as bus-down).
- [ ] Codex bridge cert review.
- [ ] Will sign-off.

---

## References

- E. T. Whittaker (1903). *On the partial differential equations of mathematical physics.* Math. Annalen 57:333–355. DOI 10.1007/BF01444290.
- E. T. Whittaker (1904). *On an Expression of the Electromagnetic Field due to Electrons by Means of Two Scalar Potential Functions.* Proc. London Math. Soc. s2-1:367–372. DOI 10.1112/plms/s2-1.1.367. *(Bridge-spec background only; not mapped in v1.)*
- N. J. Wildberger (2005). *Divine Proportions: Rational Trigonometry to Universal Geometry.* §17–18 (3D rational direction parameterization). *(Referenced for v2 deferral.)*

**End refined draft.**
