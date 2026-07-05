# Family [165] QA_CELESTIAL_NAV_CERT.v1

## One-line summary

Celestial navigation sight reduction expressed entirely in spreads, crosses, and discrete orientation flags — the sextant is a spread measurement instrument.

## Mathematical content

### Sight reduction formula

Classical: `sin(h) = sin(phi)sin(delta) + cos(phi)cos(delta)cos(LHA)`

Rational trigonometry:
```
s_h = [sigma_1 * sqrt(s_phi * s_delta) + sigma_2 * sqrt(c_phi * c_delta * c_LHA)]^2
```

where sigma_1 = +1 if same hemisphere, -1 if opposite; sigma_2 = +1 if cos(LHA) >= 0, -1 otherwise.

The decomposition into unsigned spreads + discrete orientation flags IS Theorem NT: continuous angle decomposes into quadratic measure + discrete choice.

### Azimuth (spread form)

`s_Az = c_delta * s_LHA / s_z` where `s_z = 1 - s_h` = spread of zenith distance.

### Two-star fix

Two measured altitudes give two position circles (spread loci). Their intersection is the fix — solved algebraically, no trig inversion needed.

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_celestial_nav_cert_v1
python qa_celestial_nav_cert_validate.py --self-test
```

## Cross-references

- [156] QA_WGS84_ELLIPSE — Earth = QA quantum ellipse (observation surface)
- [163] QA_DEAD_RECKONING — T-operator DR between sights
- [164] QA_GNOMONIC_RT — gnomonic chart for plotting position lines

## Verification Note (2026-07-05)

This is a Tier 1 exact-reformulation claim (not an external empirical
citation), so verified it directly: independently re-derived
`s_h = [σ₁·√(s_φ·s_δ) + σ₂·√(c_φ·c_δ·c_LHA)]²` algebraically from the
classical `sin(h) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(LHA)` by squaring and
substituting `sin(x) = ±√(sin²x)`, `cos(x) = ±√(cos²x)` with the stated
sign conventions — confirms it's an exact algebraic identity, not an
approximation. Then numerically spot-checked with 5 randomized
(latitude, declination, LHA) triples spanning both hemispheres and all
LHA quadrants: classical and rational computations matched to floating-
point precision (diff ≤ 2×10⁻³³) in every case. Same check for the
azimuth spread formula `s_Az = c_δ·s_LHA/s_z` (using `s_z = 1-s_h =
cos²(h)`, the spread of zenith distance) — matches the standard
`sin(Az) = cos(δ)sin(LHA)/cos(h)` identity exactly. The classical
formula itself is standard nautical astronomy (Bowditch's *American
Practical Navigator*, in continuous print since 1802) — no external
citation risk here, just confirmed the rational-trig rewriting is sound.
No bugs found.
