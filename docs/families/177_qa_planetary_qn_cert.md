# Family [177] QA_PLANETARY_QN_CERT.v1

## One-line summary

Quantum numbers for 10 solar system bodies: each eccentricity gives a QN tuple (b,e,d,a) with C^2+F^2=G^2, a characteristic latitude, and harmonic connections via shared prime factors.

## Mathematical content

For eccentricity epsilon, the characteristic latitude is arcsin(sqrt(2*epsilon/(1+epsilon^2))).

Key findings:
- Earth-Jupiter share b=59 (orbital QNs) — strongest gravitational pair shares a prime
- Earth shape b=101 = Uranus orbit b=101 — cross-domain harmonic
- Saturn shape (79,60,139,199): b,d,a all prime, arithmetic progression step 60
- Earth char lat 23.78 degrees near Tropic (23.44 degrees), but p~0.013 under null — Tier 2

## Cross-references

- [156] QA_WGS84_ELLIPSE — Earth shape QN canonical
- [149] QA_LAW_OF_HARMONICS — shared prime factors = harmonic resonance
- [168] QA_ELLIPSOID_GEODESIC — char latitude formula derivation

## Verification Note (2026-07-06)

**Found and fixed a real bug**: Mars_shape's declared eccentricity
(0.06489) was not real Mars data at all — it was exactly 17/262, the
QN (245,17,262,279)'s own e/d ratio, backfit and presented as if it were
the astronomical target being approximated. Mars's real shape
eccentricity (computed from its equatorial/polar radii, 3396.2/3376.2
km) is 0.10837 — the declared value was off by ~40× the fixture's own
stated 0.001 tolerance. Replaced with a genuine best-fit QN
(181,22,203,225): 181 is prime, gcd(181,22)=1, valid Pythagorean triple,
relative error 7.7×10⁻⁵ against the true value (comparable precision to
the Earth-shape fit in [156]).

**All other 9 entries independently verified against real NASA/JPL
data** and confirmed accurate within their stated tolerances: Earth
shape/orbit (already confirmed via [156]); Mars orbit (0.0934, exact);
Jupiter shape (0.35430 real vs 0.35364 declared, within tolerance) and
orbit (0.04839266, essentially exact); Saturn shape (0.43166 real vs
0.43165 declared, essentially exact) and orbit (0.0541506 real vs
0.05386 declared, within tolerance); Uranus orbit (0.04716771,
essentially exact); Moon orbit (0.0549, exact).

**Why this slipped through**: the validator's `PQ_ECC` check only
verified that a QN's e/d matched its own *declared* eccentricity field
— never that the declared field matched *reality*. Since Mars_shape's
declared value was defined as its own QN's ratio, it passed trivially.
Hardened the validator with an independently-sourced `REAL_ECCENTRICITY`
reference table (NASA Planetary Fact Sheets + Archinal et al. 2018,
DOI: 10.1007/s10569-017-9805-5) covering all 10 named bodies, and added
a new `PQ_REAL` check that cross-verifies the declared value against
it. Regression-tested that this now rejects the old buggy Mars entry.
Also fixed a stale "family [171]" self-reference in the validator's own
docstring (171 is a different, unrelated cert).
