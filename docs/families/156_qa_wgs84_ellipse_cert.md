# Family [156] QA_WGS84_ELLIPSE_CERT.v1

## One-line summary

The WGS84 reference ellipsoid (Earth's oblate spheroid) IS a QA quantum ellipse with QN (101,9,110,119), matching to 7 significant figures.

## Mathematical content

### Earth shape quantum number

The WGS84 ellipsoid has first eccentricity e = 0.0818191908... and axis ratio b/a = 0.9966471893...

QA quantum ellipse with QN (b,e,d,a) = (101, 9, 110, 119):
- Eccentricity = e/d = 9/110 = 0.0818181818...
- Relative error: 0.001% (1.2 x 10^-5)
- Axis ratio = sqrt(ab)/d = sqrt(12019)/110 = 0.9966472722...
- Relative error: 8.3 x 10^-8 (7 significant figures)
- Triple: (C,F,G) = (1980, 12019, 12181)
- Pythagorean check: 1980^2 + 12019^2 = 12181^2

### Earth orbit quantum number

Earth's orbital eccentricity = 0.0167086 (J2000.0).

QA quantum ellipse with QN (b,e,d,a) = (59, 1, 60, 61):
- Eccentricity = e/d = 1/60 = 0.0166667
- Relative error: 0.25%
- Triple: (C,F,G) = (120, 3599, 3601)
- 59 and 61 are twin primes

### Key distinction

Earth's **shape** (oblate spheroid, ecc ~ 0.082) and **orbit** (solar ellipse, ecc ~ 0.017) are different quantities requiring different QNs. The shape eccentricity is ~5x larger than the orbital eccentricity.

### Prime factor harmonic chain

| Body | QN | Shared primes |
|------|-----|---------------|
| Earth orbit | (59, 1, 60, 61) | 59, 61 |
| Moon shape | (58, 3, 61, 64) | shares 61 with Earth |
| Halley's Comet | (1, 29, 30, 59) | shares 59 with Earth |

Chain: 29 -> 58 -> 59 -> 61 (Ben's Law of Harmonics).

## Tier classification

**Tier 1 — Exact reformulation.** No empirical claim. The WGS84 ellipsoid parameters map exactly to QA quantum ellipse parameters via the identity: axis_ratio = sqrt(ab)/d, eccentricity = e/d. The match quality (7 sig figs for axis ratio) is a property of the particular QN found by exhaustive search over primitive directions.

## Checks

| Check ID | Description | Status |
|----------|-------------|--------|
| WGS_1 | Schema version | PASS |
| WGS_QN | b+e=d, b+2e=a, gcd=1 | PASS |
| WGS_TRIPLE | C=2de, F=ab, G=d*d+e*e, C^2+F^2=G^2 | PASS |
| WGS_ECC | Eccentricity within tolerance | PASS |
| WGS_AXIS | Axis ratio within tolerance | PASS |
| WGS_ORBIT | Orbit QN eccentricity match | PASS |
| WGS_W | At least 1 witness | PASS |
| WGS_F | Fail detection | PASS |

## Sources

- WGS84: NIMA Technical Report 8350.2, 3rd edition (2000)
- Quantum Ellipse: Ben Iverson, *Pythagoras and the Quantum World*, Vol. 1
- Chromogeometry: N.J. Wildberger, *Divine Proportions* (2005)
- QN fitting: Will Dale, `qa_geodesy_bridge.py` (2026-04-01)

## Validator

`qa_alphageometry_ptolemy/qa_wgs84_ellipse_cert_v1/qa_wgs84_ellipse_cert_validate.py --self-test`

## Verification Note (2026-07-06)

Independently recomputed both WGS84 reference constants from the actual
*defining* parameters (a=6378137.0 m, 1/f=298.257223563 — confirmed
these are the real WGS84 primary constants via NIMA TR8350.2, 3rd
edition, amendment 1, January 2000), not trusted from a possibly-stale
decimal copy: e² = 2f−f², e = √e² = 0.08181919084262149, and
b/a = (1−f) = 0.9966471893352525 — both match the doc's claimed values
exactly to the digits shown. Independently recomputed the Earth-orbit
eccentricity comparison against the standard J2000.0 value (0.0167086)
— also matches.

**All QA-side arithmetic independently reproduced**: shape QN
(101,9,110,119) satisfies b+e=d, b+2e=a, gcd(b,e)=1; triple
(C,F,G)=(1980,12019,12181) satisfies C²+F²=G² exactly; e/d=9/110 has
relative error 1.233×10⁻⁵ vs WGS84 (matches doc's "0.001%"); axis ratio
√(ab)/d has relative error 8.31×10⁻⁸ (matches doc's "7 significant
figures"). Orbit QN (59,1,60,61): b+e=d, b+2e=a, gcd=1, triple
(120,3599,3601) satisfies C²+F²=G², e/d=1/60 has relative error
0.251% vs the real orbital eccentricity (matches doc's "0.25%"); 59 and
61 independently confirmed prime (and twin primes). 0 mismatches
anywhere.

**Validator confirmed genuinely computing, not fixture-trusting**: read
`qa_wgs84_ellipse_cert_validate.py` in full — `WGS84_ECC` and
`WGS84_RATIO` are derived live from the same defining `a`/`f`
constants (not a copied decimal), and both the QN-structure checks and
the Pythagorean-triple check are recomputed from the fixture's raw
`b,e,d,a` at runtime. This is one of the cleaner validator designs
found this cycle — no hardening needed.

Ben Iverson's *Pythagoras and the Quantum World* (already confirmed
real in this session's audit of cert [160]) is cited for the "quantum
ellipse" framework; the specific "Vol. 1" attribution and the "prime
factor harmonic chain"/"Ben's Law of Harmonics" naming were not
independently traced to a specific page (same caveat as [160]'s Vol. 4
citation — plausible for a small self-published series, not confirmed
either way). This doesn't affect the cert's actual certified claims,
which are the QA arithmetic and the WGS84/orbital numeric matches, both
independently confirmed above.

No bugs found. `--self-test` passes on both fixtures.
