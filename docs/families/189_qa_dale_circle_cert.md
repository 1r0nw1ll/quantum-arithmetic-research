# Family [189] QA_DALE_CIRCLE_CERT.v1

## One-line summary

Dale Pond's integer circle construction: P=2W (diameter), Q=P (circumference), R=W² (area) — pi disappears in QA circular units.

## Mathematical content

### Dale Pond's circle elements (svpwiki.com, 1998)

"When Ben said there was no way to define a circle with Quantum Arithmetic, I developed three different ways to do this." — Dale Pond

| Element | Definition | Meaning |
|---------|-----------|---------|
| P | 2W | Circle diameter (QA units) |
| Q | P = 2W | Circle circumference (QA units) |
| R | W² | Circle area (QA units) |

### Key insight: QA circular units (pi=1 convention)

Dale defines QA circular units where pi=1: P=Q means diameter=circumference by **convention**, not by structural theorem. This is analogous to natural units in physics (c=1, h-bar=1). The algebraic identities P=2W, Q=P, R=W² are exact by definition. Dale's contribution is identifying W as the natural radius-like quantity for QA circles. Extends element count to 25 (with S from [190]).

### Verification (Unity Block)

(b,e,d,a) = (1,1,2,3): W=8, P=16, Q=16, R=64.

Verified for 7 Pythagorean directions with consistent results.

## Checks

| ID | Description |
|----|-------------|
| DC_1 | schema_version == 'QA_DALE_CIRCLE_CERT.v1' |
| DC_P | P = 2*W for all witnesses |
| DC_Q | Q = P (diameter = circumference in QA units) |
| DC_R | R = W*W for all witnesses |
| DC_W | W correctly derived from (b,e) |
| DC_D_A | declared d, a fields (if present) match A2-derived d=b+e, a=b+2e (hardened 2026-07-06 — previously the witness's own declared d/a were never cross-checked against anything) |
| DC_SRC | source attribution to Dale Pond |
| DC_WITNESS | >= 3 witnesses |
| DC_F | fail_ledger well-formed |

## Source grounding

- **svpwiki.com/Quantum+Arithmetic+Elements**: Dale Pond, "Unpublished Notes", December 1998
- **Ben Iverson**: original 21 elements (b–Z); circles declared impossible
- **Dale Pond**: three circle construction methods; P, Q, R elements

## Connection to other families

- **[152] Equilateral Triangle Cert**: W = equilateral side; P, Q, R derived from W
- **[133] Eisenstein Norm Cert**: F²-FW+W²=Z²; W is the Eisenstein equilateral side
- **[190] Equilateral Height Cert**: S = equilateral height; S, W, P form equilateral-circle bridge

## Fixture files

- `fixtures/dc_pass_circle.json` — 7 directions with P, Q, R verified
- `fixtures/dc_fail_bad_p.json` — FAIL fixture for testing

## Verification Note (2026-07-06)

Independently recomputed W=d*(b+3e), P=2W, Q=P, R=W*W by hand for all 7
witnesses directly from (b,e) — e.g. (5,3): d=8, b+3e=14, W=8*14=112,
P=224, R=112²=12544 — all 7 match exactly. This validator was already
one of the stronger ones in the audit cycle: unlike the Keely family
([184]-[188]), it genuinely recomputes W/P/Q/R from (b,e) rather than
cross-checking declared fields against each other.

**Found and hardened one real gap**: the witnesses' own declared `d`
and `a` fields were read but never actually verified against the A2
axiom (d=b+e, a=b+2e) — the validator silently recomputed its own
internal `d_val`/`a_val` from `b,e` and used those for the W/P/Q/R
checks, so a witness could declare an internally-inconsistent `d`/`a`
pair alongside a correct W and it would pass undetected. Added `DC_D_A`
to cross-check declared d/a against the derived formula; verified it
rejects planted wrong values for both fields. No fixture data was
actually wrong — d/a were correct in all 7 witnesses.

Note: the doc's phrase "7 Pythagorean directions" is informal — the 7
witness (b,e) pairs are not all coprime/opposite-parity in the strict
Euclid-parameterization sense (e.g. (5,3) and (7,3) are both odd-odd),
so "Pythagorean" here means "QA direction" in the project's general
sense, not literally Euclid's primitive-triple condition. This doesn't
affect any certified claim (W/P/Q/R hold for any (b,e)), so left as-is
rather than renamed.
