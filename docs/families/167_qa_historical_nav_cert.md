# Family [167] QA_HISTORICAL_NAV_CERT.v1

## One-line summary

Five independent historical navigation systems (Babylonian, Egyptian, Polynesian, Norse, Arab) used integer-ratio methods structurally equivalent to QA — Theorem NT before the name existed.

## Mathematical content

### Five civilizations, one structure

| Civilization | Era | Instrument | QA Equivalent |
|---|---|---|---|
| Babylon | ~1800 BCE | Plimpton 322 tablet | Berggren tree direction table |
| Egypt | ~2500 BCE | Seked (slope ratio) | Spread = den^2/(den^2+num^2) |
| Polynesia | ~1000 CE | Star compass (32 houses) | Mod-32 discrete bearings |
| Norse | ~800 CE | Sun stones (calcite) | Spread of polarization angle |
| Arab | ~900 CE | Kamal (finger-widths) | Integer spread increments |

### Common QA structure

All five systems operate on:
1. **Discrete direction states** (integer ratios, named houses, finger-widths)
2. **Integer arithmetic** for computation
3. **Observer projection** only at measurement and landfall

This IS Theorem NT — the boundary between continuous and discrete is crossed exactly twice.

### The historical arc

Integer methods (ancient) -> trigonometric tables (15th century) -> floating-point computation (20th century) -> integer methods (QA). A full circle. The Portuguese adoption of continuous trigonometry was a regression that introduced the very float errors QA eliminates.

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_historical_nav_cert_v1
python qa_historical_nav_cert_validate.py --self-test
```

## Cross-references

- [138] QA_PLIMPTON322 — Babylonian tablet as QA direction table
- [134] QA_EGYPTIAN_FRACTION — Egyptian unit fractions = same arithmetic
- [163] QA_DEAD_RECKONING — T-operator DR = modern version of ancient methods
- [165] QA_CELESTIAL_NAV — sextant as spread instrument (evolution of kamal/sun stone)

## Verification Note (2026-07-05)

Independently checked the two most specific numeric/citation claims:

- **Egyptian seked = 5.5 palms per cubit for the Great Pyramid**: confirmed
  exactly. Scholarship on the Rhind Mathematical Papyrus (problems 56-60)
  gives the Great Pyramid's seked as "5 palms, 2 digits" per cubit; since
  1 palm = 4 digits, 2 digits = 0.5 palm, so 5 palms + 2 digits = 5.5
  palms exactly — the cert's "5.5 palms (= 5½:7)" is a precise
  decimal-form restatement of the traditional palms+digits notation, not
  an approximation. `seked_to_spread()`'s formula (`den²/(den²+num²)`)
  is dimensionally correct: `tan(slope)=rise/run`, `spread=sin²(slope)`.
- **Ropars et al. (2012)**: confirmed real — "A depolarizer as a possible
  precise sunstone for Viking navigation by polarized skylight,"
  *Proceedings of the Royal Society A*, exact author/title/venue match.
  The underlying optics (Iceland-spar calcite can depolarize skylight to
  reveal sun position, verified within ~1% experimentally) is genuine,
  peer-reviewed physics.

**One honesty caveat worth adding** (same spirit as the Megalithic Yard
audit): the paper itself is real science about what calcite *can* do
optically — it does not, and cannot, establish that Vikings *actually*
used this method for navigation. No Viking-era sunstone has been found
in a clear navigational-use context (the closest physical evidence, a
calcite crystal on the 16th-century English Alderney shipwreck, postdates
the Viking era by centuries); the historical-use claim rests on saga
references to an unidentified "sunstone" object, not direct archaeological
proof. This cert's own framing ("Norse: Sun stones... measures sun
position through clouds") already reads as a live technique description
rather than an overclaimed historical certainty, but a one-line caveat
distinguishing "the physics works" from "Vikings are confirmed to have
used it this way" would strengthen the cert's honesty, consistent with
this project's broader practice of flagging contested historical claims.

Lewis (1972), Tibbetts (1971), and Gillings (1972) are all real,
well-established works in their respective fields (Pacific navigation,
Arab maritime history, Egyptian mathematics) — not independently
re-verified page-by-page here, but no reason to doubt given how precisely
the two most checkable numeric claims above hold up.
