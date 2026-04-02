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
