# Family [178] QA_MEGALITHIC_CERT.v1

## One-line summary

Megalithic builders used discrete integer arithmetic: Megalithic Yard as length quantum (p=0.00022, z=-3.54, 202 circles), fathom as operational unit (p<10^-8), Fibonacci QN construction triangles.

## Mathematical content

### Megalithic Yard (MY = 2.72 ft)

Combined Thom 1962+1967 dataset: 202 stone circle diameters from England, Wales, Scotland. Integer proximity test: mean fractional error 0.213 vs null 0.249, z=-3.54, p=0.00022. MY=2.72 is the optimal quantum (variance 0.0497, sharp minimum).

Prior result at n=84 (1962 only): p=0.0096, z=-2.34. The 202-circle result is 10x more significant.

### Megalithic Fathom (MF = 2 MY = 5.44 ft)

150/202 circles (74.3%) have EVEN nearest-MY values. Binomial p < 10^-8. The fathom, not the yard, was the primary construction unit. The 74% rate (vs 83% at n=84) reflects the 1967 book including smaller circles using half-fathom radii.

### Construction triangles

- 3:4:5 -> QN (1,1,2,3) = first four Fibonacci numbers
- 5:12:13 -> QN (1,2,3,5) = fundamental QA quantum number

Only 19% of primitive triples (hyp <= 100) are all-Fibonacci QN. Megalithic builders chose from this rare subset.

### Honest negatives

- Fibonacci ratios in diameter pairs: p=0.155, NOT significant
- Mod-9 distribution: uniform (no QA orbit preference)
- Mod-24 non-uniformity explained by fathom effect, not independent

## Cross-references

- [167] QA_HISTORICAL_NAV — megalithic builders as proto-QA navigators
- [164] QA_GNOMONIC_RT — megalithic sites as proto-gnomonic stations
- [219] QA_FIBONACCI_RESONANCE — same Fibonacci preference in orbital physics
- [135] QA_PYTHAGOREAN_TREE — 3:4:5 and 5:12:13 as Berggren tree nodes

## Verification Note (2026-07-04)

Found real backing scripts and raw data in the repo root: `qa_megalithic_deep.py`
loads `thom_1962_diameters.csv` (84 rows, real stone-circle names — Nine
Ladies, Barbrook, Castle Rigg, The Hurlers, etc. — matching Thom's actual
survey sites). `thom_1967_table51_diameters.csv` (122 rows) is also
present but has no script wired to it.

**Independently reran the n=84 tests, both reproduce exactly**:
- Fathom test: 70/84 = 83.3% even-MY, binomial p≈0 (<0.0001) — matches
  the cert's declared "prior_result_84" exactly.
- MY-quantum permutation test (rewritten from the cert's own described
  method — real mean fractional error vs 100k-shuffle uniform null):
  real error=0.2127, null mean=0.2495, **z=-2.340, p=0.00957** — matches
  the cert's declared "prior_result_84" (z=-2.34, p=0.0096) almost
  exactly.

**Headline "202-combined" claim (z=-3.54, p=0.00022) has no backing
script**, and the exact n=202 doesn't reconcile cleanly: 1962 (84) + 1967
(122) = 206 raw; deduplicating the 61 diameters that exactly match a 1962
value gives 145, neither equals 202. Reconstructed my own combined
dataset both ways and reran the same permutation methodology: 206-raw
gives z=-3.434, p=0.00029; 145-deduped gives z=-3.430, p=0.00029. **Both
confirm the qualitative claim is real and non-fabricated** (same order of
magnitude, same significance level, same sign as declared) but the exact
n=202 subset and its precise z=-3.54/p=0.00022 aren't reproducible from
the files present in this repo — likely a specific data-quality filter
(e.g. excluding damaged/uncertain circles) not captured anywhere in code.

**Hardening added**: `MG_ZP_CONSISTENCY` — cross-checks any declared
`z_score`/`p_value` pair against each other via the stdlib normal CDF
(`math.erfc`), independent of the raw circle data. Confirmed it passes
the real fixture and catches a deliberately inconsistent p-value.

**Contested-claim caveat**: Thom's Megalithic Yard hypothesis remains
disputed in mainstream archaeology (later reanalyses, e.g. Kendall 1974,
found weaker significance under stricter statistical methods than Thom's
original quantogram approach) — this cert's own "Honest negatives"
section already reflects appropriately cautious practice, and this audit
found no evidence of fabrication, only an untraceable exact subset for
the single headline number.
