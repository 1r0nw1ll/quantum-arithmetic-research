# [159] QA Observer Core Cert

**Schema**: `QA_OBSERVER_CORE_CERT.v1`
**Status**: PASS (1 PASS + 1 FAIL fixture)

## What it certifies

The two foundational functions of the topographic observer pipeline. Used
directly (identical `qa_mod`/`compute_qci` definitions) in 3 confirmed
domains — seismology, climate, ERA5 — plus a claimed-but-unverifiable-by-
policy 4th (finance, off-limits per project rules). **Not** used in EEG or
audio, which use different QA methods entirely (see Verification Note).

### qa_mod(x, m)

**Formula**: `((int(x) - 1) % m) + 1`

**Axiom A1 compliance**: Output is ALWAYS in `{1, ..., m}`, never 0. Verified for all integers in `[-100, 200]` with `m = 24` and `m = 9`.

| Input | m=24 | Note |
|-------|------|------|
| 1 | 1 | Identity |
| 24 | 24 | Maximum |
| 25 | 1 | Wraps |
| 0 | 24 | Zero wraps to m |
| -1 | 23 | Negative handled |

### compute_qci(labels, cmap, m, window)

For each triple `(t, t+1, t+2)`, predicts `label[t+2]` via `qa_mod(cmap[label[t]] + cmap[label[t+1]], m)` and checks match. Returns rolling mean of match fractions.

- **Deterministic**: same input always produces same output
- **Output length**: `len(labels) - 2`
- **T2 compliant**: no float-to-int feedback; `int(x)` cast on input only

### Domain witnesses (corrected 2026-07-06 — was "6 domain witnesses")

1. Seismology (46_seismic_topographic_observer.py) — confirmed, qa_mod line 34, compute_qci line 37
2. Climate (48_teleconnection_topographic_observer.py) — confirmed, qa_mod line 44, compute_qci line 48
3. ERA5 (49_forecast_coherence_observer.py) — confirmed, qa_mod line 65, compute_qci line 69
4. Finance (scripts 30-33 in `~/Desktop/qa_finance/`) — claimed, not independently verified (directory is off-limits per project rules)
5. ~~EEG (eeg_chbmit_scale.py)~~ — **removed**: does not use qa_mod/compute_qci anywhere
6. ~~Audio (qa_audio_residual_control.py)~~ — **removed**: does not use qa_mod/compute_qci anywhere

## How to run

```bash
# Unit tests (15 tests)
cd qa_lab && PYTHONPATH=. python -m pytest qa_observer/tests/ -v

# Validator self-test
cd qa_alphageometry_ptolemy/qa_observer_core_cert_v1
python qa_observer_core_cert_validate.py --self-test
```

## What breaks

- qa_mod returning 0 for any input (A1 violation)
- qa_mod output outside {1,...,m}
- compute_qci non-deterministic
- output_length != input_length - 2
- float-to-int feedback in qa_mod (T2 violation)

## Sources

- Hardy, G.H. & Wright, E.M. (2008), *An Introduction to the Theory of Numbers*, 6th ed., Oxford University Press, ISBN 978-0-19-921986-5, Ch. II — residue classes / congruences (qa_mod is a from-scratch shift of standard modular arithmetic onto a no-zero alphabet).
- Wildberger, N.J. (2005), *Divine Proportions* — QA discrete state framing (axiom A1).

## Verification Note (2026-07-06)

Ran the real 15-test suite (`cd qa_lab && PYTHONPATH=. python -m pytest
qa_observer/tests/ -v`) — all pass against the real
`qa_lab/qa_observer/core.py`.

**qa_mod arithmetic confirmed fully correct**: independently recomputed
all 8 declared cases (x=1,24,25,0,-1,48 at m=24; x=9,10 at m=9) directly
against `((int(x)-1)%m)+1` — every declared result matches exactly,
including the m=24→wraps-to-24 and negative-input cases.

**Found a real mischaracterization in the QCI example**: the fixture's
note claimed "only triple (0,1,2) matches" for the repeating-label
example, but independently running `compute_qci` on the declared
labels/cmap shows the raw per-triple match pattern is
`[1,0,0,1,0,0,1,0,0,1]` — **4 of 10** triples match (recurring every
3rd triple), not one. Separately, `known_match_rate=0.333` is not "the"
match rate at all: `compute_qci` returns a **rolling-mean series** (10
values: `[1.0, 0.5, 0.333, 0.333, ...]`), and 0.333 is only its
steady-state tail value (which happens to equal 1/3 because exactly one
of every three consecutive raw matches is a hit — a real but differently-
explained coincidence). Fixed the fixture's note and added an explicit
`raw_match_rate: 0.4` field to disambiguate.

**Found and corrected a real overclaim in the "6 domain witnesses"
claim**: grepped `eeg_chbmit_scale.py` and `qa_audio_residual_control.py`
(plus their imports, `eeg_orbit_observer_comparison.py` and
`eeg_orbit_classifier.py`) for `qa_mod`/`compute_qci` — **zero matches**
in either. Both are real, substantive QA scripts, but use different
methods entirely: EEG uses per-window topographic k-means classification
(`classify_segment_topographic`, `compute_orbit_sequence`), audio uses
matched-AC partial correlation on OFR (not the qa_mod/compute_qci scalar
rolling-match pattern at all). Confirmed the other 3 domains directly:
seismology (`46_seismic_topographic_observer.py`, qa_mod at line 34,
compute_qci at line 37 — byte-for-byte identical logic to
`qa_observer/core.py`), climate (`48_teleconnection_topographic_observer.py`,
lines 44/48), ERA5 (`49_forecast_coherence_observer.py`, lines 65/69) —
all three genuinely match. Finance (`~/Desktop/qa_finance/`) is off-limits
per project rules (frozen private scripts) and remains an unverified
claim, not a confirmed one. Removed the false EEG/audio witnesses from
the fixture and corrected "used across all 6 empirical domains" to the
accurate count (3 confirmed + 1 unverifiable-by-policy, not 6).

**Hardened the validator**: `OC_A1` previously only range-checked the
fixture's own declared `result` (never recomputing `qa_mod` itself);
`OC_QCI` only checked declared `output_length` arithmetic and a
`deterministic` flag (never recomputing `compute_qci` itself). Both now
live-import `qa_lab/qa_observer/core.py` (graceful degrade-to-warning if
not importable) and genuinely recompute, verified to reject a planted
wrong `qa_mod` result. `--self-test` passes on both fixtures with the
live recompute path active.
