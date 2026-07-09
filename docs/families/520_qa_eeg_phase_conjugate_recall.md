# [520] QA Phase-Conjugate EEG Brain-State Recall Cert

**Family ID**: 520
**Slug**: `qa_eeg_phase_conjugate_recall_cert_v1`
**Status**: Active
**Registered**: 2026-07-08

## Claim (empirically demonstrated + mechanism-certified)

The cert **[519]** phase-conjugate associative memory (built on cert **[518]**'s
exact FWM conjugator) performs **artifact-robust EEG brain-state recall**. A 10s
multi-channel EEG window maps to a **topographic QA phase vector** (per-channel
z-scored log band-power → phase in `{1,...,m}`); stored brain-states are recalled
from corrupted or partial probes — the clinically real cases of electrode dropout
and artifact contamination.

**Key mechanism** (certified deterministically, no external data): under a
**global reference-shift artifact** (`probe → qa_add(probe, φ)` — modelling EEG
re-referencing / DC offset), naive nearest-overlap classification is fooled as φ
grows, while **phase-locked classification** — scan the global compensation phase
ψ maximising overlap, read the class in that compensated frame (the [518]
phase-conjugate mirror self-locking to the distorting medium) — stays correct.
The validator proves this on two spatially-different synthetic topographies that
are *not* global shifts of each other (so phase-lock can compensate the artifact
and still separate the classes).

## Empirical record (real CHB-MIT, 7 patients)

`qa_eeg_phase_conjugate_recall.py` on `archive/.../eeg/chbmit` (chb10/12/15/16/17/21/23):

| φ (global artifact) | phase-locked | naive / direct-NN |
|---|---|---|
| 0 (clean) | 0.79–0.89 | 0.83–1.00 |
| **6** | **~0.81** (0.72–0.92) | ~0.40 (0.17–0.57) |
| **12** | **~0.85** (0.78–0.92) | ~0.44 (0.19–0.57) |

- **Seizure/baseline recall (clean): 0.83–1.00** across patients (chance 0.54–0.83).
- **Robust to 60% electrode dropout** (memory ≥ direct-NN).
- **Under a global reference-shift artifact, naive/direct classification collapses
  to near or below chance while phase-locked recall stays robust** — the [518]
  distortion-correction property delivering artifact-robust brain-state recall.

## Honest limits

- Phase-lock costs a few % on clean signal (φ=0): the ψ-scan can occasionally
  mis-lock. It trades a little clean accuracy for large artifact robustness.
- **chb16** is severely class-imbalanced (chance 0.83) — uninformative on its own;
  included for completeness, not hidden.
- Single broadband log-power feature, m=24. Richer topographic features unexplored.
- The empirical numbers **cannot be recomputed in CI** (24GB EEG data is not in the
  git tree). They are recorded here with provenance and reproduced by the committed
  script; the validator certifies the *mechanism* synthetically.
- Five of the seven patients were unlocked by fixing an unguarded EDF
  zero-range-channel division (`ZeroDivisionError`) in the project's own loader.

## Checks

| Check | Meaning |
|-------|---------|
| `PLC_MECHANISM` | phase-locked classification recovers the true class for all φ |
| `OVERLAP_MATCH` | phase-conjugate overlap == exact match count |
| `NAIVE_FAILS` | naive classification is fooled under a global shift (the problem is real) |
| `DISTORTION_ARTIFACT` | the artifact is exactly a modular shift `qa_add(probe, φ)`, invertible |
| `EMPIRICAL_WITNESS` | recorded CHB-MIT witness is internally consistent (phase-lock > naive; recall > chance) |
| `A1_RANGE` | every state in `{1,...,m}` |
| `SRC` / `F` | mapping ref present; pass/fail fixtures behave as declared |

**Fixtures**: 2 PASS (mechanism + empirical witness) + 2 FAIL. **Self-test**:
deterministic, integer-only, pure stdlib.

## Primary Sources

- Shoeb, A.H. (2009). "Application of Machine Learning to Epileptic Seizure Onset
  Detection and Treatment." MIT PhD thesis. (CHB-MIT Scalp EEG Database)
- Goldberger, A.L. et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet."
  *Circulation* 101(23):e215-e220. DOI 10.1161/01.CIR.101.23.e215
- Soffer, B.H. et al. (1986). *Opt. Lett.* 11(2):118-120. DOI 10.1364/OL.11.000118
- Owechko, Y. (1987). *IEEE J. Quantum Electron.* 25(3):619-634.

## Extension — richer per-band features (2026-07-09)

The cert uses a single broadband log-power per channel. Extending the topographic
phase vector to **per-band power (delta/theta/alpha/gamma per channel)** — 92-dim
vs 23-dim — materially improves recall and robustness (`qa_eeg_multiband_recall.py`,
chb10/chb23):

| feature | clean | 80% dropout | φ=6 phase-lock |
|---|---|---|---|
| broadband (23-d) | 0.92 / 1.00 | 0.70 / 0.92 | 0.81 / 0.71 |
| **multiband (92-d)** | **1.00 / 1.00** | **0.92 / 0.92** | **0.94 / 0.90** |

The 4× dimension gives more redundancy where broadband had headroom (extreme
dropout, systemic artifact). Naive classification stays fooled under the artifact
regardless of features (≈0.36–0.38) — phase-lock provides the robustness,
consistent with the certified mechanism. Preliminary (2 patients); the mechanism
and cert claim are unchanged.

## Companion

- Cert **[519]** (the phase-conjugate associative memory) and **[518]** (the exact
  operator + distortion-correction theorem) this application composes.
- Reference impl: `qa_eeg_phase_conjugate_recall.py` (broadband),
  `qa_eeg_multiband_recall.py` (per-band extension). Loader fix in
  `eeg_orbit_observer_comparison.py`.

**Author**: Will Dale + Claude, 2026-07-08 (multiband extension 2026-07-09).
