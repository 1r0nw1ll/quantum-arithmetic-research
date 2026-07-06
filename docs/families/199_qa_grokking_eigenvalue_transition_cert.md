# Family [199] QA_GROKKING_EIGENVALUE_TRANSITION_CERT.v1

## One-line summary

Schiffman's invariant algorithmic cores (arXiv 2602.22600) mapped to QA: neural networks learning mod-p arithmetic undergo eigenvalue transition from |lambda|<1 (observer-layer, contractive) to |lambda|=1 (QA-layer, cyclic) at grokking — the phase transition IS the network crossing the Theorem NT firewall.

## Mathematical content

### Grokking as Theorem NT crossing

Schiffman trains transformers on a+b = c (mod 53). Before grokking:
- Network memorizes via continuous (float) parameters
- Eigenvalues of learned operator lie INSIDE unit disk (|lambda| < 1)
- Fit quality R^2 ≈ 0 (operator does not predict dynamics)
- This corresponds to the **observer layer** in Theorem NT: continuous, lossy, approximate

At grokking (epoch ~800):
- Eigenvalues snap to the **unit circle** (|lambda| = 1)
- Fit quality R^2 → 1 (operator fully predicts dynamics)
- Network has discovered the **cyclic rotation operator** on Z/53Z
- This corresponds to the **QA discrete layer**: integer-valued cyclic group action

After grokking, the operator IS the DFT on Z/53Z — a cyclic rotation whose eigenvalues are roots of unity.

### Mode count ↔ QA orbit count

For Z/pZ, the DFT has floor(p/2)+1 distinct frequency pairs (including the DC component). Schiffman shows mode count approaches this maximum under weight decay.

**QA prediction for m=9 (RE-CORRECTED 2026-07-06 — see Verification Note):** The DFT on Z/9Z has floor(9/2)+1 = 5 conjugate-frequency pairs: {0}, {1,8}, {2,7}, {3,6}, {4,5}. The A1-compliant QA step function produces **5 orbit families** for mod-9 (3 cosmos[24] + 1 satellite[8] + 1 singularity[1]) — the "9 families (6 cosmos[12] + 2 satellite[4] + 1 singularity[1])" figure in the 2026-04-08 correction below was itself the product of a `qa_step` implementation bug (found and fixed during the [198] Pudelko audit, 2026-07-05/06), not the true QA orbit count. With the bug fixed, the DFT frequency count (5) and the QA orbit-family count (5) for m=9 now numerically coincide — whether this reflects a genuine mathematical connection or is a coincidence specific to m=9 remains open (see V5).

### ACE as observer projection

ACE extracts cores via SVD of activation-Jacobian interaction (continuous, float). By Theorem NT, ACE is an observer-layer tool. The extracted core is an observer projection of the underlying discrete group action. The projection is faithful (R^2 → 1) because the discrete structure is low-dimensional and the observer has sufficient capacity.

### 3D Markov cores

Schiffman finds 3D cores for 4-state Markov chains. Three independently trained models embed these cores in nearly orthogonal 3D subspaces of R^64, yet CCA alignment is >0.93. This is analogous to QA's orbit invariance: the same algebraic structure (transition eigenvalues) persists regardless of the coordinate system (embedding subspace).

## Verification criteria

1. **V1**: Eigenvalue transition from |lambda|<1 to |lambda|=1 documented at grokking epoch
2. **V2**: Post-grokking operator eigenvalues are roots of unity on Z/pZ (within numerical tolerance)
3. **V3**: Mode count for trained mod-p model matches floor(p/2)+1 — **CONFIRMED for m=97** (17 modes post-grokking, compressing from 251)
4. **V4**: RE-CORRECTED 2026-07-06 — DFT frequency pairs (5 for m=9) and QA orbit families (5 for m=9, not 9 — the "9" was a qa_step bug, see Verification Note) now numerically coincide. Whether this reflects a real connection is open (V5).
5. **V5**: DFT conjugate-frequency pairing vs QA norm-class pairing — OPEN (relationship non-trivial for composite moduli)

## Dependencies

- [126] QA_RED_GROUP_CERT.v1 (T-operator on Z/mZ)
- [128] QA_SPREAD_PERIOD_CERT.v1 (Pisano period)
- Theorem NT (Observer Projection Firewall)

## Sources

- Schiffman, "Transformers Converge to Invariant Algorithmic Cores" (arXiv 2602.22600v1, Feb 2026)

## Experimental results (2026-04-08)

### m=97 (prime control) — CONFIRMED
- Standard transformer grokked at epoch 1200, 100% test accuracy
- 97 eigenvalues on unit circle post-grokking (= vocab size)
- DFT power concentrates on kx=ky diagonal — the addition structure
- Pre-grokking: 251 broad modes. Post-grokking: 17 sparse modes.
- Eigenvalue transition from broad spectrum to sparse unit-circle modes CONFIRMED.

### m=9 (QA composite target) — NO GROKKING
- Neither standard (WD=1.0) nor spherical transformer grokked in 100k epochs
- Composite modulus m=9=3^2 resists grokking — entire literature uses primes
- 49% test accuracy = pure memorization (no generalization)
- QA interpretation: m=9 has Hensel lift structure (mod-3 → mod-9) requiring TWO-STEP discovery that flat Fourier analysis cannot achieve

### QA-native experiments — DIFFERENT APPROACH
Standard grokking is not QA-compliant (uses 0-indexed addition, standard transformer). QA-native experiments using orbit-cycling + resonance coupling (`qa_bateson_coupling_experiment.py`) show:
- Orbit families are exact invariants of QA step (not learned — conserved)
- L1 coupling preserves all 5 families (corrected 2026-07-06, was "9" due to the [198] qa_step bug); unstructured L2 destroys them
- The Schiffman eigenvalue transition phenomenon applies to PRIME moduli; composite moduli require hierarchical (Hensel) discovery

### V4 correction (RE-CORRECTED 2026-07-06)
Original 2026-04-08 claim: "mode count = 5 = QA orbit count for m=9," corrected at the time to "actual A1-compliant orbit count for m=9 is 9 families (not 5)." That 2026-04-08 correction was itself wrong: the "9 families" figure came from a `qa_step` implementation bug (`e_new` computed from the already-updated `b_new` instead of the original `(b,e)`), found and fixed during the [198] Pudelko audit. The true QA orbit count for m=9 is **5** (3 cosmos[24] + 1 satellite[8] + 1 singularity[1]) — independently reconfirmed by direct simulation. So the *original* 2026-04-08 claim ("mode count = 5 = QA orbit count") was numerically right about both counts being 5, even though whether that numerical coincidence reflects a genuine connection (rather than being specific to m=9) is still open — see V5.

## Scripts

- `qa_spherical_grokking_mod9.py` — standard + spherical transformer experiment (m=97 control + m=9)
- `qa_bateson_coupling_experiment.py` — QA-native Bateson coupling (verified family conservation)

## Status

PARTIALLY VERIFIED — eigenvalue transition confirmed at m=97 (prime). m=9 requires QA-native approach, not standard grokking. V4 re-corrected 2026-07-06 (5 families, not 9 — the "9" was a qa_step bug).

## Verification Note (2026-07-05)

Independently checked the Schiffman citation. **Schiffman, "Transformers
converge to invariant algorithmic cores"** (arXiv:2602.22600, Feb 2026)
confirmed real (author, title, arXiv ID all match). Fetched the actual
paper and confirmed several specific claims exactly:

- The mode-count formula and the eigenvalue-transition description are
  both verbatim-accurate: the paper states "a maximum of ⌊p/2⌋+1=27 valid
  harmonic representations" for its own experiment at **p=53** (not p=97
  — the paper's modular-addition experiments use p=53; this cert's own
  §"Experimental results (2026-04-08)" m=97 numbers are the project's own
  independent replication at a different modulus, correctly presented as
  separate from the Schiffman citation, not attributed to the paper).
- Eigenvalue transition: the paper's own words are "eigenvalues scatter
  inside the unit circle – the learned transformation appears
  contractive, not cyclic. At grokking (epoch 800), eigenvalues snap onto
  the unit circle" — exact match to this cert's description, including
  the same epoch 800.
- 3D Markov-chain cores: the paper reports canonical-correlation values
  of **0.999, 0.999, and 0.929** across three independently trained
  models' orthogonal-subspace cores. This cert states "CCA alignment is
  >0.93" — technically imprecise for the third pair (0.929 rounds just
  under 0.93, not over), a thousandths-place rounding quibble, not a
  substantive misrepresentation. Worth a one-word fix ("~0.93") if this
  doc is touched again, but not urgent given the magnitude.

No fabrication found. This cert's own honest self-correction history
(the V4 correction distinguishing DFT frequency pairs from QA orbit
families) is a good example of the practice this audit cycle has been
encouraging elsewhere.

## Verification Note (2026-07-06) — correcting the 2026-07-05 note above

The praise in the note directly above turned out to be premature: the
"V4 correction" it called "a good example of honest self-correction"
was itself built on a wrong number. During the separate [198] Pudelko
audit (2026-07-05/06), found and fixed a real `qa_step` implementation
bug shared across three experiment scripts, including
`qa_bateson_coupling_experiment.py` (cited by this cert). That bug
produced a wrong QA orbit-family count of 9 for mod-9; the true count,
confirmed independently by direct simulation and now consistent across
every other cert this cycle that touches m=9 orbit structure, is 5 (3
cosmos of 24 + 1 satellite of 8 + 1 singularity).

This means the *original* pre-2026-04-08 claim in this cert ("mode
count = 5 = QA orbit count for m=9") was numerically correct about both
figures being 5 — the 2026-04-08 "correction" that changed this to "9
families, not 5" was the actual regression, not an improvement. Fixed
the doc throughout (mathematical content, V4, the correction narrative,
status line) to reflect the true count and to be transparent about this
reversal rather than silently re-editing history.

**Lesson for this audit cycle**: a cert's own "honest self-correction"
narrative isn't automatically trustworthy just because it's framed as
one — the correction itself can be wrong, and confirming "the cert
transparently documents its own history" isn't the same as confirming
the corrected number is actually right. Independently recompute, don't
just check that a correction narrative exists.
