# Cert Family [485]: QA Witt Tower Z/27Z Orbit Stability Asymmetry

**Family ID**: 485
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_orbit_stability_asymmetry_cert_v1/qa_witt_tower_orbit_stability_asymmetry_cert_validate.py`
**Validated**: 2026-06-20
**MOD**: 27 (Witt Tower)
**Type**: Pure mathematics — exhaustive computation over all 729 pairs in Z/27Z

## Claim

In Z/27Z (1-indexed states {1,...,27}), with QA step b'=e, e'=((b+e−1)%27)+1 and
a=b+2e (A2, always derived, never assigned independently):

**(A)** Singularity-type pairs (a≤6): **6/729 pairs (0.82%)** — tiny minority.  
After 1 QA step, **5/6 (83.3%) escape** the a≤6 region.  
After k=27 steps, Sing-type trajectories converge to **mean a=45.5**.

**(B)** Cosmos-type pairs (a≥58): **156/729 pairs (21.4%)** — larger class.  
After 1 QA step, **100/156 (64.1%) escape** the a≥58 region.  
After k=27 steps, Cosm-type trajectories converge to **mean a=42.4**.

**(C)** Long-run asymmetry: Sing-type trajectories converge to **mean a=45.5**, which is
**+3.1 higher** than Cosm-type (42.4), despite starting much lower. The bottom drives up
more than the top sustains itself.

This is the **pure-math structural analog** of the empirical crypto finding:
- Crash-reversion (a≤6 signal, cert [482]): BTC +0.847%, ETH +1.771%
- Momentum (a≥58 signal, cert [483]): BTC +0.254%, ETH null
- Crash-reversion is 3.34× (BTC) and 11.3× (ETH) stronger than momentum

## Method

Exhaustive computation over all 729 pairs (b,e) ∈ {1,...,27}²:
1. Identify Sing-type (a=b+2e ≤ 6) and Cosm-type (a≥58) subsets
2. Apply one QA step and measure escape rates
3. Evolve k=27 steps and measure convergence of mean a
4. Compute orbit periods for all 729 pairs

## Orbit Period Structure

| Class | Orbit Periods |
|---|---|
| Sing-type (N=6) | {72} — all on the 72-period Cosmos orbit |
| Cosm-type (N=156) | {1, 8, 24, 72} — spans all orbit types |

**Notable**: All 6 Sing-type pairs lie on the period-72 orbit. They're at the LOW-a
tail of the largest orbit, not trapped in a small attractor.

**Notable**: The Cosm-type class includes the **unique fixed point (27,27)** (period=1),
which is the Singularity fixed point in the 1-indexed system (27+2×27=81, highest
possible a-coordinate). So the true mathematical Singularity is at the TOP of the a range,
not the bottom. The naming "Singularity-type" for low-a pairs follows the empirical
analogy from cert [482], not strict mathematical correspondence.

## Asymmetry Explained

Why does the bottom drive up more than the top sustains itself?

- Sing-type pairs (a≤6) can only gain a-coordinate after each step (minimum is 1+2=3;
  the step function pushes them toward the orbit equilibrium at a≈35-45)
- Cosm-type pairs (a≥58) include pairs near the high extreme that fall toward equilibrium
  AND the fixed point (27,27) which never moves. The fixed point anchors the Cosm-type
  mean near the equilibrium from above.
- After k=27 steps both converge near equilibrium (a≈40-46 range), but Sing-type starts
  lower and gets "bounced up" more energetically.

## Note on 0-indexed vs 1-indexed

The empirical crypto certs [482]/[483] use 0-indexed return-rank bins ∈ {0,...,26}
(floor(rank×27/N)). In 0-indexed {0,...,26}², the counts differ:
- Sing-type (a≤6): 16 pairs (2.19%)
- Cosm-type (a≥58): 121 pairs (16.6%)

This cert uses the canonical **A1-compliant 1-indexed system** {1,...,27}.
The asymmetry structure (Sing escapes faster AND converges higher) holds in both systems.

## Theorem NT Compliance

Pure arithmetic over Z/27Z — no observer projections. Theorem NT is not applicable
(no physical signal enters or exits). All operations are discrete integer arithmetic.

QA axiom compliance:
- **A1**: states in {1,...,27}; step e'=((b+e−1)%27)+1 is always ≥1 ✓
- **A2**: a=b+2e always derived; never assigned independently ✓
- **S1**: no `**2` used ✓
- **S2**: b, e are `int` throughout ✓
- **T1**: k = integer step count ✓

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 n_sing == 6 | exact | 6 | PASS |
| C2 sing_escape_rate >= 0.70 | ≥ 70% | 83.3% | PASS |
| C3 n_cosm == 156 | exact | 156 | PASS |
| C4 cosm_escape_rate >= 0.55 | ≥ 55% | 64.1% | PASS |
| C5 sing_k27 > cosm_k27 | sing lands higher | 45.5 > 42.4 | PASS |
| C6 sing_escape > cosm_escape | sing escapes faster | 0.833 > 0.641 | PASS |

## Primary Sources

- Hardy GH & Wright EM (2008). An Introduction to the Theory of Numbers. 6th ed. Oxford.
  doi:10.1093/oso/9780199219865.001.0001
- Wildberger NJ (2005). Divine Proportions: Rational Trigonometry to Universal Geometry.
  Wild Egg Books. ISBN 978-0-9757492-0-8

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, three-orbit partition)
- **[482]** Crypto return-rank crash-reversion (empirical analog of Sing-type escape)
- **[483]** Crypto momentum asymmetry (empirical analog of Cosm-type persistence)
