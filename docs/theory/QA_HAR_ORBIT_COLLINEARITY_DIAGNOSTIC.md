# QA HAR × Orbit-Dummy Collinearity Diagnostic

Primary source: Corsi (2009) HAR-RV; Observer 2 cascade specification in `preregistration_observer2.md` (committed SHA 99201b9, 2026-04-16).

Training fold only (first 60 % of SPY log-RV, 1993-02-01 through row 5014, n_train = 4992). Test fold NEVER touched. Matches Observer 2 index alignment exactly.

Training-fold orbit counts per scale (diagnostic sanity check): {1: {'singularity': 126, 'satellite': 479, 'cosmos': 4387}, 5: {'singularity': 460, 'satellite': 571, 'cosmos': 3961}, 22: {'singularity': 530, 'satellite': 882, 'cosmos': 3580}}.

## Section 1 — Numerical results

### A. Individual dummy R² vs classical HAR features

Model for each row: D_i = β₀ + β_d · log_RV_t + β_w · log_RV_t^(w) + β_m · log_RV_t^(m) + ε.

| Dummy | n_on (of 4992) | mean | R² | adj R² |
| --- | ---: | ---: | ---: | ---: |
| sat1 | 479 | 0.0960 | 0.0093 | 0.0087 |
| sing1 | 126 | 0.0252 | 0.1094 | 0.1089 |
| sat5 | 571 | 0.1144 | 0.0169 | 0.0163 |
| sing5 | 460 | 0.0921 | 0.3461 | 0.3457 |
| sat22 | 882 | 0.1767 | 0.0108 | 0.0102 |
| sing22 | 530 | 0.1062 | 0.4160 | 0.4157 |
| int_sat15 | 64 | 0.0128 | 0.0015 | 0.0009 |
| int_sing15 | 96 | 0.0192 | 0.1056 | 0.1051 |

### B. VIF inside the 12-variable augmented regression

Standard VIF. VIF > 10 = multicollinearity concern; VIF > 100 = near-perfect; reference thresholds are observer-side conventions.

| Variable | VIF |
| --- | ---: |
| daily | 1.2909 |
| weekly | 3.3413 |
| monthly | 3.5441 |
| sat1 | 1.1719 |
| sing1 | 4.2180 |
| sat5 | 1.1510 |
| sing5 | 2.1713 |
| sat22 | 1.0320 |
| sing22 | 2.2540 |
| int_sat15 | 1.2818 |
| int_sing15 | 4.4254 |

### C. Partial R² of the 8-dummy block beyond HAR

In-sample fits on training fold only.

| Model | params | R² | adj R² | SSR |
| --- | ---: | ---: | ---: | ---: |
| M1: HAR (const + d + w + m) | 4 | 0.1049 | 0.1044 | 37407.242 |
| M2: HAR + 8 orbit dummies | 12 | 0.1062 | 0.1042 | 37354.526 |

Partial R² of dummy block = (R²_M2 − R²_M1) / (1 − R²_M1) = **0.0014**.

F-test of joint significance of the 8 dummies: F(8, 4980) = 0.8785, p = 0.5337.

## Section 2 — Interpretation

**Test A (individual dummy R²).** Maximum dummy R² vs HAR features = 0.4160 (sing22), second-highest 0.3461 (sing5). Partition across the 8 dummies: 0 with R² ≥ 0.9 (near-perfect redundancy), 0 in 0.5–0.9 (substantial overlap), 8 below 0.5. Six of the eight sit below R² = 0.11. The two singularity-at-longer-scales dummies (sing5, sing22) show moderate linear predictability from HAR — driven by the structural fact that a long-scale rolling mean stuck in the same decile maps to singularity. Even so, none of the 8 dummies is linearly reconstructible from HAR. At the individual-dummy level the Observer-2 agent's hypothesis ("orbit dummies are a nonlinear re-coding of information already in HAR") is **not supported**: the dummies carry information HAR does not linearly capture.

**Test B (VIF in augmented model).** Maximum VIF across the 12 variables = 4.43 (int_sing15); all 12 VIFs sit below 5. Zero variables exceed the 10 threshold, let alone the 100 near-perfect threshold. There is no multicollinearity in the augmented regression. A collinearity-driven Observer-2 NULL is ruled out.

**Test C (partial R² of the 8-dummy block beyond HAR; F-test).** Partial R² = 0.00141; F(8, 4980) = 0.88, p = 0.534. The 8-dummy block is **not jointly significant even in-sample** on log_RV_{t+1}. This is the controlling finding. Individual dummies are orthogonal to HAR in their OWN domain (predicting whether an orbit class is on), but they do not predict next-step log-RV beyond what HAR already extracts. The Observer-2 agent's hypothesis gets the right verdict (dummies don't help) for the wrong reason: it is not collinearity absorption — it is that the orbit-class indicator, once HAR has set the conditional mean of log-RV, carries no additional information about where log-RV goes next. The signal in the orbit labels is about log-RV's LEVEL (high-vol regimes map to specific orbit classes), not about its FORECAST DIRECTION.

## Section 3 — Observer 3 design implication

**Verdict: Observer-2 agent hypothesis REFUTED as stated; but the NULL is real and structural, not an artefact.**

Three load-bearing facts for Observer 3:

1. Orbit dummies are NOT collinear with HAR (max VIF 4.4, max individual-dummy R² 0.42, so six of eight are effectively orthogonal). The "HAR absorbs the regime signal" story Observer 2's write-up offered is mechanistically incorrect.
2. The dummies nevertheless contribute zero forecast information for log_RV_{t+1} beyond HAR, in-sample (partial R² = 0.001, F p = 0.53). The NULL is not a generalisation failure — the signal is absent even on the training fold.
3. The moderate R² on sing5 / sing22 shows HAR linearly predicts *when orbit classes are assigned*, not *what happens after they are assigned*. Orbit class is a side-product of the HAR surface, not a predictor living above it.

Implication: Observer 3's premise ("cross-asset info is orthogonal to within-asset HAR, so cross-asset dummies will add forecast value") is **underdetermined** by this diagnostic. Within-asset orthogonality was already present — it just did not translate to forecast content. Cross-asset synchrony inherits the same risk: it can be orthogonal to HAR (plausible) and simultaneously carry zero forecast information for next-step log-RV (the observed pattern here).

Recommended Observer 3 pre-reg amendments:

- **Training-fold F-test gate.** Before running the test-fold DM, require the cross-asset synchrony block to be jointly significant on the training fold at p < 0.01. This is the minimum bar Observer 2's dummies failed (p = 0.53). If Observer 3's cross-asset block also fails this gate, Observer 3 is a NULL without consuming a test-fold observation.
- **Partial R² floor.** Require partial R² of the cross-asset block ≥ 0.01 on the training fold. Observer 2's was 0.0014 — a 7× floor filters the "orthogonal but uninformative" trap.
- **Target log-RV at scale k = 22.** The per-scale singularity enrichment grew monotonically with scale (Observer 2 finding: z = +60 at k = 22 vs +8 at k = 1). If Observer 3 forecasts daily log-RV as Observer 2 did, it reproduces the same scale-mismatch: the signal lives at the monthly scale. Switch the forecast target to monthly-averaged log-RV (or forecast daily log-RV with a cross-asset monthly-synchrony feature) so the signal bandwidth aligns with the feature bandwidth.

Proceeding with cross-asset synchrony without the first two gates risks a third consecutive NULL that does not inform the ladder because it is underpowered in exactly the way Observer 2 was underpowered — the signal you're testing doesn't live in the target's conditional mean.

